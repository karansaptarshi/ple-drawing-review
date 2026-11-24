"""Connect only to app servers named in the current evaluator task."""

import json
import re
from collections import Counter
from uuid import uuid4
import taskbridge


def response_message(data, request_id):
    text = data.decode("utf-8")
    try:
        messages = [json.loads(text)]
    except json.JSONDecodeError:
        messages = []
        for event in text.replace("\r\n", "\n").split("\n\n"):
            lines = [
                line[5:].lstrip(" ")
                for line in event.splitlines()
                if line.startswith("data:")
            ]
            if lines:
                messages.append(json.loads("\n".join(lines)))
    for message in messages:
        if isinstance(message, dict) and message.get("id") == request_id:
            if "error" in message:
                raise ValueError(str(message["error"]))
            return message["result"]
    raise ValueError("No matching response from the task app.")


async def exchange(session, url, headers, method, params=None, notification=False):
    message = {"jsonrpc": "2.0", "method": method}
    if not notification:
        message["id"] = uuid4().hex
    if params is not None:
        message["params"] = params
    raw = await session.http(url, headers=headers, body=json.dumps(message).encode())
    if not 200 <= raw["status"] < 300 or raw.get("truncated"):
        raise ValueError(f"App request failed or was truncated (HTTP {raw['status']}).")
    if notification:
        return None, raw["headers"]
    return response_message(raw["body"], message["id"]), raw["headers"]


async def app_tool(session, name, args):
    servers = {
        s["name"]: s["url"]
        for s in session.mcp_servers
        if s.get("name") and s.get("url")
    }
    key = args["server"]
    if key not in servers:
        return f"Available task app servers: {list(servers)}"
    if not hasattr(session, "app_headers"):
        session.app_headers = {}
    url = servers[key]
    try:
        if key not in session.app_headers:
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
            initial, received = await exchange(
                session,
                url,
                headers,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "drawing-review", "version": "1.0"},
                },
            )
            headers["MCP-Protocol-Version"] = initial["protocolVersion"]
            for header, value in received.items():
                if header.lower() == "mcp-session-id":
                    headers["Mcp-Session-Id"] = value
            await exchange(
                session, url, headers, "notifications/initialized", notification=True
            )
            session.app_headers[key] = headers
        headers = session.app_headers[key]
        if name == "list_app_tools":
            cursor = args.get("cursor", "")
            result, _ = await exchange(
                session,
                url,
                headers,
                "tools/list",
                {"cursor": cursor} if cursor else {},
            )
        else:
            result, _ = await exchange(
                session,
                url,
                headers,
                "tools/call",
                {"name": args["tool"], "arguments": args["arguments"]},
            )
        return json.dumps(result, ensure_ascii=False)
    except (KeyError, ValueError, TypeError, taskbridge.TaskError) as exc:
        # A mutating tool may already have taken effect. Leave any retry to the caller.
        session.app_headers.pop(key, None)
        return f"Task app returned an error: {exc}"


async def named_app_tools(session, reserved_names):
    """Offer explicitly named app tools to the worker with their published schemas."""
    mentioned = set(re.findall(r"[A-Za-z0-9_-]+", session.instruction))
    matches = []
    evidence = []
    for server in session.mcp_servers:
        if not server.get("name") or not server.get("url"):
            continue
        cursor = ""
        seen = set()
        while cursor not in seen and len(seen) < 12:
            seen.add(cursor)
            raw = await app_tool(session, "list_app_tools", {"server": server["name"], "cursor": cursor})
            try:
                page = json.loads(raw)
            except ValueError:
                break
            if not isinstance(page, dict):
                break
            for entry in page.get("tools", []):
                name = entry.get("name")
                schema = entry.get("inputSchema")
                if (name in mentioned and name not in reserved_names
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) and isinstance(schema, dict)):
                    matches.append((name, server["name"], entry.get("description", ""), schema, entry.get("annotations") or {}))
            cursor = page.get("nextCursor")
            if not cursor:
                break
    counts = Counter(row[0] for row in matches)
    schemas, destinations = [], {}
    for name, server, description, schema, annotations in matches:
        if counts[name] != 1:
            continue
        destinations[name] = server
        schemas.append({"type": "function", "function": {
            "name": name, "description": description, "parameters": schema}})
        # Materialize an explicitly requested, parameter-free lookup once. Never
        # prefetch a parameterized operation or one labelled as destructive.
        command = r"\b(?:call|invoke|use|query)\s+(?:(?:the|tool)\s+)*[`\"']?" + re.escape(name) + r"\b"
        by_reference = (len(matches) == 1 and
                        re.search(r"\b(?:call|invoke|query) (?:it|that tool|this tool)\b", session.instruction, re.I))
        is_lookup = annotations.get("readOnlyHint") is True or (
            "readOnlyHint" not in annotations and
            re.match(r"^(?:returns?|reads?|lists?|retrieves?|gets?|looks? up)\b", description.strip(), re.I)
        )
        # Leave conditional or negated app calls to the model.
        conditional = re.search(r"\b(?:if|unless|until|before|after)\b|\b(?:never|not|avoid|don't)\s+(?:call|invoke|use|query)", session.instruction, re.I)
        if (not conditional and not schema.get("properties") and not schema.get("required")
                and not annotations.get("destructiveHint") and is_lookup
                and (by_reference or re.search(command, session.instruction, re.I))):
            raw = await app_tool(session, "call_app_tool", {"server": server, "tool": name, "arguments": {}})
            try:
                result = json.loads(raw)
            except ValueError:
                continue
            if isinstance(result, dict) and not result.get("isError"):
                values = [item["text"] for item in result.get("content", [])
                          if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)]
                if values:
                    evidence.append({"server": server, "tool": name, "returned_text": "\n".join(values)[:4096]})
    return schemas, destinations, evidence


async def verify_literal_export(session, evidence):
    """Verify one explicitly requested plain-text export of one returned scalar."""
    from pathlib import PurePosixPath

    if len(evidence) != 1:
        return None
    text = session.instruction
    # A report that uses a reference number as one ingredient is a different
    # operation. Only these explicit whole-file copy instructions authorize repair.
    literal = re.search(
        r"whose entire content is (?:the |a )?(?:single )?(?:word|value|identifier|code) (?:that )?(?:the tool|it) returns\b"
        r"|\b(?:write|save|copy) the returned (?:word|value|identifier|code) (?:into|to|in)\b[^.!?]*?(?:\.txt)[`'\" ]*(?:as plain text|verbatim|only)\b",
        text, re.I,
    )
    paths = set(re.findall(r"(?<![\w./])(/[\w./-]+\.txt|[\w-]+\.txt)(?![\w/]|[.][\w./])", text, re.I))
    negated = re.search(r"\b(?:do not|don't|never|avoid)\s+(?:write|save|copy|replace|overwrite)\b", text, re.I)
    if not literal or negated or len(paths) != 1:
        return None
    path = PurePosixPath(paths.pop())
    if ".." in path.parts:
        return None
    target = str(path if path.is_absolute() else PurePosixPath(session.workdir) / path)
    value = evidence[0]["returned_text"]
    if not value or len(value) > 256 or any(c.isspace() for c in value):
        return None
    wanted = value.encode("utf-8")
    try:
        saved = await session.read_file(target)
    except taskbridge.TaskError:
        saved = None
    if saved not in (wanted, wanted + b"\n"):
        await session.write_file(target, wanted + b"\n")
        saved = await session.read_file(target)
    if saved not in (wanted, wanted + b"\n"):
        raise ValueError("The requested plain-text export failed readback.")
    return target
