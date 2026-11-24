"""Generate and verify an explicitly requested, short, single-sheet answer file."""

import asyncio
import re
from pathlib import PurePosixPath

import pypdfium2 as pdfium

from drawings import evidence, pdf_bytes, printed_lines, render
from drawing_table import ruled_table

PATH_RE = re.compile(r"(?<![\w./])(/[\w./-]+\.(?:pdf|txt))(?=[\s.,;:)\"'`]|$)", re.I)
OUTPUT_REQUEST = re.compile(
    r"\b(?:write|record|save|return|place|put|store)\b.{0,115}\b(?:in|at|to|as|into)\s+[`'\"]?(?P<path>/[\w./-]+\.txt)\b",
    re.I | re.S,
)
FORMAT_REQUEST = re.compile(
    r"\b(?:only|single|just)\b.{0,42}\b(?:answer|value|number|letter|dimension|identifier|code)\b|\b(?:answer|value|number|letter|dimension|identifier|code)\s+only\b",
    re.I,
)


def file_pair(instruction):
    paths = set(PATH_RE.findall(instruction))
    request = OUTPUT_REQUEST.search(instruction)
    if len(paths) != 2 or request is None or FORMAT_REQUEST.search(instruction) is None:
        return None
    output = request.group("path")
    inputs = [path for path in paths if path.lower().endswith(".pdf")]
    if (
        len(inputs) != 1
        or output not in paths
        or any(".." in PurePosixPath(p).parts for p in paths)
    ):
        return None
    return inputs[0], output


def views(data, question):
    with pdfium.PdfDocument(data) as document:
        if len(document) != 1:
            return []
    lines = printed_lines(data, 1)
    if re.search(r"\b(?:title block|drawing number)\b", question, re.I):
        corner = re.search(
            r"\b(lower|upper|bottom|top)[ -](left|right)\b", question, re.I
        )
        if corner:
            l, r = (590, 1000) if corner[2].lower() == "right" else (0, 410)
            t, b = (650, 1000) if corner[1].lower() in ("lower", "bottom") else (0, 350)
            return [render(data, 1, [l, t, r, b])]
    table = ruled_table(data, question, lines)
    if table is not None:
        return [table]
    detail = evidence(data, 1, question, lines)
    # Include the neighboring callout at reading scale, including at the page edges.
    output = [detail]
    marker = re.search(r"Selected printed label: (.*?)\. Nearby OCR", detail.caption)
    if marker and re.search(
        r"\b(?:detail (?:letter|identifier)|identifying letter|callout)\b",
        question,
        re.I,
    ):
        for label, (left, top, right, bottom) in lines:
            if label == marker.group(1):
                bounds = [
                    max(0, left - 85),
                    max(0, top - 28),
                    min(1000, right + 110),
                    min(1000, bottom + 30),
                ]
                output = [render(data, 1, bounds)]
                output[
                    0
                ].caption += f" Detail caption and adjacent callouts near {label}."
                break
    return output


async def try_short_drawing(session, model, client):
    pair = file_pair(session.instruction)
    if not pair or session.mcp_servers:
        return None
    source, output = pair
    available = session.time_left()
    if available is not None and available < 30:
        return None
    try:
        data = await pdf_bytes(session, source)
        images = await asyncio.to_thread(views, data, session.instruction)
        if not images:
            return None
        content = [
            {
                "type": "text",
                "text": session.instruction
                + "\nReturn exactly the requested value for the file. Write UNKNOWN if the evidence is insufficient.",
            }
        ]
        for view in images:
            content.extend([{"type": "text", "text": view.caption}, view.model_image()])
        available = session.time_left()
        timeout = 100 if available is None else min(100, max(1, available * 0.55))
        reply = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=256,
            ),
            timeout,
        )
        if reply.usage:
            await session.usage(
                model,
                reply.usage.prompt_tokens,
                reply.usage.completion_tokens,
                getattr(
                    getattr(reply.usage, "prompt_tokens_details", None),
                    "cached_tokens",
                    0,
                )
                or 0,
            )
        message = reply.choices[0].message.content or ""
        value = message.strip()
        if (
            not value
            or len(value) > 90
            or "\n" in value
            or value.lower() in {"unknown", "unclear", "n/a"}
        ):
            return None
        if re.search(
            r"(?:identifying|single|only (?:its|the)) letter\b",
            session.instruction,
            re.I,
        ) and not re.fullmatch(r"[A-Za-z]", value):
            return None
        wanted = (value + "\n").encode()
        await session.write_file(output, wanted)
        observed = await session.read_file(output)
        if observed != wanted:
            return None
        return f"Saved the drawing answer in {output} and verified the file contents."
    except (ValueError, OSError, TimeoutError):
        return None
