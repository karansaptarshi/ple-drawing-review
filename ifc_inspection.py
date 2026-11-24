"""Inspect authorized task files with the bundled IFC parser."""
import asyncio
import base64
import json
from pathlib import Path


async def inspect_ifc(session, args):
    path = args.get("path")
    class_name = args.get("class_name", "")
    if not isinstance(path, str) or not path or not isinstance(class_name, str):
        return "error: provide an IFC path and a class_name string"
    source = await session.read_result(path)
    if not source.get("ok"):
        return "error: the evaluator could not read this IFC file"
    if source.get("truncated"):
        return "error: export a smaller spatial subset of this model for inspection"
    payload = base64.b64decode(source["content_b64"], validate=True)
    if not payload.lstrip().startswith(b"ISO-10303-21;"):
        return "error: expected an uncompressed IFC STEP file"
    process = await asyncio.create_subprocess_exec(
        "node", str(Path(__file__).with_name("ifc-geometry.cjs")), class_name,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=40)
    except asyncio.TimeoutError:
        return "error: geometry inspection exceeded 40 seconds; try a smaller spatial subset"
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode or not stdout:
        return "error: IFC parser could not read the geometry. " + stderr.decode(errors="replace")[-300:]
    report = json.loads(stdout)
    if not report["world"]:
        return "No triangulated physical elements were found in this IFC file."
    return json.dumps(report, separators=(",", ":"))
