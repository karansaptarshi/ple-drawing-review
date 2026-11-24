"""Read engineering drawings through the PLE file protocol and return focused views."""

import base64
import ctypes
import json
import math
import re
import shlex
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pypdfium2 as pdfium
import taskbridge
import tesserocr
from pypdfium2 import raw


@dataclass
class DrawingView:
    caption: str
    jpeg: bytes

    def model_image(self):
        data = base64.b64encode(self.jpeg).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + data},
        }


async def pdf_bytes(session, path):
    first = await session.read_result(path)
    if not first.get("ok"):
        raise ValueError("The evaluator could not read this drawing.")
    if not first.get("truncated"):
        return base64.b64decode(first["content_b64"], validate=True)
    total = first["size"]
    if total > 64 * 1024 * 1024:
        raise ValueError(
            "Extract the required pages to a smaller PDF in the task container, then open it."
        )
    # The protocol can read 10 MB at once. Split this file inside the task's
    # temporary directory; the model never sees the binary chunks or their paths.
    part_size = 6 * 1024 * 1024
    folder = "/tmp/ple-sheet-" + uuid4().hex
    script = (
        "from pathlib import Path\nimport sys\n"
        "target=Path(sys.argv[2]); target.mkdir(mode=0o700)\n"
        'with open(sys.argv[1], "rb") as source:\n'
        " for i in range(int(sys.argv[4])):\n"
        "  (target/str(i)).write_bytes(source.read(int(sys.argv[3])))\n"
        ' if source.read(1): raise ValueError("source changed")\n'
    )
    count = math.ceil(total / part_size)
    try:
        split = await session.exec(
            shlex.join(
                ["python3", "-c", script, path, folder, str(part_size), str(count)]
            )
        )
        if split["exit_code"] != 0:
            raise ValueError(
                "Large PDF could not be split. Extract the requested page and retry."
            )
        content = bytearray()
        for i in range(count):
            content.extend(await session.read_file(f"{folder}/{i}"))
        if len(content) != total:
            raise ValueError("The drawing changed while being read.")
        return bytes(content)
    finally:
        if not session.cancelled.is_set():
            await session.exec(shlex.join(["rm", "-rf", "--", folder]))


def open_page(doc, number):
    if type(number) is not int or not 1 <= number <= len(doc):
        raise ValueError(f"Page must be between 1 and {len(doc)}.")
    return doc[number - 1]


def render(data, number, region, max_side=1536):
    if (
        not isinstance(region, list)
        or len(region) != 4
        or any(type(v) is not int for v in region)
    ):
        raise ValueError(
            "Use [left, top, right, bottom] as four integers from 0 to 1000."
        )
    l, t, r, b = region
    if not (0 <= l < r <= 1000 and 0 <= t < b <= 1000):
        raise ValueError(
            "The region must be a positive rectangle inside the displayed page."
        )
    with pdfium.PdfDocument(data) as doc:
        with closing(open_page(doc, number)) as page:
            w, h = page.get_size()
            scale = min(12, max_side / max(w * (r - l) / 1000, h * (b - t) / 1000))
            crop = (
                w * l / 1000,
                h * (1000 - b) / 1000,
                w * (1000 - r) / 1000,
                h * t / 1000,
            )
            with closing(
                page.render(
                    scale=scale, crop=crop, rev_byteorder=True, draw_annots=True
                )
            ) as bitmap:
                image = bitmap.to_pil().convert("RGB")
                encoded = BytesIO()
                image.save(
                    encoded, format="JPEG", quality=90, subsampling=0, optimize=True
                )
                return DrawingView(
                    f"Page {number}/{len(doc)}, region {region}, {image.width} x {image.height} pixels.",
                    encoded.getvalue(),
                )


def printed_lines(data, number):
    from PIL import Image

    sheet = render(data, number, [0, 0, 1000, 1000], max_side=3000)
    with (
        Image.open(BytesIO(sheet.jpeg)) as image,
        tesserocr.PyTessBaseAPI(
            path=str(Path(__file__).resolve().parent / "tessdata"),
            lang="eng",
            psm=tesserocr.PSM.SPARSE_TEXT,
        ) as api,
    ):
        api.SetImage(image)
        api.Recognize()
        iterator = api.GetIterator()
        level = tesserocr.RIL.TEXTLINE
        lines = []
        if iterator is None:
            return lines
        for item in tesserocr.iterate_level(iterator, level):
            if item.Empty(level):
                continue
            content = (item.GetUTF8Text(level) or "").strip()
            if not content:
                continue
            l, t, r, b = item.BoundingBox(level)
            box = [
                math.floor(1000 * l / image.width),
                math.floor(1000 * t / image.height),
                math.ceil(1000 * r / image.width),
                math.ceil(1000 * b / image.height),
            ]
            lines.append((content, box))
        return lines


STOP_WORDS = set(
    "a an and are at by for from give how in into is it many of on only or sheet that the to what which with".split()
)


def terms(text):
    words = re.findall(r"[a-z]+|[0-9]+", text.lower())
    return {
        w.removesuffix("s") if len(w) > 3 else w for w in words if w not in STOP_WORDS
    }


def evidence(data, number, question, lines=None):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 800:
        raise ValueError(
            "Supply a drawing question or descriptive phrase of up to 800 characters."
        )
    lines = printed_lines(data, number) if lines is None else lines
    query = terms(question)
    indexed = [(value, box, terms(value)) for value, box in lines]
    frequency = Counter(term for _, _, tokens in indexed for term in tokens)
    ranked = []
    for value, box, tokens in indexed:
        hits = query & tokens
        if hits:
            score = sum(
                math.log(1 + (len(lines) + 1) / (frequency[token] + 1))
                for token in hits
            )
            ranked.append((score * (1 + 0.2 * (len(hits) - 1)), value, box))
    if not ranked:
        sheet = render(data, number, [0, 0, 1000, 1000])
        sheet.caption += " No matching printed label. Inspect the full sheet."
        return sheet
    ranked.sort(key=lambda row: -row[0])
    _, best, (l, t, r, b) = ranked[0]
    crop = [max(0, l - 85), max(0, t - 105), min(1000, r + 85), min(1000, b + 105)]
    view = render(data, number, crop)
    nearby = [
        value
        for value, (x, y, u, v) in lines
        if crop[0] <= (x + u) / 2 <= crop[2] and crop[1] <= (y + v) / 2 <= crop[3]
    ]
    alternatives = [{"text": value[:200], "box": box} for _, value, box in ranked[1:5]]
    view.caption += (
        " Selected printed label: "
        + best[:200]
        + ". Nearby OCR may contain errors: "
        + " | ".join(nearby)[:2000]
        + ". Other possible locations: "
        + json.dumps(alternatives)
    )
    return view


def displayed_point(page, x, y):
    col, row = ctypes.c_int(), ctypes.c_int()
    ok = raw.FPDF_PageToDevice(
        page, 0, 0, 1000, 1000, 0, x, y, ctypes.byref(col), ctypes.byref(row)
    )
    if not ok:
        raise ValueError("Unable to map drawing coordinates.")
    return col.value, row.value


def locate(data, term, number):
    if not isinstance(term, str) or not term.strip() or len(term) > 160:
        raise ValueError("Provide a drawing label of 1 to 160 characters.")
    results = []
    with pdfium.PdfDocument(data) as doc:
        if type(number) is not int or not 0 <= number <= len(doc):
            raise ValueError(
                f"Use 0 to search all pages or a page from 1 to {len(doc)}."
            )
        for n in [number] if number else range(1, len(doc) + 1):
            with (
                closing(open_page(doc, n)) as page,
                closing(page.get_textpage()) as words,
                closing(words.search(term)) as hits,
            ):
                while (match := hits.get_next()) is not None:
                    corners = []
                    for i in range(words.count_rects(*match)):
                        left, bottom, right, top = words.get_rect(i)
                        corners.extend(
                            [
                                displayed_point(page, left, bottom),
                                displayed_point(page, right, top),
                            ]
                        )
                    if corners:
                        xs, ys = zip(*corners)
                        box = [
                            max(0, min(xs)),
                            max(0, min(ys)),
                            min(1000, max(xs)),
                            min(1000, max(ys)),
                        ]
                        if box[0] < box[2] and box[1] < box[3]:
                            results.append({"page": n, "box": box})
                    if len(results) == 24:
                        return json.dumps({"matches": results, "limit_reached": True})
    return json.dumps(
        {
            "matches": results,
            "limit_reached": False,
            "hint": "Use a broader region around the label to see its leader and nearby geometry."
            if results
            else "No matching text layer label. View the page to inspect outlines or scanned lettering.",
        }
    )


async def drawing_tool(session, name, args):
    try:
        data = await pdf_bytes(session, args["path"])
        if name == "view_drawing":
            return render(data, args["page"], args["region"])
        if name == "inspect_drawing_question":
            return evidence(data, args["page"], args["question"])
        return locate(data, args["text"], args["page"])
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        pdfium.PdfiumError,
        taskbridge.TaskError,
    ) as exc:
        return f"Drawing tool error: {exc}"
