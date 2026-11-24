"""Find ruled dimension tables using repeated horizontal rulings and their OCR headers."""

from io import BytesIO
import re

from PIL import Image
from drawings import render, terms


def ruled_table(data, question, ocr_lines):
    if not re.search(r"\b(?:table|tabulated|row|column)\b", question, re.I):
        return None
    overview = render(data, 1, [0, 0, 1000, 1000], max_side=1800)
    with Image.open(BytesIO(overview.jpeg)) as image:
        gray = image.convert("L")
        mask = gray.point(lambda v: int(v < 140)).tobytes()
        width, height = gray.size
    groups = []
    for y in range(1, height - 1):
        for hit in re.finditer(b"\x01{55,}", mask[y * width : (y + 1) * width]):
            start, end = hit.span()
            if end - start > width * 0.82:
                continue
            matching = next(
                (
                    item
                    for item in groups
                    if abs(item[0] - start) <= 4 and abs(item[1] - end) <= 4
                ),
                None,
            )
            if matching is None:
                groups.append([start, end, [y]])
            elif y - matching[2][-1] > 4:
                matching[2].append(y)
    choices = []
    query = terms(question)
    for x1, x2, rows in groups:
        if len(rows) < 3:
            continue
        # Split tracks when distant sections share the two x coordinates.
        blocks = [[rows[0]]]
        for y in rows[1:]:
            if y - blocks[-1][-1] > height * 0.12:
                blocks.append([])
            blocks[-1].append(y)
        for ys in blocks:
            if len(ys) < 3 or ys[-1] - ys[0] < height * 0.038:
                continue
            box = [
                round(x1 * 1000 / width),
                round(ys[0] * 1000 / height),
                round(x2 * 1000 / width),
                round(ys[-1] * 1000 / height),
            ]
            left, top, right, bottom = box
            labels = [
                s
                for s, (a, b, c, d) in ocr_lines
                if left - 3 <= (a + c) / 2 <= right + 3
                and top - 6 <= (b + d) / 2 <= bottom + 6
            ]
            nearby = [
                s
                for s, (a, b, c, d) in ocr_lines
                if left - 70 <= (a + c) / 2 <= right + 100
                and top - 35 <= (b + d) / 2 <= bottom + 95
            ]
            label_tokens = terms(" ".join(labels))
            local_tokens = terms(" ".join(nearby))
            rank = (
                2 * len(label_tokens & query)
                + len(local_tokens & query)
                + 0.2 * min(len(ys), 9)
            )
            if re.search(r"\b(?:min(?:imum)?|max(?:imum)?)\b", question, re.I):
                rank += 4 * int(
                    bool(label_tokens & {"min", "max", "minimum", "maximum"})
                )
            choices.append((rank, box))
    if not choices:
        return None
    _, (l, t, r, b) = max(choices, key=lambda x: x[0])
    view = render(
        data, 1, [max(0, l - 8), max(0, t - 12), min(1000, r + 8), min(1000, b + 12)]
    )
    view.caption += " Complete ruled table. Read the requested row and column together with the printed headers."
    return view
