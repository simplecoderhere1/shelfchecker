"""
Prototype: detect white spine labels -> crop -> OCR each crop -> order.

This is the architecture we'll port to the browser. We develop it here
(in Python, fast loop, real photos) before touching JS again.

Pipeline:
  1. Find bright, low-saturation (white/cream) blobs = label stickers
  2. Filter by size / aspect to keep label-shaped rectangles
  3. Crop each, OCR it on its own (clean horizontal line)
  4. Group top->bottom by row, left->right within row
  5. Print the ordered labels + save an annotated image

Run:  python label_crop_test.py
"""

import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

HERE = Path(__file__).parent
IN_DIR = HERE / "test_photos"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(exist_ok=True)

engine = RapidOCR()


def find_label_boxes(img):
    """Return list of (x, y, w, h) candidate white-label rectangles."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    # white/cream sticker: low saturation, high brightness
    mask = ((S < 70) & (V > 160)).astype(np.uint8) * 255
    # close small gaps so a label becomes one solid blob
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_area = w * h
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < img_area * 0.0003 or area > img_area * 0.02:
            continue                      # too tiny or too big
        ar = bw / bh
        if ar < 0.4 or ar > 4.0:
            continue                      # labels are roughly square-ish
        fill = cv2.contourArea(c) / area
        if fill < 0.55:
            continue                      # must be a fairly solid rectangle
        boxes.append((x, y, bw, bh))
    return boxes, mask


def ocr_crop(img, box, pad=6):
    x, y, w, h = box
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]
    # upscale small crops — helps OCR
    if crop.shape[1] < 240:
        sc = 240 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    res, _ = engine(crop)
    if not res:
        return "", 0.0
    txt = " ".join(r[1] for r in res).strip()
    conf = sum(float(r[2]) for r in res) / len(res)
    return txt, conf


def order_boxes(items):
    """items: list of dict with x,y,w,h. Sort into rows then L->R."""
    if not items:
        return []
    heights = sorted(it["h"] for it in items)
    medh = heights[len(heights) // 2]
    items = sorted(items, key=lambda it: it["y"] + it["h"])  # by bottom edge
    rows, row = [], [items[0]]
    for it in items[1:]:
        prev = row[-1]
        if abs((it["y"] + it["h"]) - (prev["y"] + prev["h"])) > medh * 1.4:
            rows.append(row); row = []
        row.append(it)
    rows.append(row)
    ordered = []
    for r in rows:
        ordered.extend(sorted(r, key=lambda it: it["x"]))
    return ordered


def run(path):
    print("\n" + "=" * 70)
    print("IMAGE:", path.name)
    print("=" * 70)
    img = cv2.imread(str(path))
    boxes, mask = find_label_boxes(img)
    print(f"  candidate label boxes: {len(boxes)}")

    items = []
    for (x, y, w, h) in boxes:
        txt, conf = ocr_crop(img, (x, y, w, h))
        items.append({"x": x, "y": y, "w": w, "h": h, "text": txt, "conf": conf})

    ordered = order_boxes(items)
    ordered = [it for it in ordered if it["text"]]  # drop empties

    print(f"  labels with text     : {len(ordered)}")
    print("\n  --- ordered labels (what we'd sort on) ---")
    for it in ordered:
        print(f'    [{it["conf"]:.2f}] {it["text"]!r}')

    # annotate
    ann = img.copy()
    for it in ordered:
        cv2.rectangle(ann, (it["x"], it["y"]), (it["x"] + it["w"], it["y"] + it["h"]), (0, 200, 0), 3)
    cv2.imwrite(str(OUT_DIR / f"labels_{path.stem}.jpg"), ann)
    cv2.imwrite(str(OUT_DIR / f"mask_{path.stem}.jpg"), mask)
    print(f"\n  saved -> labels_{path.stem}.jpg  and  mask_{path.stem}.jpg")


if __name__ == "__main__":
    for p in sorted(IN_DIR.glob("*.jpg")):
        run(p)
