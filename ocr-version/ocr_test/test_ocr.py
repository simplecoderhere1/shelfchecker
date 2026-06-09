"""
Throwaway feasibility test for ShelfCheck.

Question it answers: can a free, on-device OCR engine read the white
author-labels off real Champaign shelf photos well enough to sort by?

It does NOT use the cloud AI from plan.md. It runs RapidOCR (a free,
offline, pip-only OCR engine) over each photo, draws boxes on what it
found, and prints every text snippet it read with a confidence score.

Note on engine choice: the eventual web app would likely use Tesseract.js
in the browser. RapidOCR is a good proxy for "is this text physically
legible to OCR at all?" -- if a strong engine can't read a label, the
browser engine definitely can't. Treat this as the optimistic ceiling.

Usage:
    python test_ocr.py                 # reads every image in ./test_photos
    python test_ocr.py path/to/img.jpg # one specific image
"""

import sys
import re
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

HERE = Path(__file__).parent
IN_DIR = HERE / "test_photos"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(exist_ok=True)

# Looks like a library author label: "LASTNAME, First" or "WAGGONER Tim"
LABEL_RE = re.compile(r"^[A-Z][A-Za-z.\- ]+,?\s+[A-Z][a-z]", )

engine = RapidOCR()


def looks_like_label(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return False
    # mostly-uppercase surname is the tell
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.45 or bool(LABEL_RE.match(t))


def run(img_path: Path):
    print("\n" + "=" * 70)
    print(f"IMAGE: {img_path.name}")
    print("=" * 70)

    img = cv2.imread(str(img_path))
    if img is None:
        print("  !! could not read image")
        return
    h, w = img.shape[:2]
    print(f"  size: {w}x{h}")

    result, _ = engine(str(img_path))
    if not result:
        print("  !! OCR found no text at all")
        return

    # result: list of [box(4 pts), text, confidence]
    rows = []
    for box, text, conf in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        rows.append((cx, cy, text, float(conf), box))

    # label region = bottom third of the photo (where the stickers sit)
    label_band_top = h * 0.62
    labels = [r for r in rows if r[1] >= label_band_top and looks_like_label(r[2])]
    labels.sort(key=lambda r: r[0])  # left -> right

    print(f"\n  total text snippets read : {len(rows)}")
    print(f"  look like author labels  : {len(labels)}")
    if labels:
        avg = sum(r[3] for r in labels) / len(labels)
        print(f"  avg label confidence     : {avg:.2f}")

    print("\n  --- labels, left to right (what it would sort on) ---")
    for cx, cy, text, conf, _ in labels:
        flag = "" if conf >= 0.7 else "   <-- low confidence"
        print(f'    [{conf:.2f}] {text!r}{flag}')

    # annotated image for eyeballing
    annotated = img.copy()
    for cx, cy, text, conf, box in rows:
        pts = [(int(p[0]), int(p[1])) for p in box]
        is_label = cy >= label_band_top and looks_like_label(text)
        color = (0, 180, 0) if is_label else (0, 0, 255)
        for i in range(4):
            cv2.line(annotated, pts[i], pts[(i + 1) % 4], color, 2)
    out_path = OUT_DIR / f"annotated_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n  annotated image saved -> {out_path.name}")


def main():
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        targets = sorted(
            p for p in IN_DIR.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    if not targets:
        print(f"No images found. Put the shelf photos in: {IN_DIR}")
        print("Then re-run:  python test_ocr.py")
        return
    for t in targets:
        run(t)
    print("\nDone. Open the annotated_*.jpg files in ./out to see what it caught.")


if __name__ == "__main__":
    main()
