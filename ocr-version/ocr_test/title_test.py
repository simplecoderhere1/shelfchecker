"""
Can OCR read the spine TITLES (not just the labels)?
Tests the ceiling using RapidOCR (PP-OCR + angle classification, which
rotates vertical spine text before reading). Prints every text region
top->bottom with confidence so we can judge title readability, and
flags the multi-word strings that are most likely titles.

Run: python title_test.py
"""
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

HERE = Path(__file__).parent
IN_DIR = HERE / "test_photos"
# cls=True enables angle classification (handles rotated/vertical text)
engine = RapidOCR()


def run(path):
    print("\n" + "=" * 70)
    print("IMAGE:", path.name)
    print("=" * 70)
    result, _ = engine(str(path), use_cls=True)
    if not result:
        print("  no text"); return
    rows = []
    for box, text, conf in result:
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        rows.append((cy, cx, text, float(conf)))
    rows.sort()
    print(f"  {len(rows)} text regions. Likely TITLES (2+ words or long):\n")
    for cy, cx, text, conf in rows:
        words = text.split()
        is_title = len(words) >= 2 or len(text) >= 9
        mark = "TITLE" if is_title else "  -  "
        print(f"   {mark} [{conf:.2f}] {text!r}")


if __name__ == "__main__":
    for p in sorted(IN_DIR.glob("*.jpg")):
        run(p)
