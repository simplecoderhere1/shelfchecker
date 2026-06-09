"""Benchmark OCR time vs MAX_SIDE to pick the speed/quality sweet spot."""
import time, cv2
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
img0 = cv2.imread("test_photos/imagesample.jpg")
H, W = img0.shape[:2]
print(f"source {W}x{H}")

for side in [2048, 1536, 1280, 1024]:
    s = side / max(H, W)
    img = cv2.resize(img0, (round(W*s), round(H*s)), interpolation=cv2.INTER_AREA) if max(H,W) > side else img0
    # warmup-free: time 2 runs, take 2nd
    engine(img, use_cls=True)
    t = time.time()
    result, _ = engine(img, use_cls=True)
    dt = time.time() - t
    n = len(result or [])
    print(f"  MAX_SIDE={side:5d}  {img.shape[1]}x{img.shape[0]}  {dt:5.2f}s  {n} regions")
