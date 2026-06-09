"""Does use_cls cost much? Does dropping it lose the vertical titles?"""
import time, os, cv2
from rapidocr_onnxruntime import RapidOCR

print("CPU count:", os.cpu_count())
engine = RapidOCR()
img0 = cv2.imread("test_photos/imagesample.jpg")
H, W = img0.shape[:2]

for side in [2048, 1536]:
    s = side / max(H, W)
    img = cv2.resize(img0, (round(W*s), round(H*s)), interpolation=cv2.INTER_AREA)
    for cls in [True, False]:
        engine(img, use_cls=cls)  # warm
        t = time.time()
        res, _ = engine(img, use_cls=cls)
        dt = time.time() - t
        # count vertical (title-like) boxes: height > width
        vert = sum(1 for box,_,_ in (res or []) if (max(p[1] for p in box)-min(p[1] for p in box)) > (max(p[0] for p in box)-min(p[0] for p in box)))
        print(f"  side={side} cls={cls!s:5}  {dt:5.2f}s  {len(res or [])} regions, {vert} vertical")
