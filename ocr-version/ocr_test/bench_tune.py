"""Find fastest RapidOCR settings on a SIMULATED 2-core CPU (like HF free).
Sweeps rec_batch_num and thread count; reports time + region count (quality proxy).
"""
import time, cv2
from rapidocr_onnxruntime import RapidOCR

img0 = cv2.imread("test_photos/imagesample.jpg")
H, W = img0.shape[:2]
s = 1536 / max(H, W)
img = cv2.resize(img0, (round(W*s), round(H*s)), interpolation=cv2.INTER_AREA)

def bench(label, **kw):
    eng = RapidOCR(**kw)
    eng(img, use_cls=True)  # warm
    best = min(_time(eng) for _ in range(3))
    res, _ = eng(img, use_cls=True)
    print(f"  {label:38s} {best:5.2f}s  {len(res or [])} regions")

def _time(eng):
    t = time.time(); eng(img, use_cls=True); return time.time()-t

print("Simulating HF 2-core CPU (intra_op_num_threads=2), MAX_SIDE=1536:\n")
for batch in [6, 16, 32]:
    bench(f"threads=2 rec_batch={batch}",
          intra_op_num_threads=2, rec_batch_num=batch, cls_batch_num=batch)
print("\nFor reference (more threads):")
bench("threads=4 rec_batch=16", intra_op_num_threads=4, rec_batch_num=16, cls_batch_num=16)
