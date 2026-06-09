"""Does image enhancement recover OCR reads lost to low resolution?
Compares, on the SAME photo:
  - 2048px plain      (what reads well, but times out on HF)
  - 1024px plain      (what we're forced to use on HF)
  - 1024px enhanced   (CLAHE contrast + unsharp mask)
  - 1024px enhanced then upscaled to 2048 (does 'just upscale' help?)
Reports region count + mean confidence. If enhancement closes the gap to
2048, it's worth doing. If not, the limit is real resolution, not contrast.
"""
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
img0 = cv2.imread("test_photos/imagesample.jpg")
H, W = img0.shape[:2]

def resize_to(img, side):
    h, w = img.shape[:2]
    if max(h, w) <= side: return img
    s = side / max(h, w)
    return cv2.resize(img, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)

def enhance(img):
    # CLAHE on luminance (local contrast) + unsharp mask (sharpening)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img2 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img2, (0, 0), 2.0)
    return cv2.addWeighted(img2, 1.6, blur, -0.6, 0)   # unsharp

def run(label, img):
    res, _ = engine(img, use_cls=True)
    n = len(res or [])
    conf = np.mean([c for _, _, c in (res or [])]) if n else 0
    print(f"  {label:38s} {img.shape[1]}x{img.shape[0]}  {n:3d} regions  mean-conf {conf:.3f}")
    return set((t.strip().upper() for _, t, _ in (res or [])))

print(f"source {W}x{H}\n")
r2048 = run("2048 plain (ideal)",          resize_to(img0, 2048))
r1024 = run("1024 plain (HF reality)",     resize_to(img0, 1024))
r1024e = run("1024 enhanced",              enhance(resize_to(img0, 1024)))
r1024u = run("1024 enhanced -> up to 2048", resize_to(enhance(resize_to(img0, 1024)), 2048) if False else
             cv2.resize(enhance(resize_to(img0, 1024)), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

# How much of what 2048 read does each 1024 variant recover?
print(f"\n  2048 read {len(r2048)} unique strings.")
for name, s in [("1024 plain", r1024), ("1024 enhanced", r1024e), ("1024 enh+upscale", r1024u)]:
    recovered = len(r2048 & s)
    print(f"  {name:20s} shares {recovered:3d} / {len(r2048)} of the 2048 reads ({100*recovered/max(1,len(r2048)):.0f}%)")
