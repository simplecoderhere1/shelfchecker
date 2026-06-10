"""
Faithful test harness for the deployed ShelfCheck pipeline.
Replicates index.html's getResizedImageData + callGemini + detectLabelBoxes
so we can see what problems real shelf photos hit.

Kept in sync with index.html — if you change buildPrompt() or the
generationConfig there, mirror it here or the harness stops measuring
the production pipeline.
"""
import base64, io, json, sys, time, urllib.request
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

HERE = Path(__file__).parent
PROXY = "https://selfcheck-proxy.krish-sundareswar.workers.dev/"
MODEL = "gemini-3.1-flash-lite"

# Which prompt to use per file (matches what a user would pick)
SECTION = {
    "PXL_20260607_200004299.MP.jpg": "fiction",
    "PXL_20260607_200006014.jpg":    "fiction",
}  # everything else -> nonfiction

def build_prompt(section):
    # Mirrors index.html buildPrompt() — 4-element wire format with bounding box.
    if section == "nonfiction":
        return ('This is a library NONFICTION shelf photo. There may be multiple shelves.\n'
                'Read every book spine left-to-right, top shelf first, then lower shelves.\n'
                'Return ONLY a JSON array of arrays, one per book spine.\n'
                'Each inner array has exactly 4 elements:\n'
                '0: full text on the white sticker at the bottom of the spine (e.g. "973.7 HAR", "KIN"), or null if unreadable.\n'
                '1: book title from the spine, or null if unreadable.\n'
                '2: 1 if you read the white sticker clearly, 0 if unsure.\n'
                '3: bounding box of that spine as [ymin,xmin,ymax,xmax] normalized 0–1000, tight to just that one spine, never spanning neighbors or shelves.\n'
                'CRITICAL for element 0: read the sticker character-by-character exactly as printed. The cutter letters come from the AUTHOR\'S last name, NOT from the title. Never infer or guess the cutter from the title — e.g. "The New Organic Grower" by Coleman has sticker "635.04 COL", not "635.04 THE". If you cannot read it clearly, return null.\n'
                'If a sticker is partially cut off by the edge of the image or another book, DO NOT guess the missing letters or numbers. Return null.\n'
                'Never copy a neighbor\'s label.\n'
                'Example: [["973.7 HAR","Lincoln",1,[420,10,980,85]],["KIN","The Stand",1,[420,90,980,170]]]\n'
                'Example of unreadable or cropped sticker: [null, "Harry Potter", 0, [420,190,980,270]]')
    return ('This is a library FICTION shelf photo. There may be multiple shelves.\n'
            'Read every book spine left-to-right, top shelf first, then lower shelves.\n'
            'Return ONLY a JSON array of arrays, one per book spine.\n'
            'Each inner array has exactly 4 elements:\n'
            '0: full text on the white sticker at the bottom of the spine (e.g. "BLACK, Cara", "BLACK, Cara 5"), or null if unreadable.\n'
            '1: book title from the spine, or null if unreadable.\n'
            '2: 1 if you read the white sticker clearly, 0 if unsure.\n'
            '3: bounding box of that spine as [ymin,xmin,ymax,xmax] normalized 0–1000, tight to just that one spine, never spanning neighbors or shelves.\n'
            'Rules: the white bottom sticker is the primary sort key — read it first. Return null and 0 for element 2 if you cannot read the sticker. Never copy a neighbor\'s text.\n'
            'If a sticker is partially cut off by the edge of the image or another book, DO NOT guess the missing letters or numbers. Return null.\n'
            'Example: [["BLACK, Cara","Murder in Clichy",1,[420,10,980,85]],["KING, Stephen","The Stand",1,[420,90,980,170]]]\n'
            'Example of unreadable or cropped sticker: [null, "The Stand", 0, [420,190,980,270]]')

def resized_jpeg_b64(path, max_dim=2400, quality=82):
    # Mirrors getResizedImageData(): 2400px max side, q0.82, light contrast/saturation boost.
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, max_dim / max(w, h))
    if s < 1.0:
        im = im.resize((round(w*s), round(h*s)), Image.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Color(im).enhance(1.05)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode(), im.size

def call_gemini(prompt, b64):
    body = json.dumps({
        "model": MODEL,
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
    }).encode()
    req = urllib.request.Request(PROXY, data=body, headers={
        "Content-Type": "application/json",
        "Origin": "https://simplecoderhere1.github.io",
        "Referer": "https://simplecoderhere1.github.io/shelfchecker/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    }, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read())
    ms = int((time.time()-t0)*1000)

    # Mirrors callGemini()'s guarded parsing — fail loudly with the real reason.
    cands = j.get("candidates") or []
    if not cands:
        block = (j.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"no candidates (blockReason={block})")
    cand = cands[0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("response truncated (MAX_TOKENS)")
    if cand.get("finishReason") not in (None, "STOP"):
        raise RuntimeError(f"model stopped unexpectedly ({cand['finishReason']})")
    raw = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [])
    raw = raw.replace("```json", "").replace("```", "").strip()
    books = json.loads(raw)
    if not isinstance(books, list):
        raise RuntimeError("model output is not a JSON array")
    books = [b for b in books if isinstance(b, list)]   # mirrors callGemini's row filter
    return books, ms

def is_edge_cropped(book):
    # Mirrors callGemini()'s edge tagging: side- or bottom-touching boxes are
    # kept but treated as unverifiable (sticker likely cut off).
    if len(book) < 4 or not isinstance(book[3], list) or len(book[3]) != 4:
        return False
    ymin, xmin, ymax, xmax = book[3]
    return xmin <= 5 or xmax >= 995 or ymax >= 995

# ---- port of detectLabelBoxes() ----
def detect_label_boxes(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = min(1.0, 600 / W)
    sw, sh = round(W*s), round(H*s)
    im = im.resize((sw, sh), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    r, g, b = a[:,:,0], a[:,:,1], a[:,:,2]
    mx = np.maximum(np.maximum(r,g),b); mn = np.minimum(np.minimum(r,g),b)
    white = ((r>200)&(g>200)&(b>200)&((mx-mn)<50)).astype(np.uint8)
    STRIP_H = max(4, round(sh*0.025))
    blobs = []
    for y0 in range(0, sh, STRIP_H):
        y1 = min(y0+STRIP_H-1, sh-1); stripH = y1-y0+1
        col = white[y0:y1+1, :].sum(axis=0)
        thresh = stripH*0.4
        on = col >= thresh
        x = 0
        while x < sw:
            if on[x]:
                start = x
                while x < sw and on[x]: x += 1
                rw = x - start
                if sw*0.015 <= rw <= sw*0.25:
                    merged = False
                    for bl in blobs:
                        ox = min(x-1, bl['x1']) - max(start, bl['x0'])
                        if ox >= 0 and y0 <= bl['y1'] + STRIP_H*2:
                            bl['x0']=min(bl['x0'],start); bl['x1']=max(bl['x1'],x-1); bl['y1']=y1; merged=True; break
                    if not merged:
                        blobs.append({'x0':start,'y0':y0,'x1':x-1,'y1':y1})
            else:
                x += 1
    valid = [b for b in blobs if (b['x1']-b['x0']) >= (b['y1']-b['y0'])*0.8]
    return len(valid), len(blobs)

def main():
    files = sorted(p.name for p in HERE.glob("*.jpg"))
    print(f"{'file':<34} {'sec':<10} {'gem':>4} {'null':>5} {'low':>4} {'edge':>5} {'box':>4} {'match':>6}  ms")
    print("-"*98)
    for f in files:
        section = SECTION.get(f, "nonfiction")
        try:
            b64, size = resized_jpeg_b64(HERE/f)
            books, ms = call_gemini(build_prompt(section), b64)
            n = len(books)
            nulls = sum(1 for b in books if not b or b[0] in (None,"",))
            lows  = sum(1 for b in books if len(b) > 2 and b[2] != 1)
            edges = sum(1 for b in books if is_edge_cropped(b))
            boxn, raw = detect_label_boxes(HERE/f)
            match = "OK" if boxn==n else f"{boxn-n:+d}"
            print(f"{f:<34} {section:<10} {n:>4} {nulls:>5} {lows:>4} {edges:>5} {boxn:>4} {match:>6}  {ms}")
            # dump reads to per-file json for inspection
            (HERE/f"out_{f}.json").write_text(json.dumps(books, ensure_ascii=False, indent=1))
        except Exception as e:
            print(f"{f:<34} {section:<10}  ERROR: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()
