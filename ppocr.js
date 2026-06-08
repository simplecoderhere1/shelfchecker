// ─────────────────────────────────────────────────────────────
// PaddleOCR (PP-OCRv3) running directly on onnxruntime-web.
// Detection (DBNet) + Recognition (CRNN/CTC). Fully on-device.
// No wrapper libs — we control preprocessing + wasm settings.
// ─────────────────────────────────────────────────────────────

// onnxruntime-web is loaded as a UMD global (window.ort) from the SAME
// CDN/version as its wasm binaries — mismatched JS/wasm sources cause
// "(void 0) is not a function" on session create.
const ort = window.ort;

// Single-threaded wasm: our static server can't send the COOP/COEP
// headers that multi-threaded wasm (SharedArrayBuffer) requires.
// Running threaded without isolation produces silent garbage — this
// is almost certainly what broke the earlier wrapper.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';

// PP-OCRv3 preprocessing constants (from RapidOCR configs)
const DET_LIMIT = 736;          // limit_side_len, limit_type "min"
const DET_MAX = 1536;           // our extra cap for browser speed
const DET_THRESH = 0.3;         // binarize prob map
const BOX_THRESH = 0.5;         // min mean prob inside a box
const UNCLIP = 1.6;             // box expansion ratio
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const REC_H = 48, REC_W = 320;

let detSession = null, recSession = null, charset = null;

export async function init({ detUrl, recUrl, dictUrl, onStatus = () => {} }) {
  onStatus('loading detection model…');
  detSession = await ort.InferenceSession.create(detUrl, { executionProviders: ['wasm'] });
  onStatus('loading recognition model…');
  recSession = await ort.InferenceSession.create(recUrl, { executionProviders: ['wasm'] });
  onStatus('loading dictionary…');
  const txt = await (await fetch(dictUrl)).text();
  // CTC charset: index 0 = blank, then dict chars, then space
  charset = ['<blank>', ...txt.split(/\r?\n/), ' '];
  onStatus('models ready');
}

// ── helpers ──────────────────────────────────────────────────
function canvasOf(src) {
  if (src instanceof HTMLCanvasElement) return src;
  const c = document.createElement('canvas');
  c.width = src.naturalWidth || src.videoWidth || src.width;
  c.height = src.naturalHeight || src.videoHeight || src.height;
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  return c;
}

// ── DETECTION ────────────────────────────────────────────────
export async function detect(srcCanvas) {
  const src = canvasOf(srcCanvas);
  const ow = src.width, oh = src.height;

  // resize: ensure min side >= DET_LIMIT, cap max side <= DET_MAX, multiple of 32
  let ratio = 1;
  const minSide = Math.min(ow, oh), maxSide = Math.max(ow, oh);
  if (minSide < DET_LIMIT) ratio = DET_LIMIT / minSide;
  if (maxSide * ratio > DET_MAX) ratio = DET_MAX / maxSide;
  let rw = Math.max(32, Math.round((ow * ratio) / 32) * 32);
  let rh = Math.max(32, Math.round((oh * ratio) / 32) * 32);

  const rc = document.createElement('canvas');
  rc.width = rw; rc.height = rh;
  rc.getContext('2d').drawImage(src, 0, 0, rw, rh);
  const { data } = rc.getContext('2d').getImageData(0, 0, rw, rh);

  // HWC RGB -> normalized CHW float32
  const input = new Float32Array(3 * rw * rh);
  const plane = rw * rh;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    input[p]             = (data[i]     / 255 - MEAN[0]) / STD[0];
    input[plane + p]     = (data[i + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * plane + p] = (data[i + 2] / 255 - MEAN[2]) / STD[2];
  }
  const tensor = new ort.Tensor('float32', input, [1, 3, rh, rw]);
  const out = await detSession.run({ [detSession.inputNames[0]]: tensor });
  const probMap = out[detSession.outputNames[0]].data; // length rw*rh, values 0..1

  // binarize
  const bin = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) bin[i] = probMap[i] > DET_THRESH ? 1 : 0;
  dilate(bin, rw, rh);

  // connected components -> axis-aligned boxes
  const boxes = [];
  const labels = new Int32Array(plane).fill(0);
  const stack = [];
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const idx = y * rw + x;
      if (!bin[idx] || labels[idx]) continue;
      let minx = x, maxx = x, miny = y, maxy = y, area = 0, probSum = 0;
      stack.push(idx); labels[idx] = 1;
      while (stack.length) {
        const c = stack.pop();
        const cx = c % rw, cy = (c / rw) | 0;
        area++; probSum += probMap[c];
        if (cx < minx) minx = cx; if (cx > maxx) maxx = cx;
        if (cy < miny) miny = cy; if (cy > maxy) maxy = cy;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= rw || ny >= rh) continue;
          const ni = ny * rw + nx;
          if (bin[ni] && !labels[ni]) { labels[ni] = 1; stack.push(ni); }
        }
      }
      const bw = maxx - minx + 1, bh = maxy - miny + 1;
      if (bw < 4 || bh < 4) continue;
      if (probSum / area < BOX_THRESH) continue;
      // unclip (expand) by d = area*ratio/perimeter
      const A = bw * bh, L = 2 * (bw + bh);
      const d = (A * UNCLIP) / L;
      let ex0 = Math.max(0, minx - d), ey0 = Math.max(0, miny - d);
      let ex1 = Math.min(rw, maxx + d), ey1 = Math.min(rh, maxy + d);
      // back to original-image coords
      boxes.push({
        x: ex0 / rw * ow, y: ey0 / rh * oh,
        w: (ex1 - ex0) / rw * ow, h: (ey1 - ey0) / rh * oh,
        score: probSum / area,
      });
    }
  }
  return { boxes, srcCanvas: src };
}

// horizontal-biased dilation to connect characters into text lines
function dilate(bin, w, h) {
  const out = new Uint8Array(bin.length);
  const r = 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let on = 0;
      for (let dx = -r; dx <= r && !on; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < w && bin[y * w + nx]) on = 1;
      }
      out[y * w + x] = on;
    }
  }
  bin.set(out);
}

// ── RECOGNITION ──────────────────────────────────────────────
export async function recognize(src, box) {
  // crop from original image
  const sx = Math.max(0, Math.round(box.x)), sy = Math.max(0, Math.round(box.y));
  const sw = Math.min(src.width - sx, Math.round(box.w));
  const sh = Math.min(src.height - sy, Math.round(box.h));
  if (sw < 2 || sh < 2) return { text: '', score: 0 };

  const resizedW = Math.min(REC_W, Math.max(8, Math.round((REC_H * sw) / sh)));
  const rc = document.createElement('canvas');
  rc.width = REC_W; rc.height = REC_H;
  const ctx = rc.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, REC_W, REC_H);     // pad right with black
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, resizedW, REC_H);
  const { data } = ctx.getImageData(0, 0, REC_W, REC_H);

  const plane = REC_W * REC_H;
  const input = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    input[p]             = (data[i]     / 255 - 0.5) / 0.5;
    input[plane + p]     = (data[i + 1] / 255 - 0.5) / 0.5;
    input[2 * plane + p] = (data[i + 2] / 255 - 0.5) / 0.5;
  }
  const tensor = new ort.Tensor('float32', input, [1, 3, REC_H, REC_W]);
  const out = await recSession.run({ [recSession.inputNames[0]]: tensor });
  const o = out[recSession.outputNames[0]];
  const [, T, C] = o.dims;       // [1, T, num_classes]
  const d = o.data;

  let text = '', prev = -1, probSum = 0, n = 0;
  for (let t = 0; t < T; t++) {
    let best = 0, bestVal = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = d[t * C + c];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    if (best !== 0 && best !== prev) {       // skip blank + collapse repeats
      text += charset[best] || '';
      probSum += bestVal; n++;
    }
    prev = best;
  }
  return { text: text.trim(), score: n ? probSum / n : 0 };
}

// ── full pipeline ────────────────────────────────────────────
export async function runOcr(srcCanvas, onStatus = () => {}) {
  onStatus('detecting text…');
  const { boxes, srcCanvas: src } = await detect(srcCanvas);
  onStatus(`reading ${boxes.length} regions…`);
  const results = [];
  for (const box of boxes) {
    const { text, score } = await recognize(src, box);
    if (text) results.push({ text, score, box });
  }
  return results;
}
