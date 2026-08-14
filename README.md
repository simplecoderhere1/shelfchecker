# ShelfCheck

A web app for Library volunteers to find misshelved books.

Take a photo of a shelf → the app reads the white spine labels → red boxes mark books that are out of order, yellow boxes mark labels that couldn't be read clearly.

Supports **Fiction** (alphabetical by author surname) and **Nonfiction** (Dewey Decimal, including biography sections).

## How to use

Open the app, pick Fiction or Nonfiction, take a straight-on photo of one or two shelves, tap Analyze. Results appear in a few seconds.

- **Red box** = book is out of order — move it
- **Yellow box** = label was unclear — check it by hand
- **No box** = in order
- Tap any book on the photo to see what was read from its spine

The app never prints a digit it did not read. A label it cannot resolve is shown as unclear rather than guessed, because sending a volunteer to the wrong book is worse than sending them to look by hand.

## Tech

Single HTML file, no build step, no install. **Azure AI Vision (Image Analysis 4.0, `features=read`)** reads spines via a Cloudflare Worker proxy (`workers/azure-ocr-worker.js`), which holds the API key as a secret so it is never in the page. All ordering logic runs locally in JS (LIS-based, not AI).

Reading a shelf takes two passes:

1. **Full frame** — one call on the whole photo, downscaled to 2400px.
2. **Label bands** — the first pass locates the thin horizontal strip the stickers sit in (typically ~7% of frame height); that strip is cropped and re-read in three overlapping segments.

The second pass exists because Azure's text-height floor is a *proportion* of the frame, not an absolute pixel count. A 25px call number in a 4080px photo is 0.6% of frame height and falls under the floor; the same number inside a band crop is 3–8% and reads cleanly. Splitting the band into three segments matters — reading the band as one wide ribbon scores *worse* than the old blind grid.

Median time to first render is about 2.6s, worst case 3.7s.

## Running locally

```
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Related

Development, test harnesses, and ground-truth corpora live in
[`shelfcheck-ocr`](https://github.com/simplecoderhere1/shelfcheck-ocr).
