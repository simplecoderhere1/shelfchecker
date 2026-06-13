# ShelfCheck

A web app for Champaign Public Library volunteers to find misshelved books.

Take a photo of a shelf → the app reads the white spine labels with Gemini AI → red boxes mark books that are out of order, yellow boxes mark labels that couldn't be read clearly.

Supports **Fiction** (alphabetical by author surname) and **Nonfiction** (Dewey Decimal, including biography sections).

## How to use

Open the app, pick Fiction or Nonfiction, take a straight-on photo of one or two shelves, tap Analyze. Results appear in a few seconds.

- **Red box** = book is out of order — move it
- **Yellow box** = label was unclear — check it by hand
- **No box** = in order
- Tap any book on the photo to see what was read from its spine

## Tech

Single HTML file, no build step, no install. Gemini Vision (`gemini-3.1-flash-lite`) reads spines via a Cloudflare Worker proxy. All ordering logic runs locally in JS (LIS-based, not AI).

## Running locally

```
python -m http.server 8000
```

Then open `http://localhost:8000`.
