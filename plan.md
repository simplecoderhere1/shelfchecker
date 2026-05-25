# 📚 ShelfCheck — Library Reordering App
**A web app for Champaign Public Library volunteers to detect out-of-order books using AI vision.**

---

## Overview

Volunteer takes a photo of a bookshelf → AI reads the spines → App highlights which books are out of order and shows the correct sequence. Supports fiction (alphabetical by author) and nonfiction (Dewey Decimal).

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Frontend | Vanilla HTML/CSS/JS | Free |
| Camera | `getUserMedia` (browser, rear camera) | Free |
| AI Vision | Anthropic API (Claude Sonnet) → swap to Gemini 2.0 Flash after Pro expires | Free (within Claude Pro) → Free (Gemini free tier) |
| Hosting | GitHub Pages | Free |
| Build tool | None (single HTML file) | Free |

> **Note:** The Gemini 2.0 Flash free tier allows 1,500 requests/day — more than enough for a volunteer tool. Swapping from Claude → Gemini is one variable change in the JS.

---

## Core Features

### Phase 1 — MVP (build this first, ~3 days)
- [ ] **Live camera viewfinder** built into the page (rear-facing, with a shelf-framing guide overlay)
- [ ] **Capture button** that freezes the frame
- [ ] **Section selector** — Fiction or Nonfiction (picked before scanning)
- [ ] **AI spine extraction** — sends image to Claude/Gemini, gets back a list of books in left-to-right shelf order with title, author, and call number
- [ ] **Order checking logic**
  - Fiction: sort by author last name A→Z, then title A→Z
  - Nonfiction: sort by Dewey Decimal number (numeric), then by author last name
- [ ] **Results screen** — numbered list of books as AI read them, with out-of-order ones highlighted in red and a suggested correct position shown
- [ ] **Retake button** to go back to camera

### Phase 2 — Polish (after librarian feedback, ~2 days)
- [ ] **Confidence indicators** — flag spines the AI wasn't sure about (blurry, obscured, partial)
- [ ] **Manual correction** — tap a misread spine to fix the title/author before checking order
- [ ] **Multi-shelf mode** — scan multiple shelves and get a combined report
- [ ] **Share/export** — copy the out-of-order list as plain text to paste into a message
- [ ] **PWA support** — "Add to Home Screen" so it feels like a native app

### Phase 3 — Nice to have (if librarian is on board)
- [ ] Children's section support (usually E + author last name)
- [ ] Large Print section
- [ ] Save scan history (localStorage, no backend needed)

---

## AI Prompt Strategy

The hardest part is getting reliable spine extraction. The prompt sent with each image should:

1. Ask for books **in left-to-right order as they appear on the shelf**
2. Request structured JSON output: `[{ "position": 1, "title": "...", "author_last": "...", "call_number": "..." }]`
3. Instruct it to include a `"confidence": "high/medium/low"` field per book
4. Tell it to use `null` for fields it cannot read rather than guessing
5. Specify the section type so it knows what to look for (Dewey numbers vs. author labels)

Example prompt snippet:
```
You are reading a library bookshelf photo. List every book spine visible, left to right.
Return ONLY valid JSON — an array of objects with: position, title, author_last, call_number, confidence.
Section type: [FICTION | NONFICTION]
If a field is unreadable, use null. Do not guess.
```

---

## Ordering Logic

### Fiction
```
Sort by: author_last (A→Z) → then title (A→Z)
Flag: any book where author_last comes before the previous book's author_last
```

### Nonfiction (Dewey Decimal)
```
Sort by: call_number numerically (e.g. 001.2 < 025.3 < 512.7)
Then by: author_last (A→Z) within same Dewey number
Flag: any book where call_number is less than the previous book's call_number
```

---

## File Structure

```
shelfcheck/
├── index.html        ← entire app (HTML + CSS + JS in one file for simplicity)
├── README.md
└── .github/
    └── workflows/
        └── deploy.yml  ← auto-deploy to GitHub Pages on push (optional)
```

Single-file approach keeps it dead simple — Claude Code can maintain the whole thing in one shot.

---

## API Key Plan

```js
// Phase 1 — Claude (while Pro is active)
const API_URL = "https://api.anthropic.com/v1/messages";
const API_KEY = "YOUR_CLAUDE_KEY";
const MODEL = "claude-sonnet-4-20250514";

// Phase 2 — Gemini (after Claude Pro expires, ~11 days)
// Swap above to:
const API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const API_KEY = "YOUR_GEMINI_KEY";
```

Get a free Gemini API key at: https://aistudio.google.com

---

## Deployment

1. Push `index.html` to a GitHub repo
2. Enable GitHub Pages (Settings → Pages → Deploy from main branch)
3. Share the URL: `https://[username].github.io/shelfcheck`
4. Volunteers open on phone, browser asks for camera permission once, done

---

## Librarian Outreach Plan

**Don't email yet.** Build Phase 1 first (~3 days), then:
1. Record a 90-second demo video of it actually working on a real shelf
2. Email the librarian: *"Hey, I volunteer here and built a quick tool that might help with reshelving — here's a demo. Would love your feedback before I finish it."*
3. Key questions to ask them:
   - Do you use standard Dewey for all nonfiction?
   - Any special sections (graphic novels, large print, magazines)?
   - Would other volunteers actually use something like this?

This makes them a collaborator, not just an approver — much harder to say no to.

---

## Build Order (recommended)

```
Day 1:  Camera UI + capture + section selector
Day 2:  AI integration + spine extraction + JSON parsing
Day 3:  Ordering logic + results display
Day 4:  Testing on real shelves, prompt tuning
Day 5+: Polish based on what breaks
```
