# ShelfCheck — Library Reordering App

A web app for Champaign Public Library volunteers to find out-of-order books.

---

## Goal

A volunteer takes a photo of a bookshelf → the app reads the **white spine labels** → it checks whether the books are in the correct order → it draws **green/red boxes** on the photo showing which are correctly placed and which are misshelved.

Supports fiction (alphabetical by author surname) and nonfiction (Dewey Decimal).

---

## Ordering Logic

### Fiction
```
Parse the white label as "LASTNAME, First"
Sort by: last name (A→Z) → then first name (A→Z)
Flag: any book whose key sorts before the previous book's key
```

### Nonfiction (Dewey Decimal)
```
Parse the white label as a Dewey number (e.g. 001.2, 025.3, 512.7) + author cutter
Sort by: call number numerically → then author last name
Flag: any book whose number is less than the previous book's
```

---

## Librarian Outreach Plan

Build the MVP first, then:
1. Record a 90-second demo on a real shelf
2. Email the librarian: *"I volunteer here and built a quick tool that might help with reshelving — here's a demo. Would love your feedback."*
3. Ask: standard Dewey for all nonfiction? special sections (graphic novels, large print)? would volunteers use it?
