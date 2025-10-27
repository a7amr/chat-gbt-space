# chat-gbt-space 
# chat-gbt-space 
# chat-gbt-space

A tiny “space” for experimenting with a chat bot UI + helper scripts.  
This repo is **local-only** (no deployment). Use it to iterate on prompts,
handlers, and small utilities before wiring to a real model/API.

---

## What’s here
- Minimal chat UI scaffold (editable).
- Python helpers (e.g., prototypes, crawlers, data prep).
- No secrets committed; `.env` is ignored.

> Tip: keep throwaway experiments in a `sandbox/` folder so `main` stays clean.

---

## Quick start

### 1) Environment
```bash
# (Recommended) create a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install deps (add pins inside requirements.txt as needed)
pip install -r requirements.txt

# Example: start the chat UI (edit the file name if yours differs)
python app.py

# Example: run helper scripts
python spacegbtv1.py
python spacegbtv2.py
python "Full crawler.py"

.
├─ app.py                 # chat UI entry (rename or swap to your file)
├─ spacegbtv1.py          # helper script
├─ spacegbtv2.py          # helper script
├─ Full crawler.py        # data/crawling prototype
├─ requirements.txt
├─ README.md
└─ .gitignore
