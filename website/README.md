# prompt-compress demo site

Local-only demo for presentations. A React + Vite frontend and a FastAPI
backend that calls the installed `prompt_compress` library for live
compression. No deployment.

```
website/
  backend/main.py       # FastAPI; wraps prompt_compress
  frontend/             # Vite + React + TypeScript + Tailwind + Recharts
  README.md             # this file
```

## First-time setup

```bash
# from project root
source .venv/bin/activate
pip install fastapi uvicorn
# prompt_compress should already be installed (`pip install -e .` at root)

cd website/frontend
npm install
```

### macOS + Python 3.14 caveat

On Python 3.14, `site.py` silently skips `.pth` files marked with the OS
`hidden` flag. macOS sometimes sets that flag on files touched through
sandboxed tools (Claude's bash, some IDE flows), which makes the
editable install of `prompt_compress` invisible to Python and produces a
`ModuleNotFoundError: No module named 'prompt_compress'` at backend
import. Clear it once after a clean install:

```bash
# from project root
chflags nohidden .venv/lib/python3.14/site-packages/_editable_impl_prompt_compress.pth
# verify: should print "-", not "hidden"
stat -f "%Sf" .venv/lib/python3.14/site-packages/_editable_impl_prompt_compress.pth
```

The flag may need to be cleared again if the install is recreated
through one of those tools.

## Running (two terminals)

**Activate the venv in every terminal before starting either server.** If
`which uvicorn` prints `/opt/homebrew/bin/uvicorn` you're running
Homebrew's uvicorn against Homebrew's Python, which won't see the
project's `fastapi` install and will fail with `ModuleNotFoundError`.

```bash
# Terminal 1 — backend
source .venv/bin/activate                     # if not already
which uvicorn                                 # must end in .venv/bin/uvicorn
cd website/backend
uvicorn main:app --port 8000 --reload
# Startup logs should include:
#   "Backend ready. Warmup compression: ... (tier X, X.XXs)"

# Terminal 2 — frontend (no venv needed, just node)
cd website/frontend
npm run dev
# Vite serves on http://localhost:5173
```

Vite proxies `/api/*` to `http://localhost:8000`, so the frontend never
calls the backend directly — same-origin from the browser's perspective.

> The original spec called for `--workers 2`. uvicorn 0.47 needs an
> external process supervisor to honour `--workers > 1`; on this version a
> single worker is fine for a demo. The compressor is held as a
> module-level singleton so the SentenceTransformer model is loaded once
> at startup and reused for every request.

## Production build (sanity check only)

```bash
cd website/frontend
npm run build      # writes dist/
npm run preview    # serves dist/ on :4173
```

Don't deploy `dist/`. The presentation runs from `npm run dev`.

## Pages

- `#/compare` — live compression page (default). Hits `/api/compress`.
- `#/benchmark` — static benchmark view, reads `/summary.json`.
- `#/calculator` — pure-JS cost & carbon calculator.

`public/summary.json` is a copy of `data/website/summary.json` from the
project root. To refresh:

```bash
cp ../../data/website/summary.json public/summary.json
```

## Demo failure modes (verified)

| Situation | What the user sees |
|---|---|
| Backend not running | Red banner: "Backend offline. Start the API with `cd website/backend && uvicorn main:app --port 8000` and refresh." |
| Compression > 5s | Yellow "Taking longer than usual…" stays visible; no hard timeout |
| Backend raises an exception | `ok: false` returned, error displayed in red below the input |
| Validator rejects compression | Result still renders; "✗ Validator rejected" badge; original text shown in the Ours column |
| Custom prompt typed in textarea | LLMLingua column shows "comparison only for reference examples" message |
