# AeroFix AI — Setup & Run Guide

Step-by-step commands to run this project **without errors** on Windows
(PowerShell). Every command below was executed on this machine; the expected
output of each step is shown so you can confirm it worked.

> All commands are run from the project folder. Because the path contains a
> space, keep the quotes:
>
> ```powershell
> Set-Location "D:\code1\agentic ai\aerofix"
> ```

---

## What you need

| Requirement | Notes |
|---|---|
| Windows + PowerShell | Commands below are PowerShell (cmd equivalents noted where it matters) |
| Python 3.11 or 3.12 | Project metadata pins `>=3.11,<3.13`. A working `.venv` with **Python 3.10.6 already exists** in this checkout and is fully tested. `py -0p` lists installed versions. |
| Internet (first run only) | Downloads the embedding model (~100 MB) and the two HF datasets (cached under `%USERPROFILE%\.cache\huggingface`). After that everything runs **offline**. |
| Microsoft Edge | Used by the Playwright UI tests via `channel="msedge"` — **no** `playwright install` browser download is needed on Windows. |
| ~4 GB disk | torch + model + dataset caches + index |

You do **not** need any API key: without one, drafting automatically uses the
local MockLLM (grounded extractive text from retrieved chunks only).

---

## Step 1 — Open the project folder

```powershell
Set-Location "D:\code1\agentic ai\aerofix"
```

## Step 2 — Create the virtual environment

**This checkout already has `.venv` — skip to Step 3** (verify it works):

```powershell
.\.venv\Scripts\python.exe --version      # expect: Python 3.10.6
```

**Fresh install** (new machine / deleted `.venv`) — use Python 3.11 to match
the project metadata:

```powershell
py -3.11 -m venv .venv
```

> Activation is **not** required (and avoids ExecutionPolicy problems): call
> `.\.venv\Scripts\python.exe` directly for every command below. If you prefer
> an active prompt instead:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process Bypass -Force
> .\.venv\Scripts\Activate.ps1
> ```

## Step 3 — Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- Expected end: `Successfully installed ...` or a list of
  `Requirement already satisfied`, **exit code 0**.
- First fresh install downloads **torch (~2 GB)** — be patient.
- Verify nothing is broken:

```powershell
.\.venv\Scripts\python.exe -m pip check     # expect: No broken requirements found.
```

> Do **not** run `pip install -e .` — the package is used in place; running the
> commands below from the project root is all that is needed.

## Step 4 — (Optional) Configure `.env`

Everything works with zero configuration. Only copy this if you want to change
providers or add an LLM key:

```powershell
Copy-Item .env.example .env
notepad .env
```

- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` empty or unset → the app prints
  `[aerofix] OPENAI_API_KEY is not set; using the local MockLLM ...` in the
  terminal and continues (this is normal, not an error).
- Local embeddings are the default (`huggingface` + `BAAI/bge-small-en-v1.5`),
  so no key is required for indexing or searching.

## Step 5 — Build the knowledge base (first run)

Downloads the two public datasets and embeds 19,707 passages locally
(**~26 minutes** on this machine, ~8 passages/s):

```powershell
.\.venv\Scripts\python.exe scripts\ingest_datasets.py
```

Expected ending:

```
AeroFix Knowledge Base
----------------------
Documents: 5
Passages: 19707
Embeddings: 19707 (BAAI/bge-small-en-v1.5)
Ingestion wall time: ~1597
```

Confirm the manifest exists:

```powershell
.\.venv\Scripts\python.exe -c "import json; m=json.load(open('data/knowledge/manifest.json', encoding='utf-8')); print(m['passages_indexed'], m['embedding_model'])"
# expect: 19707 BAAI/bge-small-en-v1.5
```

Notes:

- **Safe to rerun / resume** — it prints `Already indexed: N  New: M` and
  continues where it stopped. Rerunning when complete is a fast no-op.
- **Never use `--rebuild`** while torch-embedded passages exist (it would wipe
  and force a full re-embed).
- Optional: additional local PDFs go in `data/raw/` and are indexed with
  `.\.venv\Scripts\python.exe ingest.py` (idempotent — rerunning adds nothing).

## Step 6 — Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Expected:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open **http://localhost:8501**. Stop the server with `Ctrl+C`.

## Step 7 — Smoke test in the browser (2 minutes)

1. In the hero, click an example chip, e.g. **`Rough idle on run-up`** — the
   fault box fills with the full sentence.
2. Click **`Search manuals & draft procedure`** (first search loads the index,
   ~20–60 s). Expected: a **“Draft steps”** card with cited steps, a
   **“Manual pages”** source card and a citation such as
   `FAA-H-8083-32, pp. …`, plus a confidence value ≥ 0.45.
3. Type an out-of-corpus question, e.g.
   `What is the part number and price of the Cessna 172 nosewheel tire?` and
   search. Expected (exactly):
   **`Insufficient verified source information.`**
4. Click **Reduce animations** in the sidebar — the draft stays on screen
   (results survive reruns), background motion freezes.
5. Optional: append `?debug=1` to the URL for the FPS/scheduler overlay.

## Step 8 — Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected last line: **`30 passed, 2 warnings in 55.18s`**, exit code `0`. This
includes the browser (Playwright + Edge) checks — allow ~1–2 minutes.

> Note: `pyproject.toml` already adds `-q` for you — don't add another one.
> With two (`-qq`) pytest hides the `30 passed` summary line; the run still
> succeeds (30 dots + exit code 0).

## Step 9 (optional) — Ask from the command line

```powershell
.\.venv\Scripts\python.exe scripts\ask.py "What does the handbook say about torque wrench inspection?"
```

Example of a healthy answer:

```
QUESTION: What does the handbook say about torque wrench inspection?
CONFIDENCE: 0.7362 (threshold 0.45)
ANSWER: ...To reach the nut, a 2-inch straight adapter must be used on an 18-inch torque wrench...
SOURCE: AMT Airframe Handbook (Remixonwin extract)
CITATION: Remixonwin/amt-airframe-handbook-dataset, p.69 (page 69)
```

Out-of-corpus questions print `ANSWER: Insufficient verified source information.`
`--json` gives machine-readable output.

---

## Command cheat sheet

```powershell
Set-Location "D:\code1\agentic ai\aerofix"                       # always first

# dependencies (fresh machine)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check

# knowledge base
.\.venv\Scripts\python.exe scripts\ingest_datasets.py            # ~26 min, rerun-safe
.\.venv\Scripts\python.exe ingest.py                              # optional: local PDFs

# run
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501   # Ctrl+C to stop

# test / verify
.\.venv\Scripts\python.exe -m pytest                              # expect 30 passed
.\.venv\Scripts\python.exe scripts\ask.py "your question"         # CLI answer

# optional maintenance
.\.venv\Scripts\python.exe -m eval.run_eval                       # retrieval report
.\.venv\Scripts\python.exe scripts\build_world_map.py             # rebuild map asset*
```

\* Only needed if you change the map geometry, and only when the Natural Earth
source file exists at the path in `scripts/build_world_map.py` — the finished
asset is already bundled in `ui/assets/world-land.json` and inside `ui/sky.html`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Port 8501 is already in use` | Kill the stale server: `Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` then start again. |
| `ModuleNotFoundError: sentence_transformers` / `datasets` / `dotenv` / `playwright` | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` (requirements.txt includes all runtime imports). |
| Terminal shows `[aerofix] OPENAI_API_KEY is not set; using the local MockLLM ...` | **Normal** — no key configured. Add `OPENAI_API_KEY=...` to `.env` to use a real provider. |
| `The procedure draft could not be created.` appears | Should not happen after the fix. The **real traceback now prints in the terminal** — look there, and re-run the command in that window to see it. |
| Changed files in `aerofix\*.py` but nothing changed in the browser | Restart Streamlit — the server keeps old Python modules in memory (`sky.html` and `app.py` reload automatically, `aerofix\*` does not). |
| Streamlit log full of `ModuleNotFoundError: No module named 'torchvision'` | **Harmless** — Streamlit's file watcher probing optional transformers modules. Ignore. |
| `execution of scripts is disabled` when activating `.venv` | Skip activation; call `.\.venv\Scripts\python.exe ...` directly (all commands above already do). |
| Playwright browser/channel error (e.g. on a machine without Edge) | Windows tests use system Edge. Elsewhere: `.\.venv\Scripts\python.exe -m playwright install chromium`. |
| First model/dataset download fails | Check internet, then re-run the same command — everything resumes from cache. Later runs work fully offline. |
| Want a clean re-embed of the index | **Don't** use `--rebuild` unless the index only contains old fastembed/mock vectors. With the current index, delete `.aerofix\chroma` and re-run `scripts\ingest_datasets.py` if you really must start over. |
| Streamlit log full of deprecation warnings (`st.components.v1.html`, `use_container_width`) | **Cosmetic only** — Streamlit API deprecation notices from upstream; they don't affect the app. |

## Fresh-machine checklist

1. `py -3.11 -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.\.venv\Scripts\python.exe -m pip check`
4. `.\.venv\Scripts\python.exe scripts\ingest_datasets.py` → `Passages: 19707`
5. `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501`
6. `.\.venv\Scripts\python.exe -m pytest` → `30 passed`

If all six succeed, the project is set up correctly.
