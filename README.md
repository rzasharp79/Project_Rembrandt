Stunning Local Image Generator
==============================

Local, private, browser-based image generation with a Python backend. Pick a checkpoint on disk, tune CFG/steps/sampler/seed/CLIP skip, add LoRAs and embeddings, and generate batches with a live filmstrip.

Version Control
---------------

- Version: 1.0.0
- Date: 2025-09-01
- Status: Preview (Diffusers engine optional; falls back to placeholder if not installed)
- SemVer: Follows semantic versioning (MAJOR.MINOR.PATCH)

Changelog
---------

- 1.0.0
  - FastAPI backend with job queue, progress, batch seeds, and PNG outputs with embedded metadata
  - React/Vite frontend with Generate screen, filmstrip, sampler/CFG/steps/seed/CLIP skip controls
  - Model Library scan for `models/` (checkpoints/loras/embeddings)
  - Static serving of built UI from backend when available

Table of Contents
-----------------

1. Overview
2. Features
3. Architecture
4. Requirements
5. Installation
6. Quick Start
7. Configuration
8. Folder Layout
9. API Overview
10. Keyboard Shortcuts
11. Development
12. Troubleshooting
13. Roadmap

Overview
--------

This app runs entirely on your machine:

- Backend: Python FastAPI service exposing generation jobs and a library index
- Frontend: React app in your browser at `http://localhost`
- Outputs: PNG files with embedded settings and metadata

Features
--------

- Prompting: Positive and optional negative prompt
- Controls: CFG, Steps, Sampler (balanced set), Seed (+ lock), CLIP skip
- Size: Portrait / Landscape / Square presets + Custom width/height
- Batch: Generate multiple images per run with stable or incrementing seeds
- Viewer: Large preview + filmstrip of batch results
- Library: Scans `models/` for checkpoints, LoRAs, embeddings
- Outputs: `outputs/YYYY-MM-DD/HHMMSS/NNN_seed_WIDTHxHEIGHT.png` with metadata

Architecture
------------

- `backend/` FastAPI app (`backend/app.py`)
  - Endpoints: `/generate`, `/job/{id}`, `/cancel/{id}`, `/image/{imageId}`, `/library/index`, `/settings`
  - Job manager: threaded jobs with progress and cancelation
- Runtime: If Diffusers/Torch are installed, backend runs real Stable Diffusion from local `.safetensors` checkpoints; otherwise it creates lightweight placeholder PNGs.
- `web/` React app (Vite)
  - Generate screen, filmstrip, and model picker wired to API
  - Dev server or static build served by backend

Requirements
------------

Minimum
- OS: Windows 10/11, macOS 13+, or Ubuntu 20.04+
- Python: 3.10 – 3.12
- Node.js: 18+ (LTS recommended), npm 9+
- Disk: ~2 GB free for outputs; models vary (1–10+ GB)
- CPU: 4+ cores, RAM 16 GB recommended

For GPU Acceleration (optional, for real generation later)
- NVIDIA GPU with 12–24 GB VRAM recommended
- Driver: 535+ (or recent) and CUDA toolkit/runtime compatible with your PyTorch build
- PyTorch with CUDA, Diffusers, Transformers, Accelerate, Safetensors

Installation
------------

One‑click install (Windows)
- Batch (CMD): `install.bat` (minimal) or `install.bat --full` (adds PyTorch/Diffusers)
- PowerShell: `powershell -ExecutionPolicy Bypass -File .\install.ps1` or `... -Full`
  - Skip UI build: add `-NoUI`

Backend (Python/FastAPI)
1) Create a virtual env and install dependencies
- `python -m venv .venv`
- Windows: `./.venv/Scripts/Activate`
- macOS/Linux: `source .venv/bin/activate`
- `pip install -r backend/requirements.txt`

2) Run the server
- `python -m uvicorn backend.app:app --reload --port 8000`

Frontend (React/Vite)
1) Install and run in dev mode
- `cd web`
- `npm install`
- Set API URL (if different):
  - PowerShell: `$env:VITE_API_URL = 'http://localhost:8000'`
  - Bash: `export VITE_API_URL=http://localhost:8000`
- `npm run dev` (http://localhost:5173)

2) Or build and serve from backend
- `npm run build` in `web/` to produce `web/dist/`
- Restart the backend; it serves `web/dist` at `http://localhost:8000`

Quick Start
-----------

1) Start backend at `:8000`.
2) Start frontend dev server at `:5173` (or build and let backend serve static files).
3) Create `models/` folders and place your checkpoint(s):
   - `models/checkpoints/`, `models/loras/`, `models/embeddings/`
4) Open the app, pick a model, set prompt and controls, click Generate.

Configuration
-------------

Settings API (`/settings`) persists a JSON file in `data/settings.json`:
- Paths: `models_root`, `outputs_root`
- Privacy: `allow_url_downloads`, `allow_online_thumbnails` (default off)
- Hardware: `device`, `precision`, `memory_guard`
- UI defaults: CFG, steps, sampler, clip skip, theme

Folder Layout
-------------

- Models root (configurable):
  - `models/checkpoints/`
  - `models/loras/`
  - `models/embeddings/`
- Outputs root:
  - `outputs/YYYY-MM-DD/HHMMSS/NNN_seed_WIDTHxHEIGHT.png`

API Overview
------------

- `POST /generate` → `{ jobId }`
- `GET /job/{jobId}` → `{ status, progress, images[], warnings[], error? }`
- `POST /cancel/{jobId}` → `{ ok: true }`
- `GET /image/{imageId}` → PNG binary
- `GET /library/index` → `{ checkpoints[], loras[], embeddings[] }`
- `GET/PUT /settings` → persisted app settings

Keyboard Shortcuts (planned)
----------------------------

- `G` Generate
- `S` Save/Star
- `Del` Delete
- `?` Toggle tooltips

Development
-----------

- Backend: edit `backend/app.py`, then hot-reload via `--reload`
- Frontend: edit files under `web/src/`; Vite dev server hot reloads
- Static build: `npm run build` outputs to `web/dist/` which backend serves automatically
 - Install scripts:
   - Batch: `install.bat` (use `--full` for GPU deps)
   - PowerShell: `install.ps1` (use `-Full` or `-NoUI`)

Troubleshooting
---------------

Backend won’t start (port in use)
- Another process is on `:8000`. Change port: `uvicorn backend.app:app --port 8001` and set `VITE_API_URL` accordingly.

Frontend can’t reach backend (CORS or 404)
- Ensure `VITE_API_URL` points to the backend (e.g., `http://localhost:8000`).
- Open browser devtools → Network tab to confirm requests and responses.

PowerShell execution policy blocks installer
- Run with: `powershell -ExecutionPolicy Bypass -File .\install.ps1`
- Or set policy for current process only: `Set-ExecutionPolicy -Scope Process Bypass`

Batch installer errors with unexpected token
- Run from CMD (not WSL bash). If still failing, use PowerShell installer instead.

Library is empty
- Check your models path: `GET /settings` → `models_root`.
- Place files under `models/checkpoints/` etc. and click “Rescan”.
- File permissions or network drives may block `os.walk()` on some systems.

Generation finishes but images don’t appear
- Confirm outputs path exists and is writable.
- Check `GET /job/{id}` → `images[]` entries. If empty, see server logs.

Slow or no GPU acceleration
- Without Diffusers/Torch installed, the app saves placeholder images (colored gradients). Install full deps to enable real generation.
- For GPU, install a CUDA-enabled PyTorch matching your driver.

“Module not found” or Python version errors
- Use Python 3.10–3.12 and recreate the venv. Run `pip install -r backend/requirements.txt` again.

Node/npm issues
- Use Node 18+ and clear installs: delete `web/node_modules` and run `npm install`.

Static build not served
- Ensure `web/dist` exists (`npm run build`). Restart backend to mount static files.

Insufficient disk space
- Free space or change `outputs_root` via `PUT /settings`. Large batches or high resolutions grow quickly.

Roadmap
-------

- Real Diffusers/PyTorch engine with model hot-swap, sampler compatibility, CLIP skip, LoRAs, embeddings
- Add-by-URL downloads (checksums, progress) and optional online thumbnails
- Presets dialog, Settings panel UI, tooltips, and accessibility polish
- Star/Delete/Copy actions and richer history management

Notes
-----

- Real generation requires installing full backend deps (PyTorch, Diffusers). Use `install.ps1 -Full` or install `backend/requirements.txt` in your venv.
- If full deps aren’t present, the backend gracefully falls back to placeholder images so the UI remains usable.
- See `_instructions_for_agent.md` for the full product brief and test checklist.

# TODO
-----
🟥 Speed up the generation time.</br>
🟥 Add a progress bar. </br>
🟥 Add a textbox that acts as a log for debugging purposes. </br>
🟥 Add a timer that starts when the user hits generate and stops after image generated. </br>
🟥 Fix the ugly UI </br>