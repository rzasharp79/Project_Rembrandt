# Stunning Local Image Generator — Implementation Brief

**Version:** 1.0  
**Date:** September 1, 2025  
**Runs:** Fully local (browser UI + local Python engine)

---

## Project one-liner and goal
- **One-liner:** A local, browser-based app that generates high-quality images from text using a checkpoint you pick from your drive, with controls for CFG, steps, sampler, seed, CLIP skip, LoRAs, positive/negative embeddings, and batch count.
- **Goal:** Deliver a gorgeous, fast, and private experience that lets you choose any compatible model locally and produce repeatable, high-fidelity results without cloud dependencies.

## Target users and main value
- **Users:** Creators, designers, and enthusiasts who want local control, custom models, and repeatable results.
- **Main value:** Point the app at a model on disk, tune a few clear controls, and get beautiful images quickly—no cloud, no login.

## Platforms and devices
- **Platform:** Web app in your browser, running on `http://localhost`.
- **Local engine:** Python service (Diffusers/PyTorch) on the same machine (Windows/Linux/macOS supported).
- **Hardware target:** NVIDIA GPU with ~12–24GB VRAM (half precision by default; CPU fallback allowed but slow).

---

## Navigation map
- **Home / Generate** → default route; shows prompt, core controls, sampler, batch, filmstrip + large viewer.  
- **Model Library** (modal/drawer) → scan + pick checkpoints / LoRAs / embeddings.  
- **Presets** (small dialog) → save/load named presets; choose which fields they affect.  
- **Settings** (panel) → hardware, paths, privacy, updates, CLIP skip default, UI theme = auto.  
- **Help** (toggleable “?” tooltips) → inline explanations next to key controls.

---

## Screen-by-screen details

### Home / Generate
**Sections**
- **Prompt**  
  - One large text box for the **Positive prompt** with live character count.  
  - Button **Add negative prompt** reveals a single-line **Negative prompt** field beneath (hidden by default).
- **Core Controls (row)**  
  - **CFG** slider + number box.  
  - **Steps** slider + number box.  
  - **Sampler** dropdown (Balanced set + Speed set): Euler a, Euler, DPM++ SDE, DPM++ 2M, UniPC; plus speed options for LCM/Turbo models (only shown if compatible).  
  - **Seed** box (+ lock icon).  
  - **CLIP skip** dropdown (0 = off, then 1,2,3…; grays out if unsupported by the current pipeline).
- **Size & Orientation**  
  - Chips: **Portrait**, **Landscape**, **Square**.  
  - Size dropdown with common presets; width/height fields unlock when **Custom** is chosen.
- **Model & Add-ons**  
  - **Model chip** showing selected checkpoint name (+ swap button opens Model Library).  
  - **LoRAs**: inline chips with tiny +/- weight nudgers and remove **×**.  
  - **Embeddings**: two lists (Positive / Negative).  
  - Type-to-add with autocomplete: `@lora/<name>` and `#embed/<name>`.
- **Batch**  
  - **Batch count** stepper (e.g., 1–8).  
  - **Generate** button (disabled while running; shows spinner and live status).
- **Viewer**  
  - **Large preview** of the selected image.  
  - **Filmstrip** of batch results and previous session images (current run first).  
  - Buttons: **Open folder**, **Copy**, **Star**, **Delete**.
- **Tooltips**  
  - Small **?** icons next to CFG, Steps, Sampler, CLIP skip, Size.

**Behaviors**
- Generate starts immediately with current settings; button turns to **Cancel** while running.  
- Filmstrip updates as images finish.  
- Clicking a filmstrip item selects it in the large viewer.

### Model Library (modal/drawer)
**Tabs**  
- **Checkpoints** (SD1.5, SDXL; optional SD3/FLUX if locally available/compatible).  
- **LoRAs**  
- **Embeddings**

**Features**  
- Top path shows the main **models folder**. Subfolders:  
  - `models/checkpoints/`  
  - `models/loras/`  
  - `models/embeddings/`
- **Smart scan** across multiple locations; supports **symlinks/shortcuts**.  
- Search bar, tags, and optional hover preview (uses sidecar preview images if present or fetched online if allowed).  
- Each item has: name, file path, size, detected type/format, and quick **Add** button.  
- Import: **Choose file…** (one-off reference, no copy), **Add by URL** (downloads into the correct subfolder if allowed in privacy settings).

### Presets (dialog)
- **Quick-save “Last used” as Default.**  
- **Named presets** list with Save/Load/Delete.  
- **Field scope selector**: choose which fields the preset controls (e.g., only sampler + steps, or everything except prompt).  
- **Lock** toggle per preset (prevents silent overwrite).

### Settings (panel)
- **Paths:** models root, outputs root.  
- **Privacy:** offline by default, manual update check; allow fetching model thumbnails/tags; allow URL downloads.  
- **Hardware:** device choice (auto-select best NVIDIA), precision (fp16), memory guard on/off.  
- **UI:** theme **Auto**, density normal, show tooltips by default.  
- **Default values:** CFG, steps, sampler, size, CLIP skip.  
- **Compatibility:** enable/disable CLIP skip where supported.

---

## Data items needed

### Local storage (files)
- **Outputs folder**: `outputs/YYYY-MM-DD/HHMMSS/` containing PNGs.  
  - Embed settings into PNG metadata (model path, prompts, CFG, steps, sampler, seed, clip skip, size, LoRAs + weights, embeddings lists, app version).  
- **Presets file(s)**: JSON storing named presets and field scopes.  
- **Library index cache**: JSON with item names, tags, paths, hashes, preview thumbnails (if allowed), last seen.

### In-memory/session
- Current settings object.  
- Current model bindings (checkpoint, tokenizer/text encoder status, active LoRAs/embeddings).  
- Batch job state and per-image progress.

---

## External connections (plain description)
- **None required for generation.** All image creation is local.  
- **Optional:**  
  - **Check for updates** button pings a release JSON once when pressed.  
  - Fetch preview images/tags for known models during library scan (only if toggled on).  
  - **Add by URL** downloads model files into your library when you paste a URL and confirm.

---

## Rules and edge cases
- If a chosen sampler isn’t compatible with the current model/pipeline, auto-swap to the closest match and show a small toast.  
- If VRAM is low, reduce batch size first; then reduce size; finally suggest CPU fallback (confirm before switching).  
- CLIP skip only appears if the active pipeline supports it; otherwise it’s disabled with a tooltip.  
- If a LoRA/embedding token name collides, show a rename prompt or namespace it automatically.  
- If model load fails (bad path, missing files), keep previous model active and show a clear error with a **Fix paths** link.  
- Deleting an image from the filmstrip moves it to a local trash folder (undo for 10 minutes).  
- URL downloads require explicit confirmation and show remaining disk space estimate.

---

## Content and sample labels
- Buttons: **Generate**, **Cancel**, **Swap model**, **Open folder**, **Copy**, **Star**, **Delete**, **Add by URL**, **Choose file…**  
- Placeholders:  
  - Positive prompt: “e.g., dramatic portrait, soft rim light, 85mm look”  
  - Negative prompt: “e.g., blurry, watermark, extra fingers”  
  - Search in library: “Find model, lora, or embedding…”  
- Tooltips (short):  
  - **CFG:** “How strongly the prompt guides the image.”  
  - **Steps:** “More steps = more detail, slower.”  
  - **Sampler:** “Different ways to refine noise into an image.”  
  - **CLIP skip:** “Skips early text-encoder layers for a different style.”

---

## Look and feel
- **Theme:** Auto-match system light/dark with smooth transitions.  
- **Style:** Modern, calm; roomy spacing; rounded corners; subtle shadows; focus rings on all interactive elements.  
- **Motion:** Tiny fades and slides (200ms). No distracting animations.  
- **Icons:** Clean line icons; clear labels next to important controls.

---

## Accessibility and inclusivity
- Keyboard navigation across all controls; visible focus states.  
- Labels tied to inputs; live progress updates announced for assistive tech.  
- Adjustable font size (OS-level honored); minimum 14px base.  
- Color contrast meets WCAG AA in both light and dark.  
- Tooltips accessible via keyboard and on focus, not hover-only.

---

## Performance expectations
- Model load: under ~10s for SD1.5; under ~20s for SDXL on first load (varies by disk).  
- Generation: first 1024-square image with SDXL on a 12–24GB GPU should feel responsive; show per-step preview in the large viewer if available.  
- UI actions: under 100ms for clicks/toggles.  
- Batch: images appear in filmstrip as they complete; never block the UI.

---

## Privacy and safety
- Offline by default; no telemetry.  
- Update check and preview fetching run **only** when toggled on.  
- Prompts and images are saved locally; option to star/keep versus delete.  
- Clear disk space warnings before large downloads.  
- Local password lock is not included by default (add later if needed).

---

## Test checklist (pass/fail)
- [ ] App launches at `http://localhost` and shows the Generate screen.  
- [ ] Models folder is detected; user can pick a checkpoint and generate an image.  
- [ ] Positive prompt only → image generates; adding negative prompt after clicking the button works.  
- [ ] CFG, Steps, Sampler, Seed, CLIP skip changes affect output; unsupported CLIP skip is disabled with a tooltip.  
- [ ] Batch count > 1 → multiple images appear in filmstrip and can be selected.  
- [ ] LoRAs can be added via chips and weight adjusted; embeddings appear under the correct Positive/Negative lists.  
- [ ] Size/orientation chips change width/height correctly.  
- [ ] Outputs are saved to `outputs/YYYY-MM-DD/HHMMSS/` with embedded metadata.  
- [ ] Presets: quick-save default works; named presets save/load; scoped fields behave as chosen.  
- [ ] Optional: model preview fetch shows thumbnails when allowed; Add by URL downloads into the right subfolder.  
- [ ] Cancel stops generation cleanly; partial images are discarded (no crash).  
- [ ] Tooltips toggle on/off and are screen-reader friendly.

---

## Assumptions I made
- You’ll store models under a main `models/` directory with subfolders; more roots can be added.  
- SD1.5 and SDXL are the primary targets; SD3/FLUX support depends on locally available, compatible pipelines or adapters.  
- Half precision (fp16) is acceptable; memory-saving attention is allowed.  
- You want PNG with embedded metadata; no sidecar files by default.  
- No in-app editing (inpaint/outpaint) on day one.

---

## Stretch goals / later ideas
- Image-to-Image, Inpaint/Outpaint tools.  
- Two-pass refine or separate refiner model for SDXL/SD3.  
- Tiled upscaling and face enhancement.  
- History panel with tags/ratings and quick compare slider.  
- Queue management (pause/reorder) and multi-GPU awareness.  
- Project workspaces and project-scoped presets.  
- Private session mode and local password lock.  
- Model compatibility advisor (warns about mismatched LoRAs/architectures).

---

# Ready-to-use prompts for a coding tool (clearly scoped tasks)

> **How to use:** Paste these task blocks into tools like Codex CLI, Claude CLI, Windsurf, or Replit Agents. They are concrete, step-by-step, and avoid jargon.

## Backend (Python + FastAPI + Diffusers) — Task Block
1. Create a FastAPI service with endpoints:
   - `POST /generate` — starts a job using fields below; returns `jobId` immediately.
   - `GET /job/{jobId}` — returns status (`queued|running|done|error|canceled`), progress percent, and list of produced images (IDs + paths) so far.
   - `GET /image/{imageId}` — serves a PNG from disk.
   - `POST /cancel/{jobId}` — cancels a running job.
2. Implement a **model loader** that caches the active pipeline and hot-swaps when `modelPath` changes. Use fp16 where supported and enable memory-efficient attention. Detect and apply CLIP skip only if the pipeline supports it.
3. Implement **sampler mapping** (see Appendix A) and validate compatibility per pipeline. If incompatible, auto-swap to the closest supported sampler and include a warning in the response.
4. Apply **LoRAs** and **embeddings** per request:
   - LoRAs: load adapters by file path with per-adapter weight in the range `[-2.0, 2.0]` (default `0.8–1.0`). Allow multiple LoRAs.
   - Embeddings (textual inversion): load from file path, register token names, and include them in the prompt processing for Positive or Negative.
5. Implement **batch generation** with seed behavior:
   - If `seedLocked = true`, reuse the same seed across the batch; otherwise, increment from a base seed.
   - Stream per-image progress milestones to `/job/{jobId}` (e.g., each 10% or each step) so the UI can update the filmstrip.
6. **Outputs:** Save to `outputs/YYYY-MM-DD/HHMMSS/`. Embed all settings and metadata into PNG. File name pattern: `NNN_seedWIDTHxHEIGHT.png` (e.g., `001_12345_1024x1024.png`).
7. **Library indexer:** On request, scan configured model roots, follow symlinks, detect item type by extension/name, and return a JSON index. If allowed by settings, attempt to load preview images/tags from local sidecars or online sources.
8. **Add by URL:** Given a URL and a target type (checkpoint/LoRA/embedding), download into the appropriate subfolder with a temp file and checksum validation. Return progress and final path.
9. **Errors:** Use clear messages; include a short `actionHint` (e.g., “Open Settings → Paths”). Return HTTP 400 for bad inputs, 409 for incompatible settings, 500 for internal errors.

### Request fields for `POST /generate`
- `modelPath` (string, required) — absolute or library path to checkpoint.
- `prompt` (string, required) — positive prompt.
- `negativePrompt` (string, optional).
- `width` (int, required) — pixels; typical 640–1024 for SDXL.
- `height` (int, required) — pixels.
- `steps` (int, required) — 1–200; default 30.
- `cfg` (float, required) — 0.0–20.0; default 7.0–8.0 depending on model.
- `sampler` (string, required) — one of: Euler a, Euler, DPM++ SDE, DPM++ 2M, UniPC; plus Speed set when compatible.
- `seed` (int, optional) — 0–2^32-1. If omitted, generate randomly.
- `seedLocked` (bool, optional) — default false. When true, use the same seed across the batch.
- `clipSkip` (int, optional) — 0 (off) or positive small integers; apply only when supported.
- `batchCount` (int, required) — 1–8.
- `loras` (array, optional) — items: { `path` (string), `weight` (float) }.
- `posEmbeddings` (array, optional) — items: { `path` (string), `token` (string) }.
- `negEmbeddings` (array, optional) — items: { `path` (string), `token` (string) }.

### Response fields for `GET /job/{jobId}`
- `status` — `queued|running|done|error|canceled`.
- `progress` — 0–100.
- `images` — array of: { `id`, `fileName`, `width`, `height`, `seed`, `sampler`, `path` } (include completed images only).
- `warnings` — array of strings (e.g., “Sampler swapped to Euler a”).
- `error` — present only when `status=error` (short description + `actionHint`).

## Frontend (React) — Task Block
1. Build the **Generate** screen exactly as specified in *Screen-by-screen details*. Use a single main column with responsive rows for controls.
2. Implement the **Sampler** dropdown with two groups: **Balanced** and **Speed**. Hide incompatible options for the current model; show a small note if a selection was auto-swapped.
3. Create the **Model Library** modal with tabs, search, tags, hover preview, and actions: **Choose file…** and **Add by URL**.
4. Build **LoRA & Embeddings UI**: chips with +/- weight for LoRAs; Positive/Negative embedding lists; type-to-add with `@lora/` and `#embed/` autocomplete from the library index.
5. Implement the **Filmstrip + Large viewer**: thumbnails populate as images complete. Actions: **Open folder**, **Copy**, **Star**, **Delete**.
6. Add the **Presets** dialog: quick-save default; named presets; field scope selector; lock toggle.
7. Add the **Settings** panel for paths, privacy toggles, hardware, defaults, and theme (Auto). Persist choices to a local JSON config.
8. Add **Tooltips** with concise explanations next to CFG, Steps, Sampler, CLIP skip, and Size. Provide a header toggle to enable/disable tooltips.

## Quality & UX — Task Block
1. Show **progress toasts** for model loads, auto-sampler swaps, and VRAM guard adjustments.
2. Use plain-English **error messages** with a short next step (e.g., “Open Settings → Paths”).
3. Add **keyboard shortcuts**: `G` generate, `S` save/star, `Del` delete, `?` toggle tooltips.
4. Ensure **accessibility**: focus outlines, semantic labels, aria-live for progress.

---

# Appendix (for implementers)

## Appendix A — Sampler mapping (target behavior)
- **Euler a** → Ancestral Euler scheduler.  
- **Euler** → Euler scheduler.  
- **DPM++ SDE** → Stochastic sampling (SDE) variant of DPM++ scheduler.  
- **DPM++ 2M** → DPM++ 2M scheduler.  
- **UniPC** → Unified Predictor-Corrector scheduler.  
- **Speed set** (when compatible): samplers/schedulers optimized for LCM/Turbo-like models.

> If a requested sampler is not available for the active model/pipeline, auto-swap to the closest supported option and add a warning string to the job response.

## Appendix B — Folder layout (expected)
- **Models root** (configurable):  
  - `models/checkpoints/` (e.g., SD1.5, SDXL)  
  - `models/loras/`  
  - `models/embeddings/`
- **Outputs root**:  
  - `outputs/YYYY-MM-DD/HHMMSS/` → PNG files per batch.
- **Config & cache**:  
  - A small local JSON config for paths and UI settings.  
  - Library index cache JSON with model entries and optional thumbnails.

## Appendix C — Preset schema (conceptual)
- **Preset** = { `name`, `locked` (bool), `scope` (list of fields), `values` (key→value pairs) }  
- **Scope examples:** `sampler`, `steps`, `cfg`, `size`, `seed`, `clipSkip`, `loras`, `embeddings`, `orientation`, `batchCount`.  
- **Behavior:** When applying a preset, only fields in `scope` override current settings.

## Appendix D — Image metadata keys (embed into PNG)
- `modelPath`, `prompt`, `negativePrompt`, `sampler`, `steps`, `cfg`, `seed`, `clipSkip`, `width`, `height`, `batchIndex`, `loras` (list of {path, weight}), `posEmbeddings` (list of {path, token}), `negEmbeddings` (list of {path, token}), `appVersion`, `createdAt`.

## Appendix E — Progress & status events (contract)
- **Job status states:** `queued`, `running`, `done`, `error`, `canceled`.  
- **Progress granularity:** at least each 10% of steps or each image completion.  
- **Event fields:** `jobId`, `status`, `progress`, `currentStep`, `totalSteps`, `lastImageId` (optional), `warnings[]`.

## Appendix F — Model & feature compatibility notes
- **SD1.5** — broad LoRA & embedding support; works well at 512–768 square; CLIP skip often available.  
- **SDXL** — sharper detail; prefers 1024 square; ensure VRAM > 12GB for comfort; CLIP skip depends on pipeline.  
- **SD3 / FLUX** — heavier; support only if locally available and compatible with your chosen libraries; LoRA/embedding ecosystems are newer.  
- **CLIP skip** — expose only when supported by the active pipeline; otherwise disable the control and show an explanatory tooltip.

## Appendix G — Error codes & example messages
- **400 Bad Request** — “Missing or invalid field: steps (must be 1–200).”  
- **409 Conflict** — “Sampler not supported by current pipeline; auto-swapped to Euler a.”  
- **500 Internal Error** — “Model load failed. Action: Open Settings → Paths and reselect the checkpoint.”  
- **507 Insufficient Storage** — “Not enough disk space to save outputs. Action: Free space or change Outputs path in Settings.”

---

## Final check for CLI-readiness
- All tasks are broken into concrete steps with clear inputs/outputs.  
- Field names and valid ranges are specified.  
- API contract (endpoints, states, and payload fields) is defined without relying on library-specific jargon.  
- UI behaviors, labels, and edge cases are explicit.  
- No external dependencies are required for generation; optional online calls are clearly marked and gated by settings.

> This markdown is structured for copy/paste into your preferred coding assistant. It avoids code, focuses on unambiguous tasks, and provides appendices for exact contracts and values.

