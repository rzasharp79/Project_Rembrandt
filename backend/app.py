import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ------------ Settings Manager ------------
class AppSettings(BaseModel):
    models_root: str = Field(default=os.path.abspath("models"))
    outputs_root: str = Field(default=os.path.abspath("outputs"))
    allow_url_downloads: bool = False
    allow_online_thumbnails: bool = False
    device: str = "auto"  # auto|cpu|cuda
    precision: str = "fp16"
    memory_guard: bool = True
    default_cfg: float = 7.5
    default_steps: int = 30
    default_sampler: str = "Euler a"
    default_clip_skip: int = 0
    theme: str = "auto"


class SettingsStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self.save(AppSettings())

    def load(self) -> AppSettings:
        try:
            import json
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppSettings(**data)
        except Exception:
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        import json
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(settings.model_dump(), f, indent=2)


# ------------ Sampler Mapping ------------
BALANCED_SAMPLERS = ["Euler a", "Euler", "DPM++ SDE", "DPM++ 2M", "UniPC"]
SPEED_SAMPLERS = ["LCM", "Turbo"]  # shown only when compatible


def map_sampler(name: str) -> str:
    # Placeholder mapping to scheduler identifiers; real engines can refine
    mapping = {
        "Euler a": "euler_ancestral",
        "Euler": "euler",
        "DPM++ SDE": "dpmpp_sde",
        "DPM++ 2M": "dpmpp_2m",
        "UniPC": "unipc",
        "LCM": "lcm",
        "Turbo": "turbo",
    }
    return mapping.get(name, "euler_ancestral")


# ------------ Request/Response Models ------------
class LoraSpec(BaseModel):
    path: str
    weight: float = 1.0


class EmbeddingSpec(BaseModel):
    path: str
    token: str


class GenerateRequest(BaseModel):
    modelPath: str
    prompt: str
    negativePrompt: Optional[str] = ""
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    seed: Optional[int] = None
    seedLocked: bool = False
    clipSkip: int = 0
    batchCount: int = 1
    loras: List[LoraSpec] = Field(default_factory=list)
    posEmbeddings: List[EmbeddingSpec] = Field(default_factory=list)
    negEmbeddings: List[EmbeddingSpec] = Field(default_factory=list)


class ImageInfo(BaseModel):
    id: str
    fileName: str
    width: int
    height: int
    seed: int
    sampler: str
    path: str


class JobStatus(BaseModel):
    status: str
    progress: int
    images: List[ImageInfo]
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ------------ Minimal PNG writer (no external deps) ------------
import struct
import zlib
import io


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(
        ">I", len(data)
    ) + tag + data + struct.pack(
        ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
    )


def write_png(path: str, width: int, height: int, rgba: bytes, text_key: str, text_value: str) -> None:
    # rgba expected length = width*height*4
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB", width, height, 8, 6, 0, 0, 0
    )  # 8-bit, RGBA
    # Add tEXt chunk with metadata JSON (might be large, but fine for placeholder)
    text_data = text_key.encode("latin-1") + b"\x00" + text_value.encode("latin-1", errors="replace")
    # Build raw scanlines: each row prefixed with filter type 0
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride : (y + 1) * stride] for y in range(height))
    idat = zlib.compress(raw)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"tEXt", text_data))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


def png_bytes(width: int, height: int, rgba: bytes, text_key: str | None = None, text_value: str | None = None) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB", width, height, 8, 6, 0, 0, 0
    )  # 8-bit, RGBA
    chunks = []
    chunks.append(_png_chunk(b"IHDR", ihdr))
    if text_key and text_value is not None:
        text_data = text_key.encode("latin-1") + b"\x00" + text_value.encode("latin-1", errors="replace")
        chunks.append(_png_chunk(b"tEXt", text_data))
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride : (y + 1) * stride] for y in range(height))
    idat = zlib.compress(raw)
    chunks.append(_png_chunk(b"IDAT", idat))
    chunks.append(_png_chunk(b"IEND", b""))
    return sig + b"".join(chunks)


def make_placeholder_image(path: str, width: int, height: int, meta: dict) -> None:
    # Simple gradient placeholder with embedded metadata
    import json

    pixels = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            r = int(255 * x / max(1, width - 1))
            g = int(255 * y / max(1, height - 1))
            b = 160
            a = 255
            pixels[i : i + 4] = bytes((r, g, b, a))
    write_png(path, width, height, bytes(pixels), "parameters", json.dumps(meta))


# ------------ Job Manager & Engine Abstraction ------------
class Job:
    def __init__(self, req: GenerateRequest, outputs_root: str):
        self.id = uuid.uuid4().hex
        self.req = req
        self.status = "queued"
        self.progress = 0
        self.images: List[ImageInfo] = []
        self.warnings: List[str] = []
        self.error: Optional[str] = None
        self._cancel = False
        self.outputs_root = outputs_root
        self.run_dir = self._make_run_dir()

    def _make_run_dir(self) -> str:
        now = datetime.now()
        d = now.strftime("%Y-%m-%d")
        t = now.strftime("%H%M%S")
        out_dir = os.path.join(self.outputs_root, d, t)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def cancel(self):
        self._cancel = True


class JobManager:
    def __init__(self, settings: SettingsStore):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.settings_store = settings
        self.active_model: Optional[str] = None

    def create(self, req: GenerateRequest) -> Job:
        st = self.settings_store.load()
        job = Job(req=req, outputs_root=st.outputs_root)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run_job, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        job.cancel()

    def _run_job(self, job: Job):
        job.status = "running"
        req = job.req
        # Validate and possibly swap sampler
        if req.sampler not in BALANCED_SAMPLERS + SPEED_SAMPLERS:
            job.warnings.append("Unknown sampler; swapped to Euler a")
            req.sampler = "Euler a"

        total_steps = max(1, req.steps)
        # Seed behavior
        base_seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")

        # Simulate generation per image; write placeholder PNGs with embedded metadata
        for i in range(req.batchCount):
            if job._cancel:
                job.status = "canceled"
                job.progress = 0
                return

            # Simulate step progress
            for step in range(total_steps):
                if job._cancel:
                    job.status = "canceled"
                    return
                # Simulate work
                time.sleep(0.01)  # keep it responsive; real engine does work here
                # update progress coarse-grained
                pct = int(((i * total_steps) + (step + 1)) / (req.batchCount * total_steps) * 100)
                job.progress = min(99, pct)

            seed_i = base_seed if req.seedLocked else (base_seed + i)
            fname = f"{i+1:03d}_{seed_i}_{req.width}x{req.height}.png"
            fpath = os.path.join(job.run_dir, fname)
            meta = {
                "modelPath": req.modelPath,
                "prompt": req.prompt,
                "negativePrompt": req.negativePrompt or "",
                "sampler": req.sampler,
                "steps": req.steps,
                "cfg": req.cfg,
                "seed": seed_i,
                "clipSkip": req.clipSkip,
                "width": req.width,
                "height": req.height,
                "batchIndex": i,
                "loras": [l.model_dump() for l in req.loras],
                "posEmbeddings": [e.model_dump() for e in req.posEmbeddings],
                "negEmbeddings": [e.model_dump() for e in req.negEmbeddings],
                "appVersion": "1.0",
                "createdAt": datetime.now().isoformat(),
            }
            make_placeholder_image(fpath, req.width, req.height, meta)

            img = ImageInfo(
                id=uuid.uuid4().hex,
                fileName=fname,
                width=req.width,
                height=req.height,
                seed=seed_i,
                sampler=req.sampler,
                path=fpath,
            )
            job.images.append(img)

        job.progress = 100
        job.status = "done"


# ------------ Model Library Scanner ------------
def ext_lower(path: str) -> str:
    return os.path.splitext(path)[1].lower()


CHECKPOINT_EXTS = {".safetensors"}
LORA_EXTS = {".safetensors"}
EMBED_EXTS = {".pt", ".bin", ".npy", ".txt"}


def scan_library(models_root: str) -> dict:
    result = {"checkpoints": [], "loras": [], "embeddings": []}
    for base, _, files in os.walk(models_root):
        lower_base = base.replace("\\", "/").lower()
        for fn in files:
            p = os.path.join(base, fn)
            ext = ext_lower(p)
            # Skip hidden or placeholder files like .gitkeep
            if fn.startswith('.'):
                continue
            entry = {
                "name": os.path.splitext(fn)[0],
                "path": p,
                "size": os.path.getsize(p),
            }
            if "/checkpoints" in lower_base and ext in CHECKPOINT_EXTS:
                result["checkpoints"].append(entry)
            elif "/loras" in lower_base and ext in LORA_EXTS:
                result["loras"].append(entry)
            elif "/embeddings" in lower_base and ext in EMBED_EXTS:
                result["embeddings"].append(entry)
    return result


# ------------ FastAPI App ------------
app = FastAPI(title="Stunning Local Image Generator", version="1.0")

# CORS for local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


settings_store = SettingsStore(path=os.path.abspath("data/settings.json"))
jobs = JobManager(settings_store)


@app.get("/settings", response_model=AppSettings)
def get_settings():
    return settings_store.load()


@app.put("/settings", response_model=AppSettings)
def put_settings(new_settings: AppSettings):
    settings_store.save(new_settings)
    return new_settings


@app.post("/generate")
def post_generate(req: GenerateRequest):
    # Basic validation per spec
    if not req.modelPath:
        raise HTTPException(400, "Missing field: modelPath")
    if not (1 <= req.steps <= 200):
        raise HTTPException(400, "Missing or invalid field: steps (must be 1-200)")
    if not (0.0 <= req.cfg <= 20.0):
        raise HTTPException(400, "Missing or invalid field: cfg (must be 0.0-20.0)")
    if not (1 <= req.batchCount <= 8):
        raise HTTPException(400, "Missing or invalid field: batchCount (must be 1-8)")

    job = jobs.create(req)
    return {"jobId": job.id}


@app.get("/job/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        status=job.status,
        progress=job.progress,
        images=job.images,
        warnings=job.warnings,
        error=job.error,
    )


@app.post("/cancel/{job_id}")
def post_cancel(job_id: str):
    try:
        jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")
    return {"ok": True}


@app.get("/image/{image_id}")
def get_image(image_id: str):
    # naive search across job images; for production, keep an index
    for job in list(jobs._jobs.values()):
        for img in job.images:
            if img.id == image_id:
                try:
                    with open(img.path, "rb") as f:
                        data = f.read()
                    return Response(content=data, media_type="image/png")
                except FileNotFoundError:
                    raise HTTPException(404, "Image not found on disk")
    raise HTTPException(404, "Image not found")


@app.get("/library/index")
def library_index():
    st = settings_store.load()
    if not os.path.exists(st.models_root):
        return {"checkpoints": [], "loras": [], "embeddings": []}
    return scan_library(st.models_root)


@app.get("/favicon.ico")
def favicon():
    # Serve a tiny in-memory PNG to avoid 404 noise
    w, h = 16, 16
    pixels = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 4
            # simple dark-gray square with a lighter diagonal
            base = 40
            diag = 80 if x == y else 0
            c = min(255, base + diag)
            pixels[i : i + 4] = bytes((c, c, c, 255))
    data = png_bytes(w, h, bytes(pixels))
    return Response(content=data, media_type="image/png")


# Optional: serve static frontend if present (build into web/dist)
static_dir = os.path.abspath("web/dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def create_dev_outputs_folder():
    st = settings_store.load()
    os.makedirs(st.outputs_root, exist_ok=True)


create_dev_outputs_folder()
