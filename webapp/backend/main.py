import os
import sys
import glob
import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

# allow importing the repository's demo/inference helpers
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "notebook"))

try:
    from inference import Inference  # type: ignore
except Exception as e:
    Inference = None  # will raise on startup if used

app = FastAPI(title="sam3d-webapp")

# static results folder
RESULTS_DIR = Path(__file__).resolve().parents[1] / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# mount frontend static files
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def find_config() -> Optional[str]:
    # try to find a pipeline.yaml under checkpoints/*
    checks = list((ROOT / "checkpoints").glob("**/pipeline.yaml"))
    if checks:
        return str(checks[0])
    # fallback to a path used in demo.py
    tag_path = ROOT / "checkpoints" / "hf" / "pipeline.yaml"
    if tag_path.exists():
        return str(tag_path)
    return None


MODEL = None


def ensure_model_loaded():
    """Lazy-load the Inference model on first use. Returns tuple (ok, message)."""
    global MODEL
    if MODEL is not None:
        return True, "loaded"
    if Inference is None:
        return False, "Could not import `Inference` from notebook/inference.py"
    cfg = os.environ.get("SAM3D_CONFIG")
    if cfg is None:
        cfg = find_config()
    if cfg is None:
        return False, "Could not find a pipeline.yaml under `checkpoints/`. Set SAM3D_CONFIG env var to the config path."
    try:
        MODEL = Inference(cfg, compile=False)
    except Exception as e:
        return False, f"Failed to instantiate Inference: {e}"
    return True, "loaded"


def pil_to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    return arr.astype(np.uint8)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<html><body><h1>sam3d webapp</h1></body></html>")


@app.get("/health")
def health():
    return {"ready": MODEL is not None}


@app.post("/infer")
async def infer(image: UploadFile = File(...), mask: Optional[UploadFile] = File(None), seed: Optional[int] = Form(None)):
    ok, msg = ensure_model_loaded()
    if not ok:
        return JSONResponse({"error": msg}, status_code=503)

    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGBA")
    img_np = pil_to_np(img)

    mask_np = None
    if mask is not None:
        mcontent = await mask.read()
        mimg = Image.open(io.BytesIO(mcontent)).convert("L")
        mask_np = (np.array(mimg) > 0).astype(np.uint8)

    try:
        out = MODEL(img_np, mask_np, seed=seed)
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {e}"}, status_code=500)

    # attempt to save a PLY if available
    ply_path = None
    # common keys observed in demos: 'gs', 'gaussian', 'gaussians'
    try:
        if isinstance(out, dict):
            if "gs" in out and hasattr(out["gs"], "save_ply"):
                ply_name = f"{uuid.uuid4().hex}.ply"
                ply_path = RESULTS_DIR / ply_name
                out["gs"].save_ply(str(ply_path))
            elif "gaussian" in out:
                g = out["gaussian"]
                if isinstance(g, (list, tuple)) and len(g) > 0 and hasattr(g[0], "save_ply"):
                    ply_name = f"{uuid.uuid4().hex}.ply"
                    ply_path = RESULTS_DIR / ply_name
                    g[0].save_ply(str(ply_path))
    except Exception as e:
        # don't fail entirely if PLY writing fails
        ply_path = None

    resp = {"ok": True}
    if ply_path is not None:
        resp["ply_url"] = f"/results/{ply_path.name}"

    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host="0.0.0.0", port=8000, reload=False)
