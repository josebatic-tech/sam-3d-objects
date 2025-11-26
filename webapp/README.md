# sam3d Webapp

This folder contains a minimal FastAPI backend and a tiny frontend to run inference using the repository's `Inference` class.

Quick start (inside the dev container):

1. Create a Python environment and install dependencies (example using pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pillow numpy
# this project also requires the repo's inference dependencies (torch, pytorch3d, etc.)
```

2. Make sure you have a valid `pipeline.yaml` config for the model under `checkpoints/` (the server tries to find `checkpoints/**/pipeline.yaml`). If your config is elsewhere, set `SAM3D_CONFIG` to point to it.

```bash
export SAM3D_CONFIG=/path/to/checkpoints/hf/pipeline.yaml
```

3. Run the backend:

```bash
python -m webapp.backend.main
```

4. Open `http://localhost:8000/` in your browser and upload an image (and optional mask).

Notes:
- Loading and running the model likely requires GPU and the project's full dependencies. If the model fails to load, check the server logs.
- Results (PLY files) are saved under `webapp/backend/static/results/` and served at `/results/<file>`.
