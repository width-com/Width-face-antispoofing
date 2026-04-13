"""
FLIP Face Anti-Spoofing Voting Service
Three models vote on whether a face image is real or spoof.
GPU accelerated with pre-cached text features.
"""

import os
import logging
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from fas import flip_mcl
import clip
from prompt_templates import spoof_templates, real_templates
from infer_one_image import (
    load_ckpt_any,
    extract_state_dict,
    infer_mcl_dims_from_state_dict,
    build_test_transform,
)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------
# Device
# ---------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------
# Model config
# ---------------------
MODEL_PATHS = {
    "cefa": "./models/cefa_flip_mcl.pth.tar",
    "wmca": "./models/wmca_flip_mcl.pth.tar",
    "surf": "./models/surf_flip_mcl.pth.tar",
}

MODEL_DOWNLOAD_LINKS = {
    "cefa": "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/koushik_srivatsan_mbzuai_ac_ae/EVYJub_HzZ5NjLaLQ_WwrfEBQcuY9yCs12knWWRxbcJToQ?e=A9wWGH",
    "surf": "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/koushik_srivatsan_mbzuai_ac_ae/EdbVYxkP21pPmIhdkl6n7joBEZyKennbpsoBloZma4FYnw?e=OJFqfQ",
}

IMAGE_SIZE = 224

# Global state
models: Dict[str, flip_mcl] = {}
cached_text_features: torch.Tensor = None
cached_logit_scale: torch.Tensor = None


def ensure_models_exist():
    """Check that all model files exist; raise with download instructions if missing."""
    missing = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            url = MODEL_DOWNLOAD_LINKS.get(name, "see docs/model_zoo.md")
            missing.append(f"  - {path}  (download from: {url})")
    if missing:
        msg = (
            "Missing model checkpoints:\n"
            + "\n".join(missing)
            + "\n\nPlease download the 0-shot WCS checkpoints from docs/model_zoo.md "
            "and place them in ./models/"
        )
        raise FileNotFoundError(msg)


def load_single_model(name: str, ckpt_path: str) -> flip_mcl:
    """Load a single flip_mcl model from checkpoint, move to DEVICE."""
    logger.info(f"Loading model: {name} from {ckpt_path}")
    ckpt_obj = load_ckpt_any(ckpt_path, device="cpu")
    state = extract_state_dict(ckpt_obj)
    in_dim, ssl_mlp_dim, ssl_emb_dim = infer_mcl_dims_from_state_dict(state)
    logger.info(f"  {name}: in_dim={in_dim}, ssl_mlp_dim={ssl_mlp_dim}, ssl_emb_dim={ssl_emb_dim}")

    model = flip_mcl(in_dim=in_dim, ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, device=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def precompute_text_features(clip_model):
    """Cache text features on DEVICE — same CLIP backbone for all models."""
    global cached_text_features, cached_logit_scale
    clip_model.eval()
    spoof_texts = clip.tokenize(spoof_templates).to(DEVICE)
    real_texts = clip.tokenize(real_templates).to(DEVICE)
    spoof_emb = clip_model.encode_text(spoof_texts).mean(dim=0)
    real_emb = clip_model.encode_text(real_texts).mean(dim=0)
    text_features = torch.stack([spoof_emb, real_emb], dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cached_text_features = text_features
    cached_logit_scale = clip_model.logit_scale.exp()
    logger.info(f"Text features cached on {DEVICE}.")


def load_all_models():
    """Load all three models into memory."""
    global models
    ensure_models_exist()
    for name, path in MODEL_PATHS.items():
        models[name] = load_single_model(name, path)
    first_model = next(iter(models.values()))
    precompute_text_features(first_model.model)
    logger.info(f"All {len(models)} models loaded on {DEVICE}.")


@torch.no_grad()
def infer_single_model(model: flip_mcl, x: torch.Tensor) -> Dict[str, Any]:
    """Run inference with a single model using cached text features on GPU."""
    image_features = model.model.encode_image(x)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = cached_logit_scale * image_features @ cached_text_features.t()
    probs = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    return {
        "label": "real" if pred == 1 else "spoof",
        "score_real": float(probs[1].item()),
        "score_spoof": float(probs[0].item()),
    }


def read_image_from_bytes(img_bytes: bytes) -> Image.Image:
    """Read image from bytes and convert to PIL Image (224x224)."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))
    return pil_img


# ---------------------
# FastAPI App
# ---------------------
app = FastAPI(title="FLIP Face Anti-Spoofing Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_all_models()


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/index.html")
async def index_fallback():
    return RedirectResponse(url="/static/index.html")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        pil_img = read_image_from_bytes(img_bytes)
    except Exception as e:
        return JSONResponse(content={
            "code": 400,
            "message": f"Invalid image: {e}",
            "data": None,
        }, status_code=400)

    try:
        x = build_test_transform()(pil_img).unsqueeze(0).to(DEVICE)

        results = {}
        for name, model in models.items():
            results[name] = infer_single_model(model, x)

        # Voting: majority vote on label
        labels = [r["label"] for r in results.values()]
        real_votes = labels.count("real")
        spoof_votes = labels.count("spoof")
        voted_label = "real" if real_votes > spoof_votes else "spoof"

        # Average score (use score_real as the unified score, higher = more likely real)
        avg_score = sum(r["score_real"] for r in results.values()) / len(results)

        return JSONResponse(content={
            "code": 200,
            "message": "success",
            "data": {
                "label": voted_label,
                "score": round(avg_score, 6),
                "cefa_score": round(results["cefa"]["score_real"], 6),
                "wmca_score": round(results["wmca"]["score_real"], 6),
                "surf_score": round(results["surf"]["score_real"], 6),
            },
        })
    except Exception as e:
        logger.exception("Inference error")
        return JSONResponse(content={
            "code": 500,
            "message": f"Inference error: {e}",
            "data": None,
        }, status_code=500)


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
