"""
FLIP Face Anti-Spoofing Voting Service
Three models vote on whether a face image is real or spoof.
Served with Flask + gunicorn.
"""

import base64
import logging
import os
import site
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR):
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    site.addsitedir(VENDOR_DIR)

import boto3
import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

from fas import flip_mcl
from infer_one_image import (
    build_test_transform,
    extract_state_dict,
    infer_mcl_dims_from_state_dict,
    load_ckpt_any,
)
from prompt_templates import real_templates, spoof_templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
INFERENCE_WORKERS = len(MODEL_PATHS)

models: Dict[str, flip_mcl] = {}
cached_text_features: Optional[torch.Tensor] = None
cached_logit_scale: Optional[torch.Tensor] = None
models_lock = threading.Lock()
s3_client = boto3.client("s3", region_name=DEFAULT_AWS_REGION)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


def ensure_models_exist():
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
    logger.info("Loading model: %s from %s", name, ckpt_path)
    ckpt_obj = load_ckpt_any(ckpt_path, device="cpu")
    state = extract_state_dict(ckpt_obj)
    in_dim, ssl_mlp_dim, ssl_emb_dim = infer_mcl_dims_from_state_dict(state)
    logger.info(
        "  %s: in_dim=%s, ssl_mlp_dim=%s, ssl_emb_dim=%s",
        name,
        in_dim,
        ssl_mlp_dim,
        ssl_emb_dim,
    )
    model = flip_mcl(
        in_dim=in_dim,
        ssl_mlp_dim=ssl_mlp_dim,
        ssl_emb_dim=ssl_emb_dim,
        device=DEVICE,
    )
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def precompute_text_features(clip_model):
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
    logger.info("Text features cached on %s.", DEVICE)


def load_all_models():
    global models
    ensure_models_exist()
    loaded_models = {}
    for name, path in MODEL_PATHS.items():
        loaded_models[name] = load_single_model(name, path)
    first_model = next(iter(loaded_models.values()))
    precompute_text_features(first_model.model)
    models = loaded_models
    logger.info("All %s models loaded on %s.", len(models), DEVICE)


def ensure_models_loaded():
    if models:
        return
    with models_lock:
        if not models:
            load_all_models()


@torch.no_grad()
def infer_single_model(model: flip_mcl, x: torch.Tensor) -> Dict[str, Any]:
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


def infer_all_models_parallel(x: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=INFERENCE_WORKERS) as executor:
        futures = {
            name: executor.submit(infer_single_model, model, x)
            for name, model in models.items()
        }
        return {name: future.result() for name, future in futures.items()}


def read_image_from_bytes(img_bytes: bytes) -> Image.Image:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))


def decode_base64_image(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_s3_image_bytes(s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def resolve_image_bytes() -> bytes:
    image = request.files.get("image")
    if image is not None:
        img_bytes = image.read()
        if img_bytes:
            return img_bytes

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    if not payload:
        payload = request.form.to_dict()

    image_base64 = payload.get("image_base64") or payload.get("base64")
    if image_base64:
        return decode_base64_image(str(image_base64))

    s3_uri = payload.get("s3_uri") or payload.get("image_s3_uri")
    if s3_uri:
        return download_s3_image_bytes(str(s3_uri))

    raise ValueError("Provide one of: image file, image_base64, or s3_uri")


@app.before_request
def warmup_models():
    ensure_models_loaded()


@app.get("/")
def root():
    return redirect("/static/index.html")


@app.get("/index.html")
def index_fallback():
    return redirect("/static/index.html")


@app.get("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(app.static_folder, filename)


@app.post("/predict")
def predict():
    try:
        img_bytes = resolve_image_bytes()
        pil_img = read_image_from_bytes(img_bytes)
    except Exception as exc:
        return (
            jsonify({"code": 400, "message": f"Invalid image: {exc}", "data": None}),
            400,
        )

    try:
        x = build_test_transform()(pil_img).unsqueeze(0).to(DEVICE)
        results = infer_all_models_parallel(x)
        labels = [result["label"] for result in results.values()]
        real_votes = labels.count("real")
        spoof_votes = labels.count("spoof")
        voted_label = "real" if real_votes > spoof_votes else "spoof"
        avg_score = sum(result["score_real"] for result in results.values()) / len(results)
        return jsonify(
            {
                "code": 200,
                "message": "success",
                "data": {
                    "label": voted_label,
                    "score": round(avg_score, 6),
                    "cefa_score": round(results["cefa"]["score_real"], 6),
                    "wmca_score": round(results["wmca"]["score_real"], 6),
                    "surf_score": round(results["surf"]["score_real"], 6),
                },
            }
        )
    except Exception as exc:
        logger.exception("Inference error")
        return (
            jsonify({"code": 500, "message": f"Inference error: {exc}", "data": None}),
            500,
        )


if __name__ == "__main__":
    from werkzeug.serving import run_simple

    port = int(os.getenv("PORT", "5010"))
    run_simple("0.0.0.0", port, app, threaded=True)
