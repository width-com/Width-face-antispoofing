# infer_one_image.py
import os
import json
import argparse
import tempfile
import zipfile
from typing import Optional, Any, Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from fas import flip_mcl
import clip
from prompt_templates import spoof_templates, real_templates


# ----------------------------
# 1) 预处理：对齐 dataset.py test
# ----------------------------
def build_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def read_image_like_dataset(img_path: str, image_size: int = 224) -> Image.Image:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"cv2.imread failed: {img_path}")

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype(np.uint8)).resize((image_size, image_size))
    return img


# ----------------------------
# 2) ckpt 加载：兼容文件/解包目录
# ----------------------------
def _is_unpacked_checkpoint_dir(dir_path: str) -> bool:
    return (os.path.isfile(os.path.join(dir_path, "data.pkl"))
            and os.path.isfile(os.path.join(dir_path, "version"))
            and os.path.exists(os.path.join(dir_path, "data")))


def _repack_dir_to_zip_for_torch_load(dir_path: str) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
    os.close(tmp_fd)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, dir_path)
                zf.write(abs_path, rel_path)
    return tmp_path


def load_ckpt_any(ckpt_path: str, device: str):
    if os.path.isfile(ckpt_path):
        return torch.load(ckpt_path, map_location=device)

    if os.path.isdir(ckpt_path):
        try:
            return torch.load(ckpt_path, map_location=device)
        except Exception:
            pass

        if _is_unpacked_checkpoint_dir(ckpt_path):
            repacked = _repack_dir_to_zip_for_torch_load(ckpt_path)
            return torch.load(repacked, map_location=device)

    raise FileNotFoundError(f"Cannot load checkpoint from: {ckpt_path}")


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        for k in ["model", "net", "network", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        return ckpt_obj
    raise ValueError(f"Unsupported checkpoint object type: {type(ckpt_obj)}")


def infer_mcl_dims_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    if "image_mlp.layer1.weight" not in state or "image_mlp.layer3.weight" not in state:
        raise ValueError("Cannot infer flip_mcl dims: missing image_mlp.layer1.weight / layer3.weight")
    w1 = state["image_mlp.layer1.weight"]  # [ssl_mlp_dim, in_dim]
    w3 = state["image_mlp.layer3.weight"]  # [ssl_emb_dim, ssl_mlp_dim]
    ssl_mlp_dim = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    ssl_emb_dim = int(w3.shape[0])
    return in_dim, ssl_mlp_dim, ssl_emb_dim


# ----------------------------
# 3) 用 CPU 复刻 flip_mcl.forward_eval（避免 CUDA kernel 不兼容）
# ----------------------------
@torch.no_grad()
def forward_eval_cpu(clip_model, x: torch.Tensor, device_for_clip: str = "cpu") -> torch.Tensor:
    """
    返回 similarity logits: [B,2]
    """
    clip_model.eval()
    clip_model.to(device_for_clip)
    x = x.to(device_for_clip)

    # tokenize prompts
    spoof_texts = clip.tokenize(spoof_templates).to(device_for_clip, non_blocking=False)
    real_texts = clip.tokenize(real_templates).to(device_for_clip, non_blocking=False)

    # text encoder
    spoof_class_embeddings = clip_model.encode_text(spoof_texts).mean(dim=0)
    real_class_embeddings = clip_model.encode_text(real_texts).mean(dim=0)

    text_features = torch.stack([spoof_class_embeddings, real_class_embeddings], dim=0)

    # image encoder
    image_features = clip_model.encode_image(x)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # similarity logits
    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    return logits


@torch.no_grad()
def infer_one_image_flip_mcl(model: flip_mcl, img_path: str, image_size: int, force_cpu_clip: bool, device: str):
    """
    - 预处理按 dataset test
    - 如果 force_cpu_clip=True：用 CPU 跑 CLIP forward_eval（规避 CUDA kernel）
    """
    pil_img = read_image_like_dataset(img_path, image_size=image_size)
    x = build_test_transform()(pil_img).unsqueeze(0)

    if force_cpu_clip:
        # 用 model.model (CLIP) 在 CPU 上算 similarity
        similarity = forward_eval_cpu(model.model, x, device_for_clip="cpu")
    else:
        # 如果你的 CUDA 环境兼容，可以直接用原 forward_eval
        model = model.to(device)
        x = x.to(device)
        similarity, _ = model.forward_eval(x)

    probs = F.softmax(similarity, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    label_map = {0: "spoof", 1: "real"}

    return {
        "mode": "flip_mcl_forward_eval_cpu" if force_cpu_clip else "flip_mcl_forward_eval",
        "image_path": img_path,
        "pred_id": pred,
        "pred_label": label_map.get(pred, str(pred)),
        "prob_spoof": float(probs[0].item()),
        "prob_real": float(probs[1].item()),
        "logits": [float(similarity[0, 0].item()), float(similarity[0, 1].item())],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="wmca_flip_mcl.pth.tar (file) or unpacked dir")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--force_cpu_clip", action="store_true", help="force CLIP encode_text/encode_image on CPU")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        device = "cpu"

    # load ckpt/state_dict
    ckpt_obj = load_ckpt_any(args.ckpt, device="cpu")  # ckpt 先放 CPU 读取最稳
    state = extract_state_dict(ckpt_obj)

    # infer mcl dims
    in_dim, ssl_mlp_dim, ssl_emb_dim = infer_mcl_dims_from_state_dict(state)
    print(f"[INFO] Inferred flip_mcl dims: in_dim={in_dim}, ssl_mlp_dim={ssl_mlp_dim}, ssl_emb_dim={ssl_emb_dim}")

    # build model
    model = flip_mcl(in_dim=in_dim, ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim)

    # load weights
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[WARN] Missing keys (strict=False): {missing}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys (strict=False): {unexpected}")

    # infer
    # 你现在 CUDA 不兼容，建议直接加 --force_cpu_clip
    result = infer_one_image_flip_mcl(
        model=model,
        img_path=args.image,
        image_size=args.image_size,
        force_cpu_clip=args.force_cpu_clip,
        device=device,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_json is not None:
        save_dir = os.path.dirname(args.save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved result to: {args.save_json}")


if __name__ == "__main__":
    main()

