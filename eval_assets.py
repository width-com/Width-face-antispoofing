"""
Evaluate 3 models on assets/faces images.
Show per-model results and voting result for each image.
"""
import os
import numpy as np
import cv2
from PIL import Image
from glob import glob

import torch
import torch.nn.functional as F

from fas import flip_mcl
import clip
from prompt_templates import spoof_templates, real_templates
from infer_one_image import (
    load_ckpt_any, extract_state_dict,
    infer_mcl_dims_from_state_dict, build_test_transform,
)

FACES_DIR = "/home/yman/workspace/FSVFM/FSFM-CVPR25-temp/assets/faces"
MODEL_PATHS = {
    "cefa": "./models/cefa_flip_mcl.pth.tar",
    "wmca": "./models/wmca_flip_mcl.pth.tar",
    "surf": "./models/surf_flip_mcl.pth.tar",
}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(name, path):
    ckpt_obj = load_ckpt_any(path, device="cpu")
    state = extract_state_dict(ckpt_obj)
    in_dim, ssl_mlp_dim, ssl_emb_dim = infer_mcl_dims_from_state_dict(state)
    model = flip_mcl(in_dim=in_dim, ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, device=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def precompute_text_features(clip_model):
    clip_model.eval()
    spoof_texts = clip.tokenize(spoof_templates).to(DEVICE)
    real_texts = clip.tokenize(real_templates).to(DEVICE)
    spoof_emb = clip_model.encode_text(spoof_texts).mean(dim=0)
    real_emb = clip_model.encode_text(real_texts).mean(dim=0)
    tf = torch.stack([spoof_emb, real_emb], dim=0)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    ls = clip_model.logit_scale.exp()
    return tf, ls


@torch.no_grad()
def infer_image(models, text_features, logit_scale, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
    x = build_test_transform()(pil_img).unsqueeze(0).to(DEVICE)

    results = {}
    for name, model in models.items():
        image_features = model.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        probs = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        results[name] = {
            "label": "real" if pred == 1 else "spoof",
            "real_score": float(probs[1].item()),
        }

    # Voting
    labels = [r["label"] for r in results.values()]
    voted = "real" if labels.count("real") > 1 else "spoof"
    avg_score = sum(r["real_score"] for r in results.values()) / 3
    results["vote"] = {"label": voted, "real_score": avg_score}
    return results


def main():
    # Load models
    print("Loading models...")
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = load_model(name, path)
    text_features, logit_scale = precompute_text_features(
        next(iter(models.values())).model
    )
    print(f"Models loaded on {DEVICE}\n")

    # Get all images
    files = sorted(glob(os.path.join(FACES_DIR, "*.jpg")) + glob(os.path.join(FACES_DIR, "*.png")))
    print(f"Found {len(files)} images\n")

    # Header
    print(f"{'Image':<16} {'CeFA':>12} {'WMCA':>12} {'SURF':>12} {'| Vote':>8}  {'Avg Score':>10}")
    print("-" * 78)

    for img_path in files:
        fname = os.path.basename(img_path)
        results = infer_image(models, text_features, logit_scale, img_path)
        if results is None:
            print(f"{fname:<16} ERROR")
            continue

        cefa = results["cefa"]
        wmca = results["wmca"]
        surf = results["surf"]
        vote = results["vote"]

        def fmt(r):
            color = r["label"].upper()
            return f"{color} {r['real_score']:.3f}"

        print(f"{fname:<16} {fmt(cefa):>12} {fmt(wmca):>12} {fmt(surf):>12} | {vote['label'].upper():<6} {vote['real_score']:.4f}")

    # Summary by category
    print("\n" + "=" * 78)
    print("Summary by category:")
    categories = {"real": [], "mask": [], "photo": [], "reply": []}
    for img_path in files:
        fname = os.path.basename(img_path).lower()
        results = infer_image(models, text_features, logit_scale, img_path)
        if results is None:
            continue
        for cat in categories:
            if fname.startswith(cat):
                categories[cat].append(results)
                break

    for cat, items in categories.items():
        if not items:
            continue
        correct = 0
        for r in items:
            voted = r["vote"]["label"]
            expected = "real" if cat == "real" else "spoof"
            if voted == expected:
                correct += 1
        print(f"  {cat:>6}: {correct}/{len(items)} correct  ({correct/len(items)*100:.0f}%)")


if __name__ == "__main__":
    main()
