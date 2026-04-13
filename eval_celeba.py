"""
Evaluate FLIP 3-model voting on CelebA-Spoof test set.
Random sample 1000 live + 1000 spoof, compute confusion matrix.
"""
import os
import random
import time
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
    load_ckpt_any,
    extract_state_dict,
    infer_mcl_dims_from_state_dict,
    build_test_transform,
)

# ---- Config ----
DATA_ROOT = "/home/yman/workspace/FSVFM/FSFM-CVPR25-temp/datasets/data/CelebA_Spoof/images/test"
LIVE_DIR = os.path.join(DATA_ROOT, "live")
SPOOF_DIR = os.path.join(DATA_ROOT, "spoof")

MODEL_PATHS = {
    "cefa": "./models/cefa_flip_mcl.pth.tar",
    "wmca": "./models/wmca_flip_mcl.pth.tar",
    "surf": "./models/surf_flip_mcl.pth.tar",
}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLE_PER_CLASS = 1000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_model(name, path):
    print(f"Loading {name} from {path}...")
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
    text_features = torch.stack([spoof_emb, real_emb], dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    return text_features, logit_scale


@torch.no_grad()
def infer_one(models, text_features, logit_scale, img_path):
    """Run 3-model voting on an image file."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
    x = build_test_transform()(pil_img).unsqueeze(0).to(DEVICE)

    votes = []
    scores = []
    for name, model in models.items():
        image_features = model.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        probs = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        votes.append("real" if pred == 1 else "spoof")
        scores.append(float(probs[1].item()))

    real_count = votes.count("real")
    label = "real" if real_count > 1 else "spoof"
    return label, scores


def main():
    # List image files
    live_files = sorted(glob(os.path.join(LIVE_DIR, "*.jpg")) + glob(os.path.join(LIVE_DIR, "*.png")))
    spoof_files = sorted(glob(os.path.join(SPOOF_DIR, "*.jpg")) + glob(os.path.join(SPOOF_DIR, "*.png")))
    print(f"Total: {len(live_files)} live, {len(spoof_files)} spoof")

    # Random sample
    live_subset = random.sample(live_files, min(SAMPLE_PER_CLASS, len(live_files)))
    spoof_subset = random.sample(spoof_files, min(SAMPLE_PER_CLASS, len(spoof_files)))
    print(f"Sampled: {len(live_subset)} live, {len(spoof_subset)} spoof")

    # Load models
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = load_model(name, path)

    text_features, logit_scale = precompute_text_features(
        next(iter(models.values())).model
    )
    print(f"Models loaded on {DEVICE}, text features cached.\n")

    # Evaluate
    tp = fp = tn = fn = 0
    skip = 0

    print("Evaluating live samples...")
    t0 = time.time()
    for i, img_path in enumerate(live_subset):
        result = infer_one(models, text_features, logit_scale, img_path)
        if result is None:
            skip += 1
            continue
        pred_label, _ = result
        if pred_label == "real":
            tp += 1
        else:
            fn += 1
        if (i + 1) % 200 == 0:
            print(f"  live {i+1}/{len(live_subset)}, elapsed {time.time()-t0:.1f}s")

    print("Evaluating spoof samples...")
    t1 = time.time()
    for i, img_path in enumerate(spoof_subset):
        result = infer_one(models, text_features, logit_scale, img_path)
        if result is None:
            skip += 1
            continue
        pred_label, _ = result
        if pred_label == "spoof":
            tn += 1
        else:
            fp += 1
        if (i + 1) % 200 == 0:
            print(f"  spoof {i+1}/{len(spoof_subset)}, elapsed {time.time()-t1:.1f}s")

    total_time = time.time() - t0
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    spoof_recall = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    print("\n" + "=" * 55)
    print("  FLIP 3-Model Voting - CelebA-Spoof Test")
    print("  Data: " + DATA_ROOT)
    print("=" * 55)
    print(f"\n  Samples: {tp+fn} live + {tn+fp} spoof = {total}  (skipped: {skip})")
    print(f"  Total time: {total_time:.1f}s ({total_time/total*1000:.1f}ms/image)")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} {'Pred Real':>12} {'Pred Spoof':>12}")
    print(f"  {'Actual Real':>20} {tp:>12} {fn:>12}")
    print(f"  {'Actual Spoof':>20} {fp:>12} {tn:>12}")
    print(f"\n  Accuracy:      {accuracy:.2f}%")
    print(f"  Precision:     {precision:.2f}%  (real)")
    print(f"  Recall:        {recall:.2f}%  (real)")
    print(f"  F1 Score:      {f1:.2f}%")
    print(f"  Spoof Recall:  {spoof_recall:.2f}%")
    print(f"\n  Live  -> Spoof (FN): {fn:>4} / {tp+fn} ({fn/(tp+fn)*100:.2f}%)")
    print(f"  Spoof -> Real  (FP): {fp:>4} / {tn+fp} ({fp/(tn+fp)*100:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
