
import sys, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, "src")
from data import load_fewrel, build_text
from model import FewShotEncoder
from engine import get_device, tokenize
from transformers import AutoTokenizer

DATA = "data/val_wiki.json"
N_REL = 6
PER = 30
SEED = 42
OUT = "figures/trained_clusters.png"


def embed(model_path, ckpt, center, texts):
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    model = FewShotEncoder(model_path, center=center).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            ids, mask = tokenize(texts[i:i+32], tok, 128, device)
            out.append(model(ids, mask).float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deberta_ckpt", required=True)
    ap.add_argument("--t5_ckpt", required=True)
    ap.add_argument("--deberta_model", default="microsoft/deberta-v3-base")
    ap.add_argument("--t5_model", default="t5-small")
    ap.add_argument("--out", default="figures/trained_clusters.png")
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    data = load_fewrel(DATA)
    rels = sorted(data.keys())[:N_REL]
    texts, labels = [], []
    for i, r in enumerate(rels):
        for it in data[r][:PER]:
            texts.append(build_text(it)); labels.append(i)
    labels = np.array(labels)

    deb = embed(a.deberta_model, a.deberta_ckpt, True, texts)
    t5 = embed(a.t5_model, a.t5_ckpt, False, texts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, X, title in [
        (axes[0], t5, "T5-small (trained, acc~0.86)"),
        (axes[1], deb, "DeBERTa-v3 (trained+centering, P checkpoint, acc~0.54)")]:
        Z = TSNE(n_components=2, perplexity=20, random_state=SEED, init="pca").fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
        ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Trained embeddings by relation: well-separated (T5) vs partial (DeBERTa)", fontsize=13)
    plt.tight_layout(); plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()