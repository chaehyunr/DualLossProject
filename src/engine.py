import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from utils import set_seed, get_logger, save_checkpoint, load_checkpoint
from data import load_fewrel, split_relations, EpisodeSampler
from model import FewShotEncoder, episode_losses, compute_prototypes, prototype_ce_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(texts, tokenizer, max_len, device):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _move(ep, tokenizer, max_len, device):
    s_ids, s_mask = tokenize(ep["support_text"], tokenizer, max_len, device)
    q_ids, q_mask = tokenize(ep["query_text"], tokenizer, max_len, device)
    s_lab = torch.from_numpy(ep["support_label"]).to(device)
    q_lab = torch.from_numpy(ep["query_label"]).to(device)
    return s_ids, s_mask, s_lab, q_ids, q_mask, q_lab


@torch.no_grad()
def evaluate(model, tokenizer, sampler, episodes, n_way, max_len, device,
             collect_per_relation=False):
    model.eval()
    accs = []
    per_rel_correct, per_rel_total = {}, {}
    for _ in range(episodes):
        ep = sampler.sample_episode()
        s_ids, s_mask, s_lab, q_ids, q_mask, q_lab = _move(ep, tokenizer, max_len, device)
        s_emb = model(s_ids, s_mask)
        q_emb = model(q_ids, q_mask)
        protos = compute_prototypes(s_emb, s_lab, n_way)
        _, _, preds = prototype_ce_loss(q_emb, q_lab, protos)
        correct = (preds == q_lab)
        accs.append(correct.float().mean().item())
        if collect_per_relation:
            for i, rel in enumerate(ep["query_relid"]):
                per_rel_total[rel] = per_rel_total.get(rel, 0) + 1
                per_rel_correct[rel] = per_rel_correct.get(rel, 0) + int(correct[i].item())
    model.train()
    accs_arr = np.array(accs)
    mean_acc = float(accs_arr.mean())
    std_acc = float(accs_arr.std(ddof=0))
    ci95 = float(1.96 * std_acc / np.sqrt(len(accs_arr))) if len(accs_arr) > 0 else 0.0
    per_rel = None
    if collect_per_relation:
        per_rel = {r: per_rel_correct[r] / per_rel_total[r]
                   for r in per_rel_total if per_rel_total[r] > 0}
    return mean_acc, std_acc, ci95, per_rel


def train_one_run(cfg):
    result_path = cfg["result_path"]
    if os.path.exists(result_path):
        return "skipped"

    set_seed(cfg["seed"])
    device = get_device()
    logger = get_logger(cfg["log_path"], name=cfg["run_id"])
    logger.info(f"START {cfg['run_id']} | device={device} | cfg={cfg}")

    train_full = load_fewrel(cfg["train_path"])
    train_data, dev_data = split_relations(train_full, dev_fraction=0.1, seed=cfg["seed"])
    test_data = load_fewrel(cfg["val_path"])

    n, k, q = cfg["n_way"], cfg["k_shot"], cfg["q_query"]
    train_sampler = EpisodeSampler(train_data, n, k, q, seed=cfg["seed"])
    dev_sampler = EpisodeSampler(dev_data, n, k, q, seed=cfg["seed"] + 1) \
        if len(dev_data) >= n else train_sampler
    test_sampler = EpisodeSampler(test_data, n, k, q, seed=cfg["seed"] + 2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],
                                                  fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # ── DeBERTa-v3  ──────────────────────────────
    is_deberta = "deberta" in cfg["model_name"].lower()
    use_whiten = is_deberta and bool(cfg.get("whiten", False))
    use_center = is_deberta and not use_whiten

    model = FewShotEncoder(cfg["model_name"],
                           center=use_center,
                           whiten=use_whiten).to(device)
    if use_whiten:
        logger.info("DeBERTa  → whitening ")
    elif use_center:
        logger.info("DeBERTa → centering")
    # ─────────────────────────────────────────────────────────────

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"],
                                  eps=1e-6, foreach=False, fused=False)


    warmup_steps = cfg.get("warmup_steps", 0)
    scheduler = None
    if warmup_steps and warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_ep, best_acc = load_checkpoint(cfg["ckpt_path"], model, optimizer, device)
    if start_ep > 0:
        logger.info(f"RESUME from episode {start_ep} (best_acc={best_acc:.4f})")

    episodes = cfg["episodes"]
    val_every = cfg["val_every"]
    ckpt_every = cfg["ckpt_every"]
    best_path = cfg["ckpt_path"].replace(".pt", "_best.pt")

    model.train()
    for ep_i in range(start_ep, episodes):
        ep = train_sampler.sample_episode()
        s_ids, s_mask, s_lab, q_ids, q_mask, q_lab = _move(ep, tokenizer, cfg["max_len"], device)
        out = episode_losses(model, s_ids, s_mask, s_lab, q_ids, q_mask, q_lab,
                             n_way=n, cl_weight=cfg["cl_weight"],
                             temperature=cfg["temperature"],
                             cl_numerator=cfg.get("cl_numerator", "P"))
        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (ep_i + 1) % 50 == 0:
            acc = (out["preds"] == out["query_label"]).float().mean().item()
            logger.info(f"ep {ep_i+1}/{episodes} | loss={out['loss'].item():.4f} "
                        f"ce={out['loss_ce'].item():.4f} cl={out['loss_cl'].item():.4f} "
                        f"train_acc={acc:.3f}")

        if (ep_i + 1) % val_every == 0:
            dev_acc, _, _, _ = evaluate(model, tokenizer, dev_sampler,
                                        cfg["eval_episodes"], n, cfg["max_len"], device)
            logger.info(f"[dev] ep {ep_i+1} acc={dev_acc:.4f} (best={best_acc:.4f})")
            if dev_acc > best_acc:
                best_acc = dev_acc
                save_checkpoint(best_path, model, optimizer, ep_i, best_acc)

        if (ep_i + 1) % ckpt_every == 0:
            save_checkpoint(cfg["ckpt_path"], model, optimizer, ep_i, best_acc)


    if os.path.exists(best_path):
        load_checkpoint(best_path, model, None, device)
        logger.info("loaded BEST checkpoint for meta-test")
    test_acc, test_std, test_ci95, per_rel = evaluate(
        model, tokenizer, test_sampler,
        cfg["test_episodes"], n, cfg["max_len"], device,
        collect_per_relation=True)
    logger.info(f"[TEST] {cfg['run_id']} acc={test_acc:.4f} "
                f"±{test_ci95:.4f} (std={test_std:.4f}, n={cfg['test_episodes']})")

    result = {
        "run_id": cfg["run_id"],
        "model_name": cfg["model_name"],
        "n_way": n, "k_shot": k,
        "cl_weight": cfg["cl_weight"], "weight_decay": cfg["weight_decay"],
        "temperature": cfg["temperature"], "seed": cfg["seed"],
        "best_dev_acc": best_acc,
        "test_acc": test_acc,
        "test_acc_std": test_std,
        "test_acc_ci95": test_ci95,
        "n_test_episodes": cfg["test_episodes"],
        "per_relation_acc": per_rel,
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"DONE {cfg['run_id']} → {result_path}")
    return "done"
