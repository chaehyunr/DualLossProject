
import argparse
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer

# 기존 코드 재사용 — 새로 구현하지 않는다.
from utils import set_seed, get_logger, save_checkpoint, load_checkpoint
from data import load_fewrel, split_relations, EpisodeSampler
from model import FewShotEncoder, episode_losses, compute_prototypes, prototype_ce_loss
from engine import get_device, tokenize, _move, evaluate   # 평가/토크나이즈 그대로 재사용


# 도메인 이름 → 데이터 파일 매핑
DOMAIN_FILES = {
    "wiki":    "val_wiki.json",
    "pubmed":  "val_pubmed.json",
    "semeval": "val_semeval.json",
}


def train_one_run_da(cfg):
    result_path = cfg["result_path"]
    if os.path.exists(result_path):
        return "skipped"

    set_seed(cfg["seed"])
    device = get_device()
    logger = get_logger(cfg["log_path"], name=cfg["run_id"])
    logger.info(f"START(DA) {cfg['run_id']} | device={device} | "
                f"dev={cfg['dev_domain']} test={cfg['eval_domain']} | cfg={cfg}")

    train_full = load_fewrel(cfg["train_path"])

    if cfg["dev_domain"] == "wiki":
        train_data, dev_data = split_relations(train_full, dev_fraction=0.1, seed=cfg["seed"])
    else:
        train_data = train_full
        dev_data = load_fewrel(cfg["dev_path"])

    test_data = load_fewrel(cfg["val_path"])

    n, k, q = cfg["n_way"], cfg["k_shot"], cfg["q_query"]
    train_sampler = EpisodeSampler(train_data, n, k, q, seed=cfg["seed"])
    dev_sampler = EpisodeSampler(dev_data, n, k, q, seed=cfg["seed"] + 1) \
        if len(dev_data) >= n else train_sampler
    test_sampler = EpisodeSampler(test_data, n, k, q, seed=cfg["seed"] + 2)

    logger.info(f"[data] train_rel={len(train_data)} dev_rel={len(dev_data)} "
                f"test_rel={len(test_data)} (n_way={n})")
    if len(test_data) < n:
        logger.warning(f"test domain relations: ({len(test_data)}) < n_way({n}). "
                       f"N-way not availble for sampling — lower n_way.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],
                                                  fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    is_deberta = "deberta" in cfg["model_name"].lower()
    use_whiten = is_deberta and bool(cfg.get("whiten", False))
    use_center = is_deberta and not use_whiten
    model = FewShotEncoder(cfg["model_name"],
                           center=use_center,
                           whiten=use_whiten).to(device)
    if use_whiten:
        logger.info("[DA] DeBERTa  → whitening )")
    elif use_center:
        logger.info("[DA] DeBERTa → centering")
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
            logger.info(f"[dev:{cfg['dev_domain']}] ep {ep_i+1} acc={dev_acc:.4f} "
                        f"(best={best_acc:.4f})")
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
    logger.info(f"[TEST:{cfg['eval_domain']}] {cfg['run_id']} acc={test_acc:.4f} "
                f"±{test_ci95:.4f} (std={test_std:.4f}, n={cfg['test_episodes']})")

    result = {
        "run_id": cfg["run_id"],
        "model_name": cfg["model_name"],
        "n_way": n, "k_shot": k,
        "dev_domain": cfg["dev_domain"],       # DA
        "eval_domain": cfg["eval_domain"],     # DA
        "cl_weight": cfg["cl_weight"], "weight_decay": cfg["weight_decay"],
        "cl_numerator": cfg.get("cl_numerator", "P"),
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
    logger.info(f"DONE(DA) {cfg['run_id']} → {result_path}")
    return "done"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--n_way", type=int, required=True)
    p.add_argument("--k_shot", type=int, required=True)
    p.add_argument("--q_query", type=int, default=5)
    # 손실(논문 수식 그대로) — C 실험은 이 둘만 바꾸면 됨
    p.add_argument("--cl_weight", type=float, default=1.0)      # α
    p.add_argument("--weight_decay", type=float, default=0.01)  # λ (= L2/Reg)
    p.add_argument("--temperature", type=float, default=0.5)    # τ
    p.add_argument("--cl_numerator", default="P", choices=["P", "N"],
                   help="P = same classifier, N = different classifier")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--test_episodes", type=int, default=600)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--whiten", action="store_true",
                   help="DeBERTa whitening, default = centering")
    p.add_argument("--run_id", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--result_dir", default="results")
    p.add_argument("--log_dir", default="results/logs")
    # --- 도메인 적응 전용 인자 ---
    p.add_argument("--eval_domain", default="pubmed",
                   choices=list(DOMAIN_FILES.keys()),
                   help="target domain. default=pubmed). wiki -> same as 1.0")
    p.add_argument("--dev_domain", default="wiki",
                   choices=list(DOMAIN_FILES.keys()),
                   help="dev. wiki=train split / semeval")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = {
        "model_name": a.model_name,
        "n_way": a.n_way, "k_shot": a.k_shot, "q_query": a.q_query,
        "cl_weight": a.cl_weight, "weight_decay": a.weight_decay,
        "cl_numerator": a.cl_numerator,
        "whiten": a.whiten,
        "temperature": a.temperature, "lr": a.lr, "seed": a.seed,
        "episodes": a.episodes, "val_every": a.val_every,
        "ckpt_every": a.ckpt_every, "eval_episodes": a.eval_episodes,
        "warmup_steps": a.warmup_steps, "grad_clip": a.grad_clip,
        "test_episodes": a.test_episodes, "max_len": a.max_len,
        "run_id": a.run_id,
        "dev_domain": a.dev_domain,
        "eval_domain": a.eval_domain,
        "train_path": os.path.join(a.data_dir, "train_wiki.json"),
        "val_path":  os.path.join(a.data_dir, DOMAIN_FILES[a.eval_domain]),
        "dev_path":  os.path.join(a.data_dir, DOMAIN_FILES[a.dev_domain]),
        "ckpt_path": os.path.join(a.ckpt_dir, a.run_id + ".pt"),
        "result_path": os.path.join(a.result_dir, a.run_id + ".json"),
        "log_path": os.path.join(a.log_dir, a.run_id + ".log"),
    }
    status = train_one_run_da(cfg)
    print(f"[{a.run_id}] {status}")


if __name__ == "__main__":
    main()