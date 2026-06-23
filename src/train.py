
import argparse
import os

from engine import train_one_run   # v5: whitening 옵션 포함(model 사용)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--n_way", type=int, required=True)
    p.add_argument("--k_shot", type=int, required=True)
    p.add_argument("--q_query", type=int, default=5)
    p.add_argument("--cl_weight", type=float, default=1.0)   # α
    p.add_argument("--weight_decay", type=float, default=0.01)  # λ (= L2/Reg)
    p.add_argument("--temperature", type=float, default=0.5)    # τ
    p.add_argument("--cl_numerator", default="P", choices=["P", "N"],
                   help="P=same classifier, N=different classifier")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="lr warmup steps.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="gradient norm clipping cutoff. DeBERTa-v3: grad explosion ")
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--test_episodes", type=int, default=600)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--whiten", action="store_true",
                   help="DeBERTa: whitening applied. default: centering.")
    p.add_argument("--run_id", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--result_dir", default="results")
    p.add_argument("--log_dir", default="results/logs")
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
        "warmup_steps": a.warmup_steps,
        "grad_clip": a.grad_clip,
        "test_episodes": a.test_episodes, "max_len": a.max_len,
        "run_id": a.run_id,
        "train_path": os.path.join(a.data_dir, "train_wiki.json"),
        "val_path": os.path.join(a.data_dir, "val_wiki.json"),
        "ckpt_path": os.path.join(a.ckpt_dir, a.run_id + ".pt"),
        "result_path": os.path.join(a.result_dir, a.run_id + ".json"),
        "log_path": os.path.join(a.log_dir, a.run_id + ".log"),
    }
    status = train_one_run(cfg)
    print(f"[{a.run_id}] {status}")


if __name__ == "__main__":
    main()
