
import os
import random
import logging

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_path: str, name: str = "train"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name + "_" + os.path.basename(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def save_checkpoint(ckpt_path: str, model, optimizer, episode: int, best_acc: float):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp_path = ckpt_path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "best_acc": best_acc,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        tmp_path,
    )
    os.replace(tmp_path, ckpt_path)


def load_checkpoint(ckpt_path: str, model, optimizer, device):
    if not os.path.exists(ckpt_path):
        return 0, 0.0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    try:
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
    except Exception:
        pass
    return ckpt["episode"] + 1, ckpt["best_acc"]