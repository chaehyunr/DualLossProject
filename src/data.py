"""
data.py
FewRel Load + Text with N-way K-shot episode sampler
"""

import json
import random

import numpy as np


def load_fewrel(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _mention_span(entity):
    try:
        idx = entity[2][0]
        return min(idx), max(idx) + 1
    except Exception:
        return None


def build_text(instance):
    tokens = list(instance["tokens"])
    spans = []
    h = _mention_span(instance.get("h"))
    t = _mention_span(instance.get("t"))
    if h:
        spans.append((h[0], h[1], "*"))
    if t:
        spans.append((t[0], t[1], "#"))
    for start, end, mark in sorted(spans, key=lambda x: x[0], reverse=True):
        if 0 <= start < end <= len(tokens):
            tokens.insert(end, mark)
            tokens.insert(start, mark)
    return " ".join(tokens)


class EpisodeSampler:
    def __init__(self, data, n_way, k_shot, q_query, seed=0):
        self.data = data
        self.relations = [r for r, items in data.items()
                          if len(items) >= (k_shot + q_query)]
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = random.Random(seed)
        if len(self.relations) < n_way:
            raise ValueError(
                f"Relations({len(self.relations)}) is smaller than n_way({n_way})"
            )

    def sample_episode(self):
        chosen = self.rng.sample(self.relations, self.n_way)
        support_text, support_label = [], []
        query_text, query_label, query_relid = [], [], []
        for label_idx, rel in enumerate(chosen):
            items = self.data[rel]
            picked = self.rng.sample(items, self.k_shot + self.q_query)
            for it in picked[:self.k_shot]:
                support_text.append(build_text(it))
                support_label.append(label_idx)
            for it in picked[self.k_shot:]:
                query_text.append(build_text(it))
                query_label.append(label_idx)
                query_relid.append(rel)
        return {
            "support_text": support_text,
            "support_label": np.array(support_label, dtype=np.int64),
            "query_text": query_text,
            "query_label": np.array(query_label, dtype=np.int64),
            "query_relid": query_relid,
        }


def split_relations(data, dev_fraction=0.1, seed=0):
    rels = sorted(data.keys())
    rng = random.Random(seed)
    rng.shuffle(rels)
    n_dev = max(1, int(len(rels) * dev_fraction))
    dev_rels = set(rels[:n_dev])
    train_data = {r: v for r, v in data.items() if r not in dev_rels}
    dev_data = {r: v for r, v in data.items() if r in dev_rels}
    return train_data, dev_data
