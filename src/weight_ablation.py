"""
figure3_ablation.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULT_DIR = os.path.join(BASE, "results", "ablation")
FIG_DIR = os.path.join(BASE, "figures")

ORDER = [
    ("ce",           "CE:1\nCL:0\nReg:0"),
    ("cl0.5",        "CE:1\nCL:0.5\nReg:0"),
    ("cl1.0",        "CE:1\nCL:1\nReg:0"),
    ("cl1.0_reg0.5", "CE:1\nCL:1\nReg:0.5"),
    ("cl1.0_reg1.0", "CE:1\nCL:1\nReg:1"),
]


def load_curve(num):
    vals = []
    for tag, _ in ORDER:
        p = os.path.join(RESULT_DIR, f"abl_{num}_t5_{tag}.json")
        if not os.path.exists(p):
            print(f"  Leakage: {p}")
            vals.append(None); continue
        d = json.load(open(p, encoding="utf-8"))
        acc = d.get("test_acc", d.get("acc"))
        vals.append(acc * 100 if acc is not None else None)
    return vals


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    labels = [lab for _, lab in ORDER]
    pv = load_curve("P")
    nv = load_curve("N")
    if all(v is None for v in pv) and all(v is None for v in nv):
        print("abl_*.json not exist — run_figure3.sh : first.")
        return

    x = range(len(ORDER))
    plt.figure(figsize=(8.5, 5))
    if any(v is not None for v in nv):
        xs = [i for i in x if nv[i] is not None]; ys = [nv[i] for i in xs]
        plt.plot(xs, ys, "o-", color="#4C72B0", linewidth=2, markersize=8,
                 label="N (paper's original formulation)")
        for i, v in zip(xs, ys):
            plt.text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#4C72B0")
    if any(v is not None for v in pv):
        xs = [i for i in x if pv[i] is not None]; ys = [pv[i] for i in xs]
        plt.plot(xs, ys, "s--", color="#C44E52", linewidth=2, markersize=8,
                 label="P (standard SupCon)")
        for i, v in zip(xs, ys):
            plt.text(i, v - 0.8, f"{v:.1f}", ha="center", va="top", fontsize=8, color="#C44E52")

    plt.xticks(list(x), labels, fontsize=8)
    plt.ylabel("Classification Accuracy (%)")
    plt.xlabel("Loss Weight Ratio (CE / CL / Reg)")
    plt.title("Impact of Loss Weight Ratios on Classification Performance\n(N: paper formulation vs P: standard SupCon)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "figure3_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")

    print("\n=== Figure 3 data ===")
    print(f"{'config':28s} {'N(%)':>7} {'P(%)':>7}")
    for i, (_, lab) in enumerate(ORDER):
        n = f"{nv[i]:.1f}" if nv[i] is not None else "-"
        p = f"{pv[i]:.1f}" if pv[i] is not None else "-"
        print(f"{lab.replace(chr(10), ' '):28s} {n:>7} {p:>7}")


if __name__ == "__main__":
    main()
