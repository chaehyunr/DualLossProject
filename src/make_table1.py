
import os, json

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(BASE, "figures")
MODELS = ["t5", "roberta", "bert", "albert", "deberta"]
SETTINGS = ["5w1s", "5w5s", "10w1s", "10w5s"]

#  Table 1 (PDF) — model: [5w1s, 5w5s, 10w1s, 10w5s]
PAPER = {
    "bert":    [72.5, 74.8, 68.3, 75.2],
    "roberta": [75.7, 78.2, 51.4, 78.1],
    "albert":  [70.8, 73.1, 76.7, 73.4],
    "deberta": [77.2, 80.3, 73.6, 70.5],
    "t5":      [83.1, 83.7, 70.2, 76.8],
}


def acc_of(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
        a = d.get("test_acc", d.get("acc"))
        return a * 100 if a is not None else None
    except Exception:
        return None


def find_acc(result_dir, candidates):
    for c in candidates:
        p = os.path.join(BASE, result_dir, c)
        if os.path.exists(p):
            a = acc_of(p)
            if a is not None:
                return a
    return None


def wiki_acc(num, model, setting):
    # results_final/t1P_t5_5w5s.json
    return find_acc("results/table1", [f"t1{num}_{model}_{setting}.json"])


def pubmed_acc(num, model, setting):
    # daP_/daN_ ... _pubmed.json, T5 P: da_t5_..._pubmed.json
    cands = [f"da{num}_{model}_{setting}_pubmed.json"]
    if num == "P":
        cands.append(f"da_{model}_{setting}_pubmed.json")
    return find_acc("results/table1_pubmed", cands)


def build(title, getter, csv_name, show_paper):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    head = f"{'model':9}" + "".join(f"{s:>9}" for s in SETTINGS)
    if show_paper:
        print("(괄호=논문값)")
    print(head)
    lines = ["model," + ",".join(SETTINGS)]
    for m in MODELS:
        cells = []
        csv_cells = []
        for i, s in enumerate(SETTINGS):
            v = getter(m, s)
            if v is None:
                cells.append(f"{'-':>9}"); csv_cells.append("-")
            elif show_paper:
                pa = PAPER[m][i]
                cells.append(f"{v:5.1f}({pa:.0f})")
                csv_cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:>9.1f}"); csv_cells.append(f"{v:.1f}")
        print(f"{m:9}" + "".join(cells))
        lines.append(f"{m}," + ",".join(csv_cells))
    os.makedirs(FIG, exist_ok=True)
    open(os.path.join(FIG, csv_name), "w").write("\n".join(lines))
    print(f"[saved] figures/{csv_name}")


if __name__ == "__main__":
    # 1.0 wiki
    build("Table 1 — FewRel 1.0 wiki, N (paper formulation)",
          lambda m, s: wiki_acc("N", m, s), "table1_wiki_N.csv", show_paper=True)
    build("Table 1 — FewRel 1.0 wiki, P (standard SupCon)",
          lambda m, s: wiki_acc("P", m, s), "table1_wiki_P.csv", show_paper=True)
    # 2.0 pubmed — N, P
    build("Table 1 — FewRel 2.0 pubmed (DA), N",
          lambda m, s: pubmed_acc("N", m, s), "table1_pubmed_N.csv", show_paper=False)
    build("Table 1 — FewRel 2.0 pubmed (DA), P",
          lambda m, s: pubmed_acc("P", m, s), "table1_pubmed_P.csv", show_paper=False)
    print("\nSuccess. figures/table1_*.csv 4 generated.")
