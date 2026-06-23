# Replicating the Dual-Loss Few-Shot Strategy (FewRel)

Independent re-implementation of the dual-loss few-shot strategy from
arXiv 2505.06145. On top of 5 pre-trained Transformer backbones (BERT, RoBERTa,
ALBERT, T5-small, DeBERTa-v3), we apply prototype classification + CE /
supervised contrastive / L2, and evaluate on FewRel under the N-way K-shot
protocol.

## 0. Directory Structure

```
Final/
├── README.md
├── refs.bib                     # 3 additional citations for the report
├── requirements.txt
├── data/                        # FewRel public split
│   ├── train_wiki.json
│   ├── val_wiki.json
│   └── val_pubmed.json
├── src/
│   ├── data.py                  # FewRel loading + N-way K-shot episode sampler
│   ├── model.py                 # encoder + prototype classification + CE/SupCon/L2 + DeBERTa centering
│   ├── engine.py                # single-run training (with resume) + meta-test
│   ├── utils.py                 # seed / logging / checkpoint
│   ├── train.py                 # training CLI (shared by Table 1 and ablation)
│   ├── train_da.py              # domain-adaptation CLI (wiki -> pubmed)
│   ├── make_table1.py           # build the 4 Table 1 variants -> figures/table1_*.csv
│   ├── weight_ablation.py       # build Figure 1 (ablation) -> figures/figure3_ablation.png
│   └── train_clusters_visualize.py   # build Figure 2 (t-SNE)
├── scripts/
│   ├── run_table1.sh            # Table 1 (wiki, 5 models x 4 settings x N/P)
│   ├── run_da.sh                # FewRel 2.0 pubmed domain adaptation
│   ├── run_ablation.sh          # T5 loss-weight ablation (P/N)
│   ├── run_clusters.sh          # build t-SNE (Figure 2)
│   ├── collect_existing.sh      # collect existing JSON into results/ (plot without re-training)
│   └── slurm_run.sh             # Alliance Canada submission (table1|da|ablation)
├── results/                     # run-result JSON (created/collected at run time)
│   ├── table1/                  # t1{N,P}_{model}_{setting}.json
│   ├── table1_pubmed/           # da{N,P}_{model}_{setting}_pubmed.json
│   └── ablation/                # abl_{N,P}_t5_{tag}.json
└── figures/                     # generated tables (CSV) and figures (PNG)
```

## 1. Environment Setup

```bash
virtualenv ~/envs/fewrel && source ~/envs/fewrel/bin/activate
pip install -r requirements.txt
```

## 2. Data / Models

- `data/`: `train_wiki.json`, `val_wiki.json`, `val_pubmed.json` (FewRel public
  split). FewRel test labels are hidden, so the meta-test is measured on the
  validation split.
- Models: the scripts use HF hub IDs directly (`roberta-base`,
  `microsoft/deberta-v3-base`, etc.). If the compute node is offline, pre-fetch
  them on the login node to cache, or change the `MODEL[...]` values in
  `scripts/run_*.sh` to local paths.

## 3. Running

Each run skips automatically if its result JSON already exists, and resumes from
a checkpoint if one is present. So even if a job is interrupted, re-submitting
the same command will run it to completion.

```bash
# (a) Reproduce from scratch -- GPU required
bash scripts/run_table1.sh     # Table 1   : FewRel 1.0 wiki, 5 models x 4 settings x {N,P}
bash scripts/run_da.sh         # Sec 3.2/4.2 : FewRel 2.0 wiki -> pubmed domain adaptation
bash scripts/run_ablation.sh   # Figure 1  : T5 loss-weight ablation (P/N)

# Cluster (Alliance Canada):
sbatch scripts/slurm_run.sh table1   # da / ablation work the same way

# (b) Tables/figures only, from existing results -- no GPU needed
bash scripts/collect_existing.sh     # collect existing JSON into results/
```

## 4. Building Tables / Figures

```bash
python src/make_table1.py        # Table 1 (wiki N/P + paper values, pubmed N/P) -> figures/table1_*.csv
python src/weight_ablation.py    # Figure 1 (ablation, P vs N)                   -> figures/figure3_ablation.png
```

The t-SNE embedding visualization (report Figure 2) is generated with
`src/train_clusters_visualize.py` (centered DeBERTa-v3 vs T5 clusters).

## 5. Output Locations

| Artifact | Generating code | Output |
|---|---|---|
| Table 1 (wiki, N/P) | `train.py` -> `make_table1.py` | `results/table1/`, `figures/table1_wiki_*.csv` |
| FewRel 2.0 (pubmed) | `train_da.py` -> `make_table1.py` | `results/table1_pubmed/`, `figures/table1_pubmed_*.csv` |
| Figure 1 (ablation) | `train.py` -> `weight_ablation.py` | `results/ablation/`, `figures/figure3_ablation.png` |
| Figure 2 (t-SNE) | `train_clusters_visualize.py` | `figures/trained_clusters_*.png` |

## 6. Key Implementation Notes

- **Classification**: prototype (ProtoNet). The paper does not specify the N-way
  classification procedure, so we adopt the FewRel standard.
- **L_reg**: handled via AdamW `weight_decay` (= lambda) instead of a separate
  term (identical gradient).
- **Contrastive loss N/P**: `--cl_numerator` switches the numerator set S(i).
  P = same class (standard SupCon), N = literal reading of the paper's formula.
  In a 5-way batch |N(i)| ~= 4|P(i)|, so under N the loss converges to 0
  (report Sec 2.3).
- **DeBERTa-v3**: batch-mean centering is applied automatically for anisotropy
  handling (`center=True` in `model.py`, when the model name contains "deberta").

## 7. Generative AI Disclosure

Generative AI was used as an aid for code cleanup and writing reproduction
scripts. All experiment design, execution, and analysis were performed and
verified by the author.
