#!/bin/bash
# collect_existing.sh 
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/table1 results/table1_pubmed results/ablation

[ -d Table1/FewRel1.0 ] && cp Table1/FewRel1.0/t1*.json   results/table1/         2>/dev/null || true
[ -d Table1/FewRel2.0 ] && cp Table1/FewRel2.0/da*.json   results/table1_pubmed/  2>/dev/null || true
[ -d Figure3 ]          && cp Figure3/abl_*.json          results/ablation/       2>/dev/null || true

echo "collections:"
echo "  table1        : $(ls results/table1/*.json 2>/dev/null | wc -l) files"
echo "  table1_pubmed : $(ls results/table1_pubmed/*.json 2>/dev/null | wc -l) files"
echo "  ablation      : $(ls results/ablation/*.json 2>/dev/null | wc -l) files"
echo "python src/make_table1.py  /  python src/weight_ablation.py"
