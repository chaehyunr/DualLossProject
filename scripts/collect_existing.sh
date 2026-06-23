#!/bin/bash
# collect_existing.sh — 이미 돌려둔 결과 JSON을 results/ 표준 경로로 모은다.
# 재학습 없이 곧바로 표/그림을 뽑고 싶을 때 사용.
# (기존 폴더 구조: Table1/FewRel1.0, Table1/FewRel2.0, Figure3)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/table1 results/table1_pubmed results/ablation

[ -d Table1/FewRel1.0 ] && cp Table1/FewRel1.0/t1*.json   results/table1/         2>/dev/null || true
[ -d Table1/FewRel2.0 ] && cp Table1/FewRel2.0/da*.json   results/table1_pubmed/  2>/dev/null || true
[ -d Figure3 ]          && cp Figure3/abl_*.json          results/ablation/       2>/dev/null || true

echo "수집 결과:"
echo "  table1        : $(ls results/table1/*.json 2>/dev/null | wc -l) files"
echo "  table1_pubmed : $(ls results/table1_pubmed/*.json 2>/dev/null | wc -l) files"
echo "  ablation      : $(ls results/ablation/*.json 2>/dev/null | wc -l) files"
echo "이제: python src/make_table1.py  /  python src/weight_ablation.py"
