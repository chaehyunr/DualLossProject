#!/bin/bash
# run_clusters.sh — t-SNE 임베딩 시각화 재현 (보고서 Figure 2)
# 학습된 T5(잘 분리, ~0.86) vs DeBERTa-v3(centering, P checkpoint, ~0.54)를
# val_wiki 임베딩의 t-SNE로 나란히 비교.
#
# 전제: run_table1.sh가 만든 best 체크포인트가 checkpoints/ 에 있어야 한다.
#   checkpoints/t1P_t5_5w5s_best.pt, checkpoints/t1P_deberta_5w5s_best.pt
# (engine.py는 학습 종료 시 *_best.pt 를 남긴다.)
set -euo pipefail
cd "$(dirname "$0")/.."

T5_CKPT="${1:-checkpoints/t1P_t5_5w5s_best.pt}"
DEBERTA_CKPT="${2:-checkpoints/t1P_deberta_5w5s_best.pt}"

python src/train_clusters_visualize.py \
  --t5_ckpt "$T5_CKPT" \
  --deberta_ckpt "$DEBERTA_CKPT" \
  --t5_model t5-small \
  --deberta_model microsoft/deberta-v3-base \
  --out figures/trained_clusters_P.png

echo "완료 → figures/trained_clusters_P.png"
