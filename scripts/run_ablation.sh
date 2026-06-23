#!/bin/bash
# run_ablation.sh — 손실 가중치 ablation 재현 (보고서 Figure 1)
# T5-small, 5-way 5-shot. CE → +CL → +Reg 순서로 P/N 두 수식 비교.
# 결과: results/ablation/abl_{N,P}_t5_{tag}.json
# 주: CE-only(cl_weight=0)는 P/N 동일하지만, 그래프가 두 곡선을 같은 시작점으로
#     그리므로 N/P 양쪽 다 생성한다.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT_DIR=results/ablation
MODEL=t5-small

# tag  cl_weight  weight_decay(=Reg)
CONFIGS=(
  "ce            0    0"
  "cl0.5         0.5  0"
  "cl1.0         1.0  0"
  "cl1.0_reg0.5  1.0  0.5"
  "cl1.0_reg1.0  1.0  1.0"
)

for num in N P; do
  for c in "${CONFIGS[@]}"; do
    read -r tag clw wd <<< "$c"
    run_id="abl_${num}_t5_${tag}"
    echo "=== ${run_id} (cl=${clw} reg=${wd}) ==="
    python src/train.py \
      --model_name "$MODEL" \
      --n_way 5 --k_shot 5 \
      --cl_weight "$clw" --weight_decay "$wd" --temperature 0.5 \
      --cl_numerator "$num" \
      --lr 2e-5 --grad_clip 1.0 \
      --episodes 2000 --test_episodes 600 \
      --result_dir "$RESULT_DIR" \
      --run_id "$run_id"
  done
done

echo "Ablation runs 완료. 그림 생성: python src/weight_ablation.py"
