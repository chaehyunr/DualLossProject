#!/bin/bash
# run_table1.sh — Table 1 재현 (FewRel 1.0 wiki)
# 5 backbones x 4 settings x {N,P} = 40 runs.
# 결과: results/table1/t1{N,P}_{model}_{setting}.json
# 이미 결과 JSON이 있으면 train.py가 자동 skip하므로, 중단 후 재실행해도 안전.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo 루트로 이동 (data/, results/ 상대경로 기준)

RESULT_DIR=results/table1

# 모델 이름(HF 허브 ID 또는 로컬 models/ 경로). 클러스터가 오프라인이면
# 미리 받은 로컬 경로로 바꾸세요 (예: models/roberta-base).
declare -A MODEL=(
  [bert]=bert-base-uncased
  [roberta]=roberta-base
  [albert]=albert-base-v2
  [t5]=t5-small
  [deberta]=microsoft/deberta-v3-base
)
# DeBERTa-v3만 lr/grad_clip를 따로 둔다(이방성·grad explosion 대응).
declare -A LR=(   [bert]=2e-5 [roberta]=2e-5 [albert]=2e-5 [t5]=2e-5 [deberta]=5e-6 )
declare -A CLIP=( [bert]=1.0  [roberta]=1.0  [albert]=1.0  [t5]=1.0  [deberta]=0.1  )

# setting: name n_way k_shot
SETTINGS=( "5w1s 5 1" "5w5s 5 5" "10w1s 10 1" "10w5s 10 5" )

for num in N P; do
  for m in bert roberta albert t5 deberta; do
    for s in "${SETTINGS[@]}"; do
      read -r name n k <<< "$s"
      run_id="t1${num}_${m}_${name}"
      echo "=== ${run_id} ==="
      python src/train.py \
        --model_name "${MODEL[$m]}" \
        --n_way "$n" --k_shot "$k" \
        --cl_weight 1.0 --weight_decay 0.01 --temperature 0.5 \
        --cl_numerator "$num" \
        --lr "${LR[$m]}" --grad_clip "${CLIP[$m]}" \
        --episodes 2000 --test_episodes 600 \
        --result_dir "$RESULT_DIR" \
        --run_id "$run_id"
    done
  done
done

echo "Table 1 runs 완료. 표 생성: python src/make_table1.py"
