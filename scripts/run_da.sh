#!/bin/bash
# run_da.sh — §3.2 / §4.2 도메인 적응 재현 (FewRel 2.0, wiki→pubmed)
# wiki로 학습한 모델을 pubmed(의학) 도메인에서 메타-테스트.
# 5 backbones x 4 settings x {N,P} = 40 runs.
# 결과: results/table1_pubmed/da{N,P}_{model}_{setting}_pubmed.json
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT_DIR=results/table1_pubmed

declare -A MODEL=(
  [bert]=bert-base-uncased
  [roberta]=roberta-base
  [albert]=albert-base-v2
  [t5]=t5-small
  [deberta]=microsoft/deberta-v3-base
)
declare -A LR=(   [bert]=2e-5 [roberta]=2e-5 [albert]=2e-5 [t5]=2e-5 [deberta]=5e-6 )
declare -A CLIP=( [bert]=1.0  [roberta]=1.0  [albert]=1.0  [t5]=1.0  [deberta]=0.1  )

SETTINGS=( "5w1s 5 1" "5w5s 5 5" "10w1s 10 1" "10w5s 10 5" )

for num in N P; do
  for m in bert roberta albert t5 deberta; do
    for s in "${SETTINGS[@]}"; do
      read -r name n k <<< "$s"
      run_id="da${num}_${m}_${name}_pubmed"
      echo "=== ${run_id} ==="
      # 학습=wiki(고정), dev=wiki 분할(실험 A), test=pubmed
      python src/train_da.py \
        --model_name "${MODEL[$m]}" \
        --n_way "$n" --k_shot "$k" \
        --cl_weight 1.0 --weight_decay 0.01 --temperature 0.5 \
        --cl_numerator "$num" \
        --lr "${LR[$m]}" --grad_clip "${CLIP[$m]}" \
        --episodes 2000 --test_episodes 600 \
        --eval_domain pubmed --dev_domain wiki \
        --result_dir "$RESULT_DIR" \
        --run_id "$run_id"
    done
  done
done

echo "DA runs 완료. 표 생성: python src/make_table1.py (pubmed 섹션)"
