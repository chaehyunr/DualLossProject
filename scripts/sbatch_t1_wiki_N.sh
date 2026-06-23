#!/bin/bash
#SBATCH --account=def-awolson
#SBATCH --job-name=t1_wiki_N
#SBATCH --array=0-4                 # 5개 백본을 배열 잡으로 병렬 (0=bert..4=deberta)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00             # 모델당 4세팅. 끊겨도 재제출하면 체크포인트에서 이어감
#SBATCH --output=slurm-%x-%a-%j.out
# ---------------------------------------------------------------
# Table 1 — domain=wiki, cl_numerator=N
# 결과: results/table1/
#        t1N_{model}_{setting}.json
# 제출: sbatch scripts/sbatch_t1_wiki_N.sh
# ---------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023 python/3.10 cuda
source ~/envs/fewrel/bin/activate     # 최초 1회: virtualenv ~/envs/fewrel && pip install -r requirements.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 컴퓨트 노드 오프라인 → 캐시/로컬에서 로드

NUM=N
DOMAIN=wiki

MODELS=(bert roberta albert t5 deberta)
declare -A HF=(   [bert]=bert-base-uncased [roberta]=roberta-base [albert]=albert-base-v2 [t5]=t5-small [deberta]=microsoft/deberta-v3-base )
declare -A LR=(   [bert]=2e-5 [roberta]=2e-5 [albert]=2e-5 [t5]=2e-5 [deberta]=5e-6 )
declare -A CLIP=( [bert]=1.0  [roberta]=1.0  [albert]=1.0  [t5]=1.0  [deberta]=0.1  )
SETTINGS=( "5w1s 5 1" "5w5s 5 5" "10w1s 10 1" "10w5s 10 5" )

m=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] array=$SLURM_ARRAY_TASK_ID model=$m num=$NUM domain=$DOMAIN"

for s in "${SETTINGS[@]}"; do
  read -r name n k <<< "$s"
  if [ "$DOMAIN" = "wiki" ]; then
    run_id="t1${NUM}_${m}_${name}"
    python src/train.py \
      --model_name "${HF[$m]}" --n_way "$n" --k_shot "$k" \
      --cl_weight 1.0 --weight_decay 0.01 --temperature 0.5 --cl_numerator "$NUM" \
      --lr "${LR[$m]}" --grad_clip "${CLIP[$m]}" \
      --episodes 2000 --test_episodes 600 \
      --result_dir results/table1 --run_id "$run_id"
  else
    run_id="da${NUM}_${m}_${name}_pubmed"
    python src/train_da.py \
      --model_name "${HF[$m]}" --n_way "$n" --k_shot "$k" \
      --cl_weight 1.0 --weight_decay 0.01 --temperature 0.5 --cl_numerator "$NUM" \
      --lr "${LR[$m]}" --grad_clip "${CLIP[$m]}" \
      --episodes 2000 --test_episodes 600 \
      --eval_domain pubmed --dev_domain wiki \
      --result_dir results/table1_pubmed --run_id "$run_id"
  fi
done
echo "[$(date)] DONE model=$m"
