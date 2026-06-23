#!/bin/bash
#SBATCH --account=def-awolson
#SBATCH --job-name=fewrel-repl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
# ----------------------------------------------------------------------
# Alliance Canada(구 Compute Canada) 제출 스크립트.
# GPU 슬롯이 끊겨도 train.py가 체크포인트에서 이어 돌고, 끝난 run은
# 결과 JSON 존재 시 자동 skip하므로, 같은 잡을 여러 번 던져 완주할 수 있다.
#
# 사용:
#   sbatch scripts/slurm_run.sh table1     # Table 1 (wiki)
#   sbatch scripts/slurm_run.sh da          # FewRel 2.0 도메인 적응
#   sbatch scripts/slurm_run.sh ablation    # 손실 가중치 ablation
# ----------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023 python/3.10 cuda
# 가상환경(최초 1회: virtualenv ~/envs/fewrel && pip install -r requirements.txt)
source ~/envs/fewrel/bin/activate

# 클러스터 컴퓨트 노드는 오프라인 → HF를 캐시/로컬에서만 로드.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK="${1:-table1}"
case "$TASK" in
  table1)   bash scripts/run_table1.sh ;;
  da)       bash scripts/run_da.sh ;;
  ablation) bash scripts/run_ablation.sh ;;
  *) echo "unknown task: $TASK (table1|da|ablation)"; exit 1 ;;
esac
