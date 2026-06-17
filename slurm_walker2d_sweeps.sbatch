#!/bin/bash
#SBATCH --job-name=walker2d-sac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-99%10
#SBATCH --nodelist=linux[56-60]
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --error=slurm-logs/%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /usr/pkg/miniconda-23.9.0/etc/profile.d/conda.sh
conda activate base

if [[ -f .venv/bin/activate ]]; then
	source .venv/bin/activate
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

TV_RHOS=(None 50 55 60 65)
KL_RHOS=(50 55 60 65 70)

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID < 50 )); then
	ALG="tv"
	SCRIPT="run.py"
	RHO_INDEX=$(( TASK_ID / 10 ))
	SEED=$(( TASK_ID % 10 ))
	RHO="${TV_RHOS[$RHO_INDEX]}"
else
	ALG="kl"
	SCRIPT="run2.py"
	KL_TASK_ID=$(( TASK_ID - 50 ))
	RHO_INDEX=$(( KL_TASK_ID / 10 ))
	SEED=$(( KL_TASK_ID % 10 ))
	RHO="${KL_RHOS[$RHO_INDEX]}"
fi

export PYTHONHASHSEED="$SEED"

OUTDIR="results/${ALG}/rho_${RHO}/seed_${SEED}"
mkdir -p "$OUTDIR"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "alg=$ALG script=$SCRIPT rho=$RHO seed=$SEED outdir=$OUTDIR"
echo "python=$(command -v python)"
python - <<'PY'
import gymnasium, torch
print("gymnasium", gymnasium.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

srun python -u "$SCRIPT" --rho "$RHO" --seed "$SEED" --outdir "$OUTDIR"
