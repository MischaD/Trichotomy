#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=Sample
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

#module load python/3.9-anaconda
module load python
module load cuda

cd /home/atuin/b180dc/b180dc10/pycharm/trichotomy
source activate /home/atuin/b180dc/b180dc10/conda/edm2


#tar xf /home/atuin/b180dc/b180dc10/data/trichotomy/images.tar -C $TMPDIR

export N_GPUS=${SLURM_GPUS_ON_NODE:-1}  # Default to 1 if not set

# Define the parameters for each run
BASE_DIR="/home/atuin/b180dc/b180dc10/pycharm/trichotomy"
PROGRAM="edm2/generate_syn_train.py"
OUTDIR_PREFIX="$TMPDIR"
DATA_BASE_DIR="/home/atuin/b180dc/b180dc10/data/trichotomy"
GUIDANCE="1.0"
FILELIST="${BASE_DIR}/datasets/eight_cxr8_train.txt"
GMODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_cond/training-state-0008388.pt"
GNET="${BASE_DIR}/baseline-runs/cxr8_cond/network-snapshot-0008388-0.100.pkl"
COND_MODE="cond"; 

TRAINING_PATH_CSV="/home/atuin/b180dc/b180dc10/data/chestxray14/eight_cxr8.csv"

case "$SLURM_ARRAY_TASK_ID" in
    0) CKPT="0083886"; OUTDIR="${OUTDIR_PREFIX}/cxr8_${CKPT}";;
    1) CKPT="0092274"; OUTDIR="${OUTDIR_PREFIX}/cxr8_${CKPT}";;
    2) CKPT="0100663"; OUTDIR="${OUTDIR_PREFIX}/cxr8_${CKPT}";;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"; exit 1;;
esac

MODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_cond/training-state-${CKPT}.pt"
NET="${BASE_DIR}/baseline-runs/cxr8_cond/network-snapshot-${CKPT}-0.100.pkl"

# Export PYTHONPATH
export PYTHONPATH=$BASE_DIR

# Log the environment for debugging
echo "==================================="
echo "Running training with:"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "FILELIST: $FILELIST"
echo "OUTDIR: $OUTDIR"
echo "model weights: $MODEL_WEIGHTS"
echo "beyondfid data path: $TRAINING_PATH_CSV"
echo "==================================="


python $PROGRAM \
    --model_weights $MODEL_WEIGHTS\
    --gmodel_weights $GMODEL_WEIGHTS\
    --net $NET\
    --gnet $GNET\
    --outdir "${OUTDIR}/cxr8_${CKPT}"\
    --cond_mode $COND_MODE\
    --guidance $GUIDANCE\
    --basedir $DATA_BASE_DIR\
    --filelist $FILELIST


beyondfid $TRAINING_PATH_CSV $TRAINING_PATH_CSV $OUTDIR --feature_extractors swav inception --metrics irs fid --output_path "${BASE_DIR}/results" --results_filename "EDM_AG_${CKPT}.json"
