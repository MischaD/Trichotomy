#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainEDM2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
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
PROGRAM="edm2/train_edm2.py"
DATA_BASE_DIR="/home/atuin/b180dc/b180dc10/data/trichotomy"
OUTDIR_PREFIX="baseline-runs"
PRESET="edm2-img512-xs"
BATCH_GPU="256"
LR="0.0012"
DECAY="17000"
FP16="False"
SEED="42"
SNAPSHOT="8Mi"
CHECKPOINT="8Mi"
STATUS="64Ki"
PSEUDO_COND_FEATURE_EXTRACTOR="swav"


#FILELIST="$BASE_DIR/datasets/eight_cxr8_train_debug.txt"; COND_MODE="cond"; OUTDIR="${OUTDIR_PREFIX}/cxr8_cond";;
#FILELIST="$BASE_DIR/datasets/eight_cxr8_train_debug.txt"; COND_MODE="cond"; OUTDIR="${OUTDIR_PREFIX}/cxr8_cond";;
# Filelist and cond_mode configurations
case "$SLURM_ARRAY_TASK_ID" in
    0) FILELIST="$BASE_DIR/datasets/eight_cxr8_train.txt"; ENCODER_NORM_MODE="cxr8"; COND_MODE="uncond"; OUTDIR="${OUTDIR_PREFIX}/cxr8_uncond";;
    1) FILELIST="$BASE_DIR/datasets/eight_cxr8_train.txt"; ENCODER_NORM_MODE="cxr8"; COND_MODE="cond"; OUTDIR="${OUTDIR_PREFIX}/cxr8_cond";;
    2) FILELIST="$BASE_DIR/datasets/eight_cxr8_train.txt"; ENCODER_NORM_MODE="cxr8"; COND_MODE="pseudocond"; OUTDIR="${OUTDIR_PREFIX}/cxr8_pseudocond";;
    3) FILELIST="$BASE_DIR/datasets/eight_mimic_train.txt"; ENCODER_NORM_MODE="mimic"; COND_MODE="uncond"; OUTDIR="${OUTDIR_PREFIX}/mimic_uncond";;
    4) FILELIST="$BASE_DIR/datasets/eight_mimic_train.txt"; ENCODER_NORM_MODE="mimic"; COND_MODE="cond"; OUTDIR="${OUTDIR_PREFIX}/mimic_cond";;
    5) FILELIST="$BASE_DIR/datasets/eight_mimic_train.txt"; ENCODER_NORM_MODE="mimic"; COND_MODE="pseudocond"; OUTDIR="${OUTDIR_PREFIX}/mimic_pseudocond";;
    6) FILELIST="$BASE_DIR/datasets/eight_chexpert_train.txt"; ENCODER_NORM_MODE="chexpert"; COND_MODE="uncond"; OUTDIR="${OUTDIR_PREFIX}/chexpert_uncond";;
    7) FILELIST="$BASE_DIR/datasets/eight_chexpert_train.txt"; ENCODER_NORM_MODE="chexpert"; COND_MODE="cond"; OUTDIR="${OUTDIR_PREFIX}/chexpert_cond";;
    8) FILELIST="$BASE_DIR/datasets/eight_chexpert_train.txt"; ENCODER_NORM_MODE="chexpert"; COND_MODE="pseudocond"; OUTDIR="${OUTDIR_PREFIX}/chexpert_pseudocond";;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"; exit 1;;
esac

# set pretrain model
if [[ $COND_MODE = "uncond" ]]
then 
    PRETRAIN_PATH="$BASE_DIR/importantmodels/edm2-img512-xs-uncond-2147483-0.045.pkl"
elif [[ $COND_MODE = "cond" ]]
then 
    PRETRAIN_PATH="$BASE_DIR/importantmodels/edm2-img512-xs-2147483-0.135.pkl"
elif [[ $COND_MODE = "pseudocond" ]]
then 
    PRETRAIN_PATH="$BASE_DIR/importantmodels/training-state-0298844.pt"
else 
    echo "Invalid Pretrain Path"; echo $COND_MODE; exit 1;
fi

# Export PYTHONPATH
export PYTHONPATH=$BASE_DIR

# Log the environment for debugging
echo "==================================="
echo "Running training with:"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "FILELIST: $FILELIST"
echo "COND_MODE: $COND_MODE"
echo "OUTDIR: $OUTDIR"
echo "N_GPUS: $N_GPUS"
echo "PROGRAM: $PROGRAM"
echo "PSEUDO_COND_FEATURE_EXTRACTOR: $PSEUDO_COND_FEATURE_EXTRACTOR"
echo "DATA_BASE_DIR: $DATA_BASE_DIR"
echo "PRETRAIN_PATH: $PRETRAIN_PATH"
echo "PRESET: $PRESET"
echo "BATCH_GPU: $BATCH_GPU"
echo "ENCODER_NORM_MODE: $ENCODER_NORM_MODE"
echo "LR: $LR"
echo "DECAY: $DECAY"
echo "FP16: $FP16"
echo "SEED: $SEED"
echo "SNAPSHOT: $SNAPSHOT"
echo "CHECKPOINT: $CHECKPOINT"
echo "STATUS: $STATUS"
echo "==================================="

# Execute the training script
torchrun --standalone --nproc_per_node=${N_GPUS} $PROGRAM \
    --pseudo_cond_feature_extractor $PSEUDO_COND_FEATURE_EXTRACTOR \
    --outdir $OUTDIR \
    --basedir $DATA_BASE_DIR \
    --filelist $FILELIST \
    --cond_mode $COND_MODE \
    --pretrain_path $PRETRAIN_PATH \
    --preset $PRESET \
    --batch-gpu $BATCH_GPU \
    --encoder_norm_mode $ENCODER_NORM_MODE \
    --lr $LR \
    --decay $DECAY \
    --fp16 $FP16 \
    --seed $SEED \
    --snapshot $SNAPSHOT \
    --checkpoint $CHECKPOINT \
    --status $STATUS
