#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=EvalFullDiADM
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

#module load python/3.9-anaconda
module load python
module load cuda
cd /home/atuin/b180dc/b180dc10/pycharm/trichotomy
source activate /home/atuin/b180dc/b180dc10/conda/edm2
export N_GPUS=${SLURM_GPUS_ON_NODE:-1}  # Default to 1 if not set


PROGRAM="generate_syn_train.py"
BASE_DIR="/home/atuin/b180dc/b180dc10/pycharm/trichotomy"
DATA_BASE_DIR="/home/atuin/b180dc/b180dc10/data/trichotomy_clean/latent"
IMG_DATA_BASE_DIR="/home/atuin/b180dc/b180dc10/data/chestxray14"

case "$SLURM_ARRAY_TASK_ID" in
    0) 
    MODE="DiADM-AG"; 
    GMODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_pseudocond/training-state-0008388.pt"; 
    GNET="${BASE_DIR}/baseline-runs/cxr8_pseudocond/network-snapshot-0008388-0.100.pkl"
    COND_MODE="pseudocond"; 
    ;;
    1) 
    MODE="DiADM"; 
    GMODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_uncond/training-state-0100663.pt"; 
    GNET="${BASE_DIR}/baseline-runs/cxr8_uncond/network-snapshot-0100663-0.100.pkl"
    COND_MODE="pseudocond"; 
    ;;
    2) 
    MODE="DiADM"; 
    GMODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_uncond/training-state-0008388.pt"; 
    GNET="${BASE_DIR}/baseline-runs/cxr8_uncond/network-snapshot-0008388-0.100.pkl"
    COND_MODE="pseudocond"; 
    ;;
    
    *) echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"; exit 1;;
esac
SAVE_NAME="${SLURM_ARRAY_TASK_ID}_${MODE}" # has to be unique

MODEL_WEIGHTS="${BASE_DIR}/baseline-runs/cxr8_pseudocond/training-state-0083886.pt"
NET="${BASE_DIR}/baseline-runs/cxr8_pseudocond/network-snapshot-0083886-0.100.pkl"

OUTDIR_PREFIX="${TMPDIR}/eval_${SAVE_NAME}"
OUTDIR="${OUTDIR_PREFIX}/id_${SAVE_NAME}" # beyondfid need double folder
GUIDANCE="1.4"
FILELIST="${BASE_DIR}/datasets/eight_cxr8_train.txt"
FILELIST_VAL="${BASE_DIR}/datasets/eight_cxr8_val.txt"
FILELIST_TEST="${BASE_DIR}/datasets/eight_cxr8_test.txt"
TRAINING_PATH_CSV="/home/atuin/b180dc/b180dc10/data/chestxray14/eight_cxr8.csv"


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
    --filelist $FILELIST\
    --mode $MODE\
    --outdir "${OUTDIR}"\
    --model_weights $MODEL_WEIGHTS\
    --gmodel_weights $GMODEL_WEIGHTS\
    --net $NET\
    --gnet $GNET\
    --cond_mode $COND_MODE\
    --guidance $GUIDANCE\
    --basedir $DATA_BASE_DIR\

beyondfid $TRAINING_PATH_CSV $TRAINING_PATH_CSV $OUTDIR_PREFIX --feature_extractors swav inception --metrics irs fid --output_path "${BASE_DIR}/results" --results_filename "eval_${SAVE_NAME}.json"

python ./chexnet/Main.py train \
    --data_dir_train $OUTDIR \
    --data_dir $IMG_DATA_BASE_DIR \
    --train_file $FILELIST \
    --val_file $FILELIST_VAL \
    --test_file $FILELIST_TEST \
    --outfile downstream/snth_checknext_${SAVE_NAME}.txt \
    --arch DENSE-NET-121 \
    --pretrained \
    --epochs 100 \
    --resize 256 \
    --crop 224 \
    --save_path ./downstream/saved_snth_models_${SAVE_NAME}
