#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=GenerateDSETrain
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
export N_GPUS=${SLURM_GPUS_ON_NODE:-1}  # Default to 1 if not set

PROGRAM="generate_with_dse.py"
BASE_DIR="/home/atuin/b180dc/b180dc10/pycharm/trichotomy"
DATA_BASE_DIR="/home/atuin/b180dc/b180dc10/data/trichotomy_ls"

MODE="DiADM"
COND_MODE="pseudocond" 
CLF_PATH="/home/atuin/b180dc/b180dc10/pycharm/trichotomy/importantmodels/m-05122024-131940.pth.tar"
PRIV_PATH="/home/atuin/b180dc/b180dc10/pycharm/trichotomy/importantmodels/Siamese_ResNet50_allcxr_checkpoint.pt"

N_PER_INDEX=32 

# SWAV is all previous experiments e.g. guidance strength
case "$SLURM_ARRAY_TASK_ID" in
    0) DS="cxr8"; FEATURE_EXTRACTOR="swav"; RUNPATHNAME="pseudocond";;
    1) DS="mimic"; FEATURE_EXTRACTOR="swav"; RUNPATHNAME="pseudocond";;
    2) DS="chexpert"; FEATURE_EXTRACTOR="swav"; RUNPATHNAME="pseudocond";;
    3) DS="cxr8"; FEATURE_EXTRACTOR="inception"; RUNPATHNAME="feinception";;
    4) DS="mimic"; FEATURE_EXTRACTOR="inception"; RUNPATHNAME="feinception";;
    5) DS="chexpert"; FEATURE_EXTRACTOR="inception"; RUNPATHNAME="feinception";;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"; exit 1;;
esac

GMODEL_WEIGHTS="${BASE_DIR}/baseline-runs/${DS}_uncond/training-state-0100663.pt"
GNET="${BASE_DIR}/baseline-runs/${DS}_uncond/network-snapshot-0100663-0.100.pkl"

MODEL_WEIGHTS="${BASE_DIR}/baseline-runs/${DS}_${RUNPATHNAME}/training-state-0083886.pt"
NET="${BASE_DIR}/baseline-runs/${DS}_${RUNPATHNAME}/network-snapshot-0083886-0.100.pkl"
SAVE_NAME="${FEATURE_EXTRACTOR}_${DS}" # has to be unique

OUTDIR_PREFIX="./dse_${SAVE_NAME}"
OUTDIR="${OUTDIR_PREFIX}/${SAVE_NAME}" # beyondfid need double folder
GUIDANCE="1.2"
FILELIST="${BASE_DIR}/datasets/eight_${DS}_train.txt"

# Export PYTHONPATH
export PYTHONPATH=$BASE_DIR

echo "==================================="
echo "Starting evaluation run"
echo "Task ID:              $SLURM_ARRAY_TASK_ID"
echo "Dataset:              $DS"
echo "Feature Extractor:    $FEATURE_EXTRACTOR"
echo "Run Name:             $RUNPATHNAME"
echo "File List:            $FILELIST"
echo "Output Directory:     $OUTDIR"
echo "Model Weights:        $MODEL_WEIGHTS"
echo "GModel Weights:       $GMODEL_WEIGHTS"
echo "Network Path:         $NET"
echo "GNetwork Path:        $GNET"
echo "Classifier Path:      $CLF_PATH"
echo "Private Path:         $PRIV_PATH"
echo "Base Directory:       $DATA_BASE_DIR"
echo "Guidance Strength:    $GUIDANCE"
echo "BeyondFID Data Path:  $TRAINING_PATH_CSV"
echo "Number per Index:     $N_PER_INDEX"
echo "Mode:                 $MODE"
echo "==================================="


python $PROGRAM \
    --n_per_index $N_PER_INDEX\
    --filelist $FILELIST\
    --target_dir "${OUTDIR}"\
    --mode $MODE\
    --guidance $GUIDANCE\
    --pseudo_cond_feature_extractor $FEATURE_EXTRACTOR\
    --model_weights $MODEL_WEIGHTS\
    --gmodel_weights $GMODEL_WEIGHTS\
    --path_net $NET\
    --path_gnet $GNET\
    --clf_path $CLF_PATH\
    --priv_path $PRIV_PATH\
    --basedir $DATA_BASE_DIR\
