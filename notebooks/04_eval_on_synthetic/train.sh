#!/bin/bash

### OPTIONS: 
# center crop only 
DATA_DIR_TRAIN="/vol/ideadata/ed52egek/data/reconstructed"

# Define the datasets and their configurations
declare -A DATASETS
DATASETS["chexpert"]="/vol/ideadata/ed52egek/data/chexpert/chexpertchestxrays-u20210408"
DATASETS["mimic"]="/vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
DATASETS["cxr8"]="/vol/ideadata/ed52egek/data/chestxray14"

# Define the corresponding train, val, and test files
declare -A FILES
FILES["chexpert_train"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_chexpert_train.txt"
FILES["chexpert_val"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_chexpert_val.txt"
FILES["chexpert_test"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_chexpert_test.txt"

FILES["mimic_train"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_mimic_train.txt"
FILES["mimic_val"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_mimic_val.txt"
FILES["mimic_test"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_mimic_test.txt"

FILES["cxr8_train"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_train.txt"
FILES["cxr8_val"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_val.txt"
FILES["cxr8_test"]="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_test.txt"

# Define the GPUs to use
GPUS=(1 2 3)

cd /vol/ideadata/ed52egek/pycharm/trichotomy/chexnet 

# Define the log file directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Common command options
COMMON_COMMAND="/vol/ideadata/ed52egek/conda/hugg/bin/python /vol/ideadata/ed52egek/pycharm/trichotomy/chexnet/Main.py train \
    --arch DENSE-NET-121 \
    --pretrained \
    --epochs 100 \
    --resize 256 \
    --crop 224"

# Initialize GPU index
GPU_INDEX=0

# Loop through each dataset and launch the training in a screen session
for DATASET_NAME in "${!DATASETS[@]}"; do
    DATA_DIR=${DATASETS[$DATASET_NAME]}
    TRAIN_FILE=${FILES["${DATASET_NAME}_train"]}
    VAL_FILE=${FILES["${DATASET_NAME}_val"]}
    TEST_FILE=${FILES["${DATASET_NAME}_test"]}
    OUT_FILE="${DATASETS[$DATASET_NAME]}_reconstructed"
    SAVE_PATH="./saved_models_reconstructed_${DATASET_NAME}"
    LOG_FILE="$LOG_DIR/${DATASET_NAME}_reconstructed_training.log"
    SCREEN_NAME="${DATASET_NAME}"

    # Assign the GPU
    GPU=${GPUS[$GPU_INDEX]}
    ((GPU_INDEX++))

    # Wrap the command
    COMMAND="sleep 2; CUDA_VISIBLE_DEVICES=$GPU $COMMON_COMMAND \
        --data_dir $DATA_DIR \
        --data_dir_train $DATA_DIR_TRAIN \
        --train_file $TRAIN_FILE \
        --val_file $VAL_FILE \
        --test_file $TEST_FILE \
        --outfile $OUT_FILE \
        --save_path $SAVE_PATH | tee $LOG_FILE; exit"

    echo "$COMMAND"
    echo "Launching for $DATASET_NAME with parameters:"
    echo "  Data directory: $DATA_DIR"
    echo "  Train file: $TRAIN_FILE"
    echo "  Val file: $VAL_FILE"
    echo "  Test file: $TEST_FILE"
    echo "  GPU: $GPU"
    echo "  Save path: $SAVE_PATH"
    echo "  Log file: $LOG_FILE"

    # Launch the command in a screen session
    screen -dmS "$SCREEN_NAME" bash -c "$COMMAND"

    echo "Launched training for $DATASET_NAME on GPU $GPU. Logging to $LOG_FILE."

    # Reset GPU_INDEX if we exceed available GPUs
    if [ $GPU_INDEX -ge ${#GPUS[@]} ]; then
        GPU_INDEX=0
    fi
done