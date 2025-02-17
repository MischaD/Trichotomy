#!/bin/bash

# Directories and paths
SCRIPT_PATH="/vol/ideadata/ed52egek/pycharm/trichotomy/chexnet/Main.py"
PYTHON_EXEC="/vol/ideadata/ed52egek/miniconda/envs/trichotomy/bin/python"

cd /vol/ideadata/ed52egek/pycharm/trichotomy/chexnet

# Dataset directories
declare -A DATASETS
DATASETS["chexpert"]="/vol/ideadata/ed52egek/data/chexpert/chexpertchestxrays-u20210408"
DATASETS["mimic"]="/vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
DATASETS["cxr8"]="/vol/ideadata/ed52egek/data/chestxray14"

# Model paths
### Full real models: (no cropping, no reconstruction)
#MODEL_MIMIC="/vol/ideadata/ed52egek/pycharm/trichotomy/chexnet/saved_models_mimic/m-06122024-153204.pth.tar"
#MODEL_CXR8="/vol/ideadata/ed52egek/pycharm/trichotomy/chexnet/saved_models_cxr8/m-05122024-131940.pth.tar"
#MODEL_CHEXPERT="/vol/ideadata/ed52egek/pycharm/trichotomy/saved_models_chexpert/m-06122024-153439.pth.tar"

MODEL_CHEXPERT="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_snth_classifiers/edm-2-ag_cxr8.pth.tar"
MODEL_CXR8="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_snth_classifiers/edm-2-ag_cxr8.pth.tar"
MODEL_MIMIC="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_snth_classifiers/edm-2-ag_cxr8.pth.tar"

# Test files
TEST_CHEXPERT="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_chexpert_test.txt"
TEST_CXR8="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_test.txt"
TEST_MIMIC="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_mimic_test.txt"

# Output directory
OUT_DIR="./results_edm2-agcxr8debug"
mkdir -p $OUT_DIR

# Arrays for iteration
models=("$MODEL_MIMIC" "$MODEL_CXR8" "$MODEL_CHEXPERT")
tests=("$TEST_MIMIC" "$TEST_CXR8" "$TEST_CHEXPERT")
model_names=("mimic" "cxr8" "chexpert")
test_names=("mimic" "cxr8" "chexpert")
datasets=("${DATASETS["mimic"]}" "${DATASETS["cxr8"]}" "${DATASETS["chexpert"]}")

# Loop through all combinations of models and test cases
for i in "${!models[@]}"; do
  for j in "${!tests[@]}"; do
    model_path=${models[$i]}
    test_file=${tests[$j]}
    data_dir=${datasets[$j]}
    outfile="$OUT_DIR/res_train${model_names[$i]}_${test_names[$j]}test.txt"
    
    echo "Running model ${model_names[$i]} on ${test_names[$j]} test case..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC $SCRIPT_PATH test \
      --data_dir $data_dir \
      --test_file $test_file \
      --model_path $model_path \
      --outfile $outfile \
      --arch DENSE-NET-121 \
      --pretrained \
      --num_classes 8 \
      --batch_size 16 \
      --resize 256 \
      --crop 224
  done
done

echo "All runs completed."