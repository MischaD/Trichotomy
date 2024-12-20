# Trichotomy of Generative Diffusion Models

## 

Step 1: Dataset preparation. Txt file in the following format

    images/00000006_000.png 1 0 0 0 0 0 0 0

Step 2: Compute Latents 

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/vol/ideadata/ed52egek/pycharm/trichotomy /vol/ideadata/ed52egek/conda/hugg/bin/python /vol/ideadata/ed52egek/pycharm/trichotomy/src/compute_latents_img.py ./src/base_experiment.py latent_mimic --batch_size 16 --output_path /vol/ideadata/ed52egek/data/trichotomy/MIMIC --filelist /vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/

Step 3: Compute Latent Normalization factor for edm2


Step 4: Compute Features 

 export CUDA_VISIBLE_DEVICES=0; beyondfid /vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8.csv "" "" --output_path /vol/idea_ramses/ed52egek/data/trichotomy/FEATURES --results_filename dummy.json --config-update=basedir=/vol/ideadata/ed52egek/data/chestxray14,feature_extractors.swav.batch_size=8 --feature_extractors swav  --master_port=12345


# Reconstruction baseline: 

## SDv2
 cd /vol/ideadata/ed52egek/pycharm/trichotomy; export PYTHONPATH=$PWD; conda activate edm2; export CUDA_VISIBLE_DEVICES=0; python /vol/ideadata/ed52egek/pycharm/trichotomy/src/compute_reconstructs.py --basedir /vol/idea_ramses/ed52egek/data/trichotomy/ --batch_size 16 --output_path /vol/ideadata/ed52egek/data/reconstructed/chestxray14/ --filelist /vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_train.txt

## Cosmos

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/vol/ideadata/ed52egek/pycharm/trichotomy \
/vol/ideadata/ed52egek/miniconda/envs/trichotomy/bin/python \
/vol/ideadata/ed52egek/pycharm/trichotomy/src/compute_latents_cosmos.py \
./src/base_experiment.py \
cosmos \
--batch_size 16 \
--basedir /vol/ideadata/ed52egek/data/chestxray14 \
--output_path /vol/ideadata/ed52egek/data/trichotomy/cosmos \
--filelist /vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8.csv

#DATASETS["chexpert"]="/vol/ideadata/ed52egek/data/chexpert/chexpertchestxrays-u20210408"
#DATASETS["mimic"]="/vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#DATASETS["cxr8"]="/vol/ideadata/ed52egek/data/chestxray14"
