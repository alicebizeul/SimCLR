#!/bin/bash 
#SBATCH -o /cluster/work/vogtlab/Group/abizeul/simclr.out
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --tmp=10G
#SBATCH --array=0-768

############################ CONDA PARAMETERS #######################
source /cluster/home/abizeul/software/anaconda/etc/profile.d/conda.sh 
conda activate infonce



ID_LEARNSTD=$(expr $SLURM_ARRAY_TASK_ID / 384)  # 2
ID_BOUND=$(expr $SLURM_ARRAY_TASK_ID / 192) # 2
ID_CLASSES=$(expr $SLURM_ARRAY_TASK_ID / 48) # 4
ID_LEARNSTD_EPOCHS=$(expr $SLURM_ARRAY_TASK_ID / 16) # 3
ID_LR=$(expr $SLURM_ARRAY_TASK_ID / 4) # 4
ID_NORM=$(expr $SLURM_ARRAY_TASK_ID / 2) # 2
ID_SUBSAMPLE=$(expr $SLURM_ARRAY_TASK_ID % 2) # 2



if [[ ${ID_LEARNSTD} == 0 ]]; then
    LEARNSTD=True
fi

if [[ ${ID_LEARNSTD} == 1 ]]; then
    LEARNSTD=False
fi

if [[ ${ID_BOUND} == 0 ]]; then
    BOUND=True
fi

if [[ ${ID_BOUND} == 1 ]]; then
    BOUND=False
fi

if [[ ${ID_CLASSES} == 0 ]]; then
    CLASSES=10
fi

if [[ ${ID_CLASSES} == 1 ]]; then
    CLASSES=32
fi

if [[ ${ID_CLASSES} == 2 ]]; then
    CLASSES=64
fi

if [[ ${ID_CLASSES} == 3 ]]; then
    CLASSES=128
fi

if [[ ${ID_LEARNSTD_EPOCHS} == 0 ]]; then
    LSTD_EPOCHS=5
    EPOCHS=15
fi

if [[ ${ID_LEARNSTD_EPOCHS} == 1 ]]; then
    LSTD_EPOCHS=10
    EPOCHS=20
fi

if [[ ${ID_LR} == 0 ]]; then
    LR=0.01
    LR_CHANGE=False
fi

if [[ ${ID_LR} == 1 ]]; then
    LR=5e-4
    LR_CHANGE=True
fi

if [[ ${ID_LR} == 2 ]]; then
    LR=1e-4
    LR_CHANGE=True
fi

if [[ ${ID_LR} == 3 ]]; then
    LR=1e-5
    LR_CHANGE=True
fi

if [[ ${ID_NORM} == 0 ]]; then
    NORM=True
fi

if [[ ${ID_NORM} == 1 ]]; then
    NORM=False
fi

if [[ ${ID_SUBSAMPLE} == 0 ]]; then
    SUBSAMPLE=True
fi

if [[ ${ID_SUBSAMPLE} == 1 ]]; then
    SUBSAMPLE=False
fi

MODEL_PATH=/cluster/home/abizeul/test/${LEARNSTD}_${LSTD_EPOCHS}_${EPOCHS}_${CLASSES}_${NORM}_${LR}_${LR_CHANGE}_${BOUND}_${SUBSAMPLE}
mkdir -p ${MODEL_PATH}

python main.py \
        --custom=True \
        --learn_std=${LEARNSTD} \
        --std_epochs=${LSTD_EPOCHS} \
        --epochs=${EPOCHS} \
        --epoch_num=${EPOCHS} \
        --classes=${CLASSES} \
        --normalize=${NORM} \
        --lr_change=${LR_CHANGE} \
        --lr=${LR} \
        --bound=${BOUND} \
        --subsample=${SUBSAMPLE} \
        --model_path=${MODEL_PATH}


python linear_evaluation.py \
        --custom=True \
        --learn_std=${LEARNSTD} \
        --std_epochs=${LSTD_EPOCHS} \
        --epochs=${EPOCHS} \
        --epoch_num=${EPOCHS} \
        --classes=${CLASSES} \
        --normalize=${NORM} \
        --lr_change=${LR_CHANGE} \
        --lr=${LR} \
        --bound=${BOUND} \
        --subsample=${SUBSAMPLE} \
        --model_path=${MODEL_PATH}


