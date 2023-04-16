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
TMP0=$(expr 384 '*' $ID_LEARNSTD)
TMP1=$(expr $SLURM_ARRAY_TASK_ID - $TMP0)
ID_BOUND=$(expr $TMP1 / 192) # 2
TMP2=$(expr 192 '*' $ID_BOUND)
TMP3=$(expr $TMP1 - $TMP2)
ID_CLASSES=$(expr $TMP3 / 48) # 4
TMP4=$(expr 48 '*' $ID_CLASSES)
TMP5=$(expr $TMP3 - $TMP4)
ID_LEARNSTD_EPOCHS=$(expr $TMP5 / 16) # 3
TMP6=$(expr 16 '*' $ID_LEARNSTD_EPOCHS)
TMP7=$(expr $TMP5 - $TMP6)
ID_LR=$(expr $TMP7 / 4) # 4
TMP8=$(expr 4 '*' $ID_LR)
TMP9=$(expr $TMP7 - $TMP8)
ID_NORM=$(expr $TMP9 / 2) # 2
ID_SUBSAMPLE=$(expr $TMP9 % 2) # 2



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


