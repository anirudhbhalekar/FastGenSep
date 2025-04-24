#!/bin/bash
#$ -cwd

echo ----------------------------------------------------------------
export LD_LIBRARY_PATH=/usr/lib/nvidia-450:$LD_LIBRARY_PATH

echo Activating environment
source /research/milsrg1/user_workspace/ab2810/miniconda3/bin/activate
cd /research/milsrg1/user_workspace/ab2810/IIB_Proj/SM_GenSep++
conda activate wv_env_env
echo Python version used:
python -V
echo Starting training...
python train.py
echo ...training finished

echo ----------------------------------------------------------------
