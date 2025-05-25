#!/bin/bash
#$ -cwd

echo ----------------------------------------------------------------

echo Activating environment
source /home/ab2810/miniconda3/bin/activate
cd /home/ab2810/rds/hpc-work/FastGenSep/
conda activate FGS_env

echo -------------------------------------------------

echo Python version used:
python -V
echo ----------------------------------------------------------------
echo Starting training...
python train.py
echo ...training finished

echo ----------------------------------------------------------------
