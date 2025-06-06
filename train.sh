#!/bin/bash
#$ -cwd

echo ----------------------------------------------------------------

echo Activating environment
source /home/ab2810/miniconda3/bin/activate
cd /home/ab2810/rds/hpc-work/FastGenSep/
conda activate FGS_env

module load cuda/11.8

echo -------------------------------------------------
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++


export TORCH_CUDA_ARCH_LIST="8.0"

which gcc 
which g++
which nvcc 

echo -------------------------------------------------
echo Python version used:
python -V
echo ----------------------------------------------------------------
echo Starting training...
python train.py
#python -c "from models.ncsnpp_utils.op import fused_act" # Debugging line to check if the custom op is compiled correctly
echo ...training finished

echo ----------------------------------------------------------------
