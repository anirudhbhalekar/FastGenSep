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

echo -------------------------------------------------
echo Python version used:
python -V
echo ----------------------------------------------------------------
echo Starting evaluation...

python separate.py \
    --model /home/ab2810/rds/hpc-work/FastGenSep/seed_checkpoint/epoch-030_si_sdr-6.211.ckpt \
    --input_dir /home/ab2810/rds/hpc-work/LibriMix/Libri2Mix/wav8k/max/test/mix_clean/\
    --output_dir /home/ab2810/rds/hpc-work/FastGenSep/results/ \
    --num_steps 32 \
    --max_count 500 \
    

