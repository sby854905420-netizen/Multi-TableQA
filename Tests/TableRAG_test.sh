#!/bin/bash
#SBATCH -A 
#SBATCH -p 
#SBATCH -N 1
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -t 0-05:30:00
#SBATCH --job-name=deepseek-run
#SBATCH --output=test-%j.log


module load Anaconda3
source 


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd 


python tests/test_nocode_TableRAG_qd.py
# python src/llm/model_finetune.py

