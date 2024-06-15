#!/bin/bash

#SBATCH --job-name=Grp-A-VulCheck         # Job name
#SBATCH --output=output_%j.txt           # Standard output and error log
#SBATCH --error=error_%j.txt             # Error log
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gpus=1                         # Number of GPUs required

conda activate ai-for-se-grp4
python3 ./create_embeddings.py
