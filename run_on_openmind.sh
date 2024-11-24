#!/bin/bash
#SBATCH --partition=cpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=16GB
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-user=amanirmk@mit.edu
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT
#SBATCH --output=test1
export HF_HUB_CACHE=/om/scratch/tmp/amanirmk
set -e
cd /om/user/amanirmk/projects/68610-project/
source /om/user/amanirmk/miniconda/bin/activate
conda activate integration
python -m integration