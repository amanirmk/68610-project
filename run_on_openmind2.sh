#!/bin/bash
#SBATCH --partition=cpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-user=amanirmk@mit.edu
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT
#SBATCH --output=batch2
export HF_HUB_CACHE=/om/scratch/tmp/amanirmk
export HF_HOME=/om/scratch/tmp/amanirmk
set -e
cd /om/user/amanirmk/projects/68610-project/
source /om/user/amanirmk/miniconda/bin/activate
conda activate integration
python -m integration --model_names \
            "Salesforce/blip-image-captioning-large" \
            "microsoft/git-base" \
            "microsoft/git-large" \
            "microsoft/git-base-coco" \
            "microsoft/git-large-coco" \
            "google/paligemma-3b-pt-224" \
            "google/paligemma-3b-pt-448" \
            "google/paligemma-3b-mix-224" \
            "google/paligemma-3b-mix-448" \
            "google/paligemma-3b-ft-ocrvqa-224" \
            "google/paligemma-3b-ft-ocrvqa-448"