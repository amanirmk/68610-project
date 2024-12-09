#!/bin/bash
#SBATCH --partition=cpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --mail-user=amanirmk@mit.edu
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT
#SBATCH --output=batch1
export HF_HUB_CACHE=/om/scratch/tmp/amanirmk
export HF_HOME=/om/scratch/tmp/amanirmk
set -e
cd /om/user/amanirmk/projects/68610-project/
source /om/user/amanirmk/miniconda/bin/activate
conda activate integration
python -m integration --do_vipr_and_sizes=False \
                      --do_nlvr_zeroshot=False \
                      --do_nlvr_finetune=False \
                      --do_vqa2_zeroshot=False \
                      --do_vqa2_finetune=True \
                      --save_intermediate=True \
                      --model_names \
                      "Salesforce/blip2-opt-2.7b"
                      # "Salesforce/blip2-opt-6.7b-coco"
                      # "Salesforce/blip-image-captioning-base" \
                      # "microsoft/git-base-textcaps" \
                      # "microsoft/git-large-textcaps" \
                      # "microsoft/git-base-msrvtt-qa" \
                      # "microsoft/git-large-msrvtt-qa" \
