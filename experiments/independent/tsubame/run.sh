#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=12:00:00
#$ -o ~/.log/independent/$JOB_ID.txt
#$ -j y

# Independent 重み学習

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate jaxmarl
cd ~/JaxMARL-overcooked-research

git pull
pip install -e . --no-deps

LAYOUT=${1:-cramped_room}
python experiments/independent/train.py ENV_KWARGS.layout=$LAYOUT WANDB_MODE=offline
