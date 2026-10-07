#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=02:00:00
#$ -o ~/.log/evaluate/$JOB_ID.txt
#$ -j y

# 評価ジョブ

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate jaxmarl
cd ~/JaxMARL-overcooked-research

git pull
pip install -e . --no-deps

python experiments/evaluation/evaluate.py --layout cramped_room
