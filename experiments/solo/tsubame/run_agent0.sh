#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=12:00:00
#$ -o ~/.log/solo/$JOB_ID.txt
#$ -j y

# Solo 学習（solo_agent0: agent_0 active / agent_1 dummy）
# agent_1 active 版は run_agent1.sh を使う

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate jaxmarl
cd ~/JaxMARL-overcooked-research

git pull
pip install -e . --no-deps

LAYOUT=${1:-cramped_room}
python experiments/solo/train.py SOLO_ACTIVE_AGENT=0 ENV_KWARGS.layout=$LAYOUT WANDB_MODE=offline
