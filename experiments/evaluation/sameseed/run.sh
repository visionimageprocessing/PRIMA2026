#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=02:00:00
#$ -o ~/.log/evaluate/$JOB_ID.txt
#$ -j y

# same-seed CC (C_i vs C_i) 評価ジョブ
# 目的: env の interact 手番優先(agent_0)由来の残余 delivery imbalance を、
#       同一 shared checkpoint 同士のペアで検証する(#21 本表とは別の補助解析)。
# 本表と同一バックエンド(Tsubame GPU)・同一 eval seed=4000 で実行し一貫性を確保する。

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate jaxmarl
cd ~/JaxMARL-overcooked-research

git pull
pip install -e . --no-deps

for ROOM in cramped_room cramped_room_3x4 cramped_room_4x5 cramped_room_5x6; do
  echo "=== same-seed eval: $ROOM ==="
  python experiments/evaluation/evaluate.py \
    --layout "$ROOM" \
    --seed 4000 \
    --eval_config experiments/evaluation/sameseed/eval_${ROOM}.yaml \
    --train_config experiments/shared/config.yaml
done
