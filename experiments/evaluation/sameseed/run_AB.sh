#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=02:00:00
#$ -o ~/.log/evaluate/$JOB_ID.txt
#$ -j y

# same-seed AB (A_i + B_i) 評価ジョブ — 田中先生依頼「学習時と同じ(知っている)ペア」。
# independent 学習 run i の agent0(=A_i) と agent1(=B_i) をそのままペアにする。
# 現状の cross-seed AB(知らない人/strangers)と対比し、familiar-partner の効果を測る。
# 本表と同一バックエンド(Tsubame GPU)・同一 eval seed=4000 で一貫性確保。

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate jaxmarl
cd ~/JaxMARL-overcooked-research

git pull
pip install -e . --no-deps

for ROOM in cramped_room cramped_room_3x4 cramped_room_4x5 cramped_room_5x6; do
  echo "=== same-seed AB eval: $ROOM ==="
  python experiments/evaluation/evaluate.py \
    --layout "$ROOM" \
    --seed 4000 \
    --eval_config experiments/evaluation/sameseed/eval_AB_${ROOM}.yaml \
    --train_config experiments/independent/config.yaml
done
