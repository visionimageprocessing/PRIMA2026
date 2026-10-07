# Independent Weight 学習

## 目的

2 agent が **別々の network 重み** で協力学習する baseline。
Shared との比較で「重み独立が学習効率・役割分担に与える影響」を測る。
Custom Rewards / Solo の基盤としても使われる。

## 実行

ローカル:
```bash
python experiments/independent/train.py
```

Tsubame:
```bash
qsub -g <your-group> experiments/independent/tsubame/run.sh
```

## 出力

`experiments/independent/results/<timestamp>/` 配下に agent ごとに同梱保存:

- `agent0/params.msgpack` + `agent0/config.yaml`
- `agent1/params.msgpack` + `agent1/config.yaml`
