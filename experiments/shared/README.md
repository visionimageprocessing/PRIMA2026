# Shared Weight 学習

## 目的

2 agent が **同一の network 重み** を共有して協力学習する baseline。
Independent との比較で「重み共有が学習効率・協調行動に与える影響」を測る。

## 実行

ローカル:
```bash
python experiments/shared/train.py
```

Tsubame:
```bash
qsub -g <your-group> experiments/shared/tsubame/run.sh
```

## 出力

`experiments/shared/results/<timestamp>/` 配下に同梱保存:

- `params.msgpack` — 学習済み重み
- `config.yaml` — 学習時に使った config（再現用）
