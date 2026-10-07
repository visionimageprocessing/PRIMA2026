# AI-AI 評価

## 目的

学習済みの重みを 2 つ読み込み、`NUM_ENVS=256` の並列環境で対戦・協力プレイさせて性能を測定する。
Shared / Independent / Custom Rewards 各重みの単独評価と、それぞれを組み合わせたクロス評価を行う。

## 実行

ローカル:
```bash
python experiments/evaluation/evaluate.py --layout cramped_room
```

Tsubame:
```bash
qsub -g <your-group> experiments/evaluation/tsubame/run.sh
```

## config

`config.yaml` の `weights` キーに各重みの path（`experiments/<exp>/results/<ts>/...` 配下）を、
`eval_pairs` に対戦組合せを書く。

## 出力

`experiments/evaluation/results/<pair>_<layout>_<timestamp>/`:

- `log.txt` — 集計（mean ± std、event count、Mean Value 等）
- `steps.csv` — 全 episode の step-by-step（long format、1 行 = 1 (episode_id, step)）
- `episodes.csv` — per-episode 集計（1 行 = 1 episode_id）
- `replay.gif` — episode_id=0 のアニメーション
- `config.yaml` — 評価時に使った train_config のコピー

## action フィールドのマッピング

`steps.csv` の `p1_action` / `p2_action` は文字列、`p1_action_id` / `p2_action_id` は整数で同一情報を併記する。
直接 CSV を開いたときの可読性 (文字列) と、pandas や Excel ピボットでの集計しやすさ (整数) の両方を取るため。

| action_id | action | 説明 |
|---|---|---|
| 0 | right | 右へ移動 |
| 1 | down  | 下へ移動 |
| 2 | left  | 左へ移動 |
| 3 | up    | 上へ移動 |
| 4 | stay  | 何もしない |
| 5 | interact | 隣のオブジェクトを操作（拾う / 置く / 配達 etc.） |

正準定義: `jaxmarl.environments.overcooked_v2.common.Actions`。

## 参照

- 評価ロジック本体: `src/overcooked_research/eval/interactive_two_ai.py`
- 関連 issue: #13 (CSV 化)
