# Solo 学習（一人学習 / pseudo-solo control）

## 目的

active agent（`SOLO_ACTIVE_AGENT` で選ぶ。default は `agent_0`）が「協力相手を観測・
予測・回避する必要がない」条件で学習する対照群を作る。2 人で学習した重み
（Independent / Shared）と評価で組み合わせ、協力学習の効果や各 agent の貢献度を
切り出すために使う。

これは **完全な 1-agent 環境ではない**。評価時に協力相手と組み合わせて投入するため、
観測形状・ネットワーク構造を 2-agent と揃える必要があり、2-agent 環境を保ったまま
dummy 側（active の相手）を無害化する **pseudo-solo control** として構成する。

## 設計

`SOLO_ACTIVE_AGENT`（config, 0 か 1）で「どちらの slot を Solo 学習者（active）にするか」を
切り替える。もう一方が dummy になる。active への影響経路「観測」「行動」「物理干渉」の 3 つを潰す:

| 経路 | 対策 | 実装 |
|---|---|---|
| 観測 | active の network を `ActorCriticRNNSolo`（派生クラス）にして、forward 内で相手 agent チャンネルをマスク。観測は egocentric なので active が `agent_0`/`agent_1` どちらでも正しくマスクされる | `src/overcooked_research/train/network.py` |
| 行動 | dummy が env に渡す action を `STAY` に固定 | `train.py` `_env_step` |
| 物理干渉 | dummy を標準 cramped_room の (0,0) 壁マスに毎 step pin（**学習時のみ**） | `train.py` `_pin_agent` |

### マスク値

相手チャンネルは forward 内で **float32 cast したうえで** `pos → -1`、`dir / inv → 0` にする。

- `pos → -1`: 「相手がどのマスにも居ない」を明示。`0` だと「相手が居ない通常観測」と
  区別が付きにくい。uint8 のまま `-1` を入れると 255 にオーバーフローするため float32 にする。
- `dir / inv → 0`: 不在の agent は向き・所持品が無いので 0 が自然。

float32 cast は派生クラスの forward 内に閉じており、`Transition.obs` は uint8 のまま。
基底 network は元々 uint8 観測を conv で float32 に昇格しているので、cast 自体は重みに影響しない。

### dummy の network / update

dummy 側の network / transition / GAE / update は **構造維持のため残す**（Independent の学習
ループをそのまま使い回す）。dummy の行動は STAY 固定・位置は壁 pin で env への作用は
固定されており、network/update は計算構造の都合で残っているだけ。**保存・評価対象は active のみ**。

### (0,0) 壁 pin の妥当性（検証済み）

標準 cramped_room で `agent_1` を (0,0) 壁に毎 step pin する方式は、ローカルで検証済み:

- 1000 step / auto-reset 2 回を跨いでも `agent_1` は全 step (0,0) に留まる
- 計 3000 step エラーなし（env は壁マスの agent を問題なく処理）
- `agent_1` の event（delivery / placement / dish_pickup / plate_pickup）は全 step 0、
  `agent_0` は通常通り event を生成

## 実行

`SOLO_ACTIVE_AGENT` で active slot を選ぶ（default 0 = `agent_0` active）。

ローカル:
```bash
python experiments/solo/train.py                     # solo_agent0（default）
python experiments/solo/train.py SOLO_ACTIVE_AGENT=1  # solo_agent1
```

Tsubame:
```bash
qsub -g <your-group> experiments/solo/tsubame/run_agent0.sh  # solo_agent0
qsub -g <your-group> experiments/solo/tsubame/run_agent1.sh  # solo_agent1
```

`run.sh` 系は `git pull` でカレントブランチを更新する。Tsubame の repo を学習対象の
ブランチに checkout した状態で投入すること（script にブランチ名はハードコードしない）。

## 出力

`experiments/solo/results/<timestamp>/seed_<index>/agent<active>/`:

- `params.msgpack` — active（Solo）の学習済み重み
- `config.yaml` — 学習時の config

dummy 側は保存しない（active=0 なら `agent0/`、active=1 なら `agent1/`）。

## 評価との接続

評価では Solo 重みを使う agent だけ `ActorCriticRNNSolo` で読み込むことで、その agent の
観測だけがマスクされる（agent 単位マスク）。協力相手は通常の `ActorCriticRNN` で読み、
通常観測を渡す。**(0,0) pin は学習時のみで、評価時は協力相手が本物なので pin しない。**
（評価コード側の対応は別途。）
