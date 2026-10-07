"""cramped_room のサイズ拡張レイアウト（3x4 / 4x5 / 5x6 を自作）。

公式の自作方法（overcooked_v2/README.md「Custom Layouts」）に従い、ASCII を
`Layout.from_string` で Layout に変換して定義する。jaxmarl は import して使うだけ
（非編集）。学習・評価は `get_layout(name)` で名前を Layout に解決して env に渡す。

田中先生 指定の配置（Slack 2026/06/22）:
  鍋(P)=真ん中（偶数幅は真ん中の一個左） / 皿(B)=一番左 / 提出口(X,緑)=一番右
  玉ねぎ(O)=上の両サイド、1鍋固定（ステーション数固定）、床だけ拡大
記号: W=壁 P=鍋 O=玉ねぎ(ingredient 0) B=皿 X=提出口 A=エージェント
"""
from jaxmarl.environments.overcooked_v2.layouts import (
    Layout,
    overcooked_v2_layouts as _builtin_layouts,
)

# interior 3x4 (grid 5x6) — 偶数幅: 鍋は中央の一個左(col2)
_CRAMPED_ROOM_3x4 = """
WWPWWW
OA  AO
W    W
W    W
WBWWXW
"""

# interior 4x5 (grid 6x7) — 奇数幅: 鍋は中央(col3)
_CRAMPED_ROOM_4x5 = """
WWWPWWW
OA   AO
W     W
W     W
W     W
WBWWWXW
"""

# interior 5x6 (grid 7x8) — 偶数幅: 鍋は中央の一個左(col3)
_CRAMPED_ROOM_5x6 = """
WWWPWWWW
OA    AO
W      W
W      W
W      W
W      W
WBWWWWXW
"""

_RECIPES = [[0, 0, 0]]  # 3 個の玉ねぎ（ingredient 0）。cramped_room と同じ固定レシピ

CUSTOM_LAYOUTS = {
    "cramped_room_3x4": Layout.from_string(_CRAMPED_ROOM_3x4, possible_recipes=_RECIPES, swap_agents=True),
    "cramped_room_4x5": Layout.from_string(_CRAMPED_ROOM_4x5, possible_recipes=_RECIPES, swap_agents=True),
    "cramped_room_5x6": Layout.from_string(_CRAMPED_ROOM_5x6, possible_recipes=_RECIPES, swap_agents=True),
}


def get_layout(layout):
    """layout 名(str)を Layout に解決する。

    - 自作レイアウト名 (cramped_room_3x4 等) -> CUSTOM_LAYOUTS の Layout
    - jaxmarl 組み込み名 (cramped_room 等)   -> 組み込みの Layout
    - Layout オブジェクト                     -> そのまま

    OvercookedV2(layout=...) は str か Layout を受けるが、自作名は組み込み辞書に
    無いのでここで Layout に解決してから渡す（jaxmarl は非編集）。
    """
    if isinstance(layout, str):
        if layout in CUSTOM_LAYOUTS:
            return CUSTOM_LAYOUTS[layout]
        return _builtin_layouts[layout]
    return layout
