"""2 AI 対戦評価のエントリーポイント"""

import argparse
import csv
import json
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes
from datetime import datetime
import os
import shutil
import yaml

from jaxmarl.environments.overcooked_v2.common import Actions
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as layouts
from jaxmarl.environments.overcooked_v2.settings import DELIVERY_REWARD
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

from overcooked_research.env import OvercookedV2WithEvents as OvercookedV2, get_layout
from overcooked_research.train import ScannedRNN, CNN, ActorCriticRNN, ActorCriticRNNSolo


# Actions enum (int → 表示名) の対応表
ACTION_NAMES = {
    int(Actions.right):    "right",
    int(Actions.down):     "down",
    int(Actions.left):     "left",
    int(Actions.up):       "up",
    int(Actions.stay):     "stay",
    int(Actions.interact): "interact",
}


# パス解決をスクリプト基準に統一（cwd 依存を避ける）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

def _resolve_checkpoint(path):
    """相対パスを _REPO_ROOT 基準で絶対化する"""
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return os.path.normpath(path)


def load_eval_config(eval_config_path):
    """per-slot 評価設定 yaml を読み込む（A2 形式）

    yaml は以下の形を想定（experiments/solo/eval/config.yaml 参照）:
        evaluations:
          - name: <評価名>
            agent_0: {policy_type: solo|coop|dummy, checkpoint: <path>}
            agent_1: {policy_type: ..., checkpoint: ...}

    policy_type:
      solo  -> ActorCriticRNNSolo（相手チャンネルをマスク）。checkpoint 必須
      coop  -> ActorCriticRNN。checkpoint 必須
      dummy -> network なし。STAY + (0,0) pin + value NaN。checkpoint 不要

    Returns:
        evaluations (list[dict]): 各 dict は {name, agent_0: spec, agent_1: spec}。
        spec は {policy_type, checkpoint?(絶対パス)}。dummy は checkpoint を持たない。
    """
    with open(eval_config_path, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    evaluations = []
    for ev in eval_cfg["evaluations"]:
        slots = {}
        for slot_key in ("agent_0", "agent_1"):
            spec = dict(ev[slot_key])
            policy_type = spec["policy_type"]
            if policy_type not in ("solo", "coop", "dummy"):
                raise ValueError(
                    f"{ev['name']}.{slot_key}: unknown policy_type {policy_type!r}"
                )
            if policy_type == "dummy":
                if spec.get("checkpoint"):
                    raise ValueError(
                        f"{ev['name']}.{slot_key}: dummy は checkpoint を持てない"
                    )
                spec["checkpoint"] = None
            else:
                if not spec.get("checkpoint"):
                    raise ValueError(
                        f"{ev['name']}.{slot_key}: {policy_type} は checkpoint 必須"
                    )
                spec["checkpoint"] = _resolve_checkpoint(spec["checkpoint"])
            slots[slot_key] = spec
        evaluations.append(
            {"name": ev["name"], "agent_0": slots["agent_0"], "agent_1": slots["agent_1"]}
        )
    return evaluations


def _infer_trained_slot(checkpoint, policy_type):
    """checkpoint の来歴（trained_slot）を導出する。

    solo  -> 同梱 config の SOLO_ACTIVE_AGENT（無ければ親フォルダ名 agentN）から agent_0/1
    coop  -> "shared"（パラメータ共有で slot 固有でない）
    dummy -> None
    """
    if policy_type == "coop":
        return "shared"
    if policy_type == "dummy":
        return None
    # solo
    bundled = os.path.join(os.path.dirname(checkpoint), "config.yaml")
    if os.path.exists(bundled):
        with open(bundled, "r", encoding="utf-8") as f:
            active = yaml.safe_load(f).get("SOLO_ACTIVE_AGENT")
        if active is not None:
            return f"agent_{int(active)}"
    parent = os.path.basename(os.path.dirname(checkpoint))  # "agent0" / "agent1"
    if parent in ("agent0", "agent1"):
        return f"agent_{parent[-1]}"
    return None


def build_policy(spec, env, config, slot_idx):
    """slot の policy 仕様から評価用 policy を構築する。

    Returns dict:
        policy_type, network(None=dummy), params(None=dummy),
        checkpoint, trained_slot
    """
    policy_type = spec["policy_type"]
    if policy_type == "dummy":
        return {
            "policy_type": "dummy", "network": None, "params": None,
            "checkpoint": None, "trained_slot": None,
        }

    action_dim = env.action_space(env.agents[slot_idx]).n
    if policy_type == "solo":
        num_ingredients = int(env.layout.num_ingredients)
        network = ActorCriticRNNSolo(action_dim, config=config, num_ingredients=num_ingredients)
    else:  # coop
        network = ActorCriticRNN(action_dim, config=config)

    with open(spec["checkpoint"], "rb") as f:
        params = from_bytes(None, f.read())

    return {
        "policy_type": policy_type, "network": network, "params": params,
        "checkpoint": spec["checkpoint"],
        "trained_slot": _infer_trained_slot(spec["checkpoint"], policy_type),
    }


def resolve_eval_config(ev, fallback_config):
    """evaluation の最初の非 dummy slot の同梱 config を env/network 構築の基準に使う。

    全 checkpoint が同じ network サイズ（GRU_HIDDEN_DIM 等）で学習されている前提。
    同梱 config が無ければ fallback_config を使う。
    """
    for slot_key in ("agent_0", "agent_1"):
        spec = ev[slot_key]
        if spec["policy_type"] != "dummy":
            bundled = os.path.join(os.path.dirname(spec["checkpoint"]), "config.yaml")
            if os.path.exists(bundled):
                with open(bundled, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
    return fallback_config


MAX_STEPS = 1000


class InteractiveOvercookedV2:

    def __init__(self, layout, config, slot0_spec, slot1_spec,
                 agent_view_size=None, max_steps=1000, no_jit=False, debug=False):
        self.debug = debug
        self.no_jit = no_jit
        self.config = config
        self.num_envs = config["NUM_ENVS"]

        self.env = OvercookedV2(
            layout=layout,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            random_agent_positions=config["ENV_KWARGS"].get("random_agent_positions", False),
            negative_rewards=config["ENV_KWARGS"].get("negative_rewards", False),
            sample_recipe_on_delivery=config["ENV_KWARGS"].get("sample_recipe_on_delivery", False),
        )
        self.viz = OvercookedV2Visualizer()

        # slot ごとに policy を構築（solo/coop は network+params、dummy は network なし）
        self.policy_0 = build_policy(slot0_spec, self.env, config, 0)
        self.policy_1 = build_policy(slot1_spec, self.env, config, 1)

        # dummy が居る slot（行動 STAY + (0,0) pin の対象）。両 slot とも非 dummy なら None
        self.dummy_slot = None
        if self.policy_0["policy_type"] == "dummy":
            self.dummy_slot = 0
        elif self.policy_1["policy_type"] == "dummy":
            self.dummy_slot = 1

        rng = jax.random.PRNGKey(0)
        rng, _rng_0, _rng_1 = jax.random.split(rng, 3)

        obs_shape = self.env.observation_space().shape
        init_x = (
            jnp.zeros((1, self.num_envs, *obs_shape)),
            jnp.zeros((1, self.num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(self.num_envs, config["GRU_HIDDEN_DIM"])

        # ネットワーク構造の確定（dummy はスキップ。params は build_policy でロード済み）
        if self.policy_0["network"] is not None:
            self.policy_0["network"].init(_rng_0, init_hstate, init_x)
        if self.policy_1["network"] is not None:
            self.policy_1["network"].init(_rng_1, init_hstate, init_x)

        # JIT
        self.jit_reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0,)))
        self.jit_step = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0, 0)))
        self.jit_get_obs = jax.jit(jax.vmap(self.env.get_obs))
        self.jit_apply_0 = (
            jax.jit(self.policy_0["network"].apply)
            if self.policy_0["network"] is not None else None
        )
        self.jit_apply_1 = (
            jax.jit(self.policy_1["network"].apply)
            if self.policy_1["network"] is not None else None
        )

    def run(self, key):
        """キーボード操作モード"""
        self.key = key
        with jax.disable_jit(self.no_jit):
            self._run()

    def _run(self):
        self._reset()
        self.viz.window.reg_key_handler(self._handle_input)
        self.viz.show(block=True)

    def _handle_input(self, event):
        if self.debug:
            print("Pressed", event.key)

        ACTION_MAPPING = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down,
            " ": Actions.interact,
            "tab": Actions.stay,
        }

        match event.key:
            case "escape":
                self.viz.window.close()
                return
            case "backspace":
                self._reset()
                return
            case key if key in ACTION_MAPPING:
                action = ACTION_MAPPING[key]
            case key:
                print(f"Key {key} not recognized")
                return

        self._step(action)

    def _redraw(self):
        if self.num_envs != 1:
            return
        self.viz.render(self.state, agent_view_size=self.env.agent_view_size)

    def _reset(self):
        """環境とRNN隠れ状態をリセット"""
        self.key, key = jax.random.split(self.key)
        reset_rng = jax.random.split(key, self.num_envs)
        obsv, state = self.jit_reset(reset_rng)
        # dummy slot を (0,0) 壁に pin して obs を再取得（学習時の reset 直後 pin と同条件）
        if self.dummy_slot is not None:
            state = self._pin_agent_state(state, self.dummy_slot)
            obsv = self.jit_get_obs(state)
        self.state = state
        self.obsv = obsv
        self.hstate_0 = ScannedRNN.initialize_carry(self.num_envs, self.config["GRU_HIDDEN_DIM"])
        self.hstate_1 = ScannedRNN.initialize_carry(self.num_envs, self.config["GRU_HIDDEN_DIM"])
        self.last_done = jnp.zeros((self.num_envs,), dtype=bool)

        self._redraw()

    def _step(self, action):
        """キーボード操作用：両エージェントに同じ action を渡す"""
        if self.num_envs != 1:
            raise ValueError("Interactive keyboard mode only supports NUM_ENVS=1.")
        self.key, subkey = jax.random.split(self.key)

        actions = {f"agent_{i}": jnp.array(action) for i in range(self.env.num_agents)}
        if self.debug:
            print("Actions: ", actions)

        obs, state, reward, done, info = jax.jit(self.env.step_env)(
            subkey, self.state, actions
        )
        self.state = state
        self.obsv = obs
        print(f"t={state.time}: reward={reward['agent_0']}, done = {done['__all__']}")

        if self.debug:
            a0_obs = obs["agent_0"]
            a0_obs = jnp.transpose(a0_obs, (2, 0, 1))
            print("Agent 0 observation: ", a0_obs)
            print("Reward: ", reward)
            print("Shaped reward: ", info["shaped_reward"])

        if done["__all__"]:
            self._reset()
        else:
            self._redraw()

    @staticmethod
    def _mask_keep_tree(old_tree, new_tree, completed_mask):
        def _keep(old_leaf, new_leaf):
            reshape = (completed_mask.shape[0],) + (1,) * (new_leaf.ndim - 1)
            return jnp.where(completed_mask.reshape(reshape), old_leaf, new_leaf)

        return jax.tree_util.tree_map(_keep, old_tree, new_tree)

    @staticmethod
    def _mask_zero_tree(tree, completed_mask):
        def _zero(leaf):
            reshape = (completed_mask.shape[0],) + (1,) * (leaf.ndim - 1)
            return jnp.where(completed_mask.reshape(reshape), jnp.zeros_like(leaf), leaf)

        return jax.tree_util.tree_map(_zero, tree)

    @staticmethod
    def _pin_agent_state(state, agent_idx, x=0, y=0):
        """state 内の agent_idx の位置を (x, y) に上書きする（dummy を壁に固定）。

        評価 env は OvercookedV2（wrapper なし）なので state.agents.pos を直接書き換える。
        pos は (num_envs, num_agents)。学習側 train.py の _pin_agent と同じ役割。
        """
        return state.replace(
            agents=state.agents.replace(
                pos=state.agents.pos.replace(
                    x=state.agents.pos.x.at[:, agent_idx].set(x),
                    y=state.agents.pos.y.at[:, agent_idx].set(y),
                )
            )
        )

    def _ai_step(self):
        """AIモードの NUM_ENVS 並列環境で 1 ステップ進める"""
        self.key, _rng_0, _rng_1, _rng_step = jax.random.split(self.key, 4)
        completed_mask = self.last_done

        obs_0 = self.obsv["agent_0"]
        obs_1 = self.obsv["agent_1"]

        # slot 0: solo/coop は network から sample、dummy は STAY 固定 + value NaN
        if self.policy_0["policy_type"] == "dummy":
            action_0 = jnp.full((self.num_envs,), int(Actions.stay), dtype=jnp.int32)
            value_0 = jnp.full((1, self.num_envs), jnp.nan)
            pi_0 = None
        else:
            ac_in_0 = (obs_0[jnp.newaxis, :], self.last_done[jnp.newaxis, :])
            self.hstate_0, pi_0, value_0 = self.jit_apply_0(
                self.policy_0["params"], self.hstate_0, ac_in_0
            )
            action_0 = pi_0.sample(seed=_rng_0)

        # slot 1
        if self.policy_1["policy_type"] == "dummy":
            action_1 = jnp.full((self.num_envs,), int(Actions.stay), dtype=jnp.int32)
            value_1 = jnp.full((1, self.num_envs), jnp.nan)
            pi_1 = None
        else:
            ac_in_1 = (obs_1[jnp.newaxis, :], self.last_done[jnp.newaxis, :])
            self.hstate_1, pi_1, value_1 = self.jit_apply_1(
                self.policy_1["params"], self.hstate_1, ac_in_1
            )
            action_1 = pi_1.sample(seed=_rng_1)

        env_act = {
            "agent_0": action_0.squeeze().flatten(),
            "agent_1": action_1.squeeze().flatten(),
        }
        rng_step = jax.random.split(_rng_step, self.num_envs)
        obsv, state, reward, done, info = self.jit_step(rng_step, self.state, env_act)

        # dummy slot を (0,0) 壁に pin → obs 再取得（学習と同条件。相手が本物の時は pin しない）
        if self.dummy_slot is not None:
            state = self._pin_agent_state(state, self.dummy_slot)
            obsv = self.jit_get_obs(state)

        reward = self._mask_zero_tree(reward, completed_mask)
        info = self._mask_zero_tree(info, completed_mask)
        done = self._mask_zero_tree(done, completed_mask)
        done["__all__"] = jnp.logical_or(done["__all__"], completed_mask)
        # 評価では env が両 agent に同じ協調 reward を配るデフォルト挙動を上書きし、
        # info["delivery"] から「delivery を実行した agent にのみ」DELIVERY_REWARD を与える
        # （completed_mask 適用後の info を使うので、完了済 episode は二重カウントしない）
        reward = {
            "agent_0": info["delivery"][..., 0].astype(jnp.float32) * DELIVERY_REWARD,
            "agent_1": info["delivery"][..., 1].astype(jnp.float32) * DELIVERY_REWARD,
        }
        for agent_key in self.env.agents:
            done[agent_key] = done["__all__"]
        obsv = self._mask_keep_tree(self.obsv, obsv, completed_mask)
        state = self._mask_keep_tree(self.state, state, completed_mask)

        self.obsv = obsv
        self.state = state
        self.last_done = done["__all__"]

        self._redraw()

        # action_0 / action_1 を NUM_ENVS 並列ぶん（shape: (num_envs,)）にして返す
        # state は OvercookedV2 の State（NUM_ENVS 並列の中身を持つ）をそのまま返す
        action_0_flat = action_0.squeeze().flatten()
        action_1_flat = action_1.squeeze().flatten()
        return reward, done, info, value_0, value_1, pi_0, pi_1, action_0_flat, action_1_flat, state

    def run_evaluation(self, key, max_steps=MAX_STEPS):
        """NUM_ENVS 個の trajectory を並列に 1 エピソード実行する"""
        self.key = key
        with jax.disable_jit(self.no_jit):
            return self._run_evaluation(max_steps)

    def _run_evaluation(self, max_steps):
        self._reset()

        trajectory_buffers = []
        for trajectory_idx in range(self.num_envs):
            trajectory_buffers.append(
                {
                    "episode": trajectory_idx,
                    "step_rewards_agent0": [],
                    "step_rewards_agent1": [],
                    "step_shaped_rewards_agent0": [],
                    "step_shaped_rewards_agent1": [],
                    "step_values_agent0": [],
                    "step_values_agent1": [],
                    "step_delivery_agent0": [],
                    "step_delivery_agent1": [],
                    "step_placement_agent0": [],
                    "step_placement_agent1": [],
                    "step_dish_pickup_agent0": [],
                    "step_dish_pickup_agent1": [],
                    "step_plate_pickup_agent0": [],
                    "step_plate_pickup_agent1": [],
                    "step_actions_agent0": [],   # int (Actions enum 値)
                    "step_actions_agent1": [],
                    "terminated": False,
                    "truncated": False,
                }
            )

        # replay.gif 用に Trajectory 0 の state 列を別途記録（メモリ節約のため Trajectory 0 のみ）
        state_seq_traj0 = []

        completed = np.zeros(self.num_envs, dtype=bool)

        for _ in range(max_steps):
            if np.all(completed):
                break

            reward, done, info, value_0, value_1, _, _, action_0, action_1, state_all = self._ai_step()

            # Trajectory 0 の state を切り出して保存（OvercookedV2 の State を tree_map で env 軸 0 を取り出す）
            if not completed[0]:
                state_traj0 = jax.tree_util.tree_map(lambda x: x[0], state_all)
                state_seq_traj0.append(state_traj0)

            reward_0 = np.asarray(reward["agent_0"])
            reward_1 = np.asarray(reward["agent_1"])
            shaped_0 = np.asarray(info["shaped_reward"]["agent_0"])
            shaped_1 = np.asarray(info["shaped_reward"]["agent_1"])
            value_0 = np.asarray(value_0.squeeze(0))
            value_1 = np.asarray(value_1.squeeze(0))
            action_0_np = np.asarray(action_0, dtype=np.int32)
            action_1_np = np.asarray(action_1, dtype=np.int32)
            delivery = np.asarray(info["delivery"], dtype=bool)
            placement = np.asarray(info["placement"], dtype=bool)
            dish_pickup = np.asarray(info["dish_pickup"], dtype=bool)
            plate_pickup = np.asarray(info["plate_pickup"], dtype=bool)
            delivery_0 = delivery[:, 0]
            delivery_1 = delivery[:, 1]
            placement_0 = placement[:, 0]
            placement_1 = placement[:, 1]
            dish_pickup_0 = dish_pickup[:, 0]
            dish_pickup_1 = dish_pickup[:, 1]
            plate_pickup_0 = plate_pickup[:, 0]
            plate_pickup_1 = plate_pickup[:, 1]
            done_all = np.asarray(done["__all__"], dtype=bool)

            for env_idx in range(self.num_envs):
                if completed[env_idx]:
                    continue

                buf = trajectory_buffers[env_idx]
                buf["step_rewards_agent0"].append(float(reward_0[env_idx]))
                buf["step_rewards_agent1"].append(float(reward_1[env_idx]))
                buf["step_shaped_rewards_agent0"].append(float(shaped_0[env_idx]))
                buf["step_shaped_rewards_agent1"].append(float(shaped_1[env_idx]))
                buf["step_values_agent0"].append(float(value_0[env_idx]))
                buf["step_values_agent1"].append(float(value_1[env_idx]))
                buf["step_delivery_agent0"].append(bool(delivery_0[env_idx]))
                buf["step_delivery_agent1"].append(bool(delivery_1[env_idx]))
                buf["step_placement_agent0"].append(bool(placement_0[env_idx]))
                buf["step_placement_agent1"].append(bool(placement_1[env_idx]))
                buf["step_dish_pickup_agent0"].append(bool(dish_pickup_0[env_idx]))
                buf["step_dish_pickup_agent1"].append(bool(dish_pickup_1[env_idx]))
                buf["step_plate_pickup_agent0"].append(bool(plate_pickup_0[env_idx]))
                buf["step_plate_pickup_agent1"].append(bool(plate_pickup_1[env_idx]))
                buf["step_actions_agent0"].append(int(action_0_np[env_idx]))
                buf["step_actions_agent1"].append(int(action_1_np[env_idx]))

                if done_all[env_idx]:
                    completed[env_idx] = True
                    buf["truncated"] = True

        all_episode_results = []
        for env_idx, buf in enumerate(trajectory_buffers):
            episode_reward_0 = sum(buf["step_rewards_agent0"])
            episode_reward_1 = sum(buf["step_rewards_agent1"])
            episode_shaped_reward_0 = sum(buf["step_shaped_rewards_agent0"])
            episode_shaped_reward_1 = sum(buf["step_shaped_rewards_agent1"])
            delivery_count_0 = sum(buf["step_delivery_agent0"])
            delivery_count_1 = sum(buf["step_delivery_agent1"])
            placement_count_0 = sum(buf["step_placement_agent0"])
            placement_count_1 = sum(buf["step_placement_agent1"])
            dish_pickup_count_0 = sum(buf["step_dish_pickup_agent0"])
            dish_pickup_count_1 = sum(buf["step_dish_pickup_agent1"])
            plate_pickup_count_0 = sum(buf["step_plate_pickup_agent0"])
            plate_pickup_count_1 = sum(buf["step_plate_pickup_agent1"])
            mean_value_0 = np.mean(buf["step_values_agent0"]) if buf["step_values_agent0"] else float("nan")
            mean_value_1 = np.mean(buf["step_values_agent1"]) if buf["step_values_agent1"] else float("nan")

            episode_result = {
                "episode": env_idx,
                "num_steps": len(buf["step_rewards_agent0"]),
                "episode_reward_agent0": episode_reward_0,
                "episode_reward_agent1": episode_reward_1,
                "episode_shaped_reward_agent0": episode_shaped_reward_0,
                "episode_shaped_reward_agent1": episode_shaped_reward_1,
                "delivery_count_agent0": delivery_count_0,
                "delivery_count_agent1": delivery_count_1,
                "placement_count_agent0": placement_count_0,
                "placement_count_agent1": placement_count_1,
                "dish_pickup_count_agent0": dish_pickup_count_0,
                "dish_pickup_count_agent1": dish_pickup_count_1,
                "plate_pickup_count_agent0": plate_pickup_count_0,
                "plate_pickup_count_agent1": plate_pickup_count_1,
                "mean_value_agent0": mean_value_0,
                "mean_value_agent1": mean_value_1,
                "step_rewards_agent0": buf["step_rewards_agent0"],
                "step_rewards_agent1": buf["step_rewards_agent1"],
                "step_shaped_rewards_agent0": buf["step_shaped_rewards_agent0"],
                "step_shaped_rewards_agent1": buf["step_shaped_rewards_agent1"],
                "step_values_agent0": buf["step_values_agent0"],
                "step_values_agent1": buf["step_values_agent1"],
                "step_delivery_agent0": buf["step_delivery_agent0"],
                "step_delivery_agent1": buf["step_delivery_agent1"],
                "step_placement_agent0": buf["step_placement_agent0"],
                "step_placement_agent1": buf["step_placement_agent1"],
                "step_dish_pickup_agent0": buf["step_dish_pickup_agent0"],
                "step_dish_pickup_agent1": buf["step_dish_pickup_agent1"],
                "step_plate_pickup_agent0": buf["step_plate_pickup_agent0"],
                "step_plate_pickup_agent1": buf["step_plate_pickup_agent1"],
                "step_actions_agent0": buf["step_actions_agent0"],
                "step_actions_agent1": buf["step_actions_agent1"],
                "terminated": buf["terminated"],
                "truncated": buf["truncated"],
            }
            all_episode_results.append(episode_result)

            print(
                f"  Trajectory {env_idx}: reward={episode_reward_0:.0f}, "
                f"delivery=({delivery_count_0}, {delivery_count_1}), "
                f"placement=({placement_count_0}, {placement_count_1}), "
                f"truncated={buf['truncated']}"
            )

        # replay.gif 用の Trajectory 0 の state 列を episode_result[0] に格納
        # （JAX オブジェクトのリストなので JSON シリアライズ不可、用途は GIF 生成のみ）
        if len(all_episode_results) > 0 and len(state_seq_traj0) > 0:
            all_episode_results[0]["state_seq"] = state_seq_traj0

        return all_episode_results


# ============================================================
# 結果の保存（demo フォルダ構造）
# ============================================================

def _write_log_txt(f, pair_name, layout_name, seed, episodes, model_0_path, model_1_path):
    """log.txt（集計）を書き出す"""
    actual_max_steps = max((len(ep["step_rewards_agent0"]) for ep in episodes), default=0)
    num_trajectories = len(episodes)

    f.write("=== Overcooked log Statistics ===\n")
    f.write(f"Map Layout: {layout_name}\n")
    f.write(f"Seed: {seed}\n")
    f.write(f"Trajectories: {num_trajectories}\n")
    f.write(f"Total Steps: {actual_max_steps}\n")

    rewards_0 = np.array([ep["episode_reward_agent0"] for ep in episodes])
    f.write(
        f"Total Score: {rewards_0.mean():.1f} +/- {rewards_0.std():.1f} "
        f"(min={rewards_0.min():.1f}, median={float(np.median(rewards_0)):.1f}, max={rewards_0.max():.1f})\n"
    )
    f.write("Average FPS (Tickrate): N/A (auto evaluation, no UI)\n")
    f.write("Average AI APS: N/A\n")
    f.write("Average AI Time: N/A\n")
    f.write("Average Step Time: N/A\n")
    f.write("\n")

    f.write("--- Pair ---\n")
    f.write(f"Pair: {pair_name}\n")
    f.write(f"Model 0: {model_0_path}\n")
    f.write(f"Model 1: {model_1_path}\n")
    f.write("\n")

    delivery_0 = np.array([ep["delivery_count_agent0"] for ep in episodes])
    delivery_1 = np.array([ep["delivery_count_agent1"] for ep in episodes])
    placement_0 = np.array([ep["placement_count_agent0"] for ep in episodes])
    placement_1 = np.array([ep["placement_count_agent1"] for ep in episodes])
    dish_pickup_0 = np.array([ep["dish_pickup_count_agent0"] for ep in episodes])
    dish_pickup_1 = np.array([ep["dish_pickup_count_agent1"] for ep in episodes])
    plate_pickup_0 = np.array([ep["plate_pickup_count_agent0"] for ep in episodes])
    plate_pickup_1 = np.array([ep["plate_pickup_count_agent1"] for ep in episodes])
    shaped_0 = np.array([ep["episode_shaped_reward_agent0"] for ep in episodes])
    shaped_1 = np.array([ep["episode_shaped_reward_agent1"] for ep in episodes])
    # dummy slot は全 episode の mean_value が NaN になるので、全 NaN 時の
    # 「Mean of empty slice」警告を抑制する（結果は nan で正しい）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_value_0 = float(np.nanmean([ep["mean_value_agent0"] for ep in episodes]))
        mean_value_1 = float(np.nanmean([ep["mean_value_agent1"] for ep in episodes]))

    f.write("--- Agent Stats (mean +/- std over trajectories) ---\n")
    f.write(
        f"Player 1 (AI BOT): "
        f"Deliveries = {delivery_0.mean():.1f} +/- {delivery_0.std():.1f}, "
        f"Placements = {placement_0.mean():.1f} +/- {placement_0.std():.1f}, "
        f"Dish Pickups = {dish_pickup_0.mean():.1f} +/- {dish_pickup_0.std():.1f}, "
        f"Plate Pickups = {plate_pickup_0.mean():.1f} +/- {plate_pickup_0.std():.1f}, "
        f"Shaped Reward = {shaped_0.mean():.1f} +/- {shaped_0.std():.1f}, "
        f"Mean Value = {mean_value_0:.2f}\n"
    )
    f.write(
        f"Player 2 (AI BOT): "
        f"Deliveries = {delivery_1.mean():.1f} +/- {delivery_1.std():.1f}, "
        f"Placements = {placement_1.mean():.1f} +/- {placement_1.std():.1f}, "
        f"Dish Pickups = {dish_pickup_1.mean():.1f} +/- {dish_pickup_1.std():.1f}, "
        f"Plate Pickups = {plate_pickup_1.mean():.1f} +/- {plate_pickup_1.std():.1f}, "
        f"Shaped Reward = {shaped_1.mean():.1f} +/- {shaped_1.std():.1f}, "
        f"Mean Value = {mean_value_1:.2f}\n"
    )
    f.write("\n")

    # Per-trajectory rewards（後で別ツールで分布を見たり外れ値検出する用の生データ）
    f.write(f"--- Per-trajectory rewards ({num_trajectories} trajectories) ---\n")
    reward_strs = [f"{r:.0f}" for r in rewards_0]
    # 16 個ずつ改行
    for i in range(0, len(reward_strs), 16):
        f.write(", ".join(reward_strs[i:i + 16]))
        if i + 16 < len(reward_strs):
            f.write(",\n")
        else:
            f.write("\n")
    f.write("\n")

    f.write("--- Files ---\n")
    f.write("config.yaml          : Training configuration used by the model\n")
    f.write("steps.csv            : Step-by-step records of all episodes (1 row = 1 (episode_id, step);\n")
    f.write("                       columns: episode_id, step, p1_action, p2_action,\n")
    f.write("                       p1_action_id, p2_action_id,\n")
    f.write("                       team_reward, reward_p1, reward_p2, p1_value, p2_value;\n")
    f.write("                       action mapping: 0=right, 1=down, 2=left, 3=up, 4=stay, 5=interact)\n")
    f.write("episodes.csv         : Per-episode summary (1 row = 1 episode_id; columns:\n")
    f.write("                       episode_id, num_steps, episode_team_reward,\n")
    f.write("                       episode_reward_p1/p2, episode_shaped_reward_p1/p2,\n")
    f.write("                       delivery/placement/dish_pickup/plate_pickup _count_p1/p2,\n")
    f.write("                       mean_value_p1/p2)\n")
    f.write("replay.gif           : Animation of episode_id=0\n")


def _write_steps_csv(f, episodes):
    """steps.csv（全 episode の step-by-step を long format で）を書き出す

    1 行 = 1 (episode_id, step)。episodes.csv とは episode_id で join 可能。
    p1_action / p2_action は文字列、p1_action_id / p2_action_id は整数（両方併記）。
    """
    writer = csv.writer(f)
    writer.writerow([
        "episode_id", "step",
        "p1_action", "p2_action",
        "p1_action_id", "p2_action_id",
        "team_reward", "reward_p1", "reward_p2",
        "p1_value", "p2_value",
    ])
    for ep in episodes:
        episode_id = ep["episode"]
        num_steps = ep["num_steps"]
        actions_0 = ep["step_actions_agent0"]
        actions_1 = ep["step_actions_agent1"]
        rewards_0 = ep["step_rewards_agent0"]
        rewards_1 = ep["step_rewards_agent1"]
        values_0 = ep["step_values_agent0"]
        values_1 = ep["step_values_agent1"]
        for step_idx in range(num_steps):
            a0_id = int(actions_0[step_idx])
            a1_id = int(actions_1[step_idx])
            a0_name = ACTION_NAMES.get(a0_id, f"unk({a0_id})")
            a1_name = ACTION_NAMES.get(a1_id, f"unk({a1_id})")
            team_reward = rewards_0[step_idx] + rewards_1[step_idx]
            writer.writerow([
                episode_id, step_idx,
                a0_name, a1_name,
                a0_id, a1_id,
                team_reward,
                rewards_0[step_idx], rewards_1[step_idx],
                values_0[step_idx], values_1[step_idx],
            ])


def _write_episodes_csv(f, episodes):
    """episodes.csv（1 行 = 1 episode の集計）を書き出す

    旧 trajectories.txt の各ブロックの先頭にあった集計値（episode reward / 各種 count /
    Mean Value）を 1 行 1 episode で保持する。episode_id で steps.csv と join 可能。
    """
    writer = csv.writer(f)
    writer.writerow([
        "episode_id", "num_steps",
        "episode_team_reward", "episode_reward_p1", "episode_reward_p2",
        "episode_shaped_reward_p1", "episode_shaped_reward_p2",
        "delivery_count_p1", "delivery_count_p2",
        "placement_count_p1", "placement_count_p2",
        "dish_pickup_count_p1", "dish_pickup_count_p2",
        "plate_pickup_count_p1", "plate_pickup_count_p2",
        "mean_value_p1", "mean_value_p2",
    ])
    for ep in episodes:
        episode_team_reward = ep["episode_reward_agent0"] + ep["episode_reward_agent1"]
        writer.writerow([
            ep["episode"],
            ep["num_steps"],
            episode_team_reward,
            ep["episode_reward_agent0"], ep["episode_reward_agent1"],
            ep["episode_shaped_reward_agent0"], ep["episode_shaped_reward_agent1"],
            ep["delivery_count_agent0"], ep["delivery_count_agent1"],
            ep["placement_count_agent0"], ep["placement_count_agent1"],
            ep["dish_pickup_count_agent0"], ep["dish_pickup_count_agent1"],
            ep["plate_pickup_count_agent0"], ep["plate_pickup_count_agent1"],
            ep["mean_value_agent0"], ep["mean_value_agent1"],
        ])


def _write_replay_gif(episode_traj0, layout_name, agent_view_size, output_path):
    """Trajectory 0 の state_seq から replay.gif を生成する"""
    state_seq = episode_traj0.get("state_seq")
    if not state_seq:
        print(f"[WARN] state_seq is empty for Trajectory 0, replay.gif will be skipped.")
        return

    # state_seq は List[State]（各要素は env_idx=0 のスライス済み）
    # OvercookedV2Visualizer.animate は List[State] を tree_stack して受け取る形式に合わせる
    stacked_state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *state_seq)
    viz = OvercookedV2Visualizer()
    viz.animate(stacked_state, agent_view_size=agent_view_size, filename=output_path)


def _write_metadata_json(path, eval_name, layout_name, seed, meta):
    """metadata.json（評価の setup を機械可読で記録）を書き出す

    CSV には「どんな設定で評価したか」（どの slot が solo/coop/dummy か、流用か）が
    残らないので、ここに事実だけを残す。WHY（slot 可搬性の根拠など）は md に書く。
    """
    data = {
        "eval_name": eval_name,
        "layout": layout_name,
        "seed": seed,
        "agent_0": meta["agent_0"],
        "agent_1": meta["agent_1"],
        "obs_channel_order": "egocentric_self_then_other",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _bundled_config_for(ev):
    """evaluation の最初の非 dummy slot の checkpoint 同梱 config.yaml パスを返す（無ければ None）"""
    for slot_key in ("agent_0", "agent_1"):
        spec = ev[slot_key]
        if spec["policy_type"] != "dummy" and spec.get("checkpoint"):
            bundled = os.path.join(os.path.dirname(spec["checkpoint"]), "config.yaml")
            if os.path.exists(bundled):
                return bundled
    return None


def save_results(
    all_results,
    eval_meta,
    evaluations,
    output_dir_base,
    layout_name,
    seed,
    agent_view_size,
    fallback_config_path,
):
    """各 evaluation ごとに demo フォルダを作って
    log.txt / config.yaml / steps.csv / episodes.csv / replay.gif / metadata.json を保存する

    出力形式:
      experiments/evaluation/results/
        <評価名>_<layout>_<timestamp>/
          ├── log.txt              (集計)
          ├── config.yaml          (最初の非 dummy slot の同梱 config、無ければ fallback)
          ├── steps.csv            (全 episode の step-by-step、1 行 = 1 (episode_id, step))
          ├── episodes.csv         (per-episode 集計、1 行 = 1 episode_id)
          ├── replay.gif           (episode_id=0 のアニメーション)
          └── metadata.json        (policy_type / checkpoint / trained_slot / used_as_slot)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ev in evaluations:
        name = ev["name"]
        episodes = all_results[name]
        meta = eval_meta[name]

        folder_name = f"{name}_{layout_name}_{timestamp}"
        output_dir = os.path.join(output_dir_base, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # log.txt（model path は checkpoint。dummy は "dummy"）
        model_0 = meta["agent_0"]["checkpoint"] or "dummy"
        model_1 = meta["agent_1"]["checkpoint"] or "dummy"
        log_path = os.path.join(output_dir, "log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            _write_log_txt(
                f,
                pair_name=name,
                layout_name=layout_name,
                seed=seed,
                episodes=episodes,
                model_0_path=model_0,
                model_1_path=model_1,
            )

        # steps.csv（全 episode の step-by-step、long format。スキーマ不変）
        steps_path = os.path.join(output_dir, "steps.csv")
        with open(steps_path, "w", encoding="utf-8", newline="") as f:
            _write_steps_csv(f, episodes)

        # episodes.csv（per-episode 集計、1 行 1 episode。スキーマ不変）
        episodes_path = os.path.join(output_dir, "episodes.csv")
        with open(episodes_path, "w", encoding="utf-8", newline="") as f:
            _write_episodes_csv(f, episodes)

        # config.yaml をコピー（同梱 config を優先、無ければ fallback）
        config_src = _bundled_config_for(ev) or fallback_config_path
        if config_src and os.path.exists(config_src):
            shutil.copy(config_src, os.path.join(output_dir, "config.yaml"))

        # metadata.json（評価 setup の事実記録）
        metadata_path = os.path.join(output_dir, "metadata.json")
        _write_metadata_json(metadata_path, name, layout_name, seed, meta)

        # replay.gif（Trajectory 0 のアニメーション）
        replay_path = os.path.join(output_dir, "replay.gif")
        if len(episodes) > 0:
            _write_replay_gif(
                episode_traj0=episodes[0],
                layout_name=layout_name,
                agent_view_size=agent_view_size,
                output_path=replay_path,
            )

        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", type=str, help="Overcooked layout", default="cramped_room"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for evaluation rollouts (must differ from training SEED in yaml to avoid evaluating on the same env initialization seen during training)",
        default=42,
    )
    parser.add_argument(
        "--agent_view_size",
        type=int,
        help="Number of cells in the agent view. If not provided, the agent will see the whole grid.",
    )
    parser.add_argument(
        "--no_jit",
        default=False,
        help="Disable JIT compilation",
        action="store_true",
    )
    parser.add_argument(
        "--debug", default=False, help="Debug mode", action="store_true"
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default=os.path.join(_REPO_ROOT, "experiments", "evaluation", "config.yaml"),
        help="per-slot eval yaml (A2): `evaluations` list, each with agent_0/agent_1 = {policy_type: solo|coop|dummy, checkpoint}.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=os.path.join(_REPO_ROOT, "experiments", "shared", "config.yaml"),
        help="yaml from which ENV_KWARGS / NUM_ENVS / network sizes are read (any of solo/shared/independent works as long as network and env settings match the weights being evaluated).",
    )
    args = parser.parse_args()

    if len(args.layout) == 0:
        raise ValueError("You must provide a layout.")
    layout = get_layout(args.layout)

    # fallback train_config: 同梱 config が見つからない時のみ使う
    with open(args.train_config, "r", encoding="utf-8") as f:
        fallback_config = yaml.safe_load(f)

    # eval_config: per-slot 評価リストを読み込み
    evaluations = load_eval_config(args.eval_config)

    # 出力ディレクトリ（experiments/evaluation/results/ に集約）
    output_dir_base = os.path.join(_REPO_ROOT, "experiments", "evaluation", "results")
    os.makedirs(output_dir_base, exist_ok=True)

    # 全 evaluation を回す
    all_results = {}
    eval_meta = {}
    for ev in evaluations:
        name = ev["name"]
        print(f"\nEvaluating: {name}")

        # env / network 構築の基準 config（最初の非 dummy slot の同梱 config、無ければ fallback）
        config = resolve_eval_config(ev, fallback_config)

        interactive = InteractiveOvercookedV2(
            layout=layout,
            config=config,
            slot0_spec=ev["agent_0"],
            slot1_spec=ev["agent_1"],
            agent_view_size=args.agent_view_size,
            no_jit=args.no_jit,
            debug=args.debug,
        )

        key = jax.random.PRNGKey(args.seed)
        episodes = interactive.run_evaluation(key)
        all_results[name] = episodes
        eval_meta[name] = {
            "agent_0": {
                "policy_type": interactive.policy_0["policy_type"],
                "checkpoint": interactive.policy_0["checkpoint"],
                "trained_slot": interactive.policy_0["trained_slot"],
                "used_as_slot": "agent_0",
            },
            "agent_1": {
                "policy_type": interactive.policy_1["policy_type"],
                "checkpoint": interactive.policy_1["checkpoint"],
                "trained_slot": interactive.policy_1["trained_slot"],
                "used_as_slot": "agent_1",
            },
        }

    # 結果保存（demo フォルダ構造で出力）
    save_results(
        all_results=all_results,
        eval_meta=eval_meta,
        evaluations=evaluations,
        output_dir_base=output_dir_base,
        layout_name=args.layout,
        seed=args.seed,
        agent_view_size=args.agent_view_size,
        fallback_config_path=args.train_config,
    )
