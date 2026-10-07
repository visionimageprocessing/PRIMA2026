"""Solo 学習（一人学習 / pseudo-solo control）

2-agent Overcooked-V2 環境を維持したまま、agent_0 が「協力相手を観測・予測・回避する
必要がない」条件で学習する対照群を作る。完全な 1-agent 環境ではない（評価で協力相手と
組み合わせるため、観測形状・ネットワーク構造を 2-agent と揃える）。

SOLO_ACTIVE_AGENT（config, 0 か 1）で「どちらの slot を Solo 学習者（active）にするか」を
切り替える。もう一方が dummy になる。active への影響経路（観測・行動・物理干渉）を断つ 3 点:
  - 観測マスク: active の network を ActorCriticRNNSolo（派生クラス）にして、forward 内で
    相手 agent チャンネルをマスクする（田中先生 2026/05/27 指示）。観測は egocentric
    （self→other 順）なので、active が agent_0 でも agent_1 でも同じクラスで正しくマスクされる。
  - 行動固定:   dummy が env に渡す action を STAY に固定する。
  - 物理無害化: dummy を標準 cramped_room の (0,0) 壁マスに毎 step pin する（学習時のみ。
    評価時は相手が本物の協力相手なので pin しない）。

dummy 側の network / transition / GAE / update は構造維持のため残すが（Independent と同じ
学習ループを使い回す）、保存・評価対象は active のみ。SOLO_ACTIVE_AGENT=1 で回すと
solo_agent1（agent_1 active / agent_0 dummy）が得られる。

設計の詳細: experiments/solo/README.md / 文字起こしなどデータ/50_Solo_Learning実装.md
ベース: experiments/independent/train.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.common import Actions

from overcooked_research.env import OvercookedV2WithEvents, get_layout
from overcooked_research.train import (
    ScannedRNN,
    ActorCriticRNN,
    ActorCriticRNNSolo,
    Transition,
    batchify,
    unbatchify,
)
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import yaml
import functools


# dummy を pin する壁マス座標（標準 cramped_room の (0,0) = 左上の壁角）
DUMMY_FIXED_X = 0
DUMMY_FIXED_Y = 0


def _pin_agent(env_state, agent_idx, x=DUMMY_FIXED_X, y=DUMMY_FIXED_Y):
    """OvercookedV2LogWrapper でラップされた env_state の agent_idx 位置を (x, y) に上書きする。

    Solo 学習では dummy 側の agent（agent_idx）を STAY 固定にするが、env の reset で位置が
    ランダム化されるため、毎 step この関数で (0,0) の壁マスに pin して active への物理干渉を
    防ぐ。内側の OvercookedV2 state は env_state.env_state にある。pos は (NUM_ENVS, num_agents)。
    """
    inner = env_state.env_state
    new_inner = inner.replace(
        agents=inner.agents.replace(
            pos=inner.agents.pos.replace(
                x=inner.agents.pos.x.at[:, agent_idx].set(x),
                y=inner.agents.pos.y.at[:, agent_idx].set(y),
            )
        )
    )
    return env_state.replace(env_state=new_inner)


def _obs_from_pinned_state(env, env_state):
    """pin 済み state から各 agent の観測を再生成する。

    OvercookedV2LogWrapper の内側 env (env._env) の get_obs を、内側 state
    (env_state.env_state) に対して呼ぶ。_pin_agent_1 の直後に使うことで、state と
    observation を常に一致させる。wrapper 構造に依存する箇所をここ 1 つに隔離する。
    """
    return jax.vmap(env._env.get_obs)(env_state.env_state)


def make_train(config):
    env_kwargs = {**config["ENV_KWARGS"], "layout": get_layout(config["ENV_KWARGS"]["layout"])}
    env = OvercookedV2WithEvents(**env_kwargs)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Solo マスク用: layout の ingredient 数（other_agent チャンネル境界の計算に使う）
    num_ingredients = int(env.layout.num_ingredients)

    # Solo 学習者（active）と dummy の slot を config で切り替える（default 0 = agent_0 active）
    solo_active = int(config.get("SOLO_ACTIVE_AGENT", 0))
    if solo_active not in (0, 1):
        raise ValueError(f"SOLO_ACTIVE_AGENT must be 0 or 1, got {solo_active}")
    solo_dummy = 1 - solo_active

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):

        # INIT NETWORK
        # agent_0: Solo（forward 内で相手チャンネルをマスクする派生クラス）
        # agent_1: 通常の ActorCriticRNN（無駄でも構造維持。行動は STAY 上書き・保存しない）
        # active slot は Solo（マスク派生クラス）、dummy slot は素の ActorCriticRNN。
        # ActorCriticRNNSolo は egocentric obs の「相手ブロック」を消すので、active が
        # agent_0 でも agent_1 でも同じクラスで正しく動く。
        if solo_active == 0:
            network_0 = ActorCriticRNNSolo(
                env.action_space(env.agents[0]).n,
                config=config,
                num_ingredients=num_ingredients,
            )
            network_1 = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config)
        else:
            network_0 = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
            network_1 = ActorCriticRNNSolo(
                env.action_space(env.agents[1]).n,
                config=config,
                num_ingredients=num_ingredients,
            )

        rng, _rng_0, _rng_1 = jax.random.split(rng, 3)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        network_params_0 = network_0.init(_rng_0, init_hstate, init_x)
        network_params_1 = network_1.init(_rng_1, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx_0 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
            tx_1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx_0 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state_0 = TrainState.create(
            apply_fn=network_0.apply,
            params=network_params_0,
            tx=tx_0,
        )
        train_state_1 = TrainState.create(
            apply_fn=network_1.apply,
            params=network_params_1,
            tx=tx_1,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        env_state = _pin_agent(env_state, solo_dummy)  # dummy を (0,0) 壁に固定
        obsv = _obs_from_pinned_state(env, env_state)  # pin 済み state から obs 再生成
        init_hstate_0 = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        init_hstate_1 = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state_0, train_state_1,
                    env_state, last_obs, last_done, update_step,
                    hstate_0, hstate_1, rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng_0 = jax.random.split(rng, 2)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                ac_in_0 = (obs_0[np.newaxis, :], last_done[np.newaxis, :])
                ac_in_1 = (obs_1[np.newaxis, :], last_done[np.newaxis, :])

                hstate_0, pi_0, value_0 = network_0.apply(train_state_0.params, hstate_0, ac_in_0)
                hstate_1, pi_1, value_1 = network_1.apply(train_state_1.params, hstate_1, ac_in_1)

                # active は policy から sample、dummy は STAY 固定。env に渡す action も
                # transition に記録する action も dummy 側は STAY で統一する（network が sample
                # した action を記録すると「STAY 固定なのに別 action を学習している」ように
                # 見えるため、実際に実行した STAY で揃える）。
                if solo_active == 0:
                    action_0 = pi_0.sample(seed=_rng_0)
                    action_1 = jnp.full_like(action_0, int(Actions.stay))
                else:
                    action_1 = pi_1.sample(seed=_rng_0)
                    action_0 = jnp.full_like(action_1, int(Actions.stay))
                log_prob_0 = pi_0.log_prob(action_0)
                log_prob_1 = pi_1.log_prob(action_1)

                env_act = {
                    "agent_0": action_0.squeeze().flatten(),
                    "agent_1": action_1.squeeze().flatten(),
                }

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                env_state = _pin_agent(env_state, solo_dummy)  # reset 後もずれないよう毎 step pin
                # pin 済み state から obs を再生成し、state と obs を常に一致させる。
                # （active は ActorCriticRNNSolo 内で相手 ch を mask するので研究信号には
                #   元々影響しないが、不整合を残さずレビュー・修論防御を簡潔にするため。）
                obsv = _obs_from_pinned_state(env, env_state)

                # shaped_reward 加算前に元報酬を保存
                original_reward_0 = reward["agent_0"]
                original_reward_1 = reward["agent_1"]

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                info_0 = {
                    "shaped_reward": info["shaped_reward"]["agent_0"],
                    "original_reward": original_reward_0,
                    "anneal_factor": jnp.full((config["NUM_ENVS"],), anneal_factor),
                    "combined_reward": reward["agent_0"],
                }
                info_1 = {
                    "shaped_reward": info["shaped_reward"]["agent_1"],
                    "original_reward": original_reward_1,
                    "anneal_factor": jnp.full((config["NUM_ENVS"],), anneal_factor),
                    "combined_reward": reward["agent_1"],
                }
                # Solo 検証用: dummy 側が学習中ずっと無害（event を出していない）ことを wandb で
                # 監視する。STAY 固定 + (0,0) pin が効いていれば全て ~0 になる。
                dummy_events = {
                    "event_delivery": info["delivery"][:, solo_dummy],
                    "event_placement": info["placement"][:, solo_dummy],
                    "event_dish_pickup": info["dish_pickup"][:, solo_dummy],
                    "event_plate_pickup": info["plate_pickup"][:, solo_dummy],
                }
                if solo_dummy == 0:
                    info_0.update(dummy_events)
                else:
                    info_1.update(dummy_events)

                transition_0 = Transition(
                    done["__all__"],
                    action_0.squeeze(),
                    value_0.squeeze(),
                    reward["agent_0"],
                    log_prob_0.squeeze(),
                    obs_0,
                    info_0,
                )
                transition_1 = Transition(
                    done["__all__"],
                    action_1.squeeze(),
                    value_1.squeeze(),
                    reward["agent_1"],
                    log_prob_1.squeeze(),
                    obs_1,
                    info_1,
                )

                runner_state = (
                    train_state_0, train_state_1,
                    env_state, obsv,
                    done["__all__"],
                    update_step,
                    hstate_0, hstate_1,
                    rng,
                )
                return runner_state, (transition_0, transition_1)

            initial_hstate_0 = runner_state[-3]  # hstate_0
            initial_hstate_1 = runner_state[-2]  # hstate_1
            runner_state, (traj_batch_0, traj_batch_1) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE — agent ごとに独立
            (
                train_state_0, train_state_1,
                env_state, last_obs, last_done, update_step,
                hstate_0, hstate_1, rng,
            ) = runner_state

            # agent_0
            last_obs_0 = last_obs["agent_0"]
            ac_in_0 = (last_obs_0[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_0 = network_0.apply(train_state_0.params, hstate_0, ac_in_0)
            last_val_0 = last_val_0.squeeze()

            # agent_1
            last_obs_1 = last_obs["agent_1"]
            ac_in_1 = (last_obs_1[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_1 = network_1.apply(train_state_1.params, hstate_1, ac_in_1)
            last_val_1 = last_val_1.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages_0, targets_0 = _calculate_gae(traj_batch_0, last_val_0)
            advantages_1, targets_1 = _calculate_gae(traj_batch_1, last_val_1)

            # UPDATE NETWORK — agent ごとに独立して更新
            def _make_update_epoch(network):
                """agent ごとの _update_epoch を生成するファクトリ"""
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                            # RERUN NETWORK
                            _, pi, value = network.apply(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )

                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, init_hstate, traj_batch, advantages, targets, rng = (
                        update_state
                    )
                    rng, _rng = jax.random.split(rng)

                    init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ENVS"], -1))
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss
                return _update_epoch

            _update_epoch_0 = _make_update_epoch(network_0)
            _update_epoch_1 = _make_update_epoch(network_1)

            # 乱数を agent ごとに分ける
            rng, rng_update_0, rng_update_1 = jax.random.split(rng, 3)

            # agent_0 の更新
            update_state_0 = (
                train_state_0, initial_hstate_0,
                traj_batch_0, advantages_0, targets_0, rng_update_0,
            )
            update_state_0, loss_info_0 = jax.lax.scan(
                _update_epoch_0, update_state_0, None, config["UPDATE_EPOCHS"]
            )
            train_state_0 = update_state_0[0]

            # agent_1 の更新（構造維持。保存・評価対象ではない）
            update_state_1 = (
                train_state_1, initial_hstate_1,
                traj_batch_1, advantages_1, targets_1, rng_update_1,
            )
            update_state_1, loss_info_1 = jax.lax.scan(
                _update_epoch_1, update_state_1, None, config["UPDATE_EPOCHS"]
            )
            train_state_1 = update_state_1[0]

            # wandb ログ — agent ごとにメトリクスを分けて記録
            metric_0 = traj_batch_0.info
            metric_1 = traj_batch_1.info

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric_0 = jax.tree_util.tree_map(lambda x: x.mean(), metric_0)
            metric_1 = jax.tree_util.tree_map(lambda x: x.mean(), metric_1)
            metric = {}
            for k, v in metric_0.items():
                metric[f"agent_0/{k}"] = v
            for k, v in metric_1.items():
                metric[f"agent_1/{k}"] = v
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state_0, train_state_1,
                env_state, last_obs, last_done,
                update_step,
                hstate_0, hstate_1,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_0, train_state_1,
            env_state, obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            0,
            init_hstate_0, init_hstate_1,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path=".", config_name="config"
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    solo_active = int(config.get("SOLO_ACTIVE_AGENT", 0))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2", "Solo"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_rnn_overcooked_v2_solo_{layout_name}_active{solo_active}",
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    # === 重みと config を seed ごとに保存（Solo は active slot のみ） ===
    # experiments/solo/results/<ts>/seed_<index>/agent<active>/{params.msgpack, config.yaml}
    from flax.serialization import to_bytes

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, "results", f"{layout_name}_{ts}")

    # runner_state: (ts_0, ts_1, env_state, obs, done, update_step, hs_0, hs_1, rng)
    # active slot の train_state は runner_state[solo_active]（0→ts_0, 1→ts_1）
    runner_state = out["runner_state"]
    for seed_idx in range(num_seeds):
        params_active = jax.tree_util.tree_map(
            lambda x: x[seed_idx], runner_state[solo_active].params
        )

        agent_dir = os.path.join(run_dir, f"seed_{seed_idx}", f"agent{solo_active}")
        os.makedirs(agent_dir, exist_ok=True)

        params_path = os.path.join(agent_dir, "params.msgpack")
        with open(params_path, "wb") as f:
            f.write(to_bytes(params_active))
        config_path = os.path.join(agent_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[Solo] Saved agent_{solo_active} (active) params + config for {num_seeds} seed(s) -> {run_dir}")


if __name__ == "__main__":
    main()
