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
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

from overcooked_research.env import OvercookedV2WithEvents, get_layout
from overcooked_research.train import ScannedRNN, ActorCriticRNN, Transition, batchify, unbatchify
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import yaml
import functools


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

        # INIT NETWORK — Independent Weight: 2つのネットワークインスタンス（モジュール）を作る
        # この時点では重みはまだ無い（設計図だけ）。実際の乱数初期化は下の network_*.init(_rng_*, ...) で行う
        network_0 = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        network_1 = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config)

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

                # SELECT ACTION — agent ごとに独立して処理
                rng, _rng_0, _rng_1 = jax.random.split(rng, 3)

                obs_0 = last_obs["agent_0"]  # (256, 5, 5, 26)
                obs_1 = last_obs["agent_1"]  # (256, 5, 5, 26)

                ac_in_0 = (obs_0[np.newaxis, :], last_done[np.newaxis, :])
                ac_in_1 = (obs_1[np.newaxis, :], last_done[np.newaxis, :])

                hstate_0, pi_0, value_0 = network_0.apply(train_state_0.params, hstate_0, ac_in_0)
                hstate_1, pi_1, value_1 = network_1.apply(train_state_1.params, hstate_1, ac_in_1)

                action_0 = pi_0.sample(seed=_rng_0)
                action_1 = pi_1.sample(seed=_rng_1)
                log_prob_0 = pi_0.log_prob(action_0)
                log_prob_1 = pi_1.log_prob(action_1)

                env_act = {"agent_0": action_0.squeeze().flatten(), "agent_1": action_1.squeeze().flatten()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

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

                # info を agent ごとに分離
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

            # agent_1 の更新
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

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2", "IndependentWeight"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_rnn_overcooked_v2_independent_{layout_name}",
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    # === 重みと config を seed × agent ごとに同梱保存 ===
    # experiments/independent/results/<ts>/seed_<index>/agent{0,1}/{params.msgpack, config.yaml}
    from flax.serialization import to_bytes

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, "results", f"{layout_name}_{ts}")

    # runner_state: (ts_0, ts_1, env_state, obs, done, update_step, hs_0, hs_1, rng)
    # vmap で NUM_SEEDS 分あるので seed ごとに分けて保存
    runner_state = out["runner_state"]
    for seed_idx in range(num_seeds):
        params_0 = jax.tree_util.tree_map(lambda x: x[seed_idx], runner_state[0].params)
        params_1 = jax.tree_util.tree_map(lambda x: x[seed_idx], runner_state[1].params)

        agent0_dir = os.path.join(run_dir, f"seed_{seed_idx}", "agent0")
        agent1_dir = os.path.join(run_dir, f"seed_{seed_idx}", "agent1")
        os.makedirs(agent0_dir, exist_ok=True)
        os.makedirs(agent1_dir, exist_ok=True)

        for agent_dir, params in [(agent0_dir, params_0), (agent1_dir, params_1)]:
            params_path = os.path.join(agent_dir, "params.msgpack")
            with open(params_path, "wb") as f:
                f.write(to_bytes(params))
            config_path = os.path.join(agent_dir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[Independent] Saved params + config for {num_seeds} seed(s) -> {run_dir}")


if __name__ == "__main__":
    main()
