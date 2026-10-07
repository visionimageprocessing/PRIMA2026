"""学習スクリプト共通の小道具"""

import jax.numpy as jnp


def batchify(x: dict, agent_list, num_actors):
    """dict[agent_name -> array] を 1 つの (num_actors, ...) array に束ねる"""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """batchify の逆操作。array を agent ごとの dict に戻す"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
