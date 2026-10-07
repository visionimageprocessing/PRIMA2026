"""Actor-Critic ネットワーク定義（CNN + GRU）

3 つの学習スクリプト（solo / shared / independent）で共有する
"""

from typing import Callable, Sequence, Any, Dict
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np
import distrax


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticRNNSolo(ActorCriticRNN):
    """Solo 学習用 Actor-Critic（ActorCriticRNN の派生クラス）

    forward の冒頭で観測の「相手 agent (other_agent_layers) チャンネル」をマスクしてから
    親クラスの forward を呼ぶ。マスクはこのネットワーククラスの内部で完結させる
    （学習スクリプト側ではマスクしない。田中先生 2026/05/27 指示「派生クラスを作って
    ネットワークの中でマスクする」に従う）。

    OvercookedV2 の DEFAULT 観測は channel 軸で
        [agent_layer, other_agent_layers, static_layers, ...]
    の順に並ぶ。agent_layer の幅は `7 + num_ingredients`
    （= pos(1) + dir(4) + inv(2 + num_ingredients)）。other_agent_layers は同じ幅で
    その直後に続くので、マスク対象は channel [other_start, other_end)。

    マスク値:
        pos     チャンネル → -1（「相手はどのマスにも居ない」を明示。0 だと「相手が
                            居ない通常観測」と区別が付きにくいため。float32 にして
                            本当に -1 を入れる。uint8 のままだと -1 が 255 に
                            オーバーフローしてしまう）
        dir/inv チャンネル → 0（不在の agent は向き・所持品が無いので 0 が自然）

    float32 への cast はこの forward 内に閉じており、`Transition.obs` は uint8 のまま
    保存される（毎回 forward で cast し直す）。基底 ActorCriticRNN は元々 uint8 観測を
    conv で float32 に昇格して計算しているため、cast 自体は重みに影響しない（昇格を
    明示するだけ）。実質変わるのはマスク値だけ。

    num_ingredients は default を持たせず必須にする。layout ごとに異なり
    （例: cramped_room は 1）、間違えるとマスク範囲がずれるため、呼び出し側で
    `env.layout.num_ingredients` を明示的に渡させる。
    """

    num_ingredients: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        agent_layer_size = 7 + self.num_ingredients
        other_start = agent_layer_size
        other_end = 2 * agent_layer_size

        obs = obs.astype(jnp.float32)
        obs = obs.at[..., other_start].set(-1.0)               # pos → -1
        obs = obs.at[..., other_start + 1:other_end].set(0.0)  # dir / inv → 0

        return super().__call__(hidden, (obs, dones))


class ActorCritic(nn.Module):
    """非 RNN 版"""
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
