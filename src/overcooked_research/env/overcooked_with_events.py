"""
OvercookedV2WithEvents

jaxmarl.environments.overcooked_v2.overcooked.OvercookedV2 を継承し、
step_env / step_agents / process_interact の 3 メソッドのみ override
して、event_log（delivery / placement / dish_pickup / plate_pickup）を
info dict に追加で流す拡張クラス
"""

from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State
from jaxmarl.environments.overcooked_v2.common import (
    ACTION_TO_DIRECTION,
    Actions,
    StaticObject,
    DynamicObject,
    Agent,
)
from jaxmarl.environments.overcooked_v2.settings import (
    DELIVERY_REWARD,
    INDICATOR_ACTIVATION_COST,
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
    SHAPED_REWARDS,
)
from jaxmarl.environments.overcooked_v2.utils import tree_select


class OvercookedV2WithEvents(OvercookedV2):
    """OvercookedV2 + event_log（delivery / placement / dish_pickup / plate_pickup）

    Override する 3 メソッド:
    - step_env: info dict に event_log を追加
    - step_agents: scan の output に event signal を含める
    - process_interact: return tuple に pot_placement / dish_pickup / plate_pickup 追加
    """

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Override: info dict に event_log を追加"""

        acts = self.action_set.take(
            indices=jnp.array([actions[f"agent_{i}"] for i in range(self.num_agents)])
        )
        state, reward, shaped_rewards, deliveries, placements, dish_pickups, plate_pickups = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        rewards = {f"agent_{i}": reward for i in range(self.num_agents)}
        shaped_rewards = {
            f"agent_{i}": shaped_reward
            for i, shaped_reward in enumerate(shaped_rewards)
        }

        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__"] = done
        event_log = {
            "delivery": deliveries,
            "placement": placements,
            "dish_pickup": dish_pickups,
            "plate_pickup": plate_pickups,
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards, **event_log},
        )

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ):
        """Override: scan の output に event signal 4 つを追加して返す

        オリジナルは return (state, reward, shaped_rewards) だが、ここでは
        return (state, reward, shaped_rewards, deliveries, placements,
                dish_pickups, plate_pickups) の 7-tuple を返す
        """
        grid = state.grid

        # Move action: オリジナルと同じ
        def _move_wrapper(agent, action):
            direction = ACTION_TO_DIRECTION[action]

            def _move(agent, dir):
                pos = agent.pos
                new_pos = pos.move_in_bounds(dir, self.width, self.height)

                new_pos = tree_select(
                    grid[new_pos.y, new_pos.x, 0] == StaticObject.EMPTY, new_pos, pos
                )

                return agent.replace(pos=new_pos, dir=direction)

            return jax.lax.cond(
                direction != -1,
                _move,
                lambda a, _: a,
                agent,
                direction,
            )

        new_agents = jax.vmap(_move_wrapper)(state.agents, actions)

        # Resolve collisions: オリジナルと同じ
        def _masked_positions(mask):
            return tree_select(mask, state.agents.pos, new_agents.pos)

        def _get_collisions(mask):
            positions = _masked_positions(mask)

            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos.y, pos.x].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1

            collisions = jax.vmap(lambda p: collision_mask[p.y, p.x])(positions)
            return collisions

        initial_mask = jnp.zeros((self.num_agents,), dtype=bool)
        mask = jax.lax.while_loop(
            lambda mask: jnp.any(_get_collisions(mask)),
            lambda mask: mask | _get_collisions(mask),
            initial_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(mask))

        # Prevent swapping: オリジナルと同じ
        def _compute_swapped_agents(original_positions, new_positions):
            original_positions = original_positions.to_array()
            new_positions = new_positions.to_array()

            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swapped_agents = jnp.any(swap_pairs, axis=0)
            return swapped_agents

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact

            def _interact(carry, agent):
                grid, correct_delivery, reward = carry

                (
                    new_grid,
                    new_agent,
                    new_correct_delivery,
                    interact_reward,
                    shaped_reward,
                    pot_placement,
                    dish_pickup,
                    plate_pickup,
                ) = self.process_interact(
                    grid, agent, new_agents.inventory, state.recipe
                )

                carry = (
                    new_grid,
                    correct_delivery | new_correct_delivery,
                    reward + interact_reward,
                )
                return carry, (new_agent, shaped_reward, new_correct_delivery, pot_placement, dish_pickup, plate_pickup)

            return jax.lax.cond(
                is_interact, _interact, lambda c, a: (c, (a, 0.0, False, False, False, False)), carry, agent
            )

        carry = (grid, False, 0.0)
        xs = (new_agents, actions)
        (new_grid, new_correct_delivery, reward), (new_agents, shaped_rewards, deliveries, placements, dish_pickups, plate_pickups) = (
            jax.lax.scan(_interact_wrapper, carry, xs)
        )

        # Update extra info: オリジナルと同じ
        def _timestep_wrapper(cell):
            def _cook(cell):
                is_cooking = cell[2] > 0
                new_extra = jax.lax.select(is_cooking, cell[2] - 1, cell[2])
                finished_cooking = is_cooking * (new_extra == 0)
                new_ingredients = cell[1] | (finished_cooking * DynamicObject.COOKED)

                return jnp.array([cell[0], new_ingredients, new_extra])

            def _indicator(cell):
                new_extra = jnp.clip(cell[2] - 1, min=0)
                return cell.at[2].set(new_extra)

            branches = (
                jnp.array(
                    [
                        StaticObject.POT,
                        StaticObject.BUTTON_RECIPE_INDICATOR,
                    ]
                )
                == cell[0]
            )

            branch_idx = jax.lax.select(
                jnp.any(branches),
                jnp.argmax(branches) + 1,
                0,
            )

            return jax.lax.switch(
                branch_idx,
                [
                    lambda x: x,
                    _cook,
                    _indicator,
                ],
                cell,
            )

        new_grid = jax.vmap(jax.vmap(_timestep_wrapper))(new_grid)

        sample_new_recipe = new_correct_delivery & self.sample_recipe_on_delivery

        key, subkey = jax.random.split(key)
        new_recipe = jax.lax.cond(
            sample_new_recipe,
            lambda _, key: self._sample_recipe(key),
            lambda r, _: r,
            state.recipe,
            subkey,
        )
        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
                recipe=new_recipe,
                new_correct_delivery=new_correct_delivery,
            ),
            reward,
            shaped_rewards,
            deliveries,
            placements,
            dish_pickups,
            plate_pickups,
        )

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
        all_inventories: jnp.ndarray,
        recipe: int,
    ):
        """Override: return tuple に pot_placement / dish_pickup / plate_pickup を追加"""

        inventory = agent.inventory
        fwd_pos = agent.get_fwd_pos()

        shaped_reward = jnp.array(0, dtype=float)

        interact_cell = grid[fwd_pos.y, fwd_pos.x]

        interact_item = interact_cell[0]
        interact_ingredients = interact_cell[1]
        interact_extra = interact_cell[2]
        plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        object_is_plate_pile = interact_item == StaticObject.PLATE_PILE
        object_is_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)

        object_is_pile = object_is_plate_pile | object_is_ingredient_pile
        object_is_pot = interact_item == StaticObject.POT
        object_is_goal = interact_item == StaticObject.GOAL
        object_is_wall = interact_item == StaticObject.WALL
        onject_is_button_recipe_indicator = (
            interact_item == StaticObject.BUTTON_RECIPE_INDICATOR
        )

        object_has_no_ingredients = interact_ingredients == 0

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        merged_ingredients = interact_ingredients + inventory

        pot_is_cooking = object_is_pot * (interact_extra > 0)
        pot_is_cooked = object_is_pot * (
            interact_ingredients & DynamicObject.COOKED != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked

        successful_dish_pickup = pot_is_cooked * inventory_is_plate
        is_dish_pickup_useful = merged_ingredients == plated_recipe
        shaped_reward += (
            successful_dish_pickup
            * is_dish_pickup_useful
            * SHAPED_REWARDS["DISH_PICKUP"]
        )

        successful_pickup = (
            object_is_pile * inventory_is_empty
            + successful_dish_pickup
            + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
        )

        successful_indicator_activation = (
            onject_is_button_recipe_indicator
            * inventory_is_empty
            * object_has_no_ingredients
        )

        pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3

        successful_pot_placement = pot_is_idle * inventory_is_ingredient * ~pot_full
        ingredient_selector = inventory | (inventory << 1)
        is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
            recipe & ingredient_selector
        )
        shaped_reward += (
            successful_pot_placement
            * is_pot_placement_useful
            * jax.lax.select(
                is_pot_placement_useful,
                1,
                -1 if self.negative_rewards else 0,
            )
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

        successful_drop = (
            object_is_wall * object_has_no_ingredients * ~inventory_is_empty
            + successful_pot_placement
        )
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        pile_ingredient = (
            object_is_plate_pile * DynamicObject.PLATE
            + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
        )

        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )
        pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3

        successful_pot_start_cooking = (
            pot_is_idle
            * ~object_has_no_ingredients
            * inventory_is_empty
            * self.start_cooking_interaction
        )
        is_pot_start_cooking_useful = interact_ingredients == recipe
        shaped_reward += (
            successful_pot_start_cooking
            * is_pot_start_cooking_useful
            * SHAPED_REWARDS["POT_START_COOKING"]
        )
        auto_cook = pot_is_idle & pot_full_after_drop & ~self.start_cooking_interaction

        use_pot_extra = successful_pot_start_cooking | auto_cook
        new_extra = (
            use_pot_extra * POT_COOK_TIME
            + successful_indicator_activation * INDICATOR_ACTIVATION_TIME
            + ~use_pot_extra * ~successful_indicator_activation * interact_extra
        )

        new_cell = jnp.array([interact_item, new_ingredients, new_extra])

        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

        new_inventory = (
            successful_pickup * (pile_ingredient + merged_ingredients)
            + no_effect * inventory
        )
        new_agent = agent.replace(inventory=new_inventory)

        is_correct_recipe = inventory == plated_recipe

        reward = jnp.array(0, dtype=float)

        reward += (
            successful_delivery
            * jax.lax.select(
                is_correct_recipe,
                1,
                -1 if self.negative_rewards else 0,
            )
            * DELIVERY_REWARD
        )

        reward -= successful_indicator_activation * INDICATOR_ACTIVATION_COST

        inventory_is_plate = new_inventory == DynamicObject.PLATE
        successful_plate_pickup = successful_pickup * inventory_is_plate
        num_plates_in_inventory = jnp.sum(all_inventories == DynamicObject.PLATE)
        num_nonempty_pots = jnp.sum(
            (grid[:, :, 0] == StaticObject.POT) & (grid[:, :, 1] != 0)
        )
        is_plate_pickup_useful = num_plates_in_inventory < num_nonempty_pots
        no_plates_on_counters = jnp.sum(grid[:, :, 1] == DynamicObject.PLATE) == 0
        shaped_reward += (
            no_plates_on_counters
            * is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )

        correct_delivery = successful_delivery & is_correct_recipe
        return new_grid, new_agent, correct_delivery, reward, shaped_reward, successful_pot_placement, successful_dish_pickup, successful_plate_pickup
