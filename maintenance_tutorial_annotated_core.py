# Annotated teaching copy: every executable source line has an explanation immediately above it.
# Updated simulator: fixed aircraft identities and individually assigned heterogeneous maintenance resources.
# Explanation: This docstring states module's purpose/contract: One supporting module for the comprehensive RL/HRL maintenance course. It is documentation text, not an executable operation.
"""One supporting module for the comprehensive RL/HRL maintenance course.

OBSERVATION_VERSION=3 adds individual resource availability and duration factors.
Fleet action field 4 is now a crew-selection bitmask, not a crew count minus one.
The first examples and user's exercise answers are namespaced by tutorial part.
The fleet classes/functions keep their public names. No import depends on Code.
All figures are teaching assumptions rather than real turbofan calibration.
"""
# Explanation: Postpone annotation evaluation so forward references can name classes defined later without runtime lookup.
from __future__ import annotations



# ========================================================================
# mini_maintenance.py
# ========================================================================


# Explanation: Use Python's pseudo-random generator for the small simulator and random policy.
import random
# Explanation: Import the marker used to declare a type alias explicitly.
from typing import TypeAlias
# Explanation: Declare Observation as a type-level nickname for a two-integer tuple; runtime values are ordinary tuples.
Observation: TypeAlias = tuple[int, int]

# Explanation: Define class MiniMaintenanceEnv: a tiny one-item simulator for state, action, reward, and episode mechanics.
class MiniMaintenanceEnv:
    # Explanation: This docstring states MiniMaintenanceEnv's purpose/contract: One aircraft with one replaceable item and two high-level actions. It is documentation text, not an executable operation.
    """One aircraft with one replaceable item and two high-level actions.

    Actions:
        0 = operate for one day
        1 = ground for one day and replace the item

    Health is an integer in {0, 1, 2, 3}; 3 is new and 0 is failed.
    A successful operation earns +5. If operation ends at health 0, it earns
    -12 instead. Grounding/replacement earns -4 and restores health to 3.
    Operating wear is 1, plus one extra unit with probability shock_prob.

    The episode ends after ``horizon`` days. Failure itself does not end it:
    the policy can ground and restore the item on a later day.
    """
    # Explanation: Define shared module constant OPERATE; other examples reuse this value.
    OPERATE = 0
    # Explanation: Define shared module constant GROUND_AND_REPLACE; other examples reuse this value.
    GROUND_AND_REPLACE = 1

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, horizon: int=12, shock_prob: float=0.0) -> None:
        # Explanation: Validate the input invariant not isinstance(horizon, int) or horizon <= 0; the guard rejects invalid values before the state-changing code runs.
        if not isinstance(horizon, int) or horizon <= 0:
            # Explanation: Raise ValueError('horizon must be a positive integer') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('horizon must be a positive integer')
        # Explanation: Branch on not 0.0 <= shock_prob <= 1.0; the indented statements handle this case, and other cases fall through or use a later branch.
        if not 0.0 <= shock_prob <= 1.0:
            # Explanation: Raise ValueError('shock_prob must be between 0 and 1') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('shock_prob must be between 0 and 1')
        # Explanation: Save this value on the instance as self.horizon; later methods read it as persistent environment/model state.
        self.horizon = horizon
        # Explanation: Save this value on the instance as self.shock_prob; later methods read it as persistent environment/model state.
        self.shock_prob = float(shock_prob)
        # Explanation: Save this value on the instance as self._rng; later methods read it as persistent environment/model state.
        self._rng = random.Random()
        # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
        self.health = 3
        # Explanation: Save this value on the instance as self.day; later methods read it as persistent environment/model state.
        self.day = 0
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = True

    # Explanation: Define reset with the shown parameters/defaults; its purpose is to start an episode and initialize its random streams and state.
    def reset(self, *, seed: int | None=None, initial_health: int=3):
        # Explanation: This docstring states reset's purpose/contract: Start an episode and return (observation, info). It is documentation text, not an executable operation.
        """Start an episode and return (observation, info).

        A seed makes subsequent stochastic wear reproducible. Calling reset
        without a seed keeps the current generator stream.
        """
        # Explanation: Branch on initial_health not in (0, 1, 2, 3); the indented statements handle this case, and other cases fall through or use a later branch.
        if initial_health not in (0, 1, 2, 3):
            # Explanation: Raise ValueError('initial_health must be 0, 1, 2, or 3') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('initial_health must be 0, 1, 2, or 3')
        # Explanation: Branch on seed is not None; the indented statements handle this case, and other cases fall through or use a later branch.
        if seed is not None:
            # Explanation: Set a pseudo-random generator's state so later stochastic outcomes can be reproduced.
            self._rng.seed(seed)
        # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
        self.health = initial_health
        # Explanation: Save this value on the instance as self.day; later methods read it as persistent environment/model state.
        self.day = 0
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = False
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), {'day': self.day})

    # Explanation: Define step with the shown parameters/defaults; its purpose is to validate one action, advance the simulation, and return the transition.
    def step(self, action: int):
        # Explanation: This docstring states step's purpose/contract: Return (next_observation, reward, terminated, truncated, info). It is documentation text, not an executable operation.
        """Return (next_observation, reward, terminated, truncated, info)."""
        # Explanation: Branch on self._needs_reset; the indented statements handle this case, and other cases fall through or use a later branch.
        if self._needs_reset:
            # Explanation: Raise RuntimeError('call reset() before step(), including after an episode') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise RuntimeError('call reset() before step(), including after an episode')
        # Explanation: Branch on action not in (self.OPERATE, self.GROUND_AND_REPLACE); the indented statements handle this case, and other cases fall through or use a later branch.
        if action not in (self.OPERATE, self.GROUND_AND_REPLACE):
            # Explanation: Raise ValueError('action must be 0 (operate) or 1 (ground/replace)') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('action must be 0 (operate) or 1 (ground/replace)')
        # Explanation: Compute and bind the right-hand value to previous_health, naming an intermediate for the surrounding algorithm.
        previous_health = self.health
        # Explanation: Branch on action == self.OPERATE; the indented statements handle this case, and other cases fall through or use a later branch.
        if action == self.OPERATE:
            # Explanation: Compute and bind the right-hand value to extra_wear, naming an intermediate for the surrounding algorithm.
            extra_wear = int(self._rng.random() < self.shock_prob)
            # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
            self.health = max(0, self.health - 1 - extra_wear)
            # Explanation: Record whether wear left the item at health zero; this Boolean selects the failure penalty and event label.
            failed = self.health == 0
            # Explanation: Give an operation a −12 failure penalty if health reached zero; otherwise the successful flight earns +5.
            reward = -12.0 if failed else 5.0
            # Explanation: Compute and bind the right-hand value to info, naming an intermediate for the surrounding algorithm.
            info = {'event': 'failed_operation' if failed else 'successful_operation', 'wear': previous_health - self.health, 'failed': failed, 'grounded': False}
        # Explanation: Handle the alternative case not selected by the preceding if/elif test.
        else:
            # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
            self.health = 3
            # Explanation: Compute the immediate transition reward and store it in reward; later checks/updates use this named intermediate.
            reward = -4.0
            # Explanation: Compute and bind the right-hand value to info, naming an intermediate for the surrounding algorithm.
            info = {'event': 'grounded_and_replaced', 'wear': 0, 'failed': False, 'grounded': True}
        # Explanation: Accumulate 1 into self.day; the previous value is retained and this amount is added.
        self.day += 1
        # Explanation: Mark the episode terminal once the number of simulated days reaches the configured horizon.
        terminated = self.day >= self.horizon
        # Explanation: Compute whether an external time limit stopped the episode and store it in truncated; later checks/updates use this named intermediate.
        truncated = False
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = terminated
        # Explanation: Compute and bind the right-hand value to info['day'], naming an intermediate for the surrounding algorithm.
        info['day'] = self.day
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), reward, terminated, truncated, info)

    # Explanation: Define _observation with the shown parameters/defaults; its purpose is to pack internal state into the public observation layout.
    def _observation(self) -> Observation:
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (self.health, self.horizon - self.day)

    # Explanation: Define render_text with the shown parameters/defaults; its purpose is to format the toy environment state as readable text.
    def render_text(self) -> str:
        # Explanation: This docstring states render_text's purpose/contract: Small human-readable display; useful for tracing a policy. It is documentation text, not an executable operation.
        """Small human-readable display; useful for tracing a policy."""
        # Explanation: Return f'day={self.day:2d}  health={self.health}  steps_left={self.horizon - self.day}' to the caller; this ends the function and supplies its result.
        return f'day={self.day:2d}  health={self.health}  steps_left={self.horizon - self.day}'

# Explanation: Define random_policy with the shown parameters/defaults; its purpose is to sample either toy action without using health, providing a random baseline.
def random_policy(observation: Observation, *, rng: random.Random) -> int:
    # Explanation: This docstring states random_policy's purpose/contract: Random baseline, independent of health. It is documentation text, not an executable operation.
    """Random baseline, independent of health."""
    # Explanation: Return rng.randrange(2) to the caller; this ends the function and supplies its result.
    return rng.randrange(2)

# Explanation: Define threshold_policy with the shown parameters/defaults; its purpose is to replace at low health and operate otherwise, providing a transparent rule baseline.
def threshold_policy(observation: Observation) -> int:
    # Explanation: This docstring states threshold_policy's purpose/contract: Simple rule baseline: replace at health 0 or 1. It is documentation text, not an executable operation.
    """Simple rule baseline: replace at health 0 or 1."""
    # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
    health, _steps_left = observation
    # Explanation: Return MiniMaintenanceEnv.GROUND_AND_REPLACE if health <= 1 else MiniMaintenanceEnv.OPERATE to the caller; this ends the function and supplies its result.
    return MiniMaintenanceEnv.GROUND_AND_REPLACE if health <= 1 else MiniMaintenanceEnv.OPERATE

# Explanation: Define rollout with the shown parameters/defaults; its purpose is to run a policy to episode end and record each state, action, reward, and event.
def rollout(env: MiniMaintenanceEnv, policy, *, seed: int=0):
    # Explanation: This docstring states rollout's purpose/contract: Execute one episode; policy receives observation and returns an action. It is documentation text, not an executable operation.
    """Execute one episode; policy receives observation and returns an action."""
    # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
    observation, _info = env.reset(seed=seed)
    # Explanation: Annotate rows as list[dict] to document the expected kind of value for readers and type checkers.
    rows: list[dict] = []
    # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
    while True:
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = policy(observation)
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        next_observation, reward, terminated, truncated, info = env.step(action)
        # Explanation: Append the new record/value to rows so it remains in the result/history.
        rows.append({'observation': observation, 'action': action, 'reward': reward, 'next_observation': next_observation, 'event': info['event'], 'terminated': terminated, 'truncated': truncated})
        # Explanation: Compute the state visible to the policy and store it in observation; later checks/updates use this named intermediate.
        observation = next_observation
        # Explanation: Check the episode/option stop condition terminated or truncated; when true, stop collecting transitions for this segment.
        if terminated or truncated:
            # Explanation: Leave the nearest loop because its stopping condition has been reached.
            break
    # Explanation: Return rows to the caller; this ends the function and supplies its result.
    return rows


# ========================================================================
# hierarchical_maintenance.py
# ========================================================================


# Explanation: Generate combinations of candidate engine and component masks.
from itertools import product
# Explanation: Define shared module constant TOY_COMPONENTS; other examples reuse this value.
TOY_COMPONENTS = ('fan', 'compressor')

# Explanation: Define valid_action with the shown parameters/defaults; its purpose is to check that grounding, engine selection, and component selection agree structurally.
def valid_action(action: tuple[int, int, int]) -> bool:
    # Explanation: This docstring states valid_action's purpose/contract: Action is (ground, engine_mask, component_mask). It is documentation text, not an executable operation.
    """Action is (ground, engine_mask, component_mask).

    Engine bits 0 and 1 represent engines 0 and 1. Component bits 0-1
    represent engine 0; bits 2-3 represent engine 1. A selected engine must
    contain at least one selected component, and no unselected engine may
    receive component work.
    """
    # Explanation: Branch on len(action) != 3 or any((type(x) is not int for x in action)); the indented statements handle this case, and other cases fall through or use a later branch.
    if len(action) != 3 or any((type(x) is not int for x in action)):
        # Explanation: Return False to reject the candidate action/input without changing simulator state.
        return False
    # Explanation: Decode the action tuple by position into grounding flag, engine mask, and component mask; this gives the later bit checks meaningful names.
    ground, engines, components = action
    # Explanation: When grounding is off, only zero engine and component masks are legal; this all-zero tuple is the toy operate action.
    if ground == 0:
        # Explanation: Return engines == 0 and components == 0 to the caller; this ends the function and supplies its result.
        return engines == 0 and components == 0
    # Explanation: Validate the input invariant ground != 1 or not 1 <= engines <= 3 or (not 1 <= components <= 15); the guard rejects invalid values before the state-changing code runs.
    if ground != 1 or not 1 <= engines <= 3 or (not 1 <= components <= 15):
        # Explanation: Return False to reject the candidate action/input without changing simulator state.
        return False
    # Explanation: This toy action has one four-bit component mask: bits 0–1 belong to engine 0 and bits 2–3 to engine 1. Decimal 3 is 0011₂; shifting it by 2*engine creates that engine's pair mask. Shift binds more tightly than &: components & 3 << 2*engine means components & (3 << (2*engine)). The AND asks whether that pair contains any selected component; 1 << engine contributes engine flag 1 or 2. Summing distinct flags reconstructs the separate two-bit engine mask; it does not store component identities.
    selected_engines = sum((1 << engine for engine in range(2) if components & 3 << 2 * engine))
    # Explanation: Accept only when the stated engine flags exactly match the engines with at least one selected component; this catches missing and extra flags.
    return engines == selected_engines

# Explanation: Define all_actions with the shown parameters/defaults; its purpose is to enumerate feasible toy actions so a policy can use an integer ID instead of a tuple.
def all_actions():
    # Explanation: This docstring states all_actions's purpose/contract: List the 16 feasible composite actions, including operate. It is documentation text, not an executable operation.
    """List the 16 feasible composite actions, including operate."""
    # Explanation: Return [(0, 0, 0)] + [(1, engines, components) for engines, components in product(range(1, 4), range(1, 16)) if valid_action((1, engines, components))] to the caller; this ends the function and supplies its result.
    return [(0, 0, 0)] + [(1, engines, components) for engines, components in product(range(1, 4), range(1, 16)) if valid_action((1, engines, components))]

# Explanation: Define class ConditionalMaintenanceEnv: a deterministic two-engine example demonstrating hierarchical actions and bitmasks.
class ConditionalMaintenanceEnv:
    # Explanation: This docstring states ConditionalMaintenanceEnv's purpose/contract: Tiny deterministic simulator for one aircraft with two engines. It is documentation text, not an executable operation.
    """Tiny deterministic simulator for one aircraft with two engines."""

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, horizon: int=6):
        # Explanation: Validate the input invariant not isinstance(horizon, int) or horizon <= 0; the guard rejects invalid values before the state-changing code runs.
        if not isinstance(horizon, int) or horizon <= 0:
            # Explanation: Raise ValueError('horizon must be positive') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('horizon must be positive')
        # Explanation: Save this value on the instance as self.horizon; later methods read it as persistent environment/model state.
        self.horizon = horizon
        # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
        self.health = [3, 3, 3, 3]
        # Explanation: Save this value on the instance as self.day; later methods read it as persistent environment/model state.
        self.day = 0
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = True

    # Explanation: Define _observation with the shown parameters/defaults; its purpose is to pack internal state into the public observation layout.
    def _observation(self):
        # Explanation: Return tuple(self.health) + (self.horizon - self.day,) to the caller; this ends the function and supplies its result.
        return tuple(self.health) + (self.horizon - self.day,)

    # Explanation: Define reset with the shown parameters/defaults; its purpose is to start an episode and initialize its random streams and state.
    def reset(self, initial_health=(3, 3, 3, 3)):
        # Explanation: Branch on len(initial_health) != 4 or any((h not in range(4) for h in initial_health)); the indented statements handle this case, and other cases fall through or use a later branch.
        if len(initial_health) != 4 or any((h not in range(4) for h in initial_health)):
            # Explanation: Raise ValueError('provide four health values in 0..3') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('provide four health values in 0..3')
        # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
        self.health = list(initial_health)
        # Explanation: Save this value on the instance as self.day; later methods read it as persistent environment/model state.
        self.day = 0
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = False
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), {})

    # Explanation: Define step with the shown parameters/defaults; its purpose is to validate one action, advance the simulation, and return the transition.
    def step(self, action):
        # Explanation: Branch on self._needs_reset; the indented statements handle this case, and other cases fall through or use a later branch.
        if self._needs_reset:
            # Explanation: Raise RuntimeError('reset before stepping') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise RuntimeError('reset before stepping')
        # Explanation: Branch on not valid_action(action); the indented statements handle this case, and other cases fall through or use a later branch.
        if not valid_action(action):
            # Explanation: Raise ValueError(f'infeasible hierarchical action: {action}') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError(f'infeasible hierarchical action: {action}')
        # Explanation: Decode the action tuple by position into grounding flag, engine mask, and component mask; this gives the later bit checks meaningful names.
        ground, engines, components = action
        # Explanation: Branch on ground; the indented statements handle this case, and other cases fall through or use a later branch.
        if ground:
            # Explanation: Visit each index from range(4); the body applies the operation once per item.
            for index in range(4):
                # Explanation: Test component bit index: 1 << index creates a mask containing only that bit, and AND is nonzero exactly when the component was selected.
                if components & 1 << index:
                    # Explanation: Update stored simulator state self.health[index] from this expression; it holds health values held by this toy state or simulator for later transitions.
                    self.health[index] = 3
            # Explanation: Compute and bind the right-hand value to n_engines, naming an intermediate for the surrounding algorithm.
            n_engines = engines.bit_count()
            # Explanation: Compute and bind the right-hand value to n_components, naming an intermediate for the surrounding algorithm.
            n_components = components.bit_count()
            # Explanation: Charge a base grounding cost of 3, plus 2 per selected engine and 1 per replaced component; larger work bundles cost more.
            reward = -3.0 - 2.0 * n_engines - 1.0 * n_components
            # Explanation: Compute and bind the right-hand value to event, naming an intermediate for the surrounding algorithm.
            event = 'grounded_and_replaced'
        # Explanation: Handle the alternative case not selected by the preceding if/elif test.
        else:
            # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
            self.health = [max(0, h - 1) for h in self.health]
            # Explanation: Compute the immediate transition reward and store it in reward; later checks/updates use this named intermediate.
            reward = -15.0 if 0 in self.health else 8.0
            # Explanation: Compute and bind the right-hand value to event, naming an intermediate for the surrounding algorithm.
            event = 'failed_operation' if reward < 0 else 'successful_operation'
        # Explanation: Accumulate 1 into self.day; the previous value is retained and this amount is added.
        self.day += 1
        # Explanation: Mark the episode terminal once the number of simulated days reaches the configured horizon.
        terminated = self.day >= self.horizon
        # Explanation: Save this value on the instance as self._needs_reset; later methods read it as persistent environment/model state.
        self._needs_reset = terminated
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), reward, terminated, False, {'event': event})


# ========================================================================
# gym_maintenance.py
# ========================================================================


# Explanation: Use Gymnasium environment/wrapper base classes under the conventional alias.
import gymnasium as gym
# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Declare and validate legal observation and action shapes.
from gymnasium import spaces

# Explanation: Define class GymMaintenanceEnv: a Gymnasium-compatible wrapper around the small maintenance example.
class GymMaintenanceEnv(gym.Env):
    # Explanation: This docstring states GymMaintenanceEnv's purpose/contract: Same decisions and rewards as MiniMaintenanceEnv, with Gymnasium spaces. It is documentation text, not an executable operation.
    """Same decisions and rewards as MiniMaintenanceEnv, with Gymnasium spaces.

    The observation is [health / 3, days_left / horizon]. Normalization keeps
    both input features between zero and one for a neural-network policy.
    """
    # Explanation: Compute and bind the right-hand value to metadata, naming an intermediate for the surrounding algorithm.
    metadata = {'render_modes': ['ansi']}

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, horizon: int=12, shock_prob: float=0.0):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__()
        # Explanation: Save this value on the instance as self.core; later methods read it as persistent environment/model state.
        self.core = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        # Explanation: Save this value on the instance as self.action_space; later methods read it as persistent environment/model state.
        self.action_space = spaces.Discrete(2)
        # Explanation: Save this value on the instance as self.observation_space; later methods read it as persistent environment/model state.
        self.observation_space = spaces.Box(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

    # Explanation: Define _encode with the shown parameters/defaults; its purpose is to perform this  encode operation.
    def _encode(self, observation):
        # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
        health, days_left = observation
        # Explanation: Return np.array([health / 3.0, days_left / self.core.horizon], dtype=np.float32) to the caller; this ends the function and supplies its result.
        return np.array([health / 3.0, days_left / self.core.horizon], dtype=np.float32)

    # Explanation: Define reset with the shown parameters/defaults; its purpose is to start an episode and initialize its random streams and state.
    def reset(self, *, seed=None, options=None):
        # Explanation: Compute and bind the right-hand value to super().reset(seed, naming an intermediate for the surrounding algorithm.
        super().reset(seed=seed)
        # Explanation: Compute and bind the right-hand value to initial_health, naming an intermediate for the surrounding algorithm.
        initial_health = 3 if options is None else options.get('initial_health', 3)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        observation, info = self.core.reset(seed=seed, initial_health=initial_health)
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (self._encode(observation), info)

    # Explanation: Define step with the shown parameters/defaults; its purpose is to validate one action, advance the simulation, and return the transition.
    def step(self, action):
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        observation, reward, terminated, truncated, info = self.core.step(int(action))
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (self._encode(observation), float(reward), terminated, truncated, info)

    # Explanation: Define render with the shown parameters/defaults; its purpose is to format simulator state for a human.
    def render(self):
        # Explanation: Return self.core.render_text() to the caller; this ends the function and supplies its result.
        return self.core.render_text()

    # Explanation: Define close with the shown parameters/defaults; its purpose is to perform this close operation.
    def close(self):
        # Explanation: Leave this block intentionally empty; pass is only a syntactic placeholder.
        pass


# ========================================================================
# fleet_maintenance/fleet_env.py
# ========================================================================


# Explanation: Generate record-class methods from annotated fields.
from dataclasses import dataclass
# Explanation: Use operations such as ceiling when calculating maintenance duration.
import math
# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Use Gymnasium environment/wrapper base classes under the conventional alias.
import gymnasium as gym
# Explanation: Declare and validate legal observation and action shapes.
from gymnasium import spaces
# Explanation: Define COMPONENTS as a shared modeling constant: the six component names in bundle-bit order.
COMPONENTS = ('fan', 'LPC', 'HPC', 'combustor', 'HPT', 'LPT')
# Explanation: Define WEAR as a shared modeling constant: baseline daily wear for each component.
WEAR = np.array([0.002, 0.0025, 0.004, 0.002, 0.003, 0.0025])
# Explanation: Define LABOR_DAYS as a shared modeling constant: labor duration for each component replacement.
LABOR_DAYS = np.array([2, 3, 4, 2, 3, 3])
# Explanation: Define PART_COST as a shared modeling constant: replacement-part cost by component type.
PART_COST = np.array([8.0, 10.0, 16.0, 8.0, 12.0, 10.0])
# Explanation: Define EFFICIENCY_BASE as a shared modeling constant: efficiency baseline at zero health.
EFFICIENCY_BASE = 0.72
# Explanation: Define EFFICIENCY_GAIN as a shared modeling constant: efficiency change from health.
EFFICIENCY_GAIN = 0.24
# Explanation: Define TEMPERATURE_BASE_C as a shared modeling constant: baseline temperature by component.
TEMPERATURE_BASE_C = np.array([40.0, 120.0, 350.0, 900.0, 650.0, 450.0])
# Explanation: Define shared module constant TEMPERATURE_HEALTH_GAIN_C; other examples reuse this value.
TEMPERATURE_HEALTH_GAIN_C = 120.0
# Explanation: Define shared module constant TEMPERATURE_EXPOSURE_GAIN_C; other examples reuse this value.
TEMPERATURE_EXPOSURE_GAIN_C = 30.0
# Explanation: Define shared module constant TEMPERATURE_MIN_C; other examples reuse this value.
TEMPERATURE_MIN_C = -50.0
# Explanation: Define shared module constant TEMPERATURE_MAX_C; other examples reuse this value.
TEMPERATURE_MAX_C = 1400.0
# Explanation: Define OBSERVATION_VERSION as a shared modeling constant: saved observation-layout version tag.
OBSERVATION_VERSION = 3

# Explanation: Apply dataclass's constructor/repr/equality behavior and freeze field reassignment; these instances are immutable records.
@dataclass(frozen=True)
# Explanation: Define class Config: an immutable record of fleet, horizon, resources, disturbances, and estimation settings.
class Config:
    # Explanation: Declare Config field aircraft as int with default 7; it controls fleet size.
    aircraft: int = 7
    # Explanation: Declare Config field horizon as int with default 90; it controls episode duration in days.
    horizon: int = 90
    # Explanation: Declare Config field crews as int with default 3; it controls available maintenance crews.
    crews: int = 3
    # Explanation: Declare Config field workshops as int with default 2; it controls parallel maintenance shops.
    workshops: int = 2
    # Explanation: A factor below one makes this crew member faster than standard; above one is slower. Empty uses a repeating expert/standard/novice teaching profile (0.8, 1.0, 1.25).
    crew_time_factors: tuple[float, ...] = ()
    # Explanation: Each workshop has its own component-work duration multiplier; empty uses repeating improved/standard/slower teaching values (0.8, 1.0, 1.2).
    workshop_time_factors: tuple[float, ...] = ()
    # Explanation: Declare Config field particles as int with default 64; it controls health-belief samples per component.
    particles: int = 64
    # Explanation: Declare Config field setup_days as int with default 7; it controls fixed setup delay before component work.
    setup_days: int = 7
    # Explanation: Declare Config field replenish_days as int with default 14; it controls inventory replenishment interval.
    replenish_days: int = 14
    # Explanation: Declare Config field bird_probability as float with default 0.002; it controls bird-event probability.
    bird_probability: float = 0.002
    # Explanation: Declare Config field hail_probability as float with default 0.003; it controls hail-event probability.
    hail_probability: float = 0.003
    # Explanation: Declare Config field weather_probability as float with default 0.03; it controls weather-event probability.
    weather_probability: float = 0.03
    # Explanation: Declare Config field observation_noise as float with default 0.025; it controls sensor noise scale.
    observation_noise: float = 0.025
    # Explanation: Declare Config field wear_scale as float with default 1.0; it controls wear multiplier.
    wear_scale: float = 1.0
    # Explanation: Declare Config field mandatory_health as float with default 0.08; it controls safety threshold for estimated lower health.
    mandatory_health: float = 0.08
    # Explanation: Declare Config field initial_health as float with default 0.88; it controls mean component health at reset.
    initial_health: float = 0.88

    # Explanation: Define __post_init__ with the shown parameters/defaults; its purpose is to validate dataclass fields after the generated initializer has assigned them.
    def __post_init__(self):
        # Explanation: Visit each k from ('aircraft', 'horizon', 'crews', 'workshops', 'particles', 'setup_days', 'replenish_days'); the body applies the operation once per item.
        for k in ('aircraft', 'horizon', 'crews', 'workshops', 'particles', 'setup_days', 'replenish_days'):
            # Explanation: Validate the input invariant not isinstance(getattr(self, k), int) or getattr(self, k) < 1; the guard rejects invalid values before the state-changing code runs.
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                # Explanation: Raise ValueError(f'{k} must be a positive integer') so invalid input/failure is explicit instead of continuing with inconsistent state.
                raise ValueError(f'{k} must be a positive integer')
        # Explanation: Check that every explicitly supplied resource profile has one finite positive factor per resource.
        for name, count in (('crew_time_factors', self.crews), ('workshop_time_factors', self.workshops)):
            # Explanation: Obtain the optional profile; an empty tuple requests the documented illustrative defaults.
            factors = getattr(self, name)
            # Explanation: Accept tuples and JSON-style lists while rejecting scalars and missing profiles with a configuration error.
            if not isinstance(factors, (tuple, list)):
                # Explanation: A profile must be a sequence even when the empty default is requested.
                raise ValueError(f'{name} must be a tuple or list of duration factors')
            # Explanation: Reject missing or invalid resource factors before constructing an environment.
            if len(factors) and (len(factors) != count or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in factors)):
                # Explanation: Fail explicitly rather than calculating durations from a malformed resource profile.
                raise ValueError(f'{name} must be empty or contain {count} finite positive numbers')
            # Explanation: Normalize accepted lists to immutable tuples so factors cannot silently change during a committed job.
            object.__setattr__(self, name, tuple(factors))
        # Explanation: Visit each k from ('bird_probability', 'hail_probability', 'weather_probability'); the body applies the operation once per item.
        for k in ('bird_probability', 'hail_probability', 'weather_probability'):
            # Explanation: Validate the input invariant not 0 <= getattr(self, k) <= 1; the guard rejects invalid values before the state-changing code runs.
            if not 0 <= getattr(self, k) <= 1:
                # Explanation: Raise ValueError(f'{k} must be in [0,1]') so invalid input/failure is explicit instead of continuing with inconsistent state.
                raise ValueError(f'{k} must be in [0,1]')
        # Explanation: Validate the input invariant not 0 < self.observation_noise or not 0 < self.wear_scale; the guard rejects invalid values before the state-changing code runs.
        if not 0 < self.observation_noise or not 0 < self.wear_scale:
            # Explanation: Raise ValueError('noise and wear_scale must be positive') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('noise and wear_scale must be positive')
        # Explanation: Validate the input invariant not 0 <= self.mandatory_health < 1 or not 0 < self.initial_health <= 1; the guard rejects invalid values before the state-changing code runs.
        if not 0 <= self.mandatory_health < 1 or not 0 < self.initial_health <= 1:
            # Explanation: Raise ValueError('invalid health bounds') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('invalid health bounds')

# Explanation: Apply dataclass's constructor/repr/equality behavior and freeze field reassignment; these instances are immutable records.
@dataclass(frozen=True)
# Explanation: Define class Plan: a record of an accepted plan and its engine/component selections and resources.
class Plan:
    # Explanation: Identify the aircraft index for this accepted plan; resolve its plate through env.aircraft_plates.
    aircraft: int
    # Explanation: Declare record field engine_mask as int; dataclass construction and attribute access carry engine-selection bitmask.
    engine_mask: int
    # Explanation: Declare record field bundles as tuple[int, int]; dataclass construction and attribute access carry the two per-engine component masks.
    bundles: tuple[int, int]
    # Explanation: Bit i identifies crew member i; a plan reserves every set bit exclusively until the job completes.
    crew_mask: int
    # Explanation: Declare record field workshop as int; dataclass construction and attribute access carry this part of the plan/job.
    workshop: int
    # Explanation: Declare record field order_start as int; dataclass construction and attribute access carry this part of the plan/job.
    order_start: int
    # Explanation: Declare record field priority as int; dataclass construction and attribute access carry this part of the plan/job.
    priority: int

    # Explanation: Expose the number of selected crew members without losing their individual identities.
    @property
    # Explanation: Count selected crew bits for reporting and compatibility with count-based callers.
    def crews(self):
        # Explanation: Each one bit contributes exactly one crew member.
        return self.crew_mask.bit_count()

# Explanation: Have dataclass generate the usual record methods while allowing instances' fields to change.
@dataclass
# Explanation: Define class Job: the live execution record of a plan, its tasks, and remaining duration.
class Job:
    # Explanation: Declare record field plan as Plan; dataclass construction and attribute access carry this part of the plan/job.
    plan: Plan
    # Explanation: Declare record field tasks as tuple[tuple[int, int], ...]; dataclass construction and attribute access carry selected (engine, component) repair pairs.
    tasks: tuple[tuple[int, int], ...]
    # Explanation: Declare record field remaining as int; dataclass construction and attribute access carry days left in this job.
    remaining: int
    # Explanation: Declare record field total_duration as int; dataclass construction and attribute access carry the job's full planned duration.
    total_duration: int
    # Explanation: Store each component's scheduled completion day relative to job start, including setup; execution uses exactly the schedule calculated at acceptance.
    completion_days: tuple[int, ...]

# Explanation: Define class FleetMaintenanceEnv: the full simulator for hidden health, noisy observations, resources, jobs, rewards, and events.
class FleetMaintenanceEnv(gym.Env):
    # Explanation: State the eight action fields and their identity/commitment semantics.
    """Action row: ground, engines, bundle0, bundle1, crew_mask, shop, order, priority.

    Engine mask bits 0/1 select engines. Each bundle has six component bits.
    Crew bit i selects member CR-(i+1); workshop remains a single resource index.
    Crew work rates add; workshop factors multiply component work, not setup.
    Resource factors are synthetic teaching assumptions, not calibrated estimates.
    aircraft_plates[i] identifies aircraft i in every state array and action row.
    Plates are synthetic fleet-local identifiers, independent of operating status.
    Reset starts a new simulated history while retaining the index-to-plate mapping.
    Resources are assigned to complete plans, in selected priority order.
    Unfunded/conflicting proposals are rejected and logged; no partial repairs.
    A busy aircraft's row is ignored: the committed plan cannot be overwritten.
    Mandatory grounding uses observed lower health quantiles, not hidden health.
    """
    # Explanation: Compute and bind the right-hand value to metadata, naming an intermediate for the surrounding algorithm.
    metadata = {'render_modes': ['ansi']}

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, config: Config | None=None, render_mode=None):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__()
        # Explanation: Save this value on the instance as self.config; later methods read it as persistent environment/model state.
        self.config = config or Config()
        # Explanation: Save this value on the instance as self.render_mode; later methods read it as persistent environment/model state.
        self.render_mode = render_mode
        # Explanation: Compute and bind the right-hand value to n, naming an intermediate for the surrounding algorithm.
        n = self.config.aircraft
        # Explanation: Give each aircraft index a fixed synthetic plate; tuple order matches health, exposure, action rows, and any future fleet-mask bit positions.
        self.aircraft_plates = tuple(f'AC-{i + 1:03d}' for i in range(n))
        # Explanation: Give every crew member a stable fleet-local identity in crew-bit order.
        self.crew_ids = tuple(f'CR-{i + 1:03d}' for i in range(self.config.crews))
        # Explanation: Give every workshop a stable identity in workshop-index order.
        self.workshop_ids = tuple(f'WS-{i + 1:03d}' for i in range(self.config.workshops))
        # Explanation: Resolve explicit crew profiles or repeat the documented expert/standard/novice teaching factors.
        self.crew_time_factors = tuple(self.config.crew_time_factors) or tuple((0.8, 1.0, 1.25)[i % 3] for i in range(self.config.crews))
        # Explanation: Resolve explicit workshop profiles or repeat the documented improved/standard/slower teaching factors.
        self.workshop_time_factors = tuple(self.config.workshop_time_factors) or tuple((0.8, 1.0, 1.2)[i % 3] for i in range(self.config.workshops))
        # Explanation: Define eight action categories per aircraft; the crew head represents all 2**crews member subsets, with zero reserved for no assignment.
        self.row_dims = np.array([2, 4, 64, 64, 1 << self.config.crews, self.config.workshops, 12, n])
        # Explanation: Save this value on the instance as self.action_space; later methods read it as persistent environment/model state.
        self.action_space = spaces.MultiDiscrete(np.tile(self.row_dims, (n, 1)))
        # Explanation: Save this value on the instance as self.local_size; later methods read it as persistent environment/model state.
        self.local_size = 70
        # Explanation: Keep the original ten public fields, then add availability and duration-factor encodings for every crew member and workshop.
        self.public_size = 10 + 2 * self.config.crews + 2 * self.config.workshops
        # Explanation: Save this value on the instance as self.observation_space; later methods read it as persistent environment/model state.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n * self.local_size + self.public_size,), dtype=np.float32)
        # Explanation: Save this value on the instance as self._done; later methods read it as persistent environment/model state.
        self._done = True

    # Explanation: Define reset with the shown parameters/defaults; its purpose is to start an episode and initialize its random streams and state.
    def reset(self, *, seed=None, options=None):
        # Explanation: Compute and bind the right-hand value to super().reset(seed, naming an intermediate for the surrounding algorithm.
        super().reset(seed=seed)
        # Explanation: Compute and bind the right-hand value to cfg, naming an intermediate for the surrounding algorithm.
        cfg = self.config
        # Explanation: Compute and bind the right-hand value to seeds, naming an intermediate for the surrounding algorithm.
        seeds = np.random.SeedSequence(seed).spawn(3)
        # Explanation: Save this value on the instance as self.physical_rng; later methods read it as persistent environment/model state.
        self.physical_rng = np.random.default_rng(seeds[0])
        # Explanation: Save this value on the instance as self.filter_rng; later methods read it as persistent environment/model state.
        self.filter_rng = np.random.default_rng(seeds[1])
        # Explanation: Compute and bind the right-hand value to tape_rng, naming an intermediate for the surrounding algorithm.
        tape_rng = np.random.default_rng(seeds[2])
        # Explanation: Save this value on the instance as self.day; later methods read it as persistent environment/model state.
        self.day = 0
        # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
        self.health = np.clip(cfg.initial_health + self.physical_rng.normal(0, 0.025, (cfg.aircraft, 2, 6)), 0.1, 1)
        # Explanation: Branch on options and 'health' in options; the indented statements handle this case, and other cases fall through or use a later branch.
        if options and 'health' in options:
            # Explanation: Convert the expression to a NumPy array and store it as health (health values held by this toy state or simulator).
            health = np.asarray(options['health'], dtype=float)
            # Explanation: Branch on health.shape != self.health.shape or np.any((health < 0) | (health > 1)); the indented statements handle this case, and other cases fall through or use a later branch.
            if health.shape != self.health.shape or np.any((health < 0) | (health > 1)):
                # Explanation: Raise ValueError('initial health must have shape (aircraft,2,6), in [0,1]') so invalid input/failure is explicit instead of continuing with inconsistent state.
                raise ValueError('initial health must have shape (aircraft,2,6), in [0,1]')
            # Explanation: Update stored simulator state self.health from this expression; it holds health values held by this toy state or simulator for later transitions.
            self.health = health.copy()
        # Explanation: Compute the aircraft-level multiplier that changes modeled wear and heat and store it in self.exposure; later checks/updates use this named intermediate.
        self.exposure = tape_rng.uniform(0.75, 1.35, cfg.aircraft)
        # Explanation: Update stored simulator state self.measurements from this expression; it holds noisy sensor readings, not hidden true health for later transitions.
        self.measurements = self._sensor_model(self.health, self.exposure[:, None, None])
        # Explanation: Accumulate self.physical_rng.normal(0, 1, self.measurements.shape) * self.sensor_noise into self.measurements; the previous value is retained and this amount is added.
        self.measurements += self.physical_rng.normal(0, 1, self.measurements.shape) * self.sensor_noise
        # Explanation: Call self._clip_measurements to delegate this state update/check to the environment or model method.
        self._clip_measurements(self.measurements)
        # Explanation: Save this value on the instance as self.last_flight_day; later methods read it as persistent environment/model state.
        self.last_flight_day = np.zeros(cfg.aircraft, dtype=int)
        # Explanation: Compute and bind the right-hand value to prior, naming an intermediate for the surrounding algorithm.
        prior = self._invert_sensor_metrics(self.measurements, self.exposure[:, None, None])
        # Explanation: Update stored simulator state self.particles from this expression; it holds sampled health beliefs maintained by the filter for later transitions.
        self.particles = np.clip(prior[..., None] + self.filter_rng.normal(0, 0.04, (*self.health.shape, cfg.particles)), 0, 1)
        # Explanation: Unpack {} into named values self.jobs: dict[int, Job] so each part can be used separately below.
        self.jobs: dict[int, Job] = {}
        # Explanation: Create self.stock as a fixed-shape array filled with the given starting value; it represents the spare-part inventory by component type.
        self.stock = np.full(6, cfg.aircraft, dtype=int)
        # Explanation: Save this value on the instance as self.event_tape; later methods read it as persistent environment/model state.
        self.event_tape = tape_rng.random((cfg.horizon, cfg.aircraft, 2, 2))
        # Explanation: Save this value on the instance as self.severity_tape; later methods read it as persistent environment/model state.
        self.severity_tape = tape_rng.uniform(0.1, 0.35, (cfg.horizon, cfg.aircraft, 2, 2))
        # Explanation: Save this value on the instance as self.weather_tape; later methods read it as persistent environment/model state.
        self.weather_tape = tape_rng.random(cfg.horizon)
        # Explanation: Save this value on the instance as self.wear_tape; later methods read it as persistent environment/model state.
        self.wear_tape = tape_rng.normal(1.0, 0.12, (cfg.horizon, cfg.aircraft, 2, 6))
        # Explanation: Save this value on the instance as self.noise_tape; later methods read it as persistent environment/model state.
        self.noise_tape = tape_rng.normal(0, cfg.observation_noise, (cfg.horizon, cfg.aircraft, 2, 6, 2))
        # Explanation: Update stored simulator state self.metrics from this expression; it holds the accumulated simulator performance counters for later transitions.
        self.metrics = {k: 0.0 for k in ('return', 'cost', 'flight_failures', 'flights', 'downtime_aircraft_days', 'groundings', 'replacements', 'rejections', 'bird_events', 'hail_events', 'weather_exposures', 'unsafe_attempts')}
        # Explanation: Save this value on the instance as self._done; later methods read it as persistent environment/model state.
        self._done = False
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), {'day': self.day, 'aircraft_plates': self.aircraft_plates, 'resources': self.resource_details()})

    # Explanation: Define split_observation with the shown parameters/defaults; its purpose is to separate each aircraft's local features from fleet-wide public values.
    def split_observation(self, observation):
        # Explanation: Compute and bind the right-hand value to a, naming an intermediate for the surrounding algorithm.
        a = np.asarray(observation)
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (a[:-self.public_size].reshape(self.config.aircraft, self.local_size), a[-self.public_size:])

    # Explanation: Define idle_action with the shown parameters/defaults; its purpose is to create the all-zero action matrix used when no plan is proposed.
    def idle_action(self):
        # Explanation: Return np.zeros((self.config.aircraft, 8), dtype=np.int64) to the caller; this ends the function and supplies its result.
        return np.zeros((self.config.aircraft, 8), dtype=np.int64)

    # Explanation: Derive crew availability from the identities reserved by committed jobs.
    def available_crew_mask(self):
        # Explanation: Start with every configured crew bit set, meaning all members are available.
        available = (1 << self.config.crews) - 1
        # Explanation: Visit each running job and remove its reserved crew identities.
        for job in self.jobs.values():
            # Explanation: Clear selected bits without changing the remaining members' identities.
            available &= ~job.plan.crew_mask
        # Explanation: Return the bitmask of crew members that can accept a new job.
        return available

    # Explanation: Define resource_state with the shown parameters/defaults; its purpose is to count uncommitted crews and list unoccupied workshops.
    def resource_state(self):
        # Explanation: Compute and bind the right-hand value to busy_shops, naming an intermediate for the surrounding algorithm.
        busy_shops = {j.plan.workshop for j in self.jobs.values()}
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (self.available_crew_mask().bit_count(), [s for s in range(self.config.workshops) if s not in busy_shops])

    # Explanation: Provide labeled resource information for tracing assignments without inserting strings into neural observations.
    def resource_details(self):
        # Explanation: Return stable identities, physical duration factors, and currently available resource selections.
        return {'crew_ids': self.crew_ids, 'crew_time_factors': self.crew_time_factors, 'available_crew_mask': self.available_crew_mask(), 'workshop_ids': self.workshop_ids, 'workshop_time_factors': self.workshop_time_factors, 'available_workshops': tuple(self.resource_state()[1])}

    # Explanation: Apply decorator property to the definition below; Python passes that class/function through the decorator.
    @property
    # Explanation: Define sensor_noise with the shown parameters/defaults; its purpose is to return the noise scale used for efficiency and temperature channels.
    def sensor_noise(self):
        # Explanation: This docstring states sensor_noise's purpose/contract: Independent sensor noise scales in efficiency fraction and Celsius. It is documentation text, not an executable operation.
        """Independent sensor noise scales in efficiency fraction and Celsius."""
        # Explanation: Return np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C]) * self.config.observation_noise to the caller; this ends the function and supplies its result.
        return np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C]) * self.config.observation_noise

    # Explanation: Apply decorator staticmethod to the definition below; Python passes that class/function through the decorator.
    @staticmethod
    # Explanation: Define _sensor_model with the shown parameters/defaults; its purpose is to map hidden health/exposure into synthetic efficiency and temperature readings.
    def _sensor_model(health, exposure):
        # Explanation: This docstring states _sensor_model's purpose/contract: Toy performance mapping; health decreases efficiency and raises heat. It is documentation text, not an executable operation.
        """Toy performance mapping; health decreases efficiency and raises heat."""
        # Explanation: Compute and bind the right-hand value to efficiency, naming an intermediate for the surrounding algorithm.
        efficiency = EFFICIENCY_BASE + EFFICIENCY_GAIN * health
        # Explanation: Compute and bind the right-hand value to temperature, naming an intermediate for the surrounding algorithm.
        temperature = TEMPERATURE_BASE_C + TEMPERATURE_HEALTH_GAIN_C * (1 - health) + TEMPERATURE_EXPOSURE_GAIN_C * (exposure - 1)
        # Explanation: Return np.stack([efficiency, temperature], axis=-1) to the caller; this ends the function and supplies its result.
        return np.stack([efficiency, temperature], axis=-1)

    # Explanation: Apply decorator staticmethod to the definition below; Python passes that class/function through the decorator.
    @staticmethod
    # Explanation: Define _clip_measurements with the shown parameters/defaults; its purpose is to keep sensor values within their modeled bounds.
    def _clip_measurements(measurements):
        # Explanation: Unpack measurements[..., 0]) into named values np.clip(measurements[..., 0], 0, 1, out so each part can be used separately below.
        np.clip(measurements[..., 0], 0, 1, out=measurements[..., 0])
        # Explanation: Unpack measurements[..., 1]) into named values np.clip(measurements[..., 1], TEMPERATURE_MIN_C, TEMPERATURE_MAX_C, out so each part can be used separately below.
        np.clip(measurements[..., 1], TEMPERATURE_MIN_C, TEMPERATURE_MAX_C, out=measurements[..., 1])

    # Explanation: Apply decorator staticmethod to the definition below; Python passes that class/function through the decorator.
    @staticmethod
    # Explanation: Define _invert_sensor_metrics with the shown parameters/defaults; its purpose is to estimate health from the two sensor equations at initialization.
    def _invert_sensor_metrics(measurements, exposure):
        # Explanation: This docstring states _invert_sensor_metrics's purpose/contract: Initialization only: fuse two noisy inverse mappings with equal weight. It is documentation text, not an executable operation.
        """Initialization only: fuse two noisy inverse mappings with equal weight."""
        # Explanation: Compute and bind the right-hand value to from_efficiency, naming an intermediate for the surrounding algorithm.
        from_efficiency = (measurements[..., 0] - EFFICIENCY_BASE) / EFFICIENCY_GAIN
        # Explanation: Compute and bind the right-hand value to from_temperature, naming an intermediate for the surrounding algorithm.
        from_temperature = 1 - (measurements[..., 1] - TEMPERATURE_BASE_C - TEMPERATURE_EXPOSURE_GAIN_C * (exposure - 1)) / TEMPERATURE_HEALTH_GAIN_C
        # Explanation: Return np.clip(0.5 * (from_efficiency + from_temperature), 0, 1) to the caller; this ends the function and supplies its result.
        return np.clip(0.5 * (from_efficiency + from_temperature), 0, 1)

    # Explanation: Define observation_metrics with the shown parameters/defaults; its purpose is to decode sensor columns without exposing simulator hidden health.
    def observation_metrics(self, observation):
        # Explanation: This docstring states observation_metrics's purpose/contract: Decode observed sensor metrics without exposing hidden health. It is documentation text, not an executable operation.
        """Decode observed sensor metrics without exposing hidden health."""
        # Explanation: Unpack the return values from self.split_observation(observation) into rows; underscore entries are intentionally ignored.
        rows, _ = self.split_observation(observation)
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return {'efficiency': rows[:, 45:57].reshape(self.config.aircraft, 2, 6).copy(), 'temperature_c': rows[:, 57:69].reshape(self.config.aircraft, 2, 6) * (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C) + TEMPERATURE_MIN_C, 'reading_age_days': np.rint(rows[:, 69] * self.config.horizon).astype(int)}

    # Explanation: Define _observation with the shown parameters/defaults; its purpose is to pack internal state into the public observation layout.
    def _observation(self):
        # Explanation: Compute and bind the right-hand value to mean, naming an intermediate for the surrounding algorithm.
        mean = self.particles.mean(-1)
        # Explanation: Compute and bind the right-hand value to std, naming an intermediate for the surrounding algorithm.
        std = self.particles.std(-1)
        # Explanation: Compute the conservative 10th-percentile health estimate and store it in low; later checks/updates use this named intermediate.
        low = np.quantile(self.particles, 0.1, axis=-1)
        # Explanation: Compute and bind the right-hand value to rows, naming an intermediate for the surrounding algorithm.
        rows = []
        # Explanation: Visit each a from range(self.config.aircraft); the body applies the operation once per item.
        for a in range(self.config.aircraft):
            # Explanation: Compute and bind the right-hand value to j, naming an intermediate for the surrounding algorithm.
            j = self.jobs.get(a)
            # Explanation: Compute and bind the right-hand value to committed, naming an intermediate for the surrounding algorithm.
            committed = [j.plan.engine_mask / 3, j.plan.bundles[0] / 63, j.plan.bundles[1] / 63, j.plan.crew_mask / ((1 << self.config.crews) - 1), j.plan.workshop / max(1, self.config.workshops - 1), j.plan.order_start / 11] if j else [0.0] * 6
            # Explanation: Append the new record/value to rows so it remains in the result/history.
            rows.append(np.concatenate([mean[a].ravel(), std[a].ravel(), low[a].ravel(), [self.exposure[a] / 1.5, float(j is not None), min(1, j.remaining / 60) if j else 0], committed, self.measurements[a, :, :, 0].ravel(), ((self.measurements[a, :, :, 1] - TEMPERATURE_MIN_C) / (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C)).ravel(), [(self.day - self.last_flight_day[a]) / self.config.horizon]]))
        # Explanation: Unpack self.resource_state() into named values free, shops so each part can be used separately below.
        free, shops = self.resource_state()
        # Explanation: Compute fleet-wide observation values shared by aircraft and store it in public; later checks/updates use this named intermediate.
        public = [self.day / self.config.horizon, free / self.config.crews, len(shops) / self.config.workshops, (self.config.replenish_days - self.day % self.config.replenish_days) / self.config.replenish_days, *np.clip(self.stock / self.config.aircraft, 0, 1)]
        # Explanation: Read the individual availability mask after all current reservations.
        crew_mask = self.available_crew_mask()
        # Explanation: Append crew availability flags, then bounded encodings of each member's duration factor; factor = 1 / encoded - 1.
        public.extend([float(bool(crew_mask & (1 << i))) for i in range(self.config.crews)] + [1 / (1 + factor) for factor in self.crew_time_factors])
        # Explanation: Append workshop availability flags and the same bounded duration-factor encoding in workshop-index order.
        public.extend([float(i in shops) for i in range(self.config.workshops)] + [1 / (1 + factor) for factor in self.workshop_time_factors])
        # Explanation: Return np.asarray(np.concatenate([np.asarray(rows).ravel(), public]), dtype=np.float32) to the caller; this ends the function and supplies its result.
        return np.asarray(np.concatenate([np.asarray(rows).ravel(), public]), dtype=np.float32)

    # Explanation: Define validate_row with the shown parameters/defaults; its purpose is to check one aircraft row's grounding/engine/bundle consistency.
    def validate_row(self, aircraft, row):
        # Explanation: Decode the action tuple by position into grounding flag, engine mask, and component mask; this gives the later bit checks meaningful names.
        ground, engines, b0, b1, crew_mask, shop, order, priority = map(int, row)
        # Explanation: Branch on not ground; the indented statements handle this case, and other cases fall through or use a later branch.
        if not ground:
            # Explanation: Return None if engines == b0 == b1 == 0 else 'work_without_grounding' to the caller; this ends the function and supplies its result.
            return None if engines == b0 == b1 == 0 else 'work_without_grounding'
        # Explanation: Derive the engine mask from whether each bundle is nonempty. int(b0 != 0) is engine 0's low bit (1); 2 * int(b1 != 0) places engine 1 in bit 1 (2). Examples: [5,0]→01₂→1, [0,8]→10₂→2, [5,8]→11₂→3. Both zero gives 0 and cannot describe grounded work. The factor 2 is a bit-position weight, not extra priority; a busy aircraft's committed job is separate and is not canceled by this proposal.
        actual = int(b0 != 0) + 2 * int(b1 != 0)
        # Explanation: Reject a grounded work row if both bundles are empty, or if the engine mask disagrees with which bundles are nonzero.
        if not actual or engines != actual:
            # Explanation: Return 'inconsistent_engine_bundle' to the caller; this ends the function and supplies its result.
            return 'inconsistent_engine_bundle'
        # Explanation: Every new maintenance job must select at least one valid crew-member bit.
        if not 0 < crew_mask < (1 << self.config.crews):
            # Explanation: Reject empty or out-of-fleet crew selections as structurally invalid.
            return 'invalid_crew_mask'
        # Explanation: Return None to signal that validation found no error; callers treat this as the successful case.
        return None

    # Explanation: Define _start_job with the shown parameters/defaults; its purpose is to check resources/spares and commit an accepted maintenance plan.
    def _start_job(self, aircraft, row):
        # Explanation: Decode engine/component selections, the individual crew bitmask, workshop index, task ordering, and priority.
        _, engines, b0, b1, crew_mask, shop, order, priority = map(int, row)
        # Explanation: Compute the proposed or accepted maintenance plan and store it in plan; later checks/updates use this named intermediate.
        plan = Plan(aircraft, engines, (b0, b1), crew_mask, shop, order, priority)
        # Explanation: Decode bundle bits into (engine, component) tasks. enumerate maps b0/b1 to engine e; b & (1 << c) tests component c. Each set bit adds one repair task; a zero bit adds none.
        tasks = [(e, c) for e, b in enumerate((b0, b1)) for c in range(6) if b & 1 << c]
        # Explanation: Sort component tasks cyclically from the requested start position. The engine/component pair maps to one of 12 positions, and modulo 12 wraps the ordering around.
        tasks.sort(key=lambda x: (x[0] * 6 + x[1] - order) % 12)
        # Explanation: Count requested repairs by component type, then compare those needs with inventory before committing anything; this prevents partial plans when stock is short.
        needed = np.bincount([c for _, c in tasks], minlength=6)
        # Explanation: Unpack self.resource_state() into named values free, shops so each part can be used separately below.
        _, shops = self.resource_state()
        # Explanation: Reject any selected member who is already reserved, even when enough other crew members remain free; workshops are exclusive too.
        if crew_mask & self.available_crew_mask() != crew_mask or shop not in shops:
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return ('resource_conflict', 0.0)
        # Explanation: Branch on np.any(needed > self.stock); the indented statements handle this case, and other cases fall through or use a later branch.
        if np.any(needed > self.stock):
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return ('spare_shortage', 0.0)
        # Explanation: Combine selected members' work rates; two standard members supply twice the rate of one standard member under this teaching assumption.
        work_rate = sum(1 / factor for i, factor in enumerate(self.crew_time_factors) if crew_mask & (1 << i))
        # Explanation: Begin with the fixed setup delay, which resource expertise does not shorten.
        duration = self.config.setup_days
        # Explanation: Build one cumulative completion threshold per component before committing any inventory or resources.
        completion_days = []
        # Explanation: Use the same sorted component sequence that execution will follow.
        for _, component in tasks:
            # Explanation: Multiply standard component labor by workshop factor, divide by combined crew work rate, and round up to at least one whole workday.
            duration += max(1, math.ceil(LABOR_DAYS[component] * self.workshop_time_factors[shop] / work_rate))
            # Explanation: Record when this particular component is repaired, including setup and all earlier work.
            completion_days.append(duration)
        # Explanation: Commit all required spares only after resource, inventory, and duration calculations succeed.
        self.stock -= needed
        # Explanation: Update stored simulator state self.jobs[aircraft] from this expression; it holds accepted maintenance plans still in progress for later transitions.
        self.jobs[aircraft] = Job(plan, tuple(tasks), duration, duration, tuple(completion_days))
        # Explanation: Compute and bind the right-hand value to cost, naming an intermediate for the surrounding algorithm.
        cost = 12.0 + engines.bit_count() * 5 + sum((PART_COST[c] + 3 * self.health[aircraft, e, c] for e, c in tasks))
        # Explanation: Accumulate 1 into self.metrics['groundings']; the previous value is retained and this amount is added.
        self.metrics['groundings'] += 1
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (None, cost)

    # Explanation: Define _update_filter with the shown parameters/defaults; its purpose is to advance health-belief particles and reweight them from sensor evidence.
    def _update_filter(self, aircraft, measurements):
        # Explanation: This docstring states _update_filter's purpose/contract: Bootstrap particle filter with noisy observations and shock-mixture prior. It is documentation text, not an executable operation.
        """Bootstrap particle filter with noisy observations and shock-mixture prior.

        The estimator does not receive true event type, severity or hidden health.
        It is a deliberately simple independent-component filter, not Emanuele's.
        """
        # Explanation: Compute and bind the right-hand value to cfg, naming an intermediate for the surrounding algorithm.
        cfg = self.config
        # Explanation: Compute and bind the right-hand value to p, naming an intermediate for the surrounding algorithm.
        p = self.particles[aircraft]
        # Explanation: Compute and bind the right-hand value to wear, naming an intermediate for the surrounding algorithm.
        wear = WEAR[None, :, None] * self.exposure[aircraft] * cfg.wear_scale
        # Explanation: Reduce p by wear * self.filter_rng.normal(1, 0.12, p.shape); this records wear/cost/work already consumed in the current transition.
        p -= wear * self.filter_rng.normal(1, 0.12, p.shape)
        # Explanation: Compute and bind the right-hand value to bird, naming an intermediate for the surrounding algorithm.
        bird = self.filter_rng.random((2, 1, cfg.particles)) < cfg.bird_probability
        # Explanation: Compute and bind the right-hand value to hail, naming an intermediate for the surrounding algorithm.
        hail = self.filter_rng.random((2, 1, cfg.particles)) < cfg.hail_probability
        # Explanation: Compute and bind the right-hand value to bseverity, naming an intermediate for the surrounding algorithm.
        bseverity = self.filter_rng.uniform(0.1, 0.35, (2, 1, cfg.particles))
        # Explanation: Compute and bind the right-hand value to hseverity, naming an intermediate for the surrounding algorithm.
        hseverity = self.filter_rng.uniform(0.1, 0.35, (2, 1, cfg.particles))
        # Explanation: Reduce p by bird * bseverity * np.array([1, 0.5, 0.15, 0, 0, 0])[None, :, None]; this records wear/cost/work already consumed in the current transition.
        p -= bird * bseverity * np.array([1, 0.5, 0.15, 0, 0, 0])[None, :, None]
        # Explanation: Reduce p by hail * hseverity * np.array([0.7, 0.4, 0.3, 0.1, 0.1, 0.1])[None, :, None]; this records wear/cost/work already consumed in the current transition.
        p -= hail * hseverity * np.array([0.7, 0.4, 0.3, 0.1, 0.1, 0.1])[None, :, None]
        # Explanation: Reduce p by (self.filter_rng.random((1, 1, cfg.particles)) < cfg.weather_probability) * wear * 2; this records wear/cost/work already consumed in the current transition.
        p -= (self.filter_rng.random((1, 1, cfg.particles)) < cfg.weather_probability) * wear * 2
        # Explanation: Unpack p) into named values np.clip(p, 0, 1, out so each part can be used separately below.
        np.clip(p, 0, 1, out=p)
        # Explanation: Compute and bind the right-hand value to predicted_efficiency, naming an intermediate for the surrounding algorithm.
        predicted_efficiency = EFFICIENCY_BASE + EFFICIENCY_GAIN * p
        # Explanation: Compute and bind the right-hand value to predicted_temperature, naming an intermediate for the surrounding algorithm.
        predicted_temperature = TEMPERATURE_BASE_C[None, :, None] + TEMPERATURE_HEALTH_GAIN_C * (1 - p) + TEMPERATURE_EXPOSURE_GAIN_C * (self.exposure[aircraft] - 1)
        # Explanation: Compute and bind the right-hand value to efficiency_error, naming an intermediate for the surrounding algorithm.
        efficiency_error = (predicted_efficiency - measurements[:, :, 0, None]) / self.sensor_noise[0]
        # Explanation: Compute and bind the right-hand value to temperature_error, naming an intermediate for the surrounding algorithm.
        temperature_error = (predicted_temperature - measurements[:, :, 1, None]) / self.sensor_noise[1]
        # Explanation: Compute unnormalized scores later converted to choice probabilities and store it in logits; later checks/updates use this named intermediate.
        logits = -0.5 * (efficiency_error ** 2 + temperature_error ** 2)
        # Explanation: Compute and bind the right-hand value to weights, naming an intermediate for the surrounding algorithm.
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        # Explanation: Compute and bind the right-hand value to weights, naming an intermediate for the surrounding algorithm.
        weights /= weights.sum(-1, keepdims=True)
        # Explanation: Visit each e from range(2); the body applies the operation once per item.
        for e in range(2):
            # Explanation: Visit each c from range(6); the body applies the operation once per item.
            for c in range(6):
                # Explanation: Compute and bind the right-hand value to idx, naming an intermediate for the surrounding algorithm.
                idx = self.filter_rng.choice(cfg.particles, cfg.particles, p=weights[e, c])
                # Explanation: Unpack np.clip(p[e, c, idx] + self.filter_rng.normal(0, 0.006, cfg.particles), 0, 1) into named values p[e, c] so each part can be used separately below.
                p[e, c] = np.clip(p[e, c, idx] + self.filter_rng.normal(0, 0.006, cfg.particles), 0, 1)

    # Explanation: Define step with the shown parameters/defaults; its purpose is to validate one action, advance the simulation, and return the transition.
    def step(self, action):
        # Explanation: Check the episode/option stop condition self._done; when true, stop collecting transitions for this segment.
        if self._done:
            # Explanation: Raise RuntimeError('reset before step, including after episode completion') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise RuntimeError('reset before step, including after episode completion')
        # Explanation: Compute and bind the right-hand value to a, naming an intermediate for the surrounding algorithm.
        a = np.asarray(action)
        # Explanation: Branch on not np.issubdtype(a.dtype, np.integer) or not self.action_space.contains(a); the indented statements handle this case, and other cases fall through or use a later branch.
        if not np.issubdtype(a.dtype, np.integer) or not self.action_space.contains(a):
            # Explanation: Raise ValueError('action outside MultiDiscrete space') so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError('action outside MultiDiscrete space')
        # Explanation: Compute and bind the right-hand value to cfg, naming an intermediate for the surrounding algorithm.
        cfg = self.config
        # Explanation: Compute the conservative 10th-percentile health estimate and store it in low; later checks/updates use this named intermediate.
        low = np.quantile(self.particles, 0.1, axis=-1)
        # Explanation: Compute whether a lower-health safety alarm is active and store it in mandatory; later checks/updates use this named intermediate.
        mandatory = low.min(axis=(1, 2)) <= cfg.mandatory_health
        # Explanation: Compute repair, downtime, rejection, and failure costs and store it in costs; later checks/updates use this named intermediate.
        costs = 0.0
        # Explanation: Compute event records generated during the transition and store it in events; later checks/updates use this named intermediate.
        events = []
        # Explanation: Compute proposed plans that were not accepted and why and store it in rejected; later checks/updates use this named intermediate.
        rejected = []
        # Explanation: Compute aircraft whose jobs finished today and store it in completed; later checks/updates use this named intermediate.
        completed = []
        # Explanation: Compute and bind the right-hand value to grounded, naming an intermediate for the surrounding algorithm.
        grounded = set(self.jobs)
        # Explanation: Visit each aircraft from sorted(range(cfg.aircraft), key=lambda i: (-int(a[i, 7]), i)); the body applies the operation once per item.
        for aircraft in sorted(range(cfg.aircraft), key=lambda i: (-int(a[i, 7]), i)):
            # Explanation: Check whether the aircraft already has a committed job so the branch can preserve or track that work.
            if aircraft in self.jobs:
                # Explanation: Skip the rest of this iteration and advance to the next loop item.
                continue
            # Explanation: Compute the validation or resource-conflict reason and store it in reason; later checks/updates use this named intermediate.
            reason = self.validate_row(aircraft, a[aircraft])
            # Explanation: Branch on reason is None and int(a[aircraft, 0]); the indented statements handle this case, and other cases fall through or use a later branch.
            if reason is None and int(a[aircraft, 0]):
                # Explanation: Unpack self._start_job(aircraft, a[aircraft]) into named values reason, cost so each part can be used separately below.
                reason, cost = self._start_job(aircraft, a[aircraft])
                # Explanation: Accumulate cost into costs; the previous value is retained and this amount is added.
                costs += cost
            # Explanation: Branch on reason; the indented statements handle this case, and other cases fall through or use a later branch.
            if reason:
                # Explanation: Append the new record/value to rejected so it remains in the result/history.
                rejected.append((aircraft, reason))
                # Explanation: Accumulate 2.0 into costs; the previous value is retained and this amount is added.
                costs += 2.0
            # Explanation: Check whether the aircraft already has a committed job so the branch can preserve or track that work.
            if mandatory[aircraft] and aircraft not in self.jobs:
                # Explanation: Call grounded.add to perform this step of the surrounding algorithm.
                grounded.add(aircraft)
                # Explanation: Accumulate float(not int(a[aircraft, 0])) into self.metrics['unsafe_attempts']; the previous value is retained and this amount is added.
                self.metrics['unsafe_attempts'] += float(not int(a[aircraft, 0]))
            # Explanation: Check whether the aircraft already has a committed job so the branch can preserve or track that work.
            if aircraft in self.jobs:
                # Explanation: Call grounded.add to perform this step of the surrounding algorithm.
                grounded.add(aircraft)
        # Explanation: Accumulate len(rejected) into self.metrics['rejections']; the previous value is retained and this amount is added.
        self.metrics['rejections'] += len(rejected)
        # Explanation: Compute the number of aircraft that flew today and store it in flights; later checks/updates use this named intermediate.
        flights = 0
        # Explanation: Compute the number of failures in this transition or episode and store it in failures; later checks/updates use this named intermediate.
        failures = 0
        # Explanation: Compute whether the shared weather event occurs today and store it in weather; later checks/updates use this named intermediate.
        weather = self.weather_tape[self.day] < cfg.weather_probability
        # Explanation: Visit each aircraft from range(cfg.aircraft); the body applies the operation once per item.
        for aircraft in range(cfg.aircraft):
            # Explanation: Branch on aircraft in grounded; the indented statements handle this case, and other cases fall through or use a later branch.
            if aircraft in grounded:
                # Explanation: Skip the rest of this iteration and advance to the next loop item.
                continue
            # Explanation: Accumulate 1 into flights; the previous value is retained and this amount is added.
            flights += 1
            # Explanation: Reduce self.health[aircraft] by WEAR * self.exposure[aircraft] * cfg.wear_scale * np.maximum(0, self.wear_tape[self.day, aircraft]) * (3 if weather else 1); this records wear/cost/work already consumed in the current transition.
            self.health[aircraft] -= WEAR * self.exposure[aircraft] * cfg.wear_scale * np.maximum(0, self.wear_tape[self.day, aircraft]) * (3 if weather else 1)
            # Explanation: Branch on weather; the indented statements handle this case, and other cases fall through or use a later branch.
            if weather:
                # Explanation: Accumulate 1 into self.metrics['weather_exposures']; the previous value is retained and this amount is added.
                self.metrics['weather_exposures'] += 1
            # Explanation: Visit each e from range(2); the body applies the operation once per item.
            for e in range(2):
                # Explanation: Visit each kind, p, impact from [('bird', cfg.bird_probability, [1, 0.5, 0.15, 0, 0, 0]), ('hail', cfg.hail_probability, [0.7, 0.4, 0.3, 0.1, 0.1, 0.1])]; the body applies the operation once per item.
                for kind, p, impact in [('bird', cfg.bird_probability, [1, 0.5, 0.15, 0, 0, 0]), ('hail', cfg.hail_probability, [0.7, 0.4, 0.3, 0.1, 0.1, 0.1])]:
                    # Explanation: Compute and bind the right-hand value to k, naming an intermediate for the surrounding algorithm.
                    k = int(kind == 'hail')
                    # Explanation: Branch on self.event_tape[self.day, aircraft, e, k] < p; the indented statements handle this case, and other cases fall through or use a later branch.
                    if self.event_tape[self.day, aircraft, e, k] < p:
                        # Explanation: Compute and bind the right-hand value to severity, naming an intermediate for the surrounding algorithm.
                        severity = self.severity_tape[self.day, aircraft, e, k]
                        # Explanation: Reduce self.health[aircraft, e] by severity * np.array(impact); this records wear/cost/work already consumed in the current transition.
                        self.health[aircraft, e] -= severity * np.array(impact)
                        # Explanation: Record the hazard with both the existing aircraft index and its fixed plate so diagnostics can be traced to the same asset.
                        events.append({'aircraft': aircraft, 'aircraft_plate': self.aircraft_plates[aircraft], 'engine': e, 'type': kind})
                        # Explanation: Accumulate 1 into self.metrics[kind + '_events']; the previous value is retained and this amount is added.
                        self.metrics[kind + '_events'] += 1
            # Explanation: Unpack self.health[aircraft]) into named values np.clip(self.health[aircraft], 0, 1, out so each part can be used separately below.
            np.clip(self.health[aircraft], 0, 1, out=self.health[aircraft])
            # Explanation: Compute and bind the right-hand value to failed, naming an intermediate for the surrounding algorithm.
            failed = bool(np.any(self.health[aircraft] <= 0))
            # Explanation: Accumulate failed into failures; the previous value is retained and this amount is added.
            failures += failed
            # Explanation: Update stored simulator state self.measurements[aircraft] from this expression; it holds noisy sensor readings, not hidden true health for later transitions.
            self.measurements[aircraft] = self._sensor_model(self.health[aircraft], self.exposure[aircraft])
            # Explanation: Accumulate self.noise_tape[self.day, aircraft] * np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C]) into self.measurements[aircraft]; the previous value is retained and this amount is added.
            self.measurements[aircraft] += self.noise_tape[self.day, aircraft] * np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C])
            # Explanation: Call self._clip_measurements to delegate this state update/check to the environment or model method.
            self._clip_measurements(self.measurements[aircraft])
            # Explanation: Save this value on the instance as self.last_flight_day[aircraft]; later methods read it as persistent environment/model state.
            self.last_flight_day[aircraft] = self.day + 1
            # Explanation: Call self._update_filter to delegate this state update/check to the environment or model method.
            self._update_filter(aircraft, self.measurements[aircraft])
        # Explanation: Visit each aircraft, j from list(self.jobs.items()); the body applies the operation once per item.
        for aircraft, j in list(self.jobs.items()):
            # Explanation: Convert total duration minus remaining days into a one-based elapsed-day count for locating the current work stage.
            elapsed = j.total_duration - j.remaining + 1
            # Explanation: Use the component completion thresholds fixed when the job accepted its specific crew members and workshop.
            for (e, c), task_day in zip(j.tasks, j.completion_days):
                # Explanation: Branch on elapsed == task_day; the indented statements handle this case, and other cases fall through or use a later branch.
                if elapsed == task_day:
                    # Explanation: When the scheduled task completes, set the replaced component's hidden health to 1.0, the simulator's fully-new value.
                    self.health[aircraft, e, c] = 1.0
                    # Explanation: Reset the belief particles for the replaced component to fully healthy because the repair outcome is known.
                    self.particles[aircraft, e, c] = 1.0
                    # Explanation: Increment the completed-component replacement counter once for this task.
                    self.metrics['replacements'] += 1
            # Explanation: Reduce j.remaining by 1; this records wear/cost/work already consumed in the current transition.
            j.remaining -= 1
            # Explanation: Branch on j.remaining == 0; the indented statements handle this case, and other cases fall through or use a later branch.
            if j.remaining == 0:
                # Explanation: Append the new record/value to completed so it remains in the result/history.
                completed.append(aircraft)
                # Explanation: Continue step; this expression contributes to its current state update or result.
                del self.jobs[aircraft]
        # Explanation: Compute and bind the right-hand value to downtime, naming an intermediate for the surrounding algorithm.
        downtime = len(grounded)
        # Explanation: Accumulate downtime * 2 + failures * 80 into costs; the previous value is retained and this amount is added.
        costs += downtime * 2 + failures * 80
        # Explanation: Compute the immediate transition reward and store it in reward; later checks/updates use this named intermediate.
        reward = (flights * 10 - costs) / (cfg.aircraft * 10)
        # Explanation: Accumulate reward into self.metrics['return']; the previous value is retained and this amount is added.
        self.metrics['return'] += reward
        # Explanation: Accumulate costs into self.metrics['cost']; the previous value is retained and this amount is added.
        self.metrics['cost'] += costs
        # Explanation: Accumulate flights into self.metrics['flights']; the previous value is retained and this amount is added.
        self.metrics['flights'] += flights
        # Explanation: Accumulate failures into self.metrics['flight_failures']; the previous value is retained and this amount is added.
        self.metrics['flight_failures'] += failures
        # Explanation: Accumulate downtime into self.metrics['downtime_aircraft_days']; the previous value is retained and this amount is added.
        self.metrics['downtime_aircraft_days'] += downtime
        # Explanation: Accumulate 1 into self.day; the previous value is retained and this amount is added.
        self.day += 1
        # Explanation: Branch on self.day % cfg.replenish_days == 0; the indented statements handle this case, and other cases fall through or use a later branch.
        if self.day % cfg.replenish_days == 0:
            # Explanation: Update stored simulator state self.stock from this expression; it holds the spare-part inventory by component type for later transitions.
            self.stock = np.minimum(cfg.aircraft, self.stock + max(1, cfg.aircraft // 2))
        # Explanation: Save this value on the instance as self._done; later methods read it as persistent environment/model state.
        self._done = self.day >= cfg.horizon
        # Explanation: Include the fixed plate mapping so aircraft indices in events, rejected proposals, and completed jobs resolve to the same named assets.
        info = {'day': self.day, 'aircraft_plates': self.aircraft_plates, 'resources': self.resource_details(), 'events': events, 'rejected': rejected, 'completed': completed, 'reward_terms': {'flight_revenue': flights * 10, 'cost': costs}, 'metrics': dict(self.metrics)}
        # Explanation: Return the current observation, reward, terminal flags, and info in the API's documented order.
        return (self._observation(), float(reward), self._done, False, info)

    # Explanation: Define render with the shown parameters/defaults; its purpose is to format simulator state for a human.
    def render(self):
        # Explanation: Show the fleet's plates in index order and identify busy aircraft by plate, while preserving the existing resource and inventory display.
        return f'day={self.day} aircraft_plates={self.aircraft_plates} busy={[self.aircraft_plates[i] for i in sorted(self.jobs)]} crews/shops={self.resource_state()} stock={self.stock.tolist()}'

# Explanation: Define threshold_action with the shown parameters/defaults; its purpose is to build a fleet plan from observed estimates while respecting busy aircraft and public resources.
def threshold_action(env, observation, threshold=0.65, bundle_threshold=0.75):
    # Explanation: This docstring states threshold_action's purpose/contract: Feasible rule using only observations, not hidden health or event diagnostics. It is documentation text, not an executable operation.
    """Feasible rule using only observations, not hidden health or event diagnostics."""
    # Explanation: Unpack env.split_observation(observation) into named values rows, public so each part can be used separately below.
    rows, public = env.split_observation(observation)
    # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
    action = env.idle_action()
    # Explanation: Decode individual crew availability from observed flags rather than reading hidden simulator state.
    free_mask = sum(1 << i for i in range(env.config.crews) if public[10 + i] > 0.5)
    # Explanation: Workshop flags follow crew flags and crew-factor encodings in the public observation.
    shop_offset = 10 + 2 * env.config.crews
    # Explanation: Rank available workshops by their known duration factors, preferring the faster workshop first.
    shops = sorted((i for i in range(env.config.workshops) if public[shop_offset + i] > 0.5), key=lambda i: env.workshop_time_factors[i])
    # Explanation: Compute the spare-part inventory by component type and store it in stock; later checks/updates use this named intermediate.
    stock = np.rint(public[4:10] * env.config.aircraft).astype(int)
    # Explanation: Compute the candidate action or option values being compared and store it in scores; later checks/updates use this named intermediate.
    scores = rows[:, 24:36].min(1)
    # Explanation: Compute and bind the right-hand value to rank, naming an intermediate for the surrounding algorithm.
    rank = np.argsort(scores)
    # Explanation: Visit each priority, aircraft from enumerate(rank[::-1]); the body applies the operation once per item.
    for priority, aircraft in enumerate(rank[::-1]):
        # Explanation: Unpack priority into named values action[aircraft, 7] so each part can be used separately below.
        action[aircraft, 7] = priority
    # Explanation: Visit each aircraft from rank; the body applies the operation once per item.
    for aircraft in rank:
        # Explanation: Skip busy or healthy aircraft and stop allocating when no crew member or workshop remains available.
        if rows[aircraft, 37] > 0.5 or scores[aircraft] > threshold or not free_mask or (not shops):
            # Explanation: Skip the rest of this iteration and advance to the next loop item.
            continue
        # Explanation: Reshape this aircraft's first 12 observation values into [engine, component]. They are policy-visible estimates, not hidden simulator truth.
        health = rows[aircraft, :12].reshape(2, 6)
        # Explanation: Build a 2×6 Boolean grid selecting each component whose estimated mean health is at or below the bundle threshold.
        selected = health <= bundle_threshold
        # Explanation: Branch on not selected.any(); the indented statements handle this case, and other cases fall through or use a later branch.
        if not selected.any():
            # Explanation: If the mean-health threshold selected nothing, flatten the 2×6 grid and force its weakest lower-health estimate to True. This fallback asks for at least one repair, if stock permits.
            selected.flat[rows[aircraft, 24:36].argmin()] = True
        # Explanation: Compute the separate six-bit component masks, one per fleet engine and store it in bundles; later checks/updates use this named intermediate.
        bundles = [0, 0]
        # Explanation: Visit each e from range(2); the body applies the operation once per item.
        for e in range(2):
            # Explanation: Visit each c from range(6); the body applies the operation once per item.
            for c in range(6):
                # Explanation: Branch on selected[e, c] and stock[c] > 0; the indented statements handle this case, and other cases fall through or use a later branch.
                if selected[e, c] and stock[c] > 0:
                    # Explanation: Set component c in engine e's bundle without erasing bits selected earlier. For the ordered components fan/LPC/HPC/combustor/HPT/LPT, shifts 1, 2, 4, 8, 16, 32 represent indices 0–5. For example, 5 is 000101₂ and selects fan plus HPC. |= combines flags, not component numbers. Bundle zero means no new work for that engine; nonzero means at least one component is requested.
                    bundles[e] |= 1 << c
                    # Explanation: Reserve one spare of component type c for this proposed task so later aircraft in priority order cannot allocate the same inventory unit.
                    stock[c] -= 1
        # Explanation: Turn two nonempty-bundle tests into engine flags: engine 0 contributes bit 0/value 1; engine 1 contributes bit 1/value 2. [5,0]→01₂ (engine 0 only), [0,8]→10₂ (engine 1 only), [5,8]→11₂ (both). Each bundle is its own six-bit component mask; this two-bit mask only records which engines have work.
        mask = int(bundles[0] != 0) + 2 * int(bundles[1] != 0)
        # Explanation: Branch on not mask; the indented statements handle this case, and other cases fall through or use a later branch.
        if not mask:
            # Explanation: Skip the rest of this iteration and advance to the next loop item.
            continue
        # Explanation: Select one available member with the fastest known individual rate; this rule baseline keeps one member per new job.
        crew = min((i for i in range(env.config.crews) if free_mask & (1 << i)), key=lambda i: env.crew_time_factors[i])
        # Explanation: Assign this specific member by bitmask and reserve the chosen workshop in the proposed joint action.
        action[aircraft] = [1, mask, *bundles, 1 << crew, shops.pop(0), int(health.argmin()), action[aircraft, 7]]
        # Explanation: Remove this identity so another proposal in the same action cannot double-book it.
        free_mask &= ~(1 << crew)
    # Explanation: Return action to the caller; this ends the function and supplies its result.
    return action


# ========================================================================
# fleet_maintenance/adapters.py
# ========================================================================


# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Use Gymnasium environment/wrapper base classes under the conventional alias.
import gymnasium as gym
# Explanation: Declare and validate legal observation and action shapes.
from gymnasium import spaces

# Explanation: Define class FlattenFleetAction: an adapter that flattens fleet actions for Stable-Baselines3 and restores them before simulation.
class FlattenFleetAction(gym.ActionWrapper):
    # Explanation: This docstring states FlattenFleetAction's purpose/contract: Stable-Baselines3 expects a one-dimensional MultiDiscrete nvec. It is documentation text, not an executable operation.
    """Stable-Baselines3 expects a one-dimensional MultiDiscrete nvec."""

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, env):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__(env)
        # Explanation: Save this value on the instance as self.action_shape; later methods read it as persistent environment/model state.
        self.action_shape = env.action_space.nvec.shape
        # Explanation: Save this value on the instance as self.action_space; later methods read it as persistent environment/model state.
        self.action_space = spaces.MultiDiscrete(env.action_space.nvec.ravel())

    # Explanation: Define action with the shown parameters/defaults; its purpose is to reshape a flat action into the simulator's matrix format.
    def action(self, action):
        # Explanation: Return np.asarray(action, dtype=np.int64).reshape(self.action_shape) to the caller; this ends the function and supplies its result.
        return np.asarray(action, dtype=np.int64).reshape(self.action_shape)

    # Explanation: Define reverse_action with the shown parameters/defaults; its purpose is to flatten a matrix action for the outer library.
    def reverse_action(self, action):
        # Explanation: Return np.asarray(action, dtype=np.int64).ravel() to the caller; this ends the function and supplies its result.
        return np.asarray(action, dtype=np.int64).ravel()


# ========================================================================
# fleet_maintenance/policies.py
# ========================================================================


# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Use tensors, neural networks, optimizers, and automatic differentiation.
import torch
# Explanation: Use PyTorch's module and layer classes to construct actor/critic networks.
from torch import nn
# Explanation: Represent a discrete choice distribution with sampling, log-probability, and entropy.
from torch.distributions import Categorical

# Explanation: Define class FleetActorCritic: a constrained action policy plus a critic that estimates state value.
class FleetActorCritic(nn.Module):
    # Explanation: This docstring states FleetActorCritic's purpose/contract: Central critic + flat, conditional, or parameter-shared aircraft actors. It is documentation text, not an executable operation.
    """Central critic + flat, conditional, or parameter-shared aircraft actors.

    shared: local aircraft estimates + public resource counts. Parameters reused
    across aircraft; a central deterministic resolver enforces resource capacity.
    conditional: global actor context and masked dependent action distributions.
    flat: shared global features with no learned conditioning on preceding choices.
    All methods use the same logical masks; scarce-resource conflicts are resolved
    by the simulator and reported separately from structural feasibility.
    All act daily and obey pre-start commitment; this is not temporal HRL.
    """

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, env: FleetMaintenanceEnv, mode='conditional', goal_size=0):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__()
        # Explanation: Branch on mode not in ('flat', 'conditional', 'shared'); the indented statements handle this case, and other cases fall through or use a later branch.
        if mode not in ('flat', 'conditional', 'shared'):
            # Explanation: Raise ValueError(mode) so invalid input/failure is explicit instead of continuing with inconsistent state.
            raise ValueError(mode)
        # Explanation: Save this value on the instance as self.mode; later methods read it as persistent environment/model state.
        self.mode = mode
        # Explanation: Save this value on the instance as self.n; later methods read it as persistent environment/model state.
        self.n = env.config.aircraft
        # Explanation: Compute each aircraft's own observation slice and store it in self.local; later checks/updates use this named intermediate.
        self.local = env.local_size
        # Explanation: Compute fleet-wide observation values shared by aircraft and store it in self.public; later checks/updates use this named intermediate.
        self.public = env.public_size
        # Explanation: Save this value on the instance as self.dims; later methods read it as persistent environment/model state.
        self.dims = env.row_dims.tolist()
        # Explanation: Compute whether a lower-health safety alarm is active and store it in self.mandatory; later checks/updates use this named intermediate.
        self.mandatory = env.config.mandatory_health
        # Explanation: Save this value on the instance as self.goal_size; later methods read it as persistent environment/model state.
        self.goal_size = goal_size
        # Explanation: Compute and bind the right-hand value to size, naming an intermediate for the surrounding algorithm.
        size = env.observation_space.shape[0]
        # Explanation: Build a shared 64-unit encoder for the full fleet observation. It converts the global state into a compact context vector; Tanh bounds each encoded feature.
        self.global_encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh())
        # Explanation: Calculate the per-aircraft actor input width: local row + broadcast public fleet data + optional 64-feature global context (omitted in shared mode) + optional worker goal.
        in_size = self.local + self.public + (0 if mode == 'shared' else 64) + goal_size
        # Explanation: Build the actor's two-layer feature network. It maps each aircraft's assembled inputs to a 64-number representation used by every action-field head.
        self.actor = nn.Sequential(nn.Linear(in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        # Explanation: Create one output head per action field; each emits as many logits as that field has categories. ModuleList registers all heads so PyTorch trains/saves their parameters.
        self.heads = nn.ModuleList((nn.Linear(64, d) for d in self.dims))
        # Explanation: Create one conditioning projection per action field. It maps the eight previously selected field values into the current 64-feature state; bias=False keeps this as a pure learned interaction.
        self.condition = nn.ModuleList((nn.Linear(8, 64, bias=False) for _ in self.dims))
        # Explanation: Build the critic on the complete fleet observation; the final single output is V(s), the estimated return from this state.
        self.critic = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    # Explanation: Define value with the shown parameters/defaults; its purpose is to run the critic to estimate the current state's value.
    def value(self, obs):
        # Explanation: Return self.critic(obs).squeeze(-1) to the caller; this ends the function and supplies its result.
        return self.critic(obs).squeeze(-1)

    # Explanation: Define act with the shown parameters/defaults; its purpose is to choose an action or calculate statistics for a supplied action.
    def act(self, obs, actions=None, deterministic=False, goal=None):
        # Explanation: Branch on obs.ndim == 1; the indented statements handle this case, and other cases fall through or use a later branch.
        if obs.ndim == 1:
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = obs.unsqueeze(0)
        # Explanation: Compute and bind the right-hand value to batch, naming an intermediate for the surrounding algorithm.
        batch = obs.shape[0]
        # Explanation: Compute each aircraft's own observation slice and store it in local; later checks/updates use this named intermediate.
        local = obs[:, :-self.public].reshape(batch, self.n, self.local)
        # Explanation: Compute fleet-wide observation values shared by aircraft and store it in public; later checks/updates use this named intermediate.
        public = obs[:, -self.public:].unsqueeze(1).expand(-1, self.n, -1)
        # Explanation: Start each aircraft's actor input with its own local row and the shared public-resource vector; both are needed to choose a feasible plan.
        features = [local, public]
        # Explanation: Branch on self.goal_size; the indented statements handle this case, and other cases fall through or use a later branch.
        if self.goal_size:
            # Explanation: Branch on goal is None; the indented statements handle this case, and other cases fall through or use a later branch.
            if goal is None:
                # Explanation: Raise ValueError('goal-conditioned worker requires a goal') so invalid input/failure is explicit instead of continuing with inconsistent state.
                raise ValueError('goal-conditioned worker requires a goal')
            # Explanation: Broadcast the manager's one-hot goal across all aircraft so every worker decision is conditioned on the same selected macro.
            features.append(goal.unsqueeze(1).expand(-1, self.n, -1))
        # Explanation: Branch on self.mode != 'shared'; the indented statements handle this case, and other cases fall through or use a later branch.
        if self.mode != 'shared':
            # Explanation: Encode the whole fleet once, add an aircraft axis, then broadcast the same global context to each aircraft's actor input.
            features.append(self.global_encoder(obs).unsqueeze(1).expand(-1, self.n, -1))
        # Explanation: Concatenate each aircraft's local/public/context/goal features and pass them through the actor MLP to get per-aircraft hidden features.
        h = self.actor(torch.cat(features, -1))
        # Explanation: Initialize the eight action-field conditioning slots to zero; as fields are chosen, their normalized values are filled in for later heads.
        previous = torch.zeros(batch, self.n, 8, device=obs.device)
        # Explanation: Start one joint log-probability accumulator per observation in the batch; each chosen field contributes to it.
        logprob = torch.zeros(batch, device=obs.device)
        # Explanation: Start an entropy accumulator with the same batch shape/device as logprob.
        entropy = torch.zeros_like(logprob)
        # Explanation: Compute and bind the right-hand value to result, naming an intermediate for the surrounding algorithm.
        result = []
        # Explanation: Read the busy indicator from local observation column 37 for every batch item and aircraft; busy aircraft already have a committed plan.
        busy = local[:, :, 37] > 0.5
        # Explanation: Branch on self.goal_size; the indented statements handle this case, and other cases fall through or use a later branch.
        if self.goal_size:
            # Explanation: Compute whether an aircraft already has a committed job and store it in busy; later checks/updates use this named intermediate.
            busy = busy | (goal[:, 0] > 0.5).unsqueeze(-1)
        # Explanation: Take the minimum of the 12 per-component lower-health estimates in columns 24–35; mark the aircraft urgent if any estimate crosses the mandatory threshold.
        urgent = local[:, :, 24:36].amin(-1) <= self.mandatory
        # Explanation: Visit each k, head from enumerate(self.heads); the body applies the operation once per item.
        for k, head in enumerate(self.heads):
            # Explanation: Produce this field's category scores. Flat mode uses h directly; conditional/shared modes add a learned projection of earlier choices and pass the sum through Tanh.
            logits = head(h if self.mode == 'flat' else torch.tanh(h + self.condition[k](previous)))
            # Explanation: Initially allow every category for every aircraft; subsequent mask rules remove choices that violate action dependencies or safety constraints.
            valid = torch.ones_like(logits, dtype=torch.bool)
            # Explanation: Temporarily invalidate every category for busy aircraft; their committed job must not be overwritten by a new plan.
            valid[busy] = False
            # Explanation: Restore the no-op choice for busy aircraft; their current job continues and this action cannot assign a second plan.
            valid[:, :, 0] |= busy
            # Explanation: Branch on k == 0; the indented statements handle this case, and other cases fall through or use a later branch.
            if k == 0:
                # Explanation: Forbid the no-grounding choice for urgent aircraft that are not already busy; a running job is not replanned.
                valid[:, :, 0] &= ~(urgent & ~busy)
            # Explanation: Handle the alternative case not selected by the preceding if/elif test.
            else:
                # Explanation: Read the already-selected first field and convert it to a Boolean: this aircraft was chosen for grounding/maintenance.
                ground = result[0] == 1
                # Explanation: Branch on k == 1; the indented statements handle this case, and other cases fall through or use a later branch.
                if k == 1:
                    # Explanation: For the engine-mask field, disallow engine mask zero when grounding was selected; a grounded aircraft needs at least one engine with a nonempty bundle.
                    valid[:, :, 0] &= ~ground
                # Explanation: Branch on k in (2, 3); the indented statements handle this case, and other cases fall through or use a later branch.
                if k in (2, 3):
                    # Explanation: For bundle heads k=2/3, select bit 0/1 from the chosen engine mask. Shift 1 by k−2, AND it with result[1], then compare with zero. Python evaluates shifts before &, so this is result[1] & (1 << (k−2)) != 0.
                    engine_on = result[1] & 1 << k - 2 != 0
                    # Explanation: Compute whether a selected engine may receive a new component bundle and store it in active; later checks/updates use this named intermediate.
                    active = ground & engine_on & ~busy
                    # Explanation: Forbid bundle zero on an active engine: once selected for work, an engine must get a nonempty component bundle.
                    valid[:, :, 0] &= ~active
                    # Explanation: Allow nonzero bundles only on active engines. The final axis is the bundle-choice axis; unsqueeze adds a broadcast dimension so each active flag gates every nonzero bundle.
                    valid[:, :, 1:] &= active.unsqueeze(-1)
                # Explanation: Branch on k >= 4; the indented statements handle this case, and other cases fall through or use a later branch.
                elif k >= 4:
                    # Explanation: For the current engine/resource field, permit nonzero categories only for a new plan that is grounded and not busy; unsqueeze broadcasts that eligibility flag across the category axis.
                    valid[:, :, 1:] &= (ground & ~busy).unsqueeze(-1)
                    # Explanation: A new grounded job requires a nonempty crew mask; other resource fields may use index zero.
                    if k == 4:
                        # Explanation: Suppress empty crew selection only for a newly proposed maintenance job.
                        valid[:, :, 0] &= ~(ground & ~busy)
                # Explanation: Branch on k == 1; the indented statements handle this case, and other cases fall through or use a later branch.
                if k == 1:
                    # Explanation: For the current engine/resource field, permit nonzero categories only for a new plan that is grounded and not busy; unsqueeze broadcasts that eligibility flag across the category axis.
                    valid[:, :, 1:] &= (ground & ~busy).unsqueeze(-1)
            # Explanation: Give invalid choices an extremely negative logit, making their softmax probability effectively zero before constructing the categorical distribution.
            dist = Categorical(logits=logits.masked_fill(~valid, -1000000000.0))
            # Explanation: When training, reuse the stored action to recompute its likelihood. Otherwise take argmax for deterministic evaluation or sample a categorical choice for exploration.
            choice = actions[:, :, k] if actions is not None else dist.probs.argmax(-1) if deterministic else dist.sample()
            # Explanation: Append the new record/value to result so it remains in the result/history.
            result.append(choice)
            # Explanation: Accumulate the selected field's log-probability across aircraft; PPO uses the joint action likelihood in its policy ratio.
            logprob += dist.log_prob(choice).sum(-1)
            # Explanation: Accumulate distribution entropy across aircraft; its bonus encourages the policy to retain exploration.
            entropy += dist.entropy().sum(-1)
            # Explanation: Create a zero tensor shaped like the prior-field buffer so only this action coordinate will be written before accumulation.
            addition = torch.zeros_like(previous)
            # Explanation: Store this selected category in slot k, scaled into roughly 0–1; normalization lets later conditioning layers consume fields with different category counts.
            addition[:, :, k] = choice / float(max(1, self.dims[k] - 1))
            # Explanation: Add the newly selected normalized field to the conditioning buffer without changing earlier field choices.
            previous = previous + addition
        # Explanation: Return the selected fields stacked on the last axis, their joint log-probability, summed entropy, and critic value; these outputs feed rollout storage and PPO.
        return (torch.stack(result, -1), logprob, entropy, self.value(obs))

# Explanation: Define class OptionActorCritic: a manager policy that chooses one of four high-level options and estimates value.
class OptionActorCritic(nn.Module):
    # Explanation: This docstring states OptionActorCritic's purpose/contract: Learned manager selects one of four fixed fleet planning options. It is documentation text, not an executable operation.
    """Learned manager selects one of four fixed fleet planning options."""

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, env):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__()
        # Explanation: Compute and bind the right-hand value to size, naming an intermediate for the surrounding algorithm.
        size = env.observation_space.shape[0]
        # Explanation: Encode the whole fleet observation into a 64-feature manager representation using two Linear/Tanh stages.
        self.encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        # Explanation: Map the manager representation to four option logits, one for each high-level option ID.
        self.actor = nn.Linear(64, 4)
        # Explanation: Map the same manager representation to one scalar state-value estimate.
        self.critic = nn.Linear(64, 1)

    # Explanation: Define value with the shown parameters/defaults; its purpose is to run the critic to estimate the current state's value.
    def value(self, obs):
        # Explanation: Return self.critic(self.encoder(obs)).squeeze(-1) to the caller; this ends the function and supplies its result.
        return self.critic(self.encoder(obs)).squeeze(-1)

    # Explanation: Define act with the shown parameters/defaults; its purpose is to choose an action or calculate statistics for a supplied action.
    def act(self, obs, actions=None, deterministic=False):
        # Explanation: Branch on obs.ndim == 1; the indented statements handle this case, and other cases fall through or use a later branch.
        if obs.ndim == 1:
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = obs.unsqueeze(0)
        # Explanation: Compute the manager's learned feature representation of the current fleet state.
        h = self.encoder(obs)
        # Explanation: Convert the four manager logits into a categorical option distribution; softmax probabilities are implicit.
        dist = Categorical(logits=self.actor(h))
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = actions if actions is not None else dist.probs.argmax(-1) if deterministic else dist.sample()
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1))

# Explanation: Define class LearnedOptionActorCritic: a jointly trained manager and goal-conditioned worker that outputs a macro plus fleet plan.
class LearnedOptionActorCritic(nn.Module):
    # Explanation: This docstring states LearnedOptionActorCritic's purpose/contract: Jointly trained manager + goal-conditioned plan worker + fixed executor. It is documentation text, not an executable operation.
    """Jointly trained manager + goal-conditioned plan worker + fixed executor.

    Macro 0 waits; macros 1..3 are latent planning contexts, not certified skills.
    The worker selects engines, components and resources once before execution.
    Both levels learn from fleet return. This is not MAXQ or option-critic and
    option collapse/credit assignment require later ablations and diagnostics.
    """

    # Explanation: Define __init__ with the shown parameters/defaults; its purpose is to initialize instance state, spaces, configuration, or neural-network layers.
    def __init__(self, env):
        # Explanation: Call super to perform this step of the surrounding algorithm.
        super().__init__()
        # Explanation: Save this value on the instance as self.manager; later methods read it as persistent environment/model state.
        self.manager = OptionActorCritic(env)
        # Explanation: Save this value on the instance as self.worker; later methods read it as persistent environment/model state.
        self.worker = FleetActorCritic(env, 'conditional', goal_size=4)
        # Explanation: Save this value on the instance as self.n; later methods read it as persistent environment/model state.
        self.n = env.config.aircraft

    # Explanation: Define value with the shown parameters/defaults; its purpose is to run the critic to estimate the current state's value.
    def value(self, obs):
        # Explanation: Return self.manager.value(obs) to the caller; this ends the function and supplies its result.
        return self.manager.value(obs)

    # Explanation: Define act with the shown parameters/defaults; its purpose is to choose an action or calculate statistics for a supplied action.
    def act(self, obs, actions=None, deterministic=False):
        # Explanation: Branch on obs.ndim == 1; the indented statements handle this case, and other cases fall through or use a later branch.
        if obs.ndim == 1:
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = obs.unsqueeze(0)
        # Explanation: Compute the manager's high-level option ID and store it in macro; later checks/updates use this named intermediate.
        macro = None if actions is None else actions[:, 0]
        # Explanation: Compute the proposed or accepted maintenance plan and store it in plan; later checks/updates use this named intermediate.
        plan = None if actions is None else actions[:, 1:].reshape(-1, self.n, 8)
        # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
        option, lp, entropy, value = self.manager.act(obs, macro, deterministic)
        # Explanation: Represent each selected option ID as a four-element one-hot goal, which becomes the worker's conditioning input.
        goal = torch.nn.functional.one_hot(option, num_classes=4).float()
        # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
        plan, worker_lp, worker_entropy, _ = self.worker.act(obs, plan, deterministic, goal)
        # Explanation: Flatten each worker plan to one row, then prepend its manager option ID so the stored composite action can later be split and rescored.
        composite = torch.cat([option.unsqueeze(-1), plan.flatten(1)], -1)
        # Explanation: Return macro-plus-plan, joint manager/worker log-probability, joint entropy, and manager value; the two policy levels are trained from the same return.
        return (composite, lp + worker_lp, entropy + worker_entropy, value)
# Multi-agent maintenance: explicit proposal ownership and a common resource broker.
MAPPO_METHODS = ('mappo_aircraft', 'mappo_component')


# Explanation: Define mappo_agent_inputs with the ownership, information, or update contract described below.
def mappo_agent_inputs(observation, aircraft, local_size, public_size, assignment):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Build restricted actor inputs from observed beliefs, never hidden health."""
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    obs = torch.as_tensor(observation, dtype=torch.float32)
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if obs.ndim == 1:
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        obs = obs.unsqueeze(0)
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if obs.ndim != 2 or obs.shape[1] != aircraft * local_size + public_size:
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('incorrect fleet observation shape')
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    local = obs[:, :aircraft * local_size].reshape(-1, aircraft, local_size)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    public = obs[:, -public_size:]
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if assignment == 'aircraft':
        # Each aircraft sees only its own row plus the common resource board.
        return torch.cat([local, public.unsqueeze(1).expand(-1, aircraft, -1)], dim=-1)
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if assignment != 'component':
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('assignment must be aircraft or component')
    # Ten own-component values: two engines x mean/std/lower/efficiency/temperature.
    indices = [[c, c + 6, c + 12, c + 18, c + 24, c + 30,
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                c + 45, c + 51, c + 57, c + 63, 36, 37, 38, 69] for c in range(6)]
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    own = torch.stack([local[:, :, ix] for ix in indices], dim=1)
    # A broadcast safety-criticality report is an explicit cross-component message.
    criticality = local[:, :, 24:36].amin(-1)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    board = criticality[:, None, :, None].expand(-1, 6, -1, -1)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    features = torch.cat([own, board], dim=-1).flatten(2)
    # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
    return torch.cat([features, public.unsqueeze(1).expand(-1, 6, -1)], dim=-1)


# Explanation: Define MultiAgentFleetEnv as the multi-agent environment or model interface.
class MultiAgentFleetEnv:
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Cooperative dictionary API around the same physical fleet simulator.

    Aircraft agents propose two six-bit bundles; six component agents each
    propose a two-engine mask for every aircraft. Both cover the same twelve
    replacement bits per aircraft. They do not learn crew/shop/order choices.
    A public request board and deterministic centralized dispatcher coordinate
    execution. This is decentralized proposal generation with centralized
    arbitration, not fully decentralized resource allocation or learned messaging.
    This small adapter is not a PettingZoo/RLlib API implementation.
    """

    # Explanation: Define __init__ with the ownership, information, or update contract described below.
    def __init__(self, config=None, assignment='aircraft'):
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if assignment not in ('aircraft', 'component'):
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise ValueError('assignment must be aircraft or component')
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.env = FleetMaintenanceEnv(config)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.assignment = assignment
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.agent_ids = self.env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.num_agents = len(self.agent_ids)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        dims = [64, 64] if assignment == 'aircraft' else [4] * self.env.config.aircraft
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.action_spaces = {agent: spaces.MultiDiscrete(dims) for agent in self.agent_ids}
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        width = self.env.local_size + self.env.public_size if assignment == 'aircraft' else 15 * self.env.config.aircraft + self.env.public_size
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.observation_spaces = {agent: spaces.Box(0, 1, shape=(width,), dtype=np.float32) for agent in self.agent_ids}
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.central_observation = None
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.coordination_deferrals = 0

    # Explanation: Define observations with the ownership, information, or update contract described below.
    def observations(self):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        views = mappo_agent_inputs(self.central_observation, self.env.config.aircraft,
                                   # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                                   self.env.local_size, self.env.public_size, self.assignment)[0].numpy()
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return {agent: views[i].copy() for i, agent in enumerate(self.agent_ids)}

    # Explanation: Define reset with the ownership, information, or update contract described below.
    def reset(self, *, seed=None, options=None):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.central_observation, info = self.env.reset(seed=seed, options=options)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.coordination_deferrals = 0
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        info.update(agent_ids=self.agent_ids, central_observation=self.central_observation.copy())
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return self.observations(), info

    # Explanation: Define compose_action with the ownership, information, or update contract described below.
    def compose_action(self, actions):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        """Merge owned requests and allocate resources using observed data only."""
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if self.central_observation is None:
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise RuntimeError('reset before composing agent proposals')
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if not isinstance(actions, dict) or set(actions) != set(self.agent_ids):
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise ValueError('supply exactly one proposal for each agent ID')
        # Explanation: Process each indicated agent, field, task, or rollout transition in order.
        for agent in self.agent_ids:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            proposal = np.asarray(actions[agent])
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if not np.issubdtype(proposal.dtype, np.integer) or not self.action_spaces[agent].contains(proposal):
                # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
                raise ValueError(f'invalid proposal for {agent}')
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        raw = np.stack([actions[agent] for agent in self.agent_ids])
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if self.assignment == 'aircraft':
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            bundles = raw.copy()
        # Explanation: Handle the alternative agent assignment or control-flow case.
        else:
            # Specialist c can set only component bit c, independently per engine.
            bundles = np.zeros((self.env.config.aircraft, 2), dtype=np.int64)
            # Explanation: Process each indicated agent, field, task, or rollout transition in order.
            for c in range(6):
                # Explanation: Process each indicated agent, field, task, or rollout transition in order.
                for engine in range(2):
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    bundles[:, engine] |= ((raw[c] >> engine) & 1) << c
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        rows, public = self.env.split_observation(self.central_observation)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        free_mask = sum(1 << i for i in range(self.env.config.crews) if public[10 + i] > 0.5)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        offset = 10 + 2 * self.env.config.crews
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        shops = [i for i in range(self.env.config.workshops) if public[offset + i] > 0.5]
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        stock = np.rint(public[4:10] * self.env.config.aircraft).astype(int)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        scores = rows[:, 24:36].min(1)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        rank = np.argsort(scores, kind='stable')
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        fleet_action = self.env.idle_action()
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        deferred = []
        # Explanation: Process each indicated agent, field, task, or rollout transition in order.
        for position, aircraft in enumerate(rank):
            # The broker collects per-aircraft criticality reports; it is not a critic.
            fleet_action[aircraft, 7] = self.env.config.aircraft - position - 1
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if rows[aircraft, 37] > 0.5 or not bundles[aircraft].any():
                # Explanation: Skip this proposal or iteration without committing resources.
                continue
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            b0, b1 = map(int, bundles[aircraft])
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            tasks = [(e, c) for e, b in enumerate((b0, b1)) for c in range(6) if b & (1 << c)]
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            needed = np.bincount([c for _, c in tasks], minlength=6)
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if not free_mask or not shops:
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                deferred.append((int(aircraft), 'resource_conflict'))
                # Explanation: Skip this proposal or iteration without committing resources.
                continue
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if np.any(needed > stock):
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                deferred.append((int(aircraft), 'spare_shortage'))
                # Explanation: Skip this proposal or iteration without committing resources.
                continue
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            crew = min((i for i in range(self.env.config.crews) if free_mask & (1 << i)), key=lambda i: self.env.crew_time_factors[i])
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            shop = min(shops, key=lambda i: self.env.workshop_time_factors[i])
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            first = min(tasks, key=lambda pair: rows[aircraft, pair[0] * 6 + pair[1]])
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            engines = int(b0 != 0) + 2 * int(b1 != 0)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            fleet_action[aircraft, :7] = [1, engines, b0, b1, 1 << crew, shop, first[0] * 6 + first[1]]
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            free_mask &= ~(1 << crew)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            shops.remove(shop)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            stock -= needed
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return fleet_action, deferred

    # Explanation: Define step with the ownership, information, or update contract described below.
    def step(self, actions):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        fleet_action, deferred = self.compose_action(actions)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.central_observation, reward, terminated, truncated, info = self.env.step(fleet_action)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.coordination_deferrals += len(deferred)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        info['metrics']['coordination_deferrals'] = self.coordination_deferrals
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        info.update(agent_ids=self.agent_ids, agent_proposals={key: np.asarray(value).tolist() for key, value in actions.items()},
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    fleet_action=fleet_action.tolist(), deferred_requests=deferred,
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    central_observation=self.central_observation.copy())
        # The cooperative reward is replicated, not divided by the number of agents.
        rewards = {agent: reward for agent in self.agent_ids}
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return self.observations(), rewards, terminated, truncated, info


# Explanation: Define MultiAgentActorCritic(nn.Module) as the multi-agent environment or model interface.
class MultiAgentActorCritic(nn.Module):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Feed-forward MAPPO prototype with explicit per-agent policy statistics.

    Aircraft agents share actor weights; six specialist actors have separate
    weights. The centralized critic has one value head per agent and receives
    the full observed fleet belief, never the simulator's hidden health.
    Actors can execute through policy(agent_observations) without the critic.
    """

    # Explanation: Define __init__ with the ownership, information, or update contract described below.
    def __init__(self, env, assignment='aircraft'):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        super().__init__()
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if assignment not in ('aircraft', 'component'):
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise ValueError('assignment must be aircraft or component')
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.assignment = assignment
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.n = env.config.aircraft
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.local = env.local_size
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.public = env.public_size
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.agent_ids = env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.num_agents = len(self.agent_ids)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.input_size = self.local + self.public if assignment == 'aircraft' else 15 * self.n + self.public
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.fields = 2 if assignment == 'aircraft' else self.n
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.categories = 64 if assignment == 'aircraft' else 4
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        count = 1 if assignment == 'aircraft' else 6
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.actors = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                                                  # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                                                  nn.Linear(64, 64), nn.Tanh(),
                                                  # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                                                  nn.Linear(64, self.fields * self.categories)) for _ in range(count)])
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        self.critic = nn.Sequential(nn.Linear(env.observation_space.shape[0], 64), nn.Tanh(),
                                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.num_agents))

    # Explanation: Define agent_inputs with the ownership, information, or update contract described below.
    def agent_inputs(self, obs):
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return mappo_agent_inputs(obs, self.n, self.local, self.public, self.assignment)

    # Explanation: Define policy with the ownership, information, or update contract described below.
    def policy(self, agent_obs, actions=None, deterministic=False):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        """Execute only restricted actors; no central observation or critic needed."""
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        x = torch.as_tensor(agent_obs, dtype=torch.float32)
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if x.ndim != 3 or x.shape[1:] != (self.num_agents, self.input_size):
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise ValueError('incorrect per-agent observation shape')
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if self.assignment == 'aircraft':
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            logits = self.actors[0](x).reshape(-1, self.num_agents, self.fields, self.categories)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            busy = (x[:, :, 37] > 0.5).unsqueeze(-1).expand(-1, -1, 2)
        # Explanation: Handle the alternative agent assignment or control-flow case.
        else:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            logits = torch.stack([actor(x[:, i]) for i, actor in enumerate(self.actors)], dim=1)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            logits = logits.reshape(-1, 6, self.n, 4)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            busy = x[:, :, :15 * self.n].reshape(-1, 6, self.n, 15)[:, :, :, 11] > 0.5
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        valid = (~busy).unsqueeze(-1) | (torch.arange(self.categories, device=x.device) == 0)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        distribution = Categorical(logits=logits.masked_fill(~valid, -1e9))
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if actions is None:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            selected = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        # Explanation: Handle the alternative agent assignment or control-flow case.
        else:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            selected = torch.as_tensor(actions, device=x.device)
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
                # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
                raise ValueError('incorrect stored agent action shape or dtype')
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if torch.any((selected < 0) | (selected >= self.categories)) or torch.any(busy & (selected != 0)):
                # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
                raise ValueError('stored agent action violates a category or busy mask')
        # Sum fields WITHIN each agent; preserve the agent axis for MAPPO clipping.
        return selected, distribution.log_prob(selected).sum(-1), distribution.entropy().sum(-1)

    # Explanation: Define active_agents with the ownership, information, or update contract described below.
    def active_agents(self, agent_obs):
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if self.assignment == 'aircraft':
            # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
            return (agent_obs[:, :, 37] <= 0.5).float()
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        busy = agent_obs[:, :, :15 * self.n].reshape(-1, 6, self.n, 15)[:, :, :, 11] > 0.5
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return (~busy).any(-1).float()

    # Explanation: Define value with the ownership, information, or update contract described below.
    def value(self, obs):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        x = torch.as_tensor(obs, dtype=torch.float32)
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return self.critic(x.unsqueeze(0) if x.ndim == 1 else x)

    # Explanation: Define act with the ownership, information, or update contract described below.
    def act(self, obs, actions=None, deterministic=False):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        selected, logp, entropy = self.policy(self.agent_inputs(obs), actions, deterministic)
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return selected, logp, entropy, self.value(obs)

    # Explanation: Define action_dict with the ownership, information, or update contract described below.
    def action_dict(self, selected):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        raw = torch.as_tensor(selected).detach().cpu().numpy()
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if raw.shape != (self.num_agents, self.fields):
            # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
            raise ValueError('incorrect unbatched agent action shape')
        # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
        return {agent: raw[i].copy() for i, agent in enumerate(self.agent_ids)}


# Explanation: Apply this ownership, validation, masking, or training step.
DCMAC_METHODS = ('dcmac_aircraft', 'ddmac_aircraft', 'dcmac_component', 'ddmac_component')
# Explanation: Apply this ownership, validation, masking, or training step.
REQUEST_METHODS = (*MAPPO_METHODS, *DCMAC_METHODS)


# Explanation: Define request_assignment with the stated policy or replay contract.
def request_assignment(method):
    # Explanation: Document the modeling assumptions and execution contract.
    """Resolve ownership for registered proposal methods; reject silent fallbacks."""
    # Explanation: Apply this ownership, validation, masking, or training step.
    if method not in REQUEST_METHODS:
        # Explanation: Apply this ownership, validation, masking, or training step.
        raise ValueError('unknown request policy method')
    # Explanation: Return the calculated policy, value, loss, or training result.
    return method.rsplit('_', 1)[1]


# Explanation: Define the global-belief DCMAC/DDMAC policy family.
class DCMACActorCritic(MultiAgentActorCritic):
    # Explanation: Construct or update factorized proposal policies from global observed beliefs.
    """Fleet adaptation of DCMAC/DDMAC; global observed beliefs enter all actors.

    DCMAC has one actor with factorized control-unit outputs. DDMAC has an
    independently parameterized actor per control unit. Both have one scalar
    centralized value network and use replay with joint importance weighting.
    The fixed dispatcher owns resources; paper-specific inspection actions,
    budget augmentation and primal-dual risk constraints are not reproduced.
    Inherits only the proposal-to-dictionary conversion from the MAPPO class.
    """

    # Explanation: Define __init__ with the stated policy or replay contract.
    def __init__(self, env, assignment='aircraft', decentralized=False):
        # Explanation: Apply this ownership, validation, masking, or training step.
        nn.Module.__init__(self)
        # Explanation: Apply this ownership, validation, masking, or training step.
        if assignment not in ('aircraft', 'component'):
            # Explanation: Apply this ownership, validation, masking, or training step.
            raise ValueError('assignment must be aircraft or component')
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.assignment = assignment
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.decentralized = decentralized
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.n = env.config.aircraft
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.local = env.local_size
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.public = env.public_size
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.agent_ids = env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.num_agents = len(self.agent_ids)
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.input_size = env.observation_space.shape[0]
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.fields = 2 if assignment == 'aircraft' else self.n
        # Explanation: Apply this ownership, validation, masking, or training step.
        self.categories = 64 if assignment == 'aircraft' else 4
        # Explanation: Apply this ownership, validation, masking, or training step.
        count = self.num_agents if decentralized else 1
        # Explanation: Apply this ownership, validation, masking, or training step.
        output = self.fields * self.categories * (1 if decentralized else self.num_agents)
        # Explanation: Construct or update factorized proposal policies from global observed beliefs.
        self.actors = nn.ModuleList([
            # Explanation: Apply this ownership, validation, masking, or training step.
            nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                          # Explanation: Apply this ownership, validation, masking, or training step.
                          nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, output))
            # Explanation: Apply this ownership, validation, masking, or training step.
            for _ in range(count)
        # Explanation: Apply this ownership, validation, masking, or training step.
        ])
        # Explanation: Calculate or update the common scalar fleet-return baseline.
        self.critic = nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                                    # Explanation: Apply this ownership, validation, masking, or training step.
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    # Explanation: Define agent_inputs with the stated policy or replay contract.
    def agent_inputs(self, obs):
        # Explanation: Construct or update factorized proposal policies from global observed beliefs.
        """Return global actor inputs; unlike MAPPO, execution needs fleet beliefs."""
        # Explanation: Apply this ownership, validation, masking, or training step.
        x = torch.as_tensor(obs, dtype=torch.float32)
        # Explanation: Apply this ownership, validation, masking, or training step.
        if x.ndim == 1:
            # Explanation: Apply this ownership, validation, masking, or training step.
            x = x.unsqueeze(0)
        # Explanation: Apply this ownership, validation, masking, or training step.
        if x.ndim != 2 or x.shape[1] != self.input_size:
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            raise ValueError('incorrect global actor observation shape')
        # Explanation: Return the calculated policy, value, loss, or training result.
        return x

    # Explanation: Define _busy_fields with the stated policy or replay contract.
    def _busy_fields(self, obs):
        # Explanation: Apply this ownership, validation, masking, or training step.
        busy = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)[:, :, 37] > 0.5
        # Explanation: Apply this ownership, validation, masking, or training step.
        if self.assignment == 'aircraft':
            # Explanation: Return the calculated policy, value, loss, or training result.
            return busy.unsqueeze(-1).expand(-1, -1, 2)
        # Explanation: Return the calculated policy, value, loss, or training result.
        return busy.unsqueeze(1).expand(-1, 6, -1)

    # Explanation: Define active_agents with the stated policy or replay contract.
    def active_agents(self, obs):
        # Explanation: Return the calculated policy, value, loss, or training result.
        return (~self._busy_fields(self.agent_inputs(obs))).any(-1).float()

    # Explanation: Define policy with the stated policy or replay contract.
    def policy(self, obs, actions=None, deterministic=False):
        # Explanation: Calculate or update the common scalar fleet-return baseline.
        """Factorized categorical proposals; no hidden state or critic evaluation."""
        # Explanation: Apply this ownership, validation, masking, or training step.
        x = self.agent_inputs(obs)
        # Explanation: Apply this ownership, validation, masking, or training step.
        if self.decentralized:
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            logits = torch.stack([actor(x) for actor in self.actors], dim=1)
        # Explanation: Apply this ownership, validation, masking, or training step.
        else:
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            logits = self.actors[0](x)
        # Explanation: Construct or update factorized proposal policies from global observed beliefs.
        logits = logits.reshape(-1, self.num_agents, self.fields, self.categories)
        # Explanation: Apply this ownership, validation, masking, or training step.
        busy = self._busy_fields(x)
        # Explanation: Apply this ownership, validation, masking, or training step.
        valid = (~busy).unsqueeze(-1) | (torch.arange(self.categories, device=x.device) == 0)
        # Explanation: Construct or update factorized proposal policies from global observed beliefs.
        distribution = Categorical(logits=logits.masked_fill(~valid, -1e9))
        # Explanation: Apply this ownership, validation, masking, or training step.
        if actions is None:
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            selected = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        # Explanation: Apply this ownership, validation, masking, or training step.
        else:
            # Explanation: Apply this ownership, validation, masking, or training step.
            selected = torch.as_tensor(actions, device=x.device)
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
                # Explanation: Apply this ownership, validation, masking, or training step.
                raise ValueError('incorrect stored agent action shape or dtype')
            # Explanation: Apply this ownership, validation, masking, or training step.
            if torch.any((selected < 0) | (selected >= self.categories)) or torch.any(busy & (selected != 0)):
                # Explanation: Apply this ownership, validation, masking, or training step.
                raise ValueError('stored agent action violates a category or busy mask')
        # Explanation: Return the calculated policy, value, loss, or training result.
        return selected, distribution.log_prob(selected).sum(-1), distribution.entropy().sum(-1)

    # Explanation: Define value with the stated policy or replay contract.
    def value(self, obs):
        # Explanation: Return the calculated policy, value, loss, or training result.
        return self.critic(self.agent_inputs(obs)).squeeze(-1)

    # Explanation: Define act with the stated policy or replay contract.
    def act(self, obs, actions=None, deterministic=False):
        # Explanation: Apply this ownership, validation, masking, or training step.
        selected, logp, entropy = self.policy(obs, actions, deterministic)
        # Explanation: Return the calculated policy, value, loss, or training result.
        return selected, logp, entropy, self.value(obs)


# Explanation: Define dcmac_losses with the stated policy or replay contract.
def dcmac_losses(logp, behavior_logp, values, next_values, rewards, dones,
                 # Explanation: Use the joint behavior-to-target likelihood correction for replay.
                 gamma=0.99, importance_clip=2.0):
    # Explanation: Store or sample original proposal transitions and their collection days.
    """One-step replay update: joint truncated ratio and detached TD advantage.

    The ratio is pi(joint proposal|belief)/mu(joint proposal|belief), not the
    product of separately truncated ratios. Truncation introduces bias but
    bounds the high-variance replay weights. No PPO clipping or GAE is used.
    """
    # Explanation: Apply this ownership, validation, masking, or training step.
    if logp.ndim != 2 or behavior_logp.shape != logp.shape:
        # Explanation: Apply this ownership, validation, masking, or training step.
        raise ValueError('proposal log probabilities must have batch x agent shape')
    # Explanation: Calculate or update the common scalar fleet-return baseline.
    if any(t.shape != (len(logp),) for t in (values, next_values, rewards, dones)):
        # Explanation: Store or sample original proposal transitions and their collection days.
        raise ValueError('one scalar value, reward and terminal flag per replay transition required')
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    if not np.isfinite(importance_clip) or importance_clip <= 0 or not 0 <= gamma <= 1:
        # Explanation: Use the joint behavior-to-target likelihood correction for replay.
        raise ValueError('positive finite importance cap and gamma in [0,1] required')
    # Explanation: Calculate or update the common scalar fleet-return baseline.
    if not all(bool(torch.isfinite(t).all()) for t in (logp, behavior_logp, values, next_values, rewards, dones)):
        # Explanation: Store or sample original proposal transitions and their collection days.
        raise ValueError('replay tensors must be finite')
    # Explanation: Apply this ownership, validation, masking, or training step.
    joint_logp = logp.sum(-1)
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    log_ratio = (joint_logp - behavior_logp.sum(-1)).detach()
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    weights = torch.exp(log_ratio.clamp(max=float(np.log(importance_clip))))
    # Explanation: Calculate or update the common scalar fleet-return baseline.
    target = rewards + gamma * (1 - dones.float()) * next_values.detach()
    # Explanation: Calculate or update the common scalar fleet-return baseline.
    delta = target - values
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    actor_loss = -(weights * joint_logp * delta.detach()).mean()
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    critic_loss = 0.5 * (weights * delta.square()).mean()
    # Explanation: Return the calculated policy, value, loss, or training result.
    return actor_loss, critic_loss, weights, target.detach()


# Explanation: Define train_dcmac with the stated policy or replay contract.
def train_dcmac(config, method='dcmac_aircraft', steps=4096, seed=0,
                # Explanation: Use the joint behavior-to-target likelihood correction for replay.
                rollout_size=128, replay_capacity=4096, importance_clip=2.0):
    # Explanation: Store or sample original proposal transitions and their collection days.
    """Off-policy DCMAC/DDMAC adaptation with uniform bounded experience replay.

    Stochastic actor sampling is the behavior policy; its original proposal
    likelihood is stored even when the dispatcher defers that proposal.
    One replay minibatch is updated per day after two samples are available.
    rollout_size names minibatch size here, preserving the common train API.
    """
    # Explanation: Store or sample original proposal transitions and their collection days.
    if method not in DCMAC_METHODS or steps < 1 or rollout_size < 2 or replay_capacity < 2:
        # Explanation: Apply this ownership, validation, masking, or training step.
        raise ValueError('registered DCMAC/DDMAC method, positive steps and batch/capacity >= 2 required')
    # Explanation: Use the joint behavior-to-target likelihood correction for replay.
    if not np.isfinite(importance_clip) or importance_clip <= 0:
        # Explanation: Use the joint behavior-to-target likelihood correction for replay.
        raise ValueError('importance_clip must be positive and finite')
    # Explanation: Apply this ownership, validation, masking, or training step.
    torch.set_num_threads(1)
    # Explanation: Apply this ownership, validation, masking, or training step.
    torch.manual_seed(seed)
    # Explanation: Apply this ownership, validation, masking, or training step.
    rng = np.random.default_rng(seed)
    # Explanation: Apply this ownership, validation, masking, or training step.
    assignment = request_assignment(method)
    # Explanation: Apply this ownership, validation, masking, or training step.
    env = MultiAgentFleetEnv(config, assignment)
    # Explanation: Apply this ownership, validation, masking, or training step.
    model = DCMACActorCritic(env.env, assignment, method.startswith('ddmac_'))
    # Explanation: Construct or update factorized proposal policies from global observed beliefs.
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    # Explanation: Calculate or update the common scalar fleet-return baseline.
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    # Explanation: Store or sample original proposal transitions and their collection days.
    replay, logs = [], []
    # Explanation: Apply this ownership, validation, masking, or training step.
    _, info = env.reset(seed=seed)
    # Explanation: Apply this ownership, validation, masking, or training step.
    obs = info['central_observation']
    # Explanation: Apply this ownership, validation, masking, or training step.
    episode = 0
    # Explanation: Apply this ownership, validation, masking, or training step.
    for day in range(1, steps + 1):
        # Explanation: Apply this ownership, validation, masking, or training step.
        with torch.no_grad():
            # Explanation: Apply this ownership, validation, masking, or training step.
            action, behavior_logp, _ = model.policy(obs)
        # Explanation: Apply this ownership, validation, masking, or training step.
        _, rewards, done, truncated, info = env.step(model.action_dict(action[0]))
        # Explanation: Apply this ownership, validation, masking, or training step.
        next_obs = info['central_observation']
        # Explanation: Apply this ownership, validation, masking, or training step.
        terminal = done or truncated
        # Explanation: Store or sample original proposal transitions and their collection days.
        transition = (obs.copy(), action[0].numpy().copy(), behavior_logp[0].numpy().copy(),
                      # Explanation: Apply this ownership, validation, masking, or training step.
                      rewards[model.agent_ids[0]], next_obs.copy(), terminal, day)
        # Explanation: Store or sample original proposal transitions and their collection days.
        if len(replay) < replay_capacity:
            # Explanation: Store or sample original proposal transitions and their collection days.
            replay.append(transition)
        # Explanation: Apply this ownership, validation, masking, or training step.
        else:
            # Explanation: Store or sample original proposal transitions and their collection days.
            replay[(day - 1) % replay_capacity] = transition
        # Explanation: Store or sample original proposal transitions and their collection days.
        if len(replay) >= 2:
            # Explanation: Store or sample original proposal transitions and their collection days.
            indices = rng.choice(len(replay), size=min(rollout_size, len(replay)), replace=False)
            # Explanation: Store or sample original proposal transitions and their collection days.
            batch = [replay[int(i)] for i in indices]
            # Explanation: Store or sample original proposal transitions and their collection days.
            states, actions, behavior, reward, following, terminal_flags, sampled_days = [
                # Explanation: Apply this ownership, validation, masking, or training step.
                torch.as_tensor(np.asarray(items)) for items in zip(*batch)
            # Explanation: Apply this ownership, validation, masking, or training step.
            ]
            # Explanation: Apply this ownership, validation, masking, or training step.
            _, logp, entropy = model.policy(states, actions)
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            values = model.value(states)
            # Explanation: Apply this ownership, validation, masking, or training step.
            with torch.no_grad():
                # Explanation: Calculate or update the common scalar fleet-return baseline.
                next_values = model.value(following)
            # Explanation: Use the joint behavior-to-target likelihood correction for replay.
            actor_loss, value_loss, weights, _ = dcmac_losses(
                # Explanation: Calculate or update the common scalar fleet-return baseline.
                logp, behavior, values, next_values, reward.float(), terminal_flags,
                # Explanation: Use the joint behavior-to-target likelihood correction for replay.
                importance_clip=importance_clip
            # Explanation: Apply this ownership, validation, masking, or training step.
            )
            # Explanation: Apply this ownership, validation, masking, or training step.
            active = model.active_agents(states)
            # Explanation: Apply this ownership, validation, masking, or training step.
            entropy_bonus = (entropy * active).sum() / active.sum().clamp_min(1)
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            policy_loss = actor_loss - 0.001 * entropy_bonus
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            if not bool(torch.isfinite(policy_loss + value_loss)):
                # Explanation: Apply this ownership, validation, masking, or training step.
                raise RuntimeError('nonfinite DCMAC/DDMAC loss')
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            actor_optimizer.zero_grad()
            # Explanation: Apply this ownership, validation, masking, or training step.
            policy_loss.backward()
            # Explanation: Construct or update factorized proposal policies from global observed beliefs.
            torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
            # Explanation: Apply this ownership, validation, masking, or training step.
            if bool(active.any()):
                # Explanation: Construct or update factorized proposal policies from global observed beliefs.
                actor_optimizer.step()
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            critic_optimizer.zero_grad()
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            value_loss.backward()
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            # Explanation: Calculate or update the common scalar fleet-return baseline.
            critic_optimizer.step()
            # Explanation: Apply this ownership, validation, masking, or training step.
            logs.append({'record_type': 'update', 'method': method, 'simulated_days': day,
                         # Explanation: Calculate or update the common scalar fleet-return baseline.
                         'policy_loss': float(actor_loss.detach()), 'value_loss': float(value_loss.detach()),
                         # Explanation: Use the joint behavior-to-target likelihood correction for replay.
                         'importance_mean': float(weights.mean()), 'importance_max': float(weights.max()),
                         # Explanation: Store or sample original proposal transitions and their collection days.
                         'replay_size': len(replay), 'batch_size': len(batch),
                         # Explanation: Store or sample original proposal transitions and their collection days.
                         'sampled_transition_days': sampled_days.tolist(),
                         # Explanation: Apply this ownership, validation, masking, or training step.
                         'num_agents': model.num_agents})
        # Explanation: Apply this ownership, validation, masking, or training step.
        obs = next_obs
        # Explanation: Apply this ownership, validation, masking, or training step.
        if terminal:
            # Explanation: Apply this ownership, validation, masking, or training step.
            logs.append({'record_type': 'episode', 'episode': episode, 'simulated_days': day, **info['metrics']})
            # Explanation: Apply this ownership, validation, masking, or training step.
            episode += 1
            # Explanation: Apply this ownership, validation, masking, or training step.
            _, info = env.reset(seed=seed + episode)
            # Explanation: Apply this ownership, validation, masking, or training step.
            obs = info['central_observation']
    # Explanation: Return the calculated policy, value, loss, or training result.
    return model, logs, steps


# Explanation: Define mappo_gae with the ownership, information, or update contract described below.
def mappo_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Daily generalized advantages with a separate value baseline per agent."""
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    values = np.asarray(values, dtype=np.float32)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    next_values = np.asarray(next_values, dtype=np.float32)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    rewards = np.asarray(rewards, dtype=np.float32)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    dones = np.asarray(dones, dtype=np.float32)
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if values.ndim != 2 or next_values.shape != values.shape or rewards.shape != (len(values),) or dones.shape != rewards.shape:
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('MAPPO rollout shapes must be time x agent, with one team reward and done per time')
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    advantages = np.zeros_like(values)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    carry = np.zeros(values.shape[1], dtype=np.float32)
    # Explanation: Process each indicated agent, field, task, or rollout transition in order.
    for t in range(len(rewards) - 1, -1, -1):
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        live = 1 - dones[t]
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        delta = rewards[t] + gamma * next_values[t] * live - values[t]
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        carry = delta + gamma * lam * live * carry
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        advantages[t] = carry
    # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
    return advantages, advantages + values


# Explanation: Define mappo_policy_loss with the ownership, information, or update contract described below.
def mappo_policy_loss(new_logp, old_logp, advantages, active, clip=0.2):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Clip each agent ratio BEFORE averaging; never form a fleet joint ratio."""
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if new_logp.ndim != 2 or not (new_logp.shape == old_logp.shape == advantages.shape == active.shape):
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('MAPPO policy tensors must have identical batch x agent shapes')
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    ratio = torch.exp(new_logp - old_logp)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
    return -(surrogate * active).sum() / active.sum().clamp_min(1)


# Explanation: Define train_mappo with the ownership, information, or update contract described below.
def train_mappo(config, method='mappo_aircraft', steps=4096, seed=0, rollout_size=128):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Train decentralized proposals with centralized observed-belief value heads.

    Educational adaptation of MAPPO, not an exact benchmark reproduction:
    feed-forward actors, four full-rollout epochs, clipped values, no recurrence,
    reward/value normalization, learned communication, or hierarchical options.
    """
    # Explanation: Check this validity or control-flow condition before executing its branch.
    if method not in MAPPO_METHODS or steps < 1 or rollout_size < 2:
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('valid MAPPO method, positive steps, and rollout_size >= 2 required')
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    torch.set_num_threads(1)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    torch.manual_seed(seed)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    assignment = 'aircraft' if method == 'mappo_aircraft' else 'component'
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    env = MultiAgentFleetEnv(config, assignment)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    model = MultiAgentActorCritic(env.env, assignment)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    _, info = env.reset(seed=seed)
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    obs = info['central_observation']
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    days, episode = 0, 0
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    logs = []
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    while days < steps:
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        data = {key: [] for key in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'done', 'active')}
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        while len(data['obs']) < rollout_size and days < steps:
            # Explanation: Control gradient recording or resource lifetime for this block.
            with torch.no_grad():
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                action, logp, _, values = model.act(torch.tensor(obs))
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                active = model.active_agents(model.agent_inputs(obs))[0]
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            _, rewards, done, truncated, info = env.step(model.action_dict(action[0]))
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            next_obs = info['central_observation']
            # Explanation: Control gradient recording or resource lifetime for this block.
            with torch.no_grad():
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                next_value = model.value(next_obs)[0].numpy()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            team_reward = rewards[model.agent_ids[0]]
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            terminal = done or truncated
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            items = (obs.copy(), action[0].numpy(), logp[0].numpy(), values[0].numpy(), next_value, team_reward, terminal, active.numpy())
            # Explanation: Process each indicated agent, field, task, or rollout transition in order.
            for key, item in zip(data, items):
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                data[key].append(item)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            days += 1
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            obs = next_obs
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if terminal:
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                logs.append({'record_type': 'episode', 'episode': episode, 'simulated_days': days, **info['metrics']})
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                episode += 1
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                _, info = env.reset(seed=seed + episode)
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                obs = info['central_observation']
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        advantages, targets = mappo_gae(data['reward'], data['value'], data['next_value'], data['done'])
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        tensors = {key: torch.as_tensor(np.asarray(value)) for key, value in data.items() if key in ('obs', 'action', 'logp', 'value', 'active')}
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        advantages = torch.tensor(advantages)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        targets = torch.tensor(targets)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        active = tensors['active']
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        counts = active.sum(0).clamp_min(1)
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        mean = (advantages * active).sum(0) / counts
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        variance = ((advantages - mean).square() * active).sum(0) / counts
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        advantages = (advantages - mean) / torch.sqrt(variance + 1e-8) * active
        # Explanation: Process each indicated agent, field, task, or rollout transition in order.
        for _ in range(4):
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            _, newlog, entropy, values = model.act(tensors['obs'], tensors['action'])
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            policy_loss = mappo_policy_loss(newlog, tensors['logp'], advantages, active)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            entropy_bonus = (entropy * active).sum() / active.sum().clamp_min(1)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            clipped_values = tensors['value'] + (values - tensors['value']).clamp(-0.2, 0.2)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            value_loss = torch.maximum((values - targets).square(), (clipped_values - targets).square()).mean()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            actor_loss = policy_loss - 0.001 * entropy_bonus
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if not torch.isfinite(actor_loss + value_loss):
                # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
                raise RuntimeError('nonfinite MAPPO loss')
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            actor_optimizer.zero_grad()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            actor_loss.backward()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            # Explanation: Do not apply optimizer momentum when every agent has only a forced no-op action.
            if bool(active.any()):
                # Explanation: Update actor weights only when this rollout contains a genuine proposal choice.
                actor_optimizer.step()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            critic_optimizer.zero_grad()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            (0.5 * value_loss).backward()
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            critic_optimizer.step()
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        logs.append({'record_type': 'update', 'simulated_days': days, 'policy_loss': float(policy_loss.detach()),
                     # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                     'value_loss': float(value_loss.detach()), 'entropy': float(entropy_bonus.detach()),
                     # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                     'num_agents': model.num_agents, 'method': method})
    # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
    return model, logs, days

# Explanation: Define shared module constant OPTION_NAMES; other examples reuse this value.
OPTION_NAMES = ('wait_one_day', 'targeted_maintenance', 'opportunistic_bundle', 'preventive_bundle')

# Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
HRL_METHODS = ('concurrent_options', 'planning_hierarchy', 'aggregate_targets',
               # Explanation: Apply this observation, planning, bookkeeping, or validation step.
               'maxq', 'option_critic', 'hierarchical_vdn', 'hrl_flat')
# Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
HRL_PPO_METHODS = ('concurrent_options', 'planning_hierarchy', 'aggregate_targets', 'hrl_flat')
# Explanation: Apply this observation, planning, bookkeeping, or validation step.
HRL_BUNDLES = (1, 2, 4, 8, 16, 32, 63)
# Explanation: Apply this observation, planning, bookkeeping, or validation step.
HRL_CATALOG = ((0, 0),) + tuple((b if e & 1 else 0, b if e & 2 else 0)
                                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                                for e in (1, 2, 3) for b in HRL_BUNDLES)
# Explanation: Apply this observation, planning, bookkeeping, or validation step.
HRL_BUDGETS = (0, 1, 2, 12)


# Explanation: Define hrl_mlp for the stated hierarchy or learning contract.
def hrl_mlp(width, output):
    # Explanation: Return the computed observation, choice, value, or training result.
    return nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64),
                         # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                         nn.Tanh(), nn.Linear(64, output))


# Explanation: Define hrl_catalog_mask for the stated hierarchy or learning contract.
def hrl_catalog_mask(observation, aircraft, local_size=70, mandatory=0.08, goals=None):
    # Explanation: Document the method semantics and limits.
    """Structural masks only; the shared broker resolves stock/resource competition."""
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    obs = torch.as_tensor(observation, dtype=torch.float32)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if obs.ndim == 1:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs = obs.unsqueeze(0)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    rows = obs[:, :aircraft * local_size].reshape(-1, aircraft, local_size)
    # Explanation: Preserve action feasibility and the commitment boundary.
    busy = rows[:, :, 37] > 0.5
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    urgent = rows[:, :, 24:36].amin(-1) <= mandatory
    # Explanation: Preserve action feasibility and the commitment boundary.
    valid = torch.ones((*busy.shape, len(HRL_CATALOG)), dtype=torch.bool, device=obs.device)
    # Explanation: Preserve action feasibility and the commitment boundary.
    valid[:, :, 0] = ~urgent | busy
    # Explanation: Preserve action feasibility and the commitment boundary.
    valid[:, :, 1:] &= ~busy.unsqueeze(-1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if goals is not None:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        goals = torch.as_tensor(goals, device=obs.device, dtype=torch.long)
        # Explanation: Preserve action feasibility and the commitment boundary.
        if goals.shape != busy.shape or torch.any((goals < 0) | (goals >= len(HRL_BUDGETS))):
            # Explanation: Preserve action feasibility and the commitment boundary.
            raise ValueError('one valid budget goal per aircraft required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        counts = torch.tensor([sum(int(b).bit_count() for b in pair) for pair in HRL_CATALOG], device=obs.device)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        budgets = torch.tensor(HRL_BUDGETS, device=obs.device)[goals]
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid &= counts <= budgets.unsqueeze(-1)
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid[:, :, 0] |= busy
    # Explanation: Return the computed observation, choice, value, or training result.
    return valid


# Explanation: Define hrl_categorical for the stated hierarchy or learning contract.
def hrl_categorical(logits, valid, selected=None, deterministic=False):
    # Explanation: Preserve action feasibility and the commitment boundary.
    if logits.shape != valid.shape or not bool(valid.any(-1).all()):
        # Explanation: Preserve action feasibility and the commitment boundary.
        raise ValueError('categorical choices require matching masks with a feasible option')
    # Explanation: Preserve action feasibility and the commitment boundary.
    dist = Categorical(logits=logits.masked_fill(~valid, -1e9))
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if selected is None:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        selected = dist.probs.argmax(-1) if deterministic else dist.sample()
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    else:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        selected = torch.as_tensor(selected, device=logits.device)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('incorrect hierarchical stored choice shape/dtype')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if torch.any((selected < 0) | (selected >= logits.shape[-1])):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('hierarchical choice outside action categories')
        # Explanation: Preserve action feasibility and the commitment boundary.
        if not bool(valid.gather(-1, selected.unsqueeze(-1)).all()):
            # Explanation: Preserve action feasibility and the commitment boundary.
            raise ValueError('hierarchical choice violates a feasibility mask')
    # Explanation: Return the computed observation, choice, value, or training result.
    return selected, dist.log_prob(selected), dist.entropy()


# Explanation: Define this aircraft-option, policy, or value-decomposition implementation.
class AircraftOptionEnv:
    # Explanation: Document the method semantics and limits.
    """Daily fleet interface with independently committed aircraft repair options.

    Catalog 0 operates; other entries request one engine, the other, or both,
    with one component type or a full six-component bundle. Both engines use
    the same bundle if selected. This is a strict subset of the older action
    space. Every new method shares this catalog and the same fixed broker.
    A new decision for a free aircraft never waits for another aircraft's job.
    """

    # Explanation: Define __init__ for the stated hierarchy or learning contract.
    def __init__(self, config=None):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.adapter = MultiAgentFleetEnv(config, 'aircraft')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.env = self.adapter.env
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.n = self.env.config.aircraft
        # Explanation: Calculate or apply this level-specific learning update.
        self.active_options = {}
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.option_history = []
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.central_observation = None

    # Explanation: Define reset for the stated hierarchy or learning contract.
    def reset(self, *, seed=None, options=None):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        _, info = self.adapter.reset(seed=seed, options=options)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.central_observation = info['central_observation']
        # Explanation: Calculate or apply this level-specific learning update.
        self.active_options.clear()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.option_history.clear()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.last_info = info
        # Explanation: Return the computed observation, choice, value, or training result.
        return self.central_observation.copy(), info

    # Explanation: Define decision_mask for the stated hierarchy or learning contract.
    def decision_mask(self):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if self.central_observation is None:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise RuntimeError('reset before reading aircraft options')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        rows, _ = self.env.split_observation(self.central_observation)
        # Explanation: Return the computed observation, choice, value, or training result.
        return rows[:, 37] <= 0.5

    # Explanation: Define step for the stated hierarchy or learning contract.
    def step(self, choices):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        choices = np.asarray(choices)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if choices.shape != (self.n,) or not np.issubdtype(choices.dtype, np.integer):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('one integer catalog action per aircraft required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if np.any((choices < 0) | (choices >= len(HRL_CATALOG))):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('unknown maintenance catalog action')
        # Explanation: Preserve action feasibility and the commitment boundary.
        free = self.decision_mask()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if np.any((~free) & (choices != 0)):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('committed aircraft options accept no new planning request')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        previous = {i: job.plan for i, job in self.env.jobs.items()}
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        day = self.env.day
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        proposals = {plate: np.asarray(HRL_CATALOG[int(choices[i])], dtype=np.int64)
                     # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                     for i, plate in enumerate(self.env.aircraft_plates)}
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        _, rewards, done, truncated, info = self.adapter.step(proposals)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for i, plan in previous.items():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if i in self.env.jobs and self.env.jobs[i].plan != plan:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                raise RuntimeError('a committed aircraft plan was changed')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        started, finished = [], []
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for i, job in self.env.jobs.items():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if i not in previous:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                record = {'aircraft': i, 'aircraft_plate': self.env.aircraft_plates[i],
                          # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                          'start_day': day, 'catalog_action': int(choices[i]),
                          # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                          'planned_duration': job.total_duration}
                # Explanation: Calculate or apply this level-specific learning update.
                self.active_options[i] = record
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                started.append(dict(record))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for i in info['completed']:
            # Explanation: Calculate or apply this level-specific learning update.
            record = dict(self.active_options.pop(i))
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            record.update(end_day=self.env.day, duration=self.env.day - record['start_day'])
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            self.option_history.append(record)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            finished.append(record)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.central_observation = info['central_observation']
        # Explanation: Calculate or apply this level-specific learning update.
        info.update(aircraft_option_started=started, aircraft_option_completed=finished,
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    decision_mask_before=free.tolist(), active_aircraft_options=sorted(self.active_options))
        # Explanation: Calculate or apply this level-specific learning update.
        info['metrics'].update(option_starts=len(self.option_history) + len(self.active_options),
                               # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                               option_completions=len(self.option_history))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.last_info = info
        # Explanation: Return the computed observation, choice, value, or training result.
        return self.central_observation.copy(), rewards[self.env.aircraft_plates[0]], done, truncated, info


# Explanation: Define this aircraft-option, policy, or value-decomposition implementation.
class AircraftPlanningHierarchy(nn.Module):
    # Explanation: Document the method semantics and limits.
    """Independent aircraft options with learned planning stages or explicit targets.

    Ground/engine/bundle selections are computational planning stages, not
    three elapsed physical epochs. Temporal abstraction is the committed job.
    All actors use observed global context; this is not local-information CTDE.
    """

    # Explanation: Define __init__ for the stated hierarchy or learning contract.
    def __init__(self, env, mode='planning_hierarchy'):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        super().__init__()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if mode not in HRL_PPO_METHODS:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('unknown aircraft planning hierarchy')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.mode = mode
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.n = env.config.aircraft
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.local = env.local_size
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.width = env.observation_space.shape[0]
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.mandatory = env.config.mandatory_health
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        features = self.width + self.local
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if mode == 'hrl_flat':
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            self.flat_policy = hrl_mlp(features, len(HRL_CATALOG))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        else:
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            self.manager = hrl_mlp(features, 4 if mode == 'aggregate_targets' else 2)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if mode == 'planning_hierarchy':
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                self.engine_policy = hrl_mlp(features + 2, 3)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                self.component_policy = hrl_mlp(features + 5, 7)
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            if mode == 'aggregate_targets':
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                self.worker = hrl_mlp(features + 4, len(HRL_CATALOG))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.critic = hrl_mlp(self.width, self.n)

    # Explanation: Define features for the stated hierarchy or learning contract.
    def features(self, observation):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs = torch.as_tensor(observation, dtype=torch.float32)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if obs.ndim == 1:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs = obs.unsqueeze(0)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if obs.ndim != 2 or obs.shape[1] != self.width:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('incorrect hierarchical fleet observation')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        local = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)
        # Explanation: Return the computed observation, choice, value, or training result.
        return obs, local, torch.cat([obs[:, None].expand(-1, self.n, -1), local], -1)

    # Explanation: Define value for the stated hierarchy or learning contract.
    def value(self, observation):
        # Explanation: Return the computed observation, choice, value, or training result.
        return self.critic(self.features(observation)[0])

    # Explanation: Define actor_parameters for the stated hierarchy or learning contract.
    def actor_parameters(self):
        # Explanation: Return the computed observation, choice, value, or training result.
        return [p for name, p in self.named_parameters() if not name.startswith('critic.')]

    # Explanation: Define act for the stated hierarchy or learning contract.
    def act(self, observation, actions=None, deterministic=False):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs, local, features = self.features(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if actions is not None:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            actions = torch.as_tensor(actions)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if actions.shape != (len(obs), self.n, 3) or actions.dtype not in (torch.int32, torch.int64):
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                raise ValueError('hierarchical actions must be integer batch x aircraft x 3')
        # Explanation: Preserve action feasibility and the commitment boundary.
        mask = hrl_catalog_mask(obs, self.n, self.local, self.mandatory)
        # Explanation: Preserve action feasibility and the commitment boundary.
        busy = local[:, :, 37] > 0.5
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        urgent = local[:, :, 24:36].amin(-1) <= self.mandatory
        # Explanation: Preserve action feasibility and the commitment boundary.
        zero = torch.zeros_like(busy, dtype=torch.long)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if self.mode == 'hrl_flat':
            # Explanation: Preserve action feasibility and the commitment boundary.
            choice, logp, entropy = hrl_categorical(self.flat_policy(features), mask,
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                None if actions is None else actions[:, :, 0], deterministic)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            stored = torch.stack([choice, zero, zero], -1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        else:
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            count = 4 if self.mode == 'aggregate_targets' else 2
            # Explanation: Preserve action feasibility and the commitment boundary.
            valid = (~busy).unsqueeze(-1).expand(-1, -1, count).clone()
            # Explanation: Preserve action feasibility and the commitment boundary.
            valid[:, :, 0] = ~urgent | busy
            # Explanation: Preserve action feasibility and the commitment boundary.
            goal, logp, entropy = hrl_categorical(self.manager(features), valid,
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                None if actions is None else actions[:, :, 0], deterministic)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if self.mode == 'planning_hierarchy':
                # Explanation: Preserve action feasibility and the commitment boundary.
                active = (goal == 1) & ~busy
                # Explanation: Preserve action feasibility and the commitment boundary.
                engine_valid = active.unsqueeze(-1).expand(-1, -1, 3).clone()
                # Explanation: Preserve action feasibility and the commitment boundary.
                engine_valid[:, :, 0] |= ~active
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                engine, lp, ent = hrl_categorical(self.engine_policy(torch.cat([
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    features, nn.functional.one_hot(goal, 2).float()], -1)), engine_valid,
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    None if actions is None else actions[:, :, 1], deterministic)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                logp, entropy = logp + lp, entropy + ent
                # Explanation: Preserve action feasibility and the commitment boundary.
                bundle_valid = active.unsqueeze(-1).expand(-1, -1, 7).clone()
                # Explanation: Preserve action feasibility and the commitment boundary.
                bundle_valid[:, :, 0] |= ~active
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                bundle, lp, ent = hrl_categorical(self.component_policy(torch.cat([
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    features, nn.functional.one_hot(goal, 2).float(),
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    nn.functional.one_hot(engine, 3).float()], -1)), bundle_valid,
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    None if actions is None else actions[:, :, 2], deterministic)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                logp, entropy = logp + lp, entropy + ent
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                stored = torch.stack([goal, engine, bundle], -1)
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            elif self.mode == 'aggregate_targets':
                # A positive goal is a maximum replacement count, not a guaranteed improvement.
                # Explanation: Preserve action feasibility and the commitment boundary.
                worker_valid = hrl_catalog_mask(obs, self.n, self.local, self.mandatory, goal)
                # Explanation: Preserve action feasibility and the commitment boundary.
                worker_valid[:, :, 0] = (goal == 0) | busy
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                choice, lp, ent = hrl_categorical(self.worker(torch.cat([
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    features, nn.functional.one_hot(goal, 4).float()], -1)), worker_valid,
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    None if actions is None else actions[:, :, 1], deterministic)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                logp, entropy = logp + lp, entropy + ent
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                stored = torch.stack([goal, choice, zero], -1)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            else:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                stored = torch.stack([goal, zero, zero], -1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if actions is not None and not torch.equal(stored, actions):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('unused hierarchical action fields must be zero')
        # Explanation: Return the computed observation, choice, value, or training result.
        return stored, logp, entropy, self.value(obs)

    # Explanation: Define catalog_actions for the stated hierarchy or learning contract.
    def catalog_actions(self, observation, actions):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        actions = torch.as_tensor(actions)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if actions.ndim == 3:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if actions.shape[0] != 1:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                raise ValueError('decode one fleet action at a time')
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            actions = actions[0]
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if actions.shape != (self.n, 3):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('incorrect hierarchical action for decoding')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if self.mode == 'hrl_flat':
            # Explanation: Return the computed observation, choice, value, or training result.
            return actions[:, 0].numpy().copy()
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        if self.mode == 'aggregate_targets':
            # Explanation: Return the computed observation, choice, value, or training result.
            return actions[:, 1].numpy().copy()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if self.mode == 'planning_hierarchy':
            # Explanation: Return the computed observation, choice, value, or training result.
            return torch.where(actions[:, 0] != 0, 1 + actions[:, 1] * 7 + actions[:, 2], 0).numpy()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        _, local, _ = self.features(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        means = local[0, :, :12].reshape(self.n, 2, 6)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        engine = means.amin(-1).argmin(-1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        selected = means[torch.arange(self.n), engine]
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        component = selected.argmin(-1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        bundle = torch.where((selected < .75).sum(-1) >= 3, 6, component)
        # Explanation: Return the computed observation, choice, value, or training result.
        return torch.where(actions[:, 0] != 0, 1 + engine * 7 + bundle, 0).numpy()


# Explanation: Define hrl_option_target for the stated hierarchy or learning contract.
def hrl_option_target(rewards, next_value, terminal, gamma=.99):
    # Explanation: Document the method semantics and limits.
    """Discount each primitive day once, including bootstrap over the actual duration."""
    # Explanation: Return the computed observation, choice, value, or training result.
    return sum(gamma ** t * float(r) for t, r in enumerate(rewards)) + (
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        0. if terminal else gamma ** len(rewards) * float(next_value))


# Explanation: Define train_aircraft_hrl for the stated hierarchy or learning contract.
def train_aircraft_hrl(config, method='planning_hierarchy', steps=4096, seed=0, rollout_size=128):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if method not in HRL_PPO_METHODS or steps < 1 or rollout_size < 2:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        raise ValueError('registered hierarchy, positive steps, rollout_size >= 2 required')
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.set_num_threads(1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.manual_seed(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    env = AircraftOptionEnv(config)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    model = AircraftPlanningHierarchy(env.env, method)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    actors = model.actor_parameters()
    # Explanation: Calculate or apply this level-specific learning update.
    actor_optimizer = torch.optim.Adam(actors, lr=3e-4)
    # Explanation: Calculate or apply this level-specific learning update.
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    obs, _ = env.reset(seed=seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    pending, ready, daily, logs = {}, [], [], []
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    episode = 0
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    for day in range(1, steps + 1):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            action, logp, _, values = model.act(obs)
        # Explanation: Preserve action feasibility and the commitment boundary.
        free = env.decision_mask()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for i in np.flatnonzero(free):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            pending[int(i)] = {'obs': obs.copy(), 'action': action[0].numpy().copy(),
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                'logp': float(logp[0, i]), 'value': float(values[0, i]), 'aircraft': int(i), 'rewards': []}
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        following, reward, done, truncated, info = env.step(model.catalog_actions(obs, action))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        terminal = done or truncated
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            next_values = model.value(following)[0].numpy()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        daily.append((obs.copy(), reward + .99 * (not terminal) * next_values))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for i in list(pending):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            pending[i]['rewards'].append(reward)
            # Explanation: Preserve action feasibility and the commitment boundary.
            completed = terminal or bool(env.decision_mask()[i])
            # A budget cutoff bootstraps an unfinished option; it does not cancel its job.
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if completed or day == steps:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                item = pending.pop(i)
                # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                item.update(target=hrl_option_target(item['rewards'], next_values[i], terminal),
                            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                            duration=len(item['rewards']), option_complete=completed)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                ready.append(item)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs = following
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if terminal:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            logs.append({'record_type': 'episode', 'simulated_days': day, **info['metrics']})
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            episode += 1
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs, _ = env.reset(seed=seed + episode)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if day % rollout_size == 0 or day == steps:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if ready:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                states = torch.tensor(np.stack([r['obs'] for r in ready]))
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                actions = torch.tensor(np.stack([r['action'] for r in ready]))
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                ids = torch.tensor([r['aircraft'] for r in ready])
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                batch_ids = torch.arange(len(ready))
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                old_logp = torch.tensor([r['logp'] for r in ready])
                # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                advantages = torch.tensor([r['target'] - r['value'] for r in ready])
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                if len(ready) > 1:
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                for _ in range(4):
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    _, new_logp, entropy, _ = model.act(states, actions)
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    ratio = torch.exp(new_logp[batch_ids, ids] - old_logp)
                    # Explanation: Calculate or apply this level-specific learning update.
                    loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
                    # Explanation: Calculate or apply this level-specific learning update.
                    loss -= .001 * entropy[batch_ids, ids].mean()
                    # Explanation: Calculate or apply this level-specific learning update.
                    if not bool(torch.isfinite(loss)):
                        # Explanation: Calculate or apply this level-specific learning update.
                        raise RuntimeError('nonfinite hierarchical policy loss')
                    # Explanation: Calculate or apply this level-specific learning update.
                    actor_optimizer.zero_grad()
                    # Explanation: Calculate or apply this level-specific learning update.
                    loss.backward()
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    torch.nn.utils.clip_grad_norm_(actors, .5)
                    # Explanation: Calculate or apply this level-specific learning update.
                    actor_optimizer.step()
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                logs.append({'record_type': 'option_update', 'method': method, 'simulated_days': day,
                    # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                    'decisions': len(ready), 'durations': [r['duration'] for r in ready],
                    # Explanation: Calculate or apply this level-specific learning update.
                    'complete': [r['option_complete'] for r in ready], 'policy_loss': float(loss.detach())})
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                ready.clear()
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            states = torch.tensor(np.stack([row[0] for row in daily]))
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            targets = torch.tensor(np.stack([row[1] for row in daily]))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            for _ in range(4):
                # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                value_loss = .5 * (model.value(states) - targets).square().mean()
                # Explanation: Calculate or apply this level-specific learning update.
                if not bool(torch.isfinite(value_loss)):
                    # Explanation: Calculate or apply this level-specific learning update.
                    raise RuntimeError('nonfinite hierarchical value loss')
                # Explanation: Calculate or apply this level-specific learning update.
                critic_optimizer.zero_grad()
                # Explanation: Calculate or apply this level-specific learning update.
                value_loss.backward()
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
                # Explanation: Calculate or apply this level-specific learning update.
                critic_optimizer.step()
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            daily.clear()
    # Explanation: Return the computed observation, choice, value, or training result.
    return model, logs, steps


# Explanation: Define this aircraft-option, policy, or value-decomposition implementation.
class MAXQMaintenance(nn.Module):
    # Explanation: Document the method semantics and limits.
    """Tabular MAXQ-0 baseline on one aircraft and the common repair catalog.

    Root chooses operate or Repair; Repair chooses an Engine task; each Engine
    chooses a component bundle then waits through its committed work. Primitive
    V values and composite completion C values are stored separately. No pseudo
    rewards or MAXQ-Q projected completion function is used. Beliefs are binned,
    so the original exact-state convergence guarantees are not claimed.
    """

    # Explanation: Define __init__ for the stated hierarchy or learning contract.
    def __init__(self, env):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        super().__init__()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if env.config.aircraft != 1:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('MAXQ baseline currently requires Config(aircraft=1)')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.n = 1
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.local = env.local_size
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.mandatory = env.config.mandatory_health
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.leaf_values = {}
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        self.completion = {}

    # Explanation: Define state for the stated hierarchy or learning contract.
    def state(self, observation):
        # Explicit approximation: quantized observed beliefs, never true health.
        # Explanation: Return the computed observation, choice, value, or training result.
        return tuple(np.rint(np.asarray(observation) * 8).astype(int).tolist())

    # Explanation: Define children for the stated hierarchy or learning contract.
    def task_children(self, task, observation):
        # Explanation: Preserve action feasibility and the commitment boundary.
        busy = observation[37] > .5
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if task == 'Root':
            # Explanation: Preserve action feasibility and the commitment boundary.
            if busy:
                # Explanation: Return the computed observation, choice, value, or training result.
                return ('Wait',)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            urgent = np.min(observation[24:36]) <= self.mandatory
            # Explanation: Return the computed observation, choice, value, or training result.
            return ('Repair',) if urgent else ('Operate', 'Repair')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if task == 'Repair':
            # Explanation: Return the computed observation, choice, value, or training result.
            return ('Engine0', 'Engine1', 'EngineBoth')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if task.startswith('Engine'):
            # Explanation: Preserve action feasibility and the commitment boundary.
            if busy:
                # Explanation: Return the computed observation, choice, value, or training result.
                return ('Wait',)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            e = ('Engine0', 'Engine1', 'EngineBoth').index(task)
            # Explanation: Return the computed observation, choice, value, or training result.
            return tuple(f'Submit{1 + 7 * e + c}' for c in range(7))
        # Explanation: Return the computed observation, choice, value, or training result.
        return ()

    # Explanation: Define value for the stated hierarchy or learning contract.
    def value(self, task, observation):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        children = self.task_children(task, observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        state = self.state(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if not children:
            # Explanation: Return the computed observation, choice, value, or training result.
            return self.leaf_values.get((task, state), 0.)
        # Explanation: Return the computed observation, choice, value, or training result.
        return max(self.q(task, observation, child) for child in children)

    # Explanation: Define q for the stated hierarchy or learning contract.
    def q(self, task, observation, child):
        # Explanation: Return the computed observation, choice, value, or training result.
        return self.value(child, observation) + self.completion.get((task, self.state(observation), child), 0.)

    # Explanation: Define update_completion for the stated hierarchy or learning contract.
    def update_completion(self, task, observation, child, next_value, duration, alpha=.15, gamma=.99):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        key = (task, self.state(observation), child)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        old = self.completion.get(key, 0.)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        self.completion[key] = old + alpha * (gamma ** duration * next_value - old)

    # Explanation: Define execute for the stated hierarchy or learning contract.
    def execute(self, task, env, observation, done, budget, rng, learn=False, epsilon=.2):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if budget < 1 or done:
            # Explanation: Return the computed observation, choice, value, or training result.
            return observation, done, [], True
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        children = self.task_children(task, observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if not children:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            action = int(task[6:]) if task.startswith('Submit') else 0
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            following, reward, done, truncated, info = env.step(np.array([action], dtype=np.int64))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if learn:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                key = (task, self.state(observation))
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                old = self.leaf_values.get(key, 0.)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                self.leaf_values[key] = old + .15 * (reward - old)
            # Explanation: Return the computed observation, choice, value, or training result.
            return following, done or truncated, [(observation.copy(), reward)], True
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        trajectory = []
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        invoked = False
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        while len(trajectory) < budget and not done:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if invoked and task != 'Root' and observation[37] <= .5:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                break
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            children = self.task_children(task, observation)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            scores = np.array([self.q(task, observation, child) for child in children])
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if learn and rng.random() < epsilon:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                selected = int(rng.integers(len(children)))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            else:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                selected = int(np.argmax(scores))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            child = children[selected]
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            following, ended, trace, child_complete = self.execute(
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                child, env, observation, done, budget - len(trajectory), rng, learn, epsilon)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            parent_terminal = ended or (task != 'Root' and following[37] <= .5)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if learn and child_complete:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                next_value = 0. if parent_terminal else self.value(task, following)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                for offset, (state_obs, _) in enumerate(trace):
                    # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                    self.update_completion(task, state_obs, child, next_value, len(trace) - offset)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            trajectory.extend(trace)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            observation, done, invoked = following, ended, True
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if task == 'Repair' and child_complete:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                break
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            if not trace:
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                raise RuntimeError('MAXQ child made no physical progress')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        complete = done or (task != 'Root' and invoked and observation[37] <= .5)
        # Explanation: Return the computed observation, choice, value, or training result.
        return observation, done, trajectory, complete

    # Explanation: Define get_extra_state for the stated hierarchy or learning contract.
    def get_extra_state(self):
        # Explanation: Return the computed observation, choice, value, or training result.
        return {'leaf_values': self.leaf_values, 'completion': self.completion}

    # Explanation: Define set_extra_state for the stated hierarchy or learning contract.
    def set_extra_state(self, state):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.leaf_values = dict(state['leaf_values'])
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        self.completion = dict(state['completion'])


# Explanation: Define train_maxq for the stated hierarchy or learning contract.
def train_maxq(config, steps=4096, seed=0):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if steps < 1:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        raise ValueError('positive MAXQ simulation budget required')
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    env = AircraftOptionEnv(config)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    model = MAXQMaintenance(env.env)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    rng = np.random.default_rng(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    days, episode, logs = 0, 0, []
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    while days < steps:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs, _ = env.reset(seed=seed + episode)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        _, terminal, trace, _ = model.execute('Root', env, obs, False, steps - days, rng, learn=True)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        days += len(trace)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        logs.append({'record_type': 'episode' if terminal else 'budget_cutoff', 'simulated_days': days,
                     # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                     'return': sum(r for _, r in trace), 'primitive_values': len(model.leaf_values),
                     # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                     'completion_values': len(model.completion)})
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        episode += 1
    # Explanation: Return the computed observation, choice, value, or training result.
    return model, logs, days


# Explanation: Define this aircraft-option, policy, or value-decomposition implementation.
class FleetOptionCritic(nn.Module):
    # Explanation: Document the method semantics and limits.
    """Option-critic adaptation with safe learned planning-mode termination.

    Options condition learned maintenance policies. Beta is exactly zero while
    a repair is committed. Changing planning mode never interrupts physical work.
    The greedy/epsilon-greedy policy over options is derived from Q_Omega.
    """

    # Explanation: Define __init__ for the stated hierarchy or learning contract.
    def __init__(self, env, options=3):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        super().__init__()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if options < 2:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('at least two planning options required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.n = env.config.aircraft
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.local = env.local_size
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.width = env.observation_space.shape[0]
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.mandatory = env.config.mandatory_health
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.options = options
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.worker = hrl_mlp(self.width + self.local + options, len(HRL_CATALOG))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.q_network = hrl_mlp(self.width, self.n * options)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.termination = hrl_mlp(self.width + self.local, options)

    # Explanation: Define features for the stated hierarchy or learning contract.
    def features(self, observation):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs = torch.as_tensor(observation, dtype=torch.float32)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if obs.ndim == 1:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs = obs.unsqueeze(0)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if obs.ndim != 2 or obs.shape[-1] != self.width:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('incorrect option-critic observation')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        local = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)
        # Explanation: Return the computed observation, choice, value, or training result.
        return obs, local, torch.cat([obs[:, None].expand(-1, self.n, -1), local], -1)

    # Explanation: Define q_values for the stated hierarchy or learning contract.
    def q_values(self, observation):
        # Explanation: Return the computed observation, choice, value, or training result.
        return self.q_network(self.features(observation)[0]).reshape(-1, self.n, self.options)

    # Explanation: Define beta for the stated hierarchy or learning contract.
    def beta(self, observation, option_ids):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        _, local, features = self.features(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        ids = torch.as_tensor(option_ids, dtype=torch.long)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if ids.shape != local.shape[:2] or torch.any((ids < 0) | (ids >= self.options)):
            # Explanation: Preserve action feasibility and the commitment boundary.
            raise ValueError('one valid option per aircraft required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        raw = torch.sigmoid(self.termination(features)).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        # Explanation: Return the computed observation, choice, value, or training result.
        return raw * (local[:, :, 37] <= .5)

    # Explanation: Define policy for the stated hierarchy or learning contract.
    def policy(self, observation, option_ids, actions=None, deterministic=False):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs, local, features = self.features(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        ids = torch.as_tensor(option_ids, dtype=torch.long)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if ids.shape != local.shape[:2] or torch.any((ids < 0) | (ids >= self.options)):
            # Explanation: Preserve action feasibility and the commitment boundary.
            raise ValueError('one valid option per aircraft required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        logits = self.worker(torch.cat([features, nn.functional.one_hot(ids, self.options).float()], -1))
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid = hrl_catalog_mask(obs, self.n, self.local, self.mandatory)
        # Explanation: Return the computed observation, choice, value, or training result.
        return hrl_categorical(logits, valid, actions, deterministic)


# Explanation: Define option_critic_target for the stated hierarchy or learning contract.
def option_critic_target(reward, following_q, option_ids, beta, terminal, gamma=.99):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    current = following_q.gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    continuation = (1 - beta) * current + beta * following_q.max(-1).values
    # Explanation: Return the computed observation, choice, value, or training result.
    return reward + gamma * (1 - terminal.float()) * continuation


# Explanation: Define hrl_epsilon_choice for the stated hierarchy or learning contract.
def hrl_epsilon_choice(values, valid, rng, epsilon=0.):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    values = np.asarray(values)
    # Explanation: Preserve action feasibility and the commitment boundary.
    valid = np.asarray(valid, dtype=bool)
    # Explanation: Preserve action feasibility and the commitment boundary.
    if values.ndim != 2 or values.shape != valid.shape or not valid.any(-1).all():
        # Explanation: Preserve action feasibility and the commitment boundary.
        raise ValueError('one nonempty choice mask per control unit required')
    # Explanation: Preserve action feasibility and the commitment boundary.
    choices = np.argmax(np.where(valid, values, -np.inf), axis=-1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    for i in range(len(choices)):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if rng.random() < epsilon:
            # Explanation: Preserve action feasibility and the commitment boundary.
            choices[i] = rng.choice(np.flatnonzero(valid[i]))
    # Explanation: Return the computed observation, choice, value, or training result.
    return choices.astype(np.int64)


# Explanation: Define train_option_critic for the stated hierarchy or learning contract.
def train_option_critic(config, steps=4096, seed=0):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if steps < 1:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        raise ValueError('positive option-critic simulation budget required')
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.set_num_threads(1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.manual_seed(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    rng = np.random.default_rng(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    env = AircraftOptionEnv(config)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    model = FleetOptionCritic(env.env)
    # Explanation: Calculate or apply this level-specific learning update.
    actor_opt = torch.optim.Adam(model.worker.parameters(), lr=3e-4)
    # Explanation: Calculate or apply this level-specific learning update.
    critic_opt = torch.optim.Adam(model.q_network.parameters(), lr=3e-4)
    # Explanation: Calculate or apply this level-specific learning update.
    beta_opt = torch.optim.Adam(model.termination.parameters(), lr=3e-4)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    obs, _ = env.reset(seed=seed)
    # Explanation: Define select for the stated hierarchy or learning contract.
    def select(observation):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            q = model.q_values(observation)[0].numpy()
        # Explanation: Return the computed observation, choice, value, or training result.
        return hrl_epsilon_choice(q, np.ones_like(q, dtype=bool), rng, .1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    ids = select(obs)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    episode, logs = 0, []
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    for day in range(1, steps + 1):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        option_ids = torch.tensor(ids).unsqueeze(0)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        actions, logp, entropy = model.policy(obs, option_ids)
        # Explanation: Preserve action feasibility and the commitment boundary.
        active = torch.tensor(env.decision_mask()).unsqueeze(0).float()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        following, reward, done, truncated, info = env.step(actions[0].detach().numpy())
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        terminal = done or truncated
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            following_q = model.q_values(following)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        beta = model.beta(following, option_ids)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        target = option_critic_target(reward, following_q, option_ids, beta.detach(), torch.tensor(terminal))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        q = model.q_values(obs).gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        advantage = (target - q).detach()
        # Explanation: Calculate or apply this level-specific learning update.
        actor_loss = -((logp * advantage + .001 * entropy) * active).sum() / active.sum().clamp_min(1)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        critic_loss = .5 * (q - target.detach()).square().mean()
        # Explanation: Preserve action feasibility and the commitment boundary.
        safe = torch.tensor(env.decision_mask()).unsqueeze(0).float() * (not terminal)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        termination_advantage = (following_q.gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
                                  # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                                  - following_q.max(-1).values + .01).detach()
        # Explanation: Calculate or apply this level-specific learning update.
        termination_loss = (beta * termination_advantage * safe).sum() / safe.sum().clamp_min(1)
        # Explanation: Calculate or apply this level-specific learning update.
        if not bool(torch.isfinite(actor_loss + critic_loss + termination_loss)):
            # Explanation: Calculate or apply this level-specific learning update.
            raise RuntimeError('nonfinite option-critic loss')
        # Explanation: Calculate or apply this level-specific learning update.
        actor_opt.zero_grad()
        # Explanation: Calculate or apply this level-specific learning update.
        actor_loss.backward()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        torch.nn.utils.clip_grad_norm_(model.worker.parameters(), .5)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if bool(active.any()):
            # Explanation: Calculate or apply this level-specific learning update.
            actor_opt.step()
        # Explanation: Calculate or apply this level-specific learning update.
        critic_opt.zero_grad()
        # Explanation: Calculate or apply this level-specific learning update.
        critic_loss.backward()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        torch.nn.utils.clip_grad_norm_(model.q_network.parameters(), .5)
        # Explanation: Calculate or apply this level-specific learning update.
        critic_opt.step()
        # Explanation: Calculate or apply this level-specific learning update.
        beta_opt.zero_grad()
        # Explanation: Calculate or apply this level-specific learning update.
        termination_loss.backward()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        torch.nn.utils.clip_grad_norm_(model.termination.parameters(), .5)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if bool(safe.any()):
            # Explanation: Calculate or apply this level-specific learning update.
            beta_opt.step()
        # Explanation: Preserve action feasibility and the commitment boundary.
        terminate = (rng.random(config.aircraft) < beta.detach()[0].numpy()) & env.decision_mask()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        next_ids = select(following)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        ids = np.where(terminate, next_ids, ids)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        logs.append({'record_type': 'update', 'simulated_days': day,
                     # Explanation: Calculate or apply this level-specific learning update.
                     'policy_loss': float(actor_loss.detach()), 'value_loss': float(critic_loss.detach()),
                     # Explanation: Calculate or apply this level-specific learning update.
                     'termination_loss': float(termination_loss.detach()),
                     # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                     'safe_termination_samples': int(safe.sum()), 'option_terminations': int(terminate.sum())})
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs = following
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if terminal:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            logs.append({'record_type': 'episode', 'simulated_days': day, **info['metrics']})
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            episode += 1
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs, _ = env.reset(seed=seed + episode)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            ids = select(obs)
    # Explanation: Return the computed observation, choice, value, or training result.
    return model, logs, steps


# Explanation: Define this aircraft-option, policy, or value-decomposition implementation.
class HierarchicalVDN(nn.Module):
    # Explanation: Document the method semantics and limits.
    """Two-level value-decomposition adaptation, not an exact HAVEN reproduction.

    Per-aircraft high policies select replacement budgets; local low policies
    choose catalog requests. Both levels sum agent Q values. The low reward
    receives a detached share of the macro TD advantage (coefficient 0.1).
    Macro epochs are three days; busy aircraft retain goals and committed plans.
    """

    # Explanation: Define __init__ for the stated hierarchy or learning contract.
    def __init__(self, env):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        super().__init__()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.n = env.config.aircraft
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.local = env.local_size
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.public = env.public_size
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.mandatory = env.config.mandatory_health
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.high_agents = nn.ModuleList([hrl_mlp(self.local + self.public, 4) for _ in range(self.n)])
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self.low_agents = nn.ModuleList([hrl_mlp(self.local + self.public + 4, len(HRL_CATALOG)) for _ in range(self.n)])

    # Explanation: Define features for the stated hierarchy or learning contract.
    def features(self, observation):
        # Explanation: Return the computed observation, choice, value, or training result.
        return mappo_agent_inputs(observation, self.n, self.local, self.public, 'aircraft')

    # Explanation: Define high_values for the stated hierarchy or learning contract.
    def high_values(self, observation):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        x = self.features(observation)
        # Explanation: Return the computed observation, choice, value, or training result.
        return torch.stack([agent(x[:, i]) for i, agent in enumerate(self.high_agents)], 1)

    # Explanation: Define low_values for the stated hierarchy or learning contract.
    def low_values(self, observation, goals):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        x = self.features(observation)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        goals = torch.as_tensor(goals, dtype=torch.long)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if goals.shape != x.shape[:2] or torch.any((goals < 0) | (goals >= 4)):
            # Explanation: Preserve action feasibility and the commitment boundary.
            raise ValueError('one valid hierarchy goal per aircraft required')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        x = torch.cat([x, nn.functional.one_hot(goals, 4).float()], -1)
        # Explanation: Return the computed observation, choice, value, or training result.
        return torch.stack([agent(x[:, i]) for i, agent in enumerate(self.low_agents)], 1)

    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    @staticmethod
    # Explanation: Define mix for the stated hierarchy or learning contract.
    def mix(values):
        # Explanation: Return the computed observation, choice, value, or training result.
        return values.sum(-1)

    # Explanation: Define choose_goals for the stated hierarchy or learning contract.
    def choose_goals(self, observation, previous, rng, epsilon=0.):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        local = self.features(observation)[0]
        # Explanation: Preserve action feasibility and the commitment boundary.
        busy = local[:, 37].numpy() > .5
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        urgent = local[:, 24:36].amin(-1).numpy() <= self.mandatory
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid = np.ones((self.n, 4), dtype=bool)
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid[urgent & ~busy, 0] = False
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Preserve action feasibility and the commitment boundary.
            choices = hrl_epsilon_choice(self.high_values(observation)[0].numpy(), valid, rng, epsilon)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if previous is not None:
            # Explanation: Preserve action feasibility and the commitment boundary.
            choices[busy] = np.asarray(previous)[busy]
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        else:
            # Explanation: Preserve action feasibility and the commitment boundary.
            choices[busy] = 0
        # Explanation: Return the computed observation, choice, value, or training result.
        return choices

    # Explanation: Define choose_actions for the stated hierarchy or learning contract.
    def choose_actions(self, observation, goals, rng, epsilon=0.):
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        goals = torch.tensor(np.asarray(goals)).unsqueeze(0)
        # Explanation: Preserve action feasibility and the commitment boundary.
        valid = hrl_catalog_mask(observation, self.n, self.local, self.mandatory, goals)
        # An urgent aircraft may outgrow an old zero-budget goal before the next macro epoch.
        # Explanation: Preserve action feasibility and the commitment boundary.
        if not bool(valid.any(-1).all()):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            raise ValueError('a stale hierarchy goal must be refreshed after an urgency alarm')
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Return the computed observation, choice, value, or training result.
            return hrl_epsilon_choice(self.low_values(observation, goals)[0].numpy(), valid[0].numpy(), rng, epsilon)


# Explanation: Define train_hierarchical_vdn for the stated hierarchy or learning contract.
def train_hierarchical_vdn(config, steps=4096, seed=0, macro_days=3):
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if steps < 1 or macro_days < 1:
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        raise ValueError('positive simulation budget and macro duration required')
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.set_num_threads(1)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    torch.manual_seed(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    rng = np.random.default_rng(seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    env = AircraftOptionEnv(config)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    model = HierarchicalVDN(env.env)
    # Explanation: Calculate or apply this level-specific learning update.
    high_opt = torch.optim.Adam(model.high_agents.parameters(), lr=3e-4)
    # Explanation: Calculate or apply this level-specific learning update.
    low_opt = torch.optim.Adam(model.low_agents.parameters(), lr=3e-4)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    obs, _ = env.reset(seed=seed)
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    goals, days, episode, logs = None, 0, 0, []
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    while days < steps:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        goals = model.choose_goals(obs, goals, rng, .1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        start, window = obs.copy(), []
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for _ in range(min(macro_days, steps - days)):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            actions = model.choose_actions(obs, goals, rng, .1)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            following, reward, done, truncated, info = env.step(actions)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            terminal = done or truncated
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            window.append((obs.copy(), goals.copy(), actions.copy(), reward, following.copy(), terminal))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            days += 1
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs = following
            # End the macro on a new urgency alarm so zero-budget goals never block safety.
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            rows, _ = env.env.split_observation(obs)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            urgent = rows[:, 24:36].min(-1) <= config.mandatory_health
            # Explanation: Preserve action feasibility and the commitment boundary.
            if terminal or np.any(urgent & env.decision_mask() & (goals == 0)):
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                break
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        next_goals = model.choose_goals(obs, goals, rng)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        self_n = config.aircraft
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        selected_goals = torch.tensor(goals).reshape(1, self_n, 1)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        high_value = model.mix(model.high_values(start).gather(-1, selected_goals).squeeze(-1))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        with torch.no_grad():
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            next_high = model.mix(model.high_values(obs).gather(-1, torch.tensor(next_goals).reshape(1, self_n, 1)).squeeze(-1))
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            target = hrl_option_target([r[3] for r in window], float(next_high[0]), terminal)
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        high_advantage = (target - high_value).detach()
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        high_loss = .5 * (high_value - target).square().mean()
        # Explanation: Calculate or apply this level-specific learning update.
        low_loss = torch.zeros(())
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        for t, (state, old_goals, action, reward, following, ended) in enumerate(window):
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            next_goal = next_goals if t == len(window) - 1 else old_goals
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            q = model.mix(model.low_values(state, torch.tensor(old_goals).unsqueeze(0)).gather(
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                -1, torch.tensor(action).reshape(1, self_n, 1)).squeeze(-1))
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            with torch.no_grad():
                # Explanation: Preserve action feasibility and the commitment boundary.
                valid = hrl_catalog_mask(following, self_n, env.env.local_size, config.mandatory_health,
                                         # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                                         torch.tensor(next_goal).unsqueeze(0))
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                future = model.low_values(following, torch.tensor(next_goal).unsqueeze(0))
                # Explanation: Preserve action feasibility and the commitment boundary.
                next_low = model.mix(future.masked_fill(~valid, -1e9).max(-1).values)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                shaped_reward = reward + .1 * high_advantage / len(window)
                # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
                low_target = shaped_reward + .99 * (not ended) * next_low
            # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
            low_loss = low_loss + .5 * (q - low_target).square().mean() / len(window)
        # Explanation: Calculate or apply this level-specific learning update.
        if not bool(torch.isfinite(high_loss + low_loss)):
            # Explanation: Calculate or apply this level-specific learning update.
            raise RuntimeError('nonfinite hierarchical value-decomposition loss')
        # Explanation: Calculate or apply this level-specific learning update.
        high_opt.zero_grad()
        # Explanation: Calculate or apply this level-specific learning update.
        high_loss.backward()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        torch.nn.utils.clip_grad_norm_(model.high_agents.parameters(), .5)
        # Explanation: Calculate or apply this level-specific learning update.
        high_opt.step()
        # Explanation: Calculate or apply this level-specific learning update.
        low_opt.zero_grad()
        # Explanation: Calculate or apply this level-specific learning update.
        low_loss.backward()
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        torch.nn.utils.clip_grad_norm_(model.low_agents.parameters(), .5)
        # Explanation: Calculate or apply this level-specific learning update.
        low_opt.step()
        # Explanation: Account for actual primitive time, subtask completion, or a detached learning target.
        logs.append({'record_type': 'macro_update', 'simulated_days': days, 'duration': len(window),
                     # Explanation: Calculate or apply this level-specific learning update.
                     'high_loss': float(high_loss.detach()), 'low_loss': float(low_loss.detach()),
                     # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                     'high_advantage': float(high_advantage[0]), 'goals': goals.tolist()})
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if terminal:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            logs.append({'record_type': 'episode', 'simulated_days': days, **info['metrics']})
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            episode += 1
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            obs, _ = env.reset(seed=seed + episode)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            goals = None
    # Explanation: Return the computed observation, choice, value, or training result.
    return model, logs, days


# Explanation: Define evaluate_hrl for the stated hierarchy or learning contract.
def evaluate_hrl(config, method, model, seeds):
    # Explanation: Document the method semantics and limits.
    """Evaluate hierarchy controllers with a daily physical clock and fixed weights."""
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    if method not in HRL_METHODS or model is None:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        raise ValueError('registered hierarchy and trained model required')
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    records = []
    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
    for seed in seeds:
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        env = AircraftOptionEnv(config)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        obs, _ = env.reset(seed=int(seed))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        rng = np.random.default_rng(int(seed))
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        decisions, free_decisions, terminations = 0, 0, 0
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        if method == 'maxq':
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            _, _, trace, _ = model.execute('Root', env, obs, False, config.horizon, rng, learn=False)
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            decisions = len(trace)
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        else:
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            goals, option_ids, macro_clock = None, None, 0
            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
            while True:
                # Explanation: Preserve action feasibility and the commitment boundary.
                free_decisions += int(env.decision_mask().sum())
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                with torch.no_grad():
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    if method in HRL_PPO_METHODS:
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        action = model.act(obs, deterministic=True)[0]
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        choices = model.catalog_actions(obs, action)
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    elif method == 'option_critic':
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        if option_ids is None:
                            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                            option_ids = model.q_values(obs)[0].argmax(-1).numpy()
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        ids = torch.tensor(option_ids).unsqueeze(0)
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        choices = model.policy(obs, ids, deterministic=True)[0][0].numpy()
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    else:
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        if goals is None or macro_clock == 0:
                            # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                            goals = model.choose_goals(obs, goals, rng)
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        choices = model.choose_actions(obs, goals, rng)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                obs, _, done, truncated, info = env.step(choices)
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                decisions += 1
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                if method == 'option_critic' and not (done or truncated):
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    with torch.no_grad():
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        beta = model.beta(obs, torch.tensor(option_ids).unsqueeze(0))[0].numpy()
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        next_ids = model.q_values(obs)[0].argmax(-1).numpy()
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    change = (beta >= .5) & env.decision_mask()
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    option_ids = np.where(change, next_ids, option_ids)
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    terminations += int(change.sum())
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                if method == 'hierarchical_vdn':
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    macro_clock = (macro_clock + 1) % 3
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    rows, _ = env.env.split_observation(obs)
                    # Explanation: Preserve action feasibility and the commitment boundary.
                    if np.any((rows[:, 24:36].min(-1) <= config.mandatory_health) & env.decision_mask() & (goals == 0)):
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        macro_clock = 0
                # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                if done or truncated:
                    # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                    break
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        metrics = env.last_info['metrics']
        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
        records.append({'seed': int(seed), 'decisions': decisions,
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        'learned_planning_opportunities': free_decisions,
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        'planning_mode_terminations': terminations,
                        # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                        'availability': metrics['flights'] / (config.aircraft * config.horizon), **metrics})
    # Explanation: Return the computed observation, choice, value, or training result.
    return records


# Explanation: Calculate or apply this level-specific learning update.
ALL_LEARNED_METHODS = ('flat', 'conditional', 'shared', 'options', 'learned_options',
                       # Explanation: Apply this observation, planning, bookkeeping, or validation step.
                       *REQUEST_METHODS, *HRL_METHODS)


# Explanation: Define execute_option with the shown parameters/defaults; its purpose is to execute a fleet option for several days while accepted jobs remain committed.
def execute_option(env, option, obs, gamma=0.99, max_days=14, planned_action=None):
    # Explanation: This docstring states execute_option's purpose/contract: Plan once, then keep all active jobs committed while the fleet advances. It is documentation text, not an executable operation.
    """Plan once, then keep all active jobs committed while the fleet advances.

    Manager returns at first task completion, a new observed urgency alarm,
    episode end, or max_days. It never consumes true event diagnostics.
    Waiting without any accepted job lasts one day.
    Other operating aircraft continue their modeled flights during this option.
    Longer manager intervals change decision opportunities, so this is a timing
    experiment as well as an architecture experiment, not a matched PPO ablation.
    """
    # Explanation: Compute the selected high-level action and store it in option; later checks/updates use this named intermediate.
    option = int(option)
    # Explanation: Branch on option not in range(4); the indented statements handle this case, and other cases fall through or use a later branch.
    if option not in range(4):
        # Explanation: Raise ValueError('unknown option') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('unknown option')
    # Explanation: Branch on max_days < 1; the indented statements handle this case, and other cases fall through or use a later branch.
    if max_days < 1:
        # Explanation: Raise ValueError('max_days must be positive') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('max_days must be positive')
    # Explanation: Compute and bind the right-hand value to thresholds, naming an intermediate for the surrounding algorithm.
    thresholds = {1: (0.62, 0.62), 2: (0.65, 0.77), 3: (0.78, 0.85)}
    # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
    action = planned_action if planned_action is not None else env.idle_action() if option == 0 else threshold_action(env, obs, *thresholds[option])
    # Explanation: Unpack the return values from env.split_observation(obs) into before; underscore entries are intentionally ignored.
    before, _ = env.split_observation(obs)
    # Explanation: Compute alarms already present before option execution and store it in before_alarm; later checks/updates use this named intermediate.
    before_alarm = before[:, 24:36].min(1) <= env.config.mandatory_health
    # Explanation: Compute the primitive rewards collected during an option and store it in rewards; later checks/updates use this named intermediate.
    rewards = []
    # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
    while True:
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        obs, reward, done, truncated, info = env.step(action)
        # Explanation: Append the new record/value to rewards so it remains in the result/history.
        rewards.append(reward)
        # Explanation: Unpack the return values from env.split_observation(obs) into rows; underscore entries are intentionally ignored.
        rows, _ = env.split_observation(obs)
        # Explanation: Compute a newly appearing alarm that can end option execution and store it in new_alarm; later checks/updates use this named intermediate.
        new_alarm = np.any((rows[:, 24:36].min(1) <= env.config.mandatory_health) & ~before_alarm)
        # Explanation: Check the episode/option stop condition option == 0 or done or truncated or info['completed'] or new_alarm or (len(rewards) >= max_days) or (not env.jobs); when true, stop collecting transitions for this segment.
        if option == 0 or done or truncated or info['completed'] or new_alarm or (len(rewards) >= max_days) or (not env.jobs):
            # Explanation: Leave the nearest loop because its stopping condition has been reached.
            break
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = env.idle_action()
    # Explanation: Compute the option return after applying time-discount factors and store it in discounted; later checks/updates use this named intermediate.
    discounted = sum((gamma ** i * r for i, r in enumerate(rewards)))
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (obs, discounted, done, truncated, info, len(rewards), sum(rewards))


# ========================================================================
# fleet_maintenance/train.py
# ========================================================================


# Explanation: Convert configuration records to dictionaries for saving experiment metadata.
from dataclasses import asdict
# Explanation: Construct output paths and create directories portably.
from pathlib import Path
# Explanation: Parse command-line settings for training and evaluation.
import argparse
# Explanation: Serialize experiment results in a portable text format.
import json
# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Use tensors, neural networks, optimizers, and automatic differentiation.
import torch

# Explanation: Define make_model with the shown parameters/defaults; its purpose is to select the neural architecture named by the training method.
def make_model(env, method):
    # Explanation: Select the explicit hierarchy and its matching checkpoint architecture.
    if method in HRL_PPO_METHODS:
        # Explanation: These modes share a catalog and daily aircraft-option interface.
        return AircraftPlanningHierarchy(env, method)
    # Explanation: MAXQ is a small-instance tabular value decomposition, not a neural actor.
    if method == 'maxq':
        # Explanation: Enforce the supported single-aircraft scope.
        return MAXQMaintenance(env)
    # Explanation: Option-critic learns internal policies, option values and safe termination.
    if method == 'option_critic':
        # Explanation: Construct a planning-mode option-critic model.
        return FleetOptionCritic(env)
    # Explanation: Hierarchical VDN learns summed per-agent values at both levels.
    if method == 'hierarchical_vdn':
        # Explanation: Construct the separate high/low agent networks.
        return HierarchicalVDN(env)
    # Explanation: Reconstruct global-belief off-policy models for training and checkpoint reload.
    if method in DCMAC_METHODS:
        # Explanation: Select separate actors only for the DDMAC methods.
        return DCMACActorCritic(env, request_assignment(method), method.startswith('ddmac_'))
    # Explanation: Select the explicit multi-agent model and its agent-to-action assignment.
    if method in MAPPO_METHODS:
        # Explanation: Both assignments use the same physical simulator and a centralized training critic.
        return MultiAgentActorCritic(env, 'aircraft' if method == 'mappo_aircraft' else 'component')
    # Explanation: Branch on method == 'learned_options'; the indented statements handle this case, and other cases fall through or use a later branch.
    if method == 'learned_options':
        # Explanation: Return LearnedOptionActorCritic(env) to the caller; this ends the function and supplies its result.
        return LearnedOptionActorCritic(env)
    # Explanation: Return OptionActorCritic(env) if method == 'options' else FleetActorCritic(env, method) to the caller; this ends the function and supplies its result.
    return OptionActorCritic(env) if method == 'options' else FleetActorCritic(env, method)

# Explanation: Define load_checkpoint with the shown parameters/defaults; its purpose is to rebuild a compatible model and load saved parameters for evaluation.
def load_checkpoint(path):
    # Explanation: This docstring states load_checkpoint's purpose/contract: Reload one of this package's own checkpoints for independent evaluation. It is documentation text, not an executable operation.
    """Reload one of this package's own checkpoints for independent evaluation."""
    # Explanation: Compute and bind the right-hand value to checkpoint, naming an intermediate for the surrounding algorithm.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    # Explanation: Branch on checkpoint.get('observation_version') != OBSERVATION_VERSION; the indented statements handle this case, and other cases fall through or use a later branch.
    if checkpoint.get('observation_version') != OBSERVATION_VERSION:
        # Explanation: Raise ValueError('Checkpoint uses an older observation layout. Retrain with efficiency/temperature inputs.') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('Checkpoint uses an incompatible observation/action layout. Retrain with individual crew masks and resource factors (version 3).')
    # Explanation: Compute and bind the right-hand value to config, naming an intermediate for the surrounding algorithm.
    config = Config(**checkpoint['config'])
    # Explanation: Compute and bind the right-hand value to model, naming an intermediate for the surrounding algorithm.
    model = make_model(FleetMaintenanceEnv(config), checkpoint['method'])
    # Explanation: Call model.load_state_dict to perform this step of the surrounding algorithm.
    model.load_state_dict(checkpoint['state_dict'])
    # Explanation: Call model.eval to perform this step of the surrounding algorithm.
    model.eval()
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (config, checkpoint['method'], model)

# Explanation: Define smdp_gae with the shown parameters/defaults; its purpose is to compute generalized advantages when each manager decision can span multiple primitive days.
def smdp_gae(rewards, values, next_values, dones, durations, gamma=0.99, lam=0.95):
    # Explanation: This docstring states smdp_gae's purpose/contract: Backward GAE with gamma**duration and lambda**duration per decision. It is documentation text, not an executable operation.
    """Backward GAE with gamma**duration and lambda**duration per decision.

    rewards already contain the discounted primitive rewards inside each option.
    A one-day decision recovers ordinary GAE. Episode boundaries block bootstrap.
    """
    # Explanation: Initialize advantage with zeros as a neutral container for the GAE estimate used by PPO; later lines fill or update it.
    advantage = np.zeros(len(rewards), dtype=np.float32)
    # Explanation: Compute the later-decision advantage propagated backward and store it in carry; later checks/updates use this named intermediate.
    carry = 0.0
    # Explanation: Visit each t from range(len(rewards) - 1, -1, -1); the body applies the operation once per item.
    for t in range(len(rewards) - 1, -1, -1):
        # Explanation: Compute and bind the right-hand value to live, naming an intermediate for the surrounding algorithm.
        live = 1.0 - float(dones[t])
        # Explanation: TD residual = option reward + next value discounted across all primitive days − current value. live is zero at terminal boundaries, so no terminal bootstrap is added.
        delta = rewards[t] + gamma ** durations[t] * next_values[t] * live - values[t]
        # Explanation: Build backward GAE; discount and lambda are each adjusted for the option's duration, and live prevents advantages crossing episode boundaries.
        carry = delta + (gamma * lam) ** durations[t] * live * carry
        # Explanation: Compute the GAE estimate used by PPO and store it in advantage[t]; later checks/updates use this named intermediate.
        advantage[t] = carry
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (advantage, advantage + np.asarray(values))

# Explanation: Dispatch daily-budget training to fleet PPO, per-agent MAPPO, or off-policy replay actor-critic.
def train(config, method='conditional', steps=4096, seed=0, rollout_size=128):
    # Explanation: This docstring states train's purpose/contract: Interaction budget counts actual simulated days, not manager decisions. It is documentation text, not an executable operation.
    """Interaction budget counts actual simulated days, not manager decisions."""
    # Explanation: Route DCMAC/DDMAC to replay-based training rather than PPO.
    # Explanation: Route each hierarchy to its actual update algorithm.
    if method in HRL_PPO_METHODS:
        # Explanation: Use independent aircraft option durations with daily fleet decisions.
        return train_aircraft_hrl(config, method, steps, seed, rollout_size)
    # Explanation: MAXQ updates tabular primitive and completion values.
    if method == 'maxq':
        # Explanation: Preserve the shared simulation-budget contract.
        return train_maxq(config, steps, seed)
    # Explanation: Option-critic uses intra-option and termination gradients.
    if method == 'option_critic':
        # Explanation: Train the safe planning-mode termination model.
        return train_option_critic(config, steps, seed)
    # Explanation: Hierarchical value decomposition has separate macro and primitive updates.
    if method == 'hierarchical_vdn':
        # Explanation: Train the high/low cooperative value learners.
        return train_hierarchical_vdn(config, steps, seed)
    if method in DCMAC_METHODS:
        # Explanation: Preserve the shared training API and daily interaction budget.
        return train_dcmac(config, method, steps, seed, rollout_size)
    # Explanation: MAPPO uses per-agent rollout statistics and clipped policy ratios.
    if method in MAPPO_METHODS:
        # Explanation: Route MAPPO through its explicit agent rollout and update loop.
        return train_mappo(config, method, steps, seed, rollout_size)
    # Explanation: Branch on steps < 1 or rollout_size < 2; the indented statements handle this case, and other cases fall through or use a later branch.
    if steps < 1 or rollout_size < 2:
        # Explanation: Raise ValueError('positive budget and rollout>=2 required') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('positive budget and rollout>=2 required')
    # Explanation: Use torch.set_num_threads for the array/tensor operation required at this point.
    torch.set_num_threads(1)
    # Explanation: Use torch.manual_seed for the array/tensor operation required at this point.
    torch.manual_seed(seed)
    # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
    env = FleetMaintenanceEnv(config)
    # Explanation: Compute and bind the right-hand value to model, naming an intermediate for the surrounding algorithm.
    model = make_model(env, method)
    # Explanation: Compute and bind the right-hand value to optimizer, naming an intermediate for the surrounding algorithm.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # Explanation: Compute and bind the right-hand value to rng, naming an intermediate for the surrounding algorithm.
    rng = np.random.default_rng(seed)
    # Explanation: Compute the current training episode index and store it in episode; later checks/updates use this named intermediate.
    episode = 0
    # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
    obs, _ = env.reset(seed=seed)
    # Explanation: Compute and bind the right-hand value to simulated, naming an intermediate for the surrounding algorithm.
    simulated = 0
    # Explanation: Compute and bind the right-hand value to logs, naming an intermediate for the surrounding algorithm.
    logs = []
    # Explanation: Branch on simulated < steps; the indented statements handle this case, and other cases fall through or use a later branch.
    while simulated < steps:
        # Explanation: Compute and bind the right-hand value to data, naming an intermediate for the surrounding algorithm.
        data = {k: [] for k in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'done', 'duration')}
        # Explanation: Branch on len(data['obs']) < rollout_size and simulated < steps; the indented statements handle this case, and other cases fall through or use a later branch.
        while len(data['obs']) < rollout_size and simulated < steps:
            # Explanation: Enter the context manager for this block; it controls temporary setup/cleanup around the operation.
            with torch.no_grad():
                # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
                action, logp, _, value = model.act(torch.tensor(obs))
            # Explanation: Compute and bind the right-hand value to chosen, naming an intermediate for the surrounding algorithm.
            chosen = action[0].numpy()
            # Explanation: Branch on method in ('options', 'learned_options'); the indented statements handle this case, and other cases fall through or use a later branch.
            if method in ('options', 'learned_options'):
                # Explanation: Compute the manager's high-level option ID and store it in macro; later checks/updates use this named intermediate.
                macro = int(chosen) if method == 'options' else int(chosen[0])
                # Explanation: Compute the proposed or accepted maintenance plan and store it in plan; later checks/updates use this named intermediate.
                plan = None if method == 'options' else chosen[1:].reshape(config.aircraft, 8)
                # Explanation: Unpack the return values from execute_option(env, macro, obs, max_days=min(14, steps - simulated), planned_action=plan) into next_obs, reward, done, truncated, info, duration; underscore entries are intentionally ignored.
                next_obs, reward, done, truncated, info, duration, _ = execute_option(env, macro, obs, max_days=min(14, steps - simulated), planned_action=plan)
            # Explanation: Handle the alternative case not selected by the preceding if/elif test.
            else:
                # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
                next_obs, reward, done, truncated, info = env.step(chosen)
                # Explanation: Compute the primitive days consumed by a job or option and store it in duration; later checks/updates use this named intermediate.
                duration = 1
            # Explanation: Compute and bind the right-hand value to terminal, naming an intermediate for the surrounding algorithm.
            terminal = done or truncated
            # Explanation: Enter the context manager for this block; it controls temporary setup/cleanup around the operation.
            with torch.no_grad():
                # Explanation: Compute and bind the right-hand value to nv, naming an intermediate for the surrounding algorithm.
                nv = float(model.value(torch.tensor(next_obs).unsqueeze(0))[0])
            # Explanation: Visit each key, item from [('obs', obs.copy()), ('action', np.asarray(chosen).copy()), ('logp', float(logp[0])), ('value', float(value[0])), ('next_value', nv), ('reward', reward), ('done', terminal), ('duration', duration)]; the body applies the operation once per item.
            for key, item in [('obs', obs.copy()), ('action', np.asarray(chosen).copy()), ('logp', float(logp[0])), ('value', float(value[0])), ('next_value', nv), ('reward', reward), ('done', terminal), ('duration', duration)]:
                # Explanation: Append the new record/value to data[key] so it remains in the result/history.
                data[key].append(item)
            # Explanation: Accumulate duration into simulated; the previous value is retained and this amount is added.
            simulated += duration
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = next_obs
            # Explanation: Branch on terminal; the indented statements handle this case, and other cases fall through or use a later branch.
            if terminal:
                # Explanation: Append the new record/value to logs so it remains in the result/history.
                logs.append({'episode': episode, 'simulated_days': simulated, **info['metrics']})
                # Explanation: Accumulate 1 into episode; the previous value is retained and this amount is added.
                episode += 1
                # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
                obs, _ = env.reset(seed=seed + episode)
        # Explanation: Unpack smdp_gae(data['reward'], data['value'], data['next_value'], data['done'], data['duration']) into named values advantages, targets so each part can be used separately below.
        advantages, targets = smdp_gae(data['reward'], data['value'], data['next_value'], data['done'], data['duration'])
        # Explanation: Compute and bind the right-hand value to tensors, naming an intermediate for the surrounding algorithm.
        tensors = {k: torch.tensor(np.asarray(v)) for k, v in data.items() if k in ('obs', 'action', 'logp')}
        # Explanation: Compute and bind the right-hand value to advantages, naming an intermediate for the surrounding algorithm.
        advantages = torch.tensor(advantages)
        # Explanation: Branch on len(advantages) > 1; the indented statements handle this case, and other cases fall through or use a later branch.
        if len(advantages) > 1:
            # Explanation: Compute and bind the right-hand value to advantages, naming an intermediate for the surrounding algorithm.
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
        # Explanation: Compute and bind the right-hand value to targets, naming an intermediate for the surrounding algorithm.
        targets = torch.tensor(targets, dtype=torch.float32)
        # Explanation: Visit each _ from range(4); the body applies the operation once per item.
        for _ in range(4):
            # Explanation: Compute and bind the right-hand value to ids, naming an intermediate for the surrounding algorithm.
            ids = rng.permutation(len(targets))
            # Explanation: Visit each start from range(0, len(ids), 64); the body applies the operation once per item.
            for start in range(0, len(ids), 64):
                # Explanation: Compute and bind the right-hand value to idx, naming an intermediate for the surrounding algorithm.
                idx = ids[start:start + 64]
                # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
                _, newlog, entropy, values = model.act(tensors['obs'][idx], tensors['action'][idx])
                # Explanation: Exponentiate the log-probability difference to obtain πnew(a|s)/πold(a|s), PPO's policy-update ratio.
                ratio = torch.exp(newlog - tensors['logp'][idx])
                # Explanation: Take the smaller unclipped/clipped objective; clipping the ratio to 0.8–1.2 limits how far this minibatch moves the policy likelihood.
                surrogate = torch.minimum(ratio * advantages[idx], ratio.clamp(0.8, 1.2) * advantages[idx])
                # Explanation: Combine policy loss, half-weighted squared critic error, and a small negative entropy term that rewards exploration.
                loss = -surrogate.mean() + 0.5 * ((values - targets[idx]) ** 2).mean() - 0.001 * entropy.mean()
                # Explanation: Branch on not torch.isfinite(loss); the indented statements handle this case, and other cases fall through or use a later branch.
                if not torch.isfinite(loss):
                    # Explanation: Raise RuntimeError('nonfinite PPO loss') so invalid input/failure is explicit instead of continuing with inconsistent state.
                    raise RuntimeError('nonfinite PPO loss')
                # Explanation: Clear gradients from the previous minibatch because PyTorch accumulates gradients by default.
                optimizer.zero_grad()
                # Explanation: Differentiate this minibatch loss with respect to model parameters and populate their gradients.
                loss.backward()
                # Explanation: Rescale gradients if their combined norm exceeds 0.5, limiting unstable parameter updates.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # Explanation: Apply one Adam parameter update using the gradients just computed.
                optimizer.step()
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (model, logs, simulated)

# Explanation: Define evaluate with the shown parameters/defaults; its purpose is to run fixed-seed episodes and record simulator performance metrics.
# Explanation: Define evaluate with the ownership, information, or update contract described below.
def evaluate(config, method, model=None, seeds=range(10000, 10010)):
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    """Evaluate fixed weights on held-out seeds; proposal methods use their common broker."""
    # Explanation: Check this validity or control-flow condition before executing its branch.
    # Explanation: Preserve per-aircraft option state while evaluating each new hierarchy.
    if method in HRL_METHODS:
        # Explanation: Evaluation uses no optimizer and honors committed work.
        return evaluate_hrl(config, method, model, seeds)
    if method != 'threshold' and model is None:
        # Explanation: Stop with an explicit error rather than accepting invalid multi-agent data.
        raise ValueError('evaluation requires a trained model')
    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
    records = []
    # Explanation: Process each indicated agent, field, task, or rollout transition in order.
    for seed in seeds:
        # Explanation: Check this validity or control-flow condition before executing its branch.
        if method in REQUEST_METHODS:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            env = MultiAgentFleetEnv(config, request_assignment(method))
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            _, info = env.reset(seed=int(seed))
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            obs = info['central_observation']
        # Explanation: Handle the alternative agent assignment or control-flow case.
        else:
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            env = FleetMaintenanceEnv(config)
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            obs, _ = env.reset(seed=int(seed))
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        decisions = 0
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        while True:
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if method == 'threshold':
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                obs, _, done, truncated, info = env.step(threshold_action(env, obs))
            # Explanation: Handle the alternative agent assignment or control-flow case.
            else:
                # Explanation: Control gradient recording or resource lifetime for this block.
                with torch.no_grad():
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    # Explanation: MAPPO deployment executes restricted actors without consulting the centralized training critic.
                    if method in REQUEST_METHODS:
                        # Explanation: Stack exactly the decentralized observations exposed by the adapter.
                        views = np.stack(list(env.observations().values())) if method in MAPPO_METHODS else obs
                        # Explanation: MAPPO consumes restricted per-agent views; DCMAC/DDMAC need the observed global belief.
                        # Explanation: Run the proposal policies alone; the value network is not required for execution.
                        action, _, _ = model.policy(torch.tensor(views).unsqueeze(0) if method in MAPPO_METHODS else torch.tensor(views), deterministic=True)
                    # Explanation: Preserve the original evaluation interface for the single-agent and hierarchical baselines.
                    else:
                        # Explanation: Obtain the baseline action from its existing policy.
                        action, _, _, _ = model.act(torch.tensor(obs), deterministic=True)
                # Explanation: Check this validity or control-flow condition before executing its branch.
                if method in REQUEST_METHODS:
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    _, _, done, truncated, info = env.step(model.action_dict(action[0]))
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    obs = info['central_observation']
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                elif method in ('options', 'learned_options'):
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    chosen = action[0].numpy()
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    macro = int(chosen) if method == 'options' else int(chosen[0])
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    plan = None if method == 'options' else chosen[1:].reshape(config.aircraft, 8)
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    obs, _, done, truncated, info, _, _ = execute_option(env, macro, obs, planned_action=plan)
                # Explanation: Handle the alternative agent assignment or control-flow case.
                else:
                    # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                    obs, _, done, truncated, info = env.step(action[0].numpy())
            # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
            decisions += 1
            # Explanation: Check this validity or control-flow condition before executing its branch.
            if done or truncated:
                # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
                break
        # Explanation: Execute this step of the multi-agent observation, proposal, or learning calculation.
        records.append({'seed': int(seed), 'decisions': decisions, 'availability': info['metrics']['flights'] / (config.aircraft * config.horizon), **info['metrics']})
    # Explanation: Return the specified observations, actions, policy statistics, or training results to the caller.
    return records

# Explanation: Define main with the shown parameters/defaults; its purpose is to parse command-line settings, train/evaluate methods, and write results.
def main():
    # Explanation: Compute and bind the right-hand value to parser, naming an intermediate for the surrounding algorithm.
    parser = argparse.ArgumentParser()
    # Explanation: Unpack ['flat', 'conditional', 'shared', 'options', 'learned_options', 'all'], default='all') into named values parser.add_argument('--method', choices so each part can be used separately below.
    parser.add_argument('--method', choices=[*ALL_LEARNED_METHODS, 'all'], default='all')
    # Explanation: Unpack int, default=4096) into named values parser.add_argument('--steps', type so each part can be used separately below.
    parser.add_argument('--steps', type=int, default=4096)
    # Explanation: Unpack int, default=7) into named values parser.add_argument('--aircraft', type so each part can be used separately below.
    parser.add_argument('--aircraft', type=int, default=7)
    # Explanation: Allow the number of individually identified crew members to be configured independently of fleet size.
    parser.add_argument('--crews', type=int, default=3)
    # Explanation: Allow the number of individually identified workshops to be configured independently of fleet size.
    parser.add_argument('--workshops', type=int, default=2)
    # Explanation: Accept one positive duration factor per crew member; omission uses the documented teaching profile.
    parser.add_argument('--crew-time-factors', type=float, nargs='+', default=())
    # Explanation: Accept one positive duration factor per workshop; omission uses the documented teaching profile.
    parser.add_argument('--workshop-time-factors', type=float, nargs='+', default=())
    # Explanation: Unpack int, default=90) into named values parser.add_argument('--horizon', type so each part can be used separately below.
    parser.add_argument('--horizon', type=int, default=90)
    # Explanation: Unpack int, default=0) into named values parser.add_argument('--seed', type so each part can be used separately below.
    parser.add_argument('--seed', type=int, default=0)
    # Explanation: Unpack int, default=10) into named values parser.add_argument('--eval-episodes', type so each part can be used separately below.
    parser.add_argument('--eval-episodes', type=int, default=10)
    # Explanation: Unpack 'runs/smoke') into named values parser.add_argument('--output', default so each part can be used separately below.
    parser.add_argument('--output', default='runs/smoke')
    # Explanation: Compute and bind the right-hand value to args, naming an intermediate for the surrounding algorithm.
    args = parser.parse_args()
    # Explanation: Compute and bind the right-hand value to config, naming an intermediate for the surrounding algorithm.
    config = Config(aircraft=args.aircraft, horizon=args.horizon, crews=args.crews, workshops=args.workshops, crew_time_factors=tuple(args.crew_time_factors), workshop_time_factors=tuple(args.workshop_time_factors))
    # Explanation: Compute and bind the right-hand value to destination, naming an intermediate for the surrounding algorithm.
    destination = Path(args.output)
    # Explanation: Compute and bind the right-hand value to destination.mkdir(parents, naming an intermediate for the surrounding algorithm.
    destination.mkdir(parents=True, exist_ok=True)
    # Explanation: Compute and bind the right-hand value to result, naming an intermediate for the surrounding algorithm.
    result = {'config': asdict(config), 'training_seed': args.seed, 'requested_steps': args.steps, 'observation_version': OBSERVATION_VERSION, 'status': 'preliminary teaching experiment; not a paper reproduction or convergence claim', 'comparison_note': 'new HRL methods share a restricted 22-action catalog and broker; hrl_flat matches planning_hierarchy authority; concurrent_options has a fixed worker; target/option/VDN clocks and objectives differ; MAXQ is single-aircraft; legacy and MAPPO/DCMAC use broader action spaces', 'evaluation': {'threshold': evaluate(config, 'threshold', seeds=range(10000, 10000 + args.eval_episodes))}}
    # Explanation: Compute and bind the right-hand value to methods, naming an intermediate for the surrounding algorithm.
    methods = list(ALL_LEARNED_METHODS) if args.method == 'all' else [args.method]
    # Explanation: MAXQ is deliberately restricted to one aircraft; make skipped scope explicit.
    if args.method == 'all' and config.aircraft != 1:
        # Explanation: Do not silently run a different environment for MAXQ.
        methods.remove('maxq')
        # Explanation: Record why this method is not part of the selected fleet run.
        result['skipped_methods'] = {'maxq': 'requires aircraft=1'}
    # Explanation: Visit each method from methods; the body applies the operation once per item.
    for method in methods:
        # Explanation: Unpack train(config, method, args.steps, args.seed) into named values model, logs, days so each part can be used separately below.
        model, logs, days = train(config, method, args.steps, args.seed)
        # Explanation: Use torch.save for the array/tensor operation required at this point.
        torch.save({'state_dict': model.state_dict(), 'config': asdict(config), 'method': method, 'training_seed': args.seed, 'simulated_days': days, 'observation_version': OBSERVATION_VERSION}, destination / f'{method}.pt')
        # Explanation: Compute and bind the right-hand value to result['evaluation'][method], naming an intermediate for the surrounding algorithm.
        result['evaluation'][method] = evaluate(config, method, model, seeds=range(10000, 10000 + args.eval_episodes))
        # Explanation: Compute and bind the right-hand value to result[method + '_training'], naming an intermediate for the surrounding algorithm.
        result[method + '_training'] = logs
        # Explanation: Unpack True) into named values print(method, 'days', days, 'mean test return', np.mean([r['return'] for r in result['evaluation'][method]], flush so each part can be used separately below.
        print(method, 'days', days, 'mean test return', np.mean([r['return'] for r in result['evaluation'][method]]), flush=True)
    # Explanation: Unpack 2), encoding='utf-8') into named values destination / 'results.json').write_text(json.dumps(result, indent so each part can be used separately below.
    (destination / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    # Explanation: Compute and bind the right-hand value to summary, naming an intermediate for the surrounding algorithm.
    summary = ['# Preliminary comparison', '', result['status'], '', result['comparison_note'], '', '| Method | Return | Availability | Failed flights | Rejected plans | Decisions |', '|---|---:|---:|---:|---:|---:|']
    # Explanation: Visit each method, records from result['evaluation'].items(); the body applies the operation once per item.
    for method, records in result['evaluation'].items():
        # Explanation: Compute and bind the right-hand value to mean, naming an intermediate for the surrounding algorithm.
        mean = lambda k: np.mean([r[k] for r in records])
        # Explanation: Append the new record/value to summary so it remains in the result/history.
        summary.append(f"| {method} | {mean('return'):.2f} | {mean('availability'):.3f} | {mean('flight_failures'):.1f} | {mean('rejections'):.1f} | {mean('decisions'):.1f} |")
    # Explanation: Unpack 'utf-8') into named values destination / 'summary.md').write_text('\n'.join(summary) + '\n', encoding so each part can be used separately below.
    (destination / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    # Explanation: Show progress/result information to the person running the script.
    print('Saved', destination.resolve())


# Explanation: Define OPERATE_UNTIL_LOW as a shared modeling constant: the option ID for operation until health is low.
OPERATE_UNTIL_LOW = 0
# Explanation: Define REPLACE_NOW as a shared modeling constant: the option ID for immediate replacement.
REPLACE_NOW = 1



