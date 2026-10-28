# Maintenance tutorial core; full line-by-line explanations are in the code reports.
# Updated simulator: fixed aircraft identities and individually assigned heterogeneous maintenance resources.
"""One supporting module for the comprehensive RL/HRL maintenance course.

OBSERVATION_VERSION=3 adds individual resource availability and duration factors.
Fleet action field 4 is now a crew-selection bitmask, not a crew count minus one.
The first examples and user's exercise answers are namespaced by tutorial part.
The fleet classes/functions keep their public names. No import depends on Code.
All figures are teaching assumptions rather than real turbofan calibration.
"""
from __future__ import annotations


# ========================================================================
# mini_maintenance.py
# ========================================================================


import random
from typing import TypeAlias
Observation: TypeAlias = tuple[int, int]

class MiniMaintenanceEnv:
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
    OPERATE = 0
    GROUND_AND_REPLACE = 1

    def __init__(self, horizon: int=12, shock_prob: float=0.0) -> None:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('horizon must be a positive integer')
        if not 0.0 <= shock_prob <= 1.0:
            raise ValueError('shock_prob must be between 0 and 1')
        self.horizon = horizon
        self.shock_prob = float(shock_prob)
        self._rng = random.Random()
        self.health = 3
        self.day = 0
        self._needs_reset = True

    def reset(self, *, seed: int | None=None, initial_health: int=3):
        """Start an episode and return (observation, info).

        A seed makes subsequent stochastic wear reproducible. Calling reset
        without a seed keeps the current generator stream.
        """
        if initial_health not in (0, 1, 2, 3):
            raise ValueError('initial_health must be 0, 1, 2, or 3')
        if seed is not None:
            self._rng.seed(seed)
        self.health = initial_health
        self.day = 0
        self._needs_reset = False
        return (self._observation(), {'day': self.day})

    def step(self, action: int):
        """Return (next_observation, reward, terminated, truncated, info)."""
        if self._needs_reset:
            raise RuntimeError('call reset() before step(), including after an episode')
        if action not in (self.OPERATE, self.GROUND_AND_REPLACE):
            raise ValueError('action must be 0 (operate) or 1 (ground/replace)')
        previous_health = self.health
        if action == self.OPERATE:
            extra_wear = int(self._rng.random() < self.shock_prob)
            self.health = max(0, self.health - 1 - extra_wear)
            failed = self.health == 0
            reward = -12.0 if failed else 5.0
            info = {'event': 'failed_operation' if failed else 'successful_operation', 'wear': previous_health - self.health, 'failed': failed, 'grounded': False}
        else:
            self.health = 3
            reward = -4.0
            info = {'event': 'grounded_and_replaced', 'wear': 0, 'failed': False, 'grounded': True}
        self.day += 1
        terminated = self.day >= self.horizon
        truncated = False
        self._needs_reset = terminated
        info['day'] = self.day
        return (self._observation(), reward, terminated, truncated, info)

    def _observation(self) -> Observation:
        return (self.health, self.horizon - self.day)

    def render_text(self) -> str:
        """Small human-readable display; useful for tracing a policy."""
        return f'day={self.day:2d}  health={self.health}  steps_left={self.horizon - self.day}'

def random_policy(observation: Observation, *, rng: random.Random) -> int:
    """Random baseline, independent of health."""
    return rng.randrange(2)

def threshold_policy(observation: Observation) -> int:
    """Simple rule baseline: replace at health 0 or 1."""
    health, _steps_left = observation
    return MiniMaintenanceEnv.GROUND_AND_REPLACE if health <= 1 else MiniMaintenanceEnv.OPERATE

def rollout(env: MiniMaintenanceEnv, policy, *, seed: int=0):
    """Execute one episode; policy receives observation and returns an action."""
    observation, _info = env.reset(seed=seed)
    rows: list[dict] = []
    while True:
        action = policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        rows.append({'observation': observation, 'action': action, 'reward': reward, 'next_observation': next_observation, 'event': info['event'], 'terminated': terminated, 'truncated': truncated})
        observation = next_observation
        if terminated or truncated:
            break
    return rows


# ========================================================================
# hierarchical_maintenance.py
# ========================================================================


from itertools import product
TOY_COMPONENTS = ('fan', 'compressor')

def valid_action(action: tuple[int, int, int]) -> bool:
    """Action is (ground, engine_mask, component_mask).

    Engine bits 0 and 1 represent engines 0 and 1. Component bits 0-1
    represent engine 0; bits 2-3 represent engine 1. A selected engine must
    contain at least one selected component, and no unselected engine may
    receive component work.
    """
    if len(action) != 3 or any((type(x) is not int for x in action)):
        return False
    ground, engines, components = action
    if ground == 0:
        return engines == 0 and components == 0
    if ground != 1 or not 1 <= engines <= 3 or (not 1 <= components <= 15):
        return False
    selected_engines = sum((1 << engine for engine in range(2) if components & 3 << 2 * engine))
    return engines == selected_engines

def all_actions():
    """List the 16 feasible composite actions, including operate."""
    return [(0, 0, 0)] + [(1, engines, components) for engines, components in product(range(1, 4), range(1, 16)) if valid_action((1, engines, components))]

class ConditionalMaintenanceEnv:
    """Tiny deterministic simulator for one aircraft with two engines."""

    def __init__(self, horizon: int=6):
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('horizon must be positive')
        self.horizon = horizon
        self.health = [3, 3, 3, 3]
        self.day = 0
        self._needs_reset = True

    def _observation(self):
        return tuple(self.health) + (self.horizon - self.day,)

    def reset(self, initial_health=(3, 3, 3, 3)):
        if len(initial_health) != 4 or any((h not in range(4) for h in initial_health)):
            raise ValueError('provide four health values in 0..3')
        self.health = list(initial_health)
        self.day = 0
        self._needs_reset = False
        return (self._observation(), {})

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError('reset before stepping')
        if not valid_action(action):
            raise ValueError(f'infeasible hierarchical action: {action}')
        ground, engines, components = action
        if ground:
            for index in range(4):
                if components & 1 << index:
                    self.health[index] = 3
            n_engines = engines.bit_count()
            n_components = components.bit_count()
            reward = -3.0 - 2.0 * n_engines - 1.0 * n_components
            event = 'grounded_and_replaced'
        else:
            self.health = [max(0, h - 1) for h in self.health]
            reward = -15.0 if 0 in self.health else 8.0
            event = 'failed_operation' if reward < 0 else 'successful_operation'
        self.day += 1
        terminated = self.day >= self.horizon
        self._needs_reset = terminated
        return (self._observation(), reward, terminated, False, {'event': event})


# ========================================================================
# gym_maintenance.py
# ========================================================================


import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GymMaintenanceEnv(gym.Env):
    """Same decisions and rewards as MiniMaintenanceEnv, with Gymnasium spaces.

    The observation is [health / 3, days_left / horizon]. Normalization keeps
    both input features between zero and one for a neural-network policy.
    """
    metadata = {'render_modes': ['ansi']}

    def __init__(self, horizon: int=12, shock_prob: float=0.0):
        super().__init__()
        self.core = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

    def _encode(self, observation):
        health, days_left = observation
        return np.array([health / 3.0, days_left / self.core.horizon], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_health = 3 if options is None else options.get('initial_health', 3)
        observation, info = self.core.reset(seed=seed, initial_health=initial_health)
        return (self._encode(observation), info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.core.step(int(action))
        return (self._encode(observation), float(reward), terminated, truncated, info)

    def render(self):
        return self.core.render_text()

    def close(self):
        pass


# ========================================================================
# fleet_maintenance/fleet_env.py
# ========================================================================


from dataclasses import dataclass
import math
COMPONENTS = ('fan', 'LPC', 'HPC', 'combustor', 'HPT', 'LPT')
WEAR = np.array([0.002, 0.0025, 0.004, 0.002, 0.003, 0.0025])
LABOR_DAYS = np.array([2, 3, 4, 2, 3, 3])
PART_COST = np.array([8.0, 10.0, 16.0, 8.0, 12.0, 10.0])
EFFICIENCY_BASE = 0.72
EFFICIENCY_GAIN = 0.24
TEMPERATURE_BASE_C = np.array([40.0, 120.0, 350.0, 900.0, 650.0, 450.0])
TEMPERATURE_HEALTH_GAIN_C = 120.0
TEMPERATURE_EXPOSURE_GAIN_C = 30.0
TEMPERATURE_MIN_C = -50.0
TEMPERATURE_MAX_C = 1400.0
OBSERVATION_VERSION = 3

@dataclass(frozen=True)
class Config:
    aircraft: int = 7
    horizon: int = 90
    crews: int = 3
    workshops: int = 2
    # Factors below one speed up work; empty profiles select synthetic defaults.
    crew_time_factors: tuple[float, ...] = ()
    workshop_time_factors: tuple[float, ...] = ()
    particles: int = 64
    setup_days: int = 7
    replenish_days: int = 14
    bird_probability: float = 0.002
    hail_probability: float = 0.003
    weather_probability: float = 0.03
    observation_noise: float = 0.025
    wear_scale: float = 1.0
    mandatory_health: float = 0.08
    initial_health: float = 0.88

    def __post_init__(self):
        for k in ('aircraft', 'horizon', 'crews', 'workshops', 'particles', 'setup_days', 'replenish_days'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f'{k} must be a positive integer')
        for name, count in (('crew_time_factors', self.crews), ('workshop_time_factors', self.workshops)):
            factors = getattr(self, name)
            if not isinstance(factors, (tuple, list)):
                raise ValueError(f'{name} must be a tuple or list of duration factors')
            if len(factors) and (len(factors) != count or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in factors)):
                raise ValueError(f'{name} must be empty or contain {count} finite positive numbers')
            # Accepted-job resource profiles must remain immutable.
            object.__setattr__(self, name, tuple(factors))
        for k in ('bird_probability', 'hail_probability', 'weather_probability'):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(f'{k} must be in [0,1]')
        if not 0 < self.observation_noise or not 0 < self.wear_scale:
            raise ValueError('noise and wear_scale must be positive')
        if not 0 <= self.mandatory_health < 1 or not 0 < self.initial_health <= 1:
            raise ValueError('invalid health bounds')

@dataclass(frozen=True)
class Plan:
    aircraft: int
    engine_mask: int
    bundles: tuple[int, int]
    crew_mask: int
    workshop: int
    order_start: int
    priority: int

    @property
    def crews(self):
        return self.crew_mask.bit_count()

@dataclass
class Job:
    plan: Plan
    tasks: tuple[tuple[int, int], ...]
    remaining: int
    total_duration: int
    completion_days: tuple[int, ...]

class FleetMaintenanceEnv(gym.Env):
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
    metadata = {'render_modes': ['ansi']}

    def __init__(self, config: Config | None=None, render_mode=None):
        super().__init__()
        self.config = config or Config()
        self.render_mode = render_mode
        n = self.config.aircraft
        self.aircraft_plates = tuple(f'AC-{i + 1:03d}' for i in range(n))
        self.crew_ids = tuple(f'CR-{i + 1:03d}' for i in range(self.config.crews))
        self.workshop_ids = tuple(f'WS-{i + 1:03d}' for i in range(self.config.workshops))
        self.crew_time_factors = tuple(self.config.crew_time_factors) or tuple((0.8, 1.0, 1.25)[i % 3] for i in range(self.config.crews))
        self.workshop_time_factors = tuple(self.config.workshop_time_factors) or tuple((0.8, 1.0, 1.2)[i % 3] for i in range(self.config.workshops))
        self.row_dims = np.array([2, 4, 64, 64, 1 << self.config.crews, self.config.workshops, 12, n])
        self.action_space = spaces.MultiDiscrete(np.tile(self.row_dims, (n, 1)))
        self.local_size = 70
        self.public_size = 10 + 2 * self.config.crews + 2 * self.config.workshops
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n * self.local_size + self.public_size,), dtype=np.float32)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        # Physical events, belief updates, and exposure tapes use separate random streams.
        seeds = np.random.SeedSequence(seed).spawn(3)
        self.physical_rng = np.random.default_rng(seeds[0])
        self.filter_rng = np.random.default_rng(seeds[1])
        tape_rng = np.random.default_rng(seeds[2])
        self.day = 0
        self.health = np.clip(cfg.initial_health + self.physical_rng.normal(0, 0.025, (cfg.aircraft, 2, 6)), 0.1, 1)
        if options and 'health' in options:
            health = np.asarray(options['health'], dtype=float)
            if health.shape != self.health.shape or not np.isfinite(health).all() or np.any((health < 0) | (health > 1)):
                raise ValueError('initial health must have shape (aircraft,2,6), in [0,1]')
            self.health = health.copy()
        self.exposure = tape_rng.uniform(0.75, 1.35, cfg.aircraft)
        self.measurements = self._sensor_model(self.health, self.exposure[:, None, None])
        self.measurements += self.physical_rng.normal(0, 1, self.measurements.shape) * self.sensor_noise
        self._clip_measurements(self.measurements)
        self.last_flight_day = np.zeros(cfg.aircraft, dtype=int)
        prior = self._invert_sensor_metrics(self.measurements, self.exposure[:, None, None])
        self.particles = np.clip(prior[..., None] + self.filter_rng.normal(0, 0.04, (*self.health.shape, cfg.particles)), 0, 1)
        self.jobs: dict[int, Job] = {}
        self.stock = np.full(6, cfg.aircraft, dtype=int)
        self.event_tape = tape_rng.random((cfg.horizon, cfg.aircraft, 2, 2))
        self.severity_tape = tape_rng.uniform(0.1, 0.35, (cfg.horizon, cfg.aircraft, 2, 2))
        self.weather_tape = tape_rng.random(cfg.horizon)
        self.wear_tape = tape_rng.normal(1.0, 0.12, (cfg.horizon, cfg.aircraft, 2, 6))
        self.noise_tape = tape_rng.normal(0, cfg.observation_noise, (cfg.horizon, cfg.aircraft, 2, 6, 2))
        self.metrics = {k: 0.0 for k in ('return', 'cost', 'flight_failures', 'flights', 'downtime_aircraft_days', 'groundings', 'replacements', 'rejections', 'bird_events', 'hail_events', 'weather_exposures', 'unsafe_attempts')}
        self._done = False
        return (self._observation(), {'day': self.day, 'aircraft_plates': self.aircraft_plates, 'resources': self.resource_details()})

    def split_observation(self, observation):
        a = np.asarray(observation)
        return (a[:-self.public_size].reshape(self.config.aircraft, self.local_size), a[-self.public_size:])

    def idle_action(self):
        return np.zeros((self.config.aircraft, 8), dtype=np.int64)

    def available_crew_mask(self):
        available = (1 << self.config.crews) - 1
        for job in self.jobs.values():
            available &= ~job.plan.crew_mask
        return available

    def resource_state(self):
        busy_shops = {j.plan.workshop for j in self.jobs.values()}
        return (self.available_crew_mask().bit_count(), [s for s in range(self.config.workshops) if s not in busy_shops])

    def resource_details(self):
        return {'crew_ids': self.crew_ids, 'crew_time_factors': self.crew_time_factors, 'available_crew_mask': self.available_crew_mask(), 'workshop_ids': self.workshop_ids, 'workshop_time_factors': self.workshop_time_factors, 'available_workshops': tuple(self.resource_state()[1])}

    @property
    def sensor_noise(self):
        """Independent sensor noise scales in efficiency fraction and Celsius."""
        return np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C]) * self.config.observation_noise

    @staticmethod
    def _sensor_model(health, exposure):
        """Toy performance mapping; health decreases efficiency and raises heat."""
        efficiency = EFFICIENCY_BASE + EFFICIENCY_GAIN * health
        temperature = TEMPERATURE_BASE_C + TEMPERATURE_HEALTH_GAIN_C * (1 - health) + TEMPERATURE_EXPOSURE_GAIN_C * (exposure - 1)
        return np.stack([efficiency, temperature], axis=-1)

    @staticmethod
    def _clip_measurements(measurements):
        np.clip(measurements[..., 0], 0, 1, out=measurements[..., 0])
        np.clip(measurements[..., 1], TEMPERATURE_MIN_C, TEMPERATURE_MAX_C, out=measurements[..., 1])

    @staticmethod
    def _invert_sensor_metrics(measurements, exposure):
        """Initialization only: fuse two noisy inverse mappings with equal weight."""
        from_efficiency = (measurements[..., 0] - EFFICIENCY_BASE) / EFFICIENCY_GAIN
        from_temperature = 1 - (measurements[..., 1] - TEMPERATURE_BASE_C - TEMPERATURE_EXPOSURE_GAIN_C * (exposure - 1)) / TEMPERATURE_HEALTH_GAIN_C
        return np.clip(0.5 * (from_efficiency + from_temperature), 0, 1)

    def observation_metrics(self, observation):
        """Decode observed sensor metrics without exposing hidden health."""
        rows, _ = self.split_observation(observation)
        return {'efficiency': rows[:, 45:57].reshape(self.config.aircraft, 2, 6).copy(), 'temperature_c': rows[:, 57:69].reshape(self.config.aircraft, 2, 6) * (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C) + TEMPERATURE_MIN_C, 'reading_age_days': np.rint(rows[:, 69] * self.config.horizon).astype(int)}

    def _observation(self):
        mean = self.particles.mean(-1)
        std = self.particles.std(-1)
        low = np.quantile(self.particles, 0.1, axis=-1)
        rows = []
        for a in range(self.config.aircraft):
            j = self.jobs.get(a)
            committed = [j.plan.engine_mask / 3, j.plan.bundles[0] / 63, j.plan.bundles[1] / 63, j.plan.crew_mask / ((1 << self.config.crews) - 1), j.plan.workshop / max(1, self.config.workshops - 1), j.plan.order_start / 11] if j else [0.0] * 6
            rows.append(np.concatenate([mean[a].ravel(), std[a].ravel(), low[a].ravel(), [self.exposure[a] / 1.5, float(j is not None), min(1, j.remaining / 60) if j else 0], committed, self.measurements[a, :, :, 0].ravel(), ((self.measurements[a, :, :, 1] - TEMPERATURE_MIN_C) / (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C)).ravel(), [(self.day - self.last_flight_day[a]) / self.config.horizon]]))
        free, shops = self.resource_state()
        public = [self.day / self.config.horizon, free / self.config.crews, len(shops) / self.config.workshops, (self.config.replenish_days - self.day % self.config.replenish_days) / self.config.replenish_days, *np.clip(self.stock / self.config.aircraft, 0, 1)]
        crew_mask = self.available_crew_mask()
        public.extend([float(bool(crew_mask & (1 << i))) for i in range(self.config.crews)] + [1 / (1 + factor) for factor in self.crew_time_factors])
        public.extend([float(i in shops) for i in range(self.config.workshops)] + [1 / (1 + factor) for factor in self.workshop_time_factors])
        return np.asarray(np.concatenate([np.asarray(rows).ravel(), public]), dtype=np.float32)

    def validate_row(self, aircraft, row):
        ground, engines, b0, b1, crew_mask, shop, order, priority = map(int, row)
        if not ground:
            return None if engines == b0 == b1 == 0 else 'work_without_grounding'
        actual = int(b0 != 0) + 2 * int(b1 != 0)
        if not actual or engines != actual:
            return 'inconsistent_engine_bundle'
        if not 0 < crew_mask < (1 << self.config.crews):
            return 'invalid_crew_mask'
        return None

    def _start_job(self, aircraft, row):
        _, engines, b0, b1, crew_mask, shop, order, priority = map(int, row)
        plan = Plan(aircraft, engines, (b0, b1), crew_mask, shop, order, priority)
        tasks = [(e, c) for e, b in enumerate((b0, b1)) for c in range(6) if b & 1 << c]
        tasks.sort(key=lambda x: (x[0] * 6 + x[1] - order) % 12)
        needed = np.bincount([c for _, c in tasks], minlength=6)
        _, shops = self.resource_state()
        if crew_mask & self.available_crew_mask() != crew_mask or shop not in shops:
            return ('resource_conflict', 0.0)
        if np.any(needed > self.stock):
            return ('spare_shortage', 0.0)
        work_rate = sum(1 / factor for i, factor in enumerate(self.crew_time_factors) if crew_mask & (1 << i))
        duration = self.config.setup_days
        completion_days = []
        for _, component in tasks:
            duration += max(1, math.ceil(LABOR_DAYS[component] * self.workshop_time_factors[shop] / work_rate))
            completion_days.append(duration)
        self.stock -= needed
        self.jobs[aircraft] = Job(plan, tuple(tasks), duration, duration, tuple(completion_days))
        cost = 12.0 + engines.bit_count() * 5 + sum((PART_COST[c] + 3 * self.health[aircraft, e, c] for e, c in tasks))
        self.metrics['groundings'] += 1
        return (None, cost)

    def _update_filter(self, aircraft, measurements):
        """Bootstrap particle filter with noisy observations and shock-mixture prior.

        The estimator does not receive true event type, severity or hidden health.
        It is a deliberately simple independent-component filter, not Emanuele's.
        """
        cfg = self.config
        p = self.particles[aircraft]
        wear = WEAR[None, :, None] * self.exposure[aircraft] * cfg.wear_scale
        p -= wear * self.filter_rng.normal(1, 0.12, p.shape)
        bird = self.filter_rng.random((2, 1, cfg.particles)) < cfg.bird_probability
        hail = self.filter_rng.random((2, 1, cfg.particles)) < cfg.hail_probability
        bseverity = self.filter_rng.uniform(0.1, 0.35, (2, 1, cfg.particles))
        hseverity = self.filter_rng.uniform(0.1, 0.35, (2, 1, cfg.particles))
        p -= bird * bseverity * np.array([1, 0.5, 0.15, 0, 0, 0])[None, :, None]
        p -= hail * hseverity * np.array([0.7, 0.4, 0.3, 0.1, 0.1, 0.1])[None, :, None]
        p -= (self.filter_rng.random((1, 1, cfg.particles)) < cfg.weather_probability) * wear * 2
        np.clip(p, 0, 1, out=p)
        predicted_efficiency = EFFICIENCY_BASE + EFFICIENCY_GAIN * p
        predicted_temperature = TEMPERATURE_BASE_C[None, :, None] + TEMPERATURE_HEALTH_GAIN_C * (1 - p) + TEMPERATURE_EXPOSURE_GAIN_C * (self.exposure[aircraft] - 1)
        efficiency_error = (predicted_efficiency - measurements[:, :, 0, None]) / self.sensor_noise[0]
        temperature_error = (predicted_temperature - measurements[:, :, 1, None]) / self.sensor_noise[1]
        logits = -0.5 * (efficiency_error ** 2 + temperature_error ** 2)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        for e in range(2):
            for c in range(6):
                idx = self.filter_rng.choice(cfg.particles, cfg.particles, p=weights[e, c])
                p[e, c] = np.clip(p[e, c, idx] + self.filter_rng.normal(0, 0.006, cfg.particles), 0, 1)

    def step(self, action):
        if self._done:
            raise RuntimeError('reset before step, including after episode completion')
        a = np.asarray(action)
        if not np.issubdtype(a.dtype, np.integer) or not self.action_space.contains(a):
            raise ValueError('action outside MultiDiscrete space')
        cfg = self.config
        low = np.quantile(self.particles, 0.1, axis=-1)
        mandatory = low.min(axis=(1, 2)) <= cfg.mandatory_health
        costs = 0.0
        events = []
        rejected = []
        completed = []
        grounded = set(self.jobs)
        for aircraft in sorted(range(cfg.aircraft), key=lambda i: (-int(a[i, 7]), i)):
            if aircraft in self.jobs:
                continue
            reason = self.validate_row(aircraft, a[aircraft])
            if reason is None and int(a[aircraft, 0]):
                reason, cost = self._start_job(aircraft, a[aircraft])
                costs += cost
            if reason:
                rejected.append((aircraft, reason))
                costs += 2.0
            if mandatory[aircraft] and aircraft not in self.jobs:
                grounded.add(aircraft)
                self.metrics['unsafe_attempts'] += float(not int(a[aircraft, 0]))
            if aircraft in self.jobs:
                grounded.add(aircraft)
        self.metrics['rejections'] += len(rejected)
        flights = 0
        failures = 0
        weather = self.weather_tape[self.day] < cfg.weather_probability
        for aircraft in range(cfg.aircraft):
            if aircraft in grounded:
                continue
            flights += 1
            self.health[aircraft] -= WEAR * self.exposure[aircraft] * cfg.wear_scale * np.maximum(0, self.wear_tape[self.day, aircraft]) * (3 if weather else 1)
            if weather:
                self.metrics['weather_exposures'] += 1
            for e in range(2):
                for kind, p, impact in [('bird', cfg.bird_probability, [1, 0.5, 0.15, 0, 0, 0]), ('hail', cfg.hail_probability, [0.7, 0.4, 0.3, 0.1, 0.1, 0.1])]:
                    k = int(kind == 'hail')
                    if self.event_tape[self.day, aircraft, e, k] < p:
                        severity = self.severity_tape[self.day, aircraft, e, k]
                        self.health[aircraft, e] -= severity * np.array(impact)
                        events.append({'aircraft': aircraft, 'aircraft_plate': self.aircraft_plates[aircraft], 'engine': e, 'type': kind})
                        self.metrics[kind + '_events'] += 1
            np.clip(self.health[aircraft], 0, 1, out=self.health[aircraft])
            failed = bool(np.any(self.health[aircraft] <= 0))
            failures += failed
            self.measurements[aircraft] = self._sensor_model(self.health[aircraft], self.exposure[aircraft])
            self.measurements[aircraft] += self.noise_tape[self.day, aircraft] * np.array([EFFICIENCY_GAIN, TEMPERATURE_HEALTH_GAIN_C])
            self._clip_measurements(self.measurements[aircraft])
            self.last_flight_day[aircraft] = self.day + 1
            self._update_filter(aircraft, self.measurements[aircraft])
        for aircraft, j in list(self.jobs.items()):
            elapsed = j.total_duration - j.remaining + 1
            for (e, c), task_day in zip(j.tasks, j.completion_days):
                if elapsed == task_day:
                    self.health[aircraft, e, c] = 1.0
                    self.particles[aircraft, e, c] = 1.0
                    self.metrics['replacements'] += 1
            j.remaining -= 1
            if j.remaining == 0:
                completed.append(aircraft)
                del self.jobs[aircraft]
        downtime = len(grounded)
        costs += downtime * 2 + failures * 80
        reward = (flights * 10 - costs) / (cfg.aircraft * 10)
        self.metrics['return'] += reward
        self.metrics['cost'] += costs
        self.metrics['flights'] += flights
        self.metrics['flight_failures'] += failures
        self.metrics['downtime_aircraft_days'] += downtime
        self.day += 1
        if self.day % cfg.replenish_days == 0:
            self.stock = np.minimum(cfg.aircraft, self.stock + max(1, cfg.aircraft // 2))
        self._done = self.day >= cfg.horizon
        info = {'day': self.day, 'aircraft_plates': self.aircraft_plates, 'resources': self.resource_details(), 'events': events, 'rejected': rejected, 'completed': completed, 'reward_terms': {'flight_revenue': flights * 10, 'cost': costs}, 'metrics': dict(self.metrics)}
        return (self._observation(), float(reward), self._done, False, info)

    def render(self):
        return f'day={self.day} aircraft_plates={self.aircraft_plates} busy={[self.aircraft_plates[i] for i in sorted(self.jobs)]} crews/shops={self.resource_state()} stock={self.stock.tolist()}'

def threshold_action(env, observation, threshold=0.65, bundle_threshold=0.75):
    """Feasible rule using only observations, not hidden health or event diagnostics."""
    rows, public = env.split_observation(observation)
    action = env.idle_action()
    free_mask = sum(1 << i for i in range(env.config.crews) if public[10 + i] > 0.5)
    shop_offset = 10 + 2 * env.config.crews
    shops = sorted((i for i in range(env.config.workshops) if public[shop_offset + i] > 0.5), key=lambda i: env.workshop_time_factors[i])
    stock = np.rint(public[4:10] * env.config.aircraft).astype(int)
    scores = rows[:, 24:36].min(1)
    rank = np.argsort(scores)
    for priority, aircraft in enumerate(rank[::-1]):
        action[aircraft, 7] = priority
    for aircraft in rank:
        if rows[aircraft, 37] > 0.5 or scores[aircraft] > threshold or not free_mask or (not shops):
            continue
        health = rows[aircraft, :12].reshape(2, 6)
        selected = health <= bundle_threshold
        if not selected.any():
            selected.flat[rows[aircraft, 24:36].argmin()] = True
        bundles = [0, 0]
        for e in range(2):
            for c in range(6):
                if selected[e, c] and stock[c] > 0:
                    bundles[e] |= 1 << c
                    stock[c] -= 1
        mask = int(bundles[0] != 0) + 2 * int(bundles[1] != 0)
        if not mask:
            continue
        crew = min((i for i in range(env.config.crews) if free_mask & (1 << i)), key=lambda i: env.crew_time_factors[i])
        action[aircraft] = [1, mask, *bundles, 1 << crew, shops.pop(0), int(health.argmin()), action[aircraft, 7]]
        free_mask &= ~(1 << crew)
    return action


# ========================================================================
# fleet_maintenance/adapters.py
# ========================================================================


class FlattenFleetAction(gym.ActionWrapper):
    """Stable-Baselines3 expects a one-dimensional MultiDiscrete nvec."""

    def __init__(self, env):
        super().__init__(env)
        self.action_shape = env.action_space.nvec.shape
        self.action_space = spaces.MultiDiscrete(env.action_space.nvec.ravel())

    def action(self, action):
        return np.asarray(action, dtype=np.int64).reshape(self.action_shape)

    def reverse_action(self, action):
        return np.asarray(action, dtype=np.int64).ravel()


# ========================================================================
# fleet_maintenance/policies.py
# ========================================================================


import torch
from torch import nn
from torch.distributions import Categorical

def _fleet_observation_tensor(observation, width, error_message):
    """Validate global inputs without allocating per-aircraft actor features."""
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.ndim != 2 or obs.shape[1] != width:
        raise ValueError(error_message)
    return obs


class FleetActorCritic(nn.Module):
    """Central critic + flat, conditional, or parameter-shared aircraft actors.

    shared: local aircraft estimates + public resource counts. Parameters reused
    across aircraft; a central deterministic resolver enforces resource capacity.
    conditional: global actor context and masked dependent action distributions.
    flat: shared global features with no learned conditioning on preceding choices.
    All methods use the same logical masks; scarce-resource conflicts are resolved
    by the simulator and reported separately from structural feasibility.
    All act daily and obey pre-start commitment; this is not temporal HRL.
    """

    def __init__(self, env: FleetMaintenanceEnv, mode='conditional', goal_size=0):
        super().__init__()
        if mode not in ('flat', 'conditional', 'shared'):
            raise ValueError(mode)
        self.mode = mode
        self.n = env.config.aircraft
        self.local = env.local_size
        self.public = env.public_size
        self.dims = env.row_dims.tolist()
        self.mandatory = env.config.mandatory_health
        self.goal_size = goal_size
        size = env.observation_space.shape[0]
        # Keep legacy registered layers/state keys even when this mode does not execute them.
        self.global_encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh())
        in_size = self.local + self.public + (0 if mode == 'shared' else 64) + goal_size
        self.actor = nn.Sequential(nn.Linear(in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.heads = nn.ModuleList((nn.Linear(64, d) for d in self.dims))
        self.condition = nn.ModuleList((nn.Linear(8, 64, bias=False) for _ in self.dims))
        self.critic = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, actions=None, deterministic=False, goal=None):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]
        local = obs[:, :-self.public].reshape(batch, self.n, self.local)
        public = obs[:, -self.public:].unsqueeze(1).expand(-1, self.n, -1)
        features = [local, public]
        if self.goal_size:
            if goal is None:
                raise ValueError('goal-conditioned worker requires a goal')
            features.append(goal.unsqueeze(1).expand(-1, self.n, -1))
        if self.mode != 'shared':
            features.append(self.global_encoder(obs).unsqueeze(1).expand(-1, self.n, -1))
        h = self.actor(torch.cat(features, -1))
        previous = None if self.mode == 'flat' else torch.zeros(batch, self.n, 8, device=obs.device)
        logprob = torch.zeros(batch, device=obs.device)
        entropy = torch.zeros_like(logprob)
        result = []
        busy = local[:, :, 37] > 0.5
        if self.goal_size:
            busy = busy | (goal[:, 0] > 0.5).unsqueeze(-1)
        urgent = local[:, :, 24:36].amin(-1) <= self.mandatory
        for k, head in enumerate(self.heads):
            logits = head(h if self.mode == 'flat' else torch.tanh(h + self.condition[k](previous)))
            valid = torch.ones_like(logits, dtype=torch.bool)
            valid[busy] = False
            valid[:, :, 0] |= busy
            if k == 0:
                valid[:, :, 0] &= ~(urgent & ~busy)
            else:
                ground = result[0] == 1
                if k == 1:
                    valid[:, :, 0] &= ~ground
                if k in (2, 3):
                    engine_on = result[1] & 1 << k - 2 != 0
                    active = ground & engine_on & ~busy
                    valid[:, :, 0] &= ~active
                    valid[:, :, 1:] &= active.unsqueeze(-1)
                elif k >= 4:
                    valid[:, :, 1:] &= (ground & ~busy).unsqueeze(-1)
                    if k == 4:
                        valid[:, :, 0] &= ~(ground & ~busy)
                if k == 1:
                    valid[:, :, 1:] &= (ground & ~busy).unsqueeze(-1)
            dist = Categorical(logits=logits.masked_fill(~valid, -1000000000.0))
            choice = actions[:, :, k] if actions is not None else dist.probs.argmax(-1) if deterministic else dist.sample()
            result.append(choice)
            logprob += dist.log_prob(choice).sum(-1)
            entropy += dist.entropy().sum(-1)
            if self.mode != 'flat' and k < len(self.heads) - 1:
                # Earlier conditioning layers save their inputs for backward; do not mutate in place.
                addition = torch.zeros_like(previous)
                addition[:, :, k] = choice / float(max(1, self.dims[k] - 1))
                previous = previous + addition
        return (torch.stack(result, -1), logprob, entropy, self.value(obs))

class OptionActorCritic(nn.Module):
    """Learned manager selects one of four fixed fleet planning options."""

    def __init__(self, env):
        super().__init__()
        size = env.observation_space.shape[0]
        self.encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 4)
        self.critic = nn.Linear(64, 1)

    def value(self, obs):
        return self.critic(self.encoder(obs)).squeeze(-1)

    def act(self, obs, actions=None, deterministic=False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        dist = Categorical(logits=self.actor(h))
        action = actions if actions is not None else dist.probs.argmax(-1) if deterministic else dist.sample()
        return (action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1))

class LearnedOptionActorCritic(nn.Module):
    """Jointly trained manager + goal-conditioned plan worker + fixed executor.

    Macro 0 waits; macros 1..3 are latent planning contexts, not certified skills.
    The worker selects engines, components and resources once before execution.
    Both levels learn from fleet return. This is not MAXQ or option-critic and
    option collapse/credit assignment require later ablations and diagnostics.
    """

    def __init__(self, env):
        super().__init__()
        self.manager = OptionActorCritic(env)
        self.worker = FleetActorCritic(env, 'conditional', goal_size=4)
        self.n = env.config.aircraft

    def value(self, obs):
        return self.manager.value(obs)

    def act(self, obs, actions=None, deterministic=False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        macro = None if actions is None else actions[:, 0]
        plan = None if actions is None else actions[:, 1:].reshape(-1, self.n, 8)
        option, lp, entropy, value = self.manager.act(obs, macro, deterministic)
        goal = torch.nn.functional.one_hot(option, num_classes=4).float()
        plan, worker_lp, worker_entropy, _ = self.worker.act(obs, plan, deterministic, goal)
        composite = torch.cat([option.unsqueeze(-1), plan.flatten(1)], -1)
        return (composite, lp + worker_lp, entropy + worker_entropy, value)
# Multi-agent maintenance: explicit proposal ownership and a common resource broker.
MAPPO_METHODS = ('mappo_aircraft', 'mappo_component')


def mappo_agent_inputs(observation, aircraft, local_size, public_size, assignment):
    """Build restricted actor inputs from observed beliefs, never hidden health."""
    obs = _fleet_observation_tensor(observation, aircraft * local_size + public_size,
                                    'incorrect fleet observation shape')
    local = obs[:, :aircraft * local_size].reshape(-1, aircraft, local_size)
    public = obs[:, -public_size:]
    if assignment == 'aircraft':
        # Each aircraft sees only its own row plus the common resource board.
        return torch.cat([local, public.unsqueeze(1).expand(-1, aircraft, -1)], dim=-1)
    if assignment != 'component':
        raise ValueError('assignment must be aircraft or component')
    # Ten own-component values: two engines x mean/std/lower/efficiency/temperature.
    indices = [[c, c + 6, c + 12, c + 18, c + 24, c + 30,
                c + 45, c + 51, c + 57, c + 63, 36, 37, 38, 69] for c in range(6)]
    own = torch.stack([local[:, :, ix] for ix in indices], dim=1)
    # A broadcast safety-criticality report is an explicit cross-component message.
    criticality = local[:, :, 24:36].amin(-1)
    board = criticality[:, None, :, None].expand(-1, 6, -1, -1)
    features = torch.cat([own, board], dim=-1).flatten(2)
    return torch.cat([features, public.unsqueeze(1).expand(-1, 6, -1)], dim=-1)


class MultiAgentFleetEnv:
    """Cooperative dictionary API around the same physical fleet simulator.

    Aircraft agents propose two six-bit bundles; six component agents each
    propose a two-engine mask for every aircraft. Both cover the same twelve
    replacement bits per aircraft. They do not learn crew/shop/order choices.
    A public request board and deterministic centralized dispatcher coordinate
    execution. This is decentralized proposal generation with centralized
    arbitration, not fully decentralized resource allocation or learned messaging.
    This small adapter is not a PettingZoo/RLlib API implementation.
    """

    def __init__(self, config=None, assignment='aircraft'):
        if assignment not in ('aircraft', 'component'):
            raise ValueError('assignment must be aircraft or component')
        self.env = FleetMaintenanceEnv(config)
        self.assignment = assignment
        self.agent_ids = self.env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        self.num_agents = len(self.agent_ids)
        dims = [64, 64] if assignment == 'aircraft' else [4] * self.env.config.aircraft
        self.action_spaces = {agent: spaces.MultiDiscrete(dims) for agent in self.agent_ids}
        width = self.env.local_size + self.env.public_size if assignment == 'aircraft' else 15 * self.env.config.aircraft + self.env.public_size
        self.observation_spaces = {agent: spaces.Box(0, 1, shape=(width,), dtype=np.float32) for agent in self.agent_ids}
        self.central_observation = None
        self.coordination_deferrals = 0

    def observations(self):
        views = mappo_agent_inputs(self.central_observation, self.env.config.aircraft,
                                   self.env.local_size, self.env.public_size, self.assignment)[0].numpy()
        return {agent: views[i].copy() for i, agent in enumerate(self.agent_ids)}

    def reset(self, *, seed=None, options=None):
        self.central_observation, info = self.env.reset(seed=seed, options=options)
        self.coordination_deferrals = 0
        info.update(agent_ids=self.agent_ids, central_observation=self.central_observation.copy())
        return self.observations(), info

    def compose_action(self, actions):
        """Merge owned requests and allocate resources using observed data only."""
        if self.central_observation is None:
            raise RuntimeError('reset before composing agent proposals')
        if not isinstance(actions, dict) or set(actions) != set(self.agent_ids):
            raise ValueError('supply exactly one proposal for each agent ID')
        for agent in self.agent_ids:
            proposal = np.asarray(actions[agent])
            if not np.issubdtype(proposal.dtype, np.integer) or not self.action_spaces[agent].contains(proposal):
                raise ValueError(f'invalid proposal for {agent}')
        raw = np.stack([actions[agent] for agent in self.agent_ids])
        if self.assignment == 'aircraft':
            bundles = raw.copy()
        else:
            # Specialist c can set only component bit c, independently per engine.
            bundles = np.zeros((self.env.config.aircraft, 2), dtype=np.int64)
            for c in range(6):
                for engine in range(2):
                    bundles[:, engine] |= ((raw[c] >> engine) & 1) << c
        rows, public = self.env.split_observation(self.central_observation)
        free_mask = sum(1 << i for i in range(self.env.config.crews) if public[10 + i] > 0.5)
        offset = 10 + 2 * self.env.config.crews
        shops = [i for i in range(self.env.config.workshops) if public[offset + i] > 0.5]
        stock = np.rint(public[4:10] * self.env.config.aircraft).astype(int)
        scores = rows[:, 24:36].min(1)
        rank = np.argsort(scores, kind='stable')
        fleet_action = self.env.idle_action()
        deferred = []
        for position, aircraft in enumerate(rank):
            # The broker collects per-aircraft criticality reports; it is not a critic.
            fleet_action[aircraft, 7] = self.env.config.aircraft - position - 1
            if rows[aircraft, 37] > 0.5 or not bundles[aircraft].any():
                continue
            b0, b1 = map(int, bundles[aircraft])
            tasks = [(e, c) for e, b in enumerate((b0, b1)) for c in range(6) if b & (1 << c)]
            needed = np.bincount([c for _, c in tasks], minlength=6)
            if not free_mask or not shops:
                deferred.append((int(aircraft), 'resource_conflict'))
                continue
            if np.any(needed > stock):
                deferred.append((int(aircraft), 'spare_shortage'))
                continue
            crew = min((i for i in range(self.env.config.crews) if free_mask & (1 << i)), key=lambda i: self.env.crew_time_factors[i])
            shop = min(shops, key=lambda i: self.env.workshop_time_factors[i])
            first = min(tasks, key=lambda pair: rows[aircraft, pair[0] * 6 + pair[1]])
            engines = int(b0 != 0) + 2 * int(b1 != 0)
            fleet_action[aircraft, :7] = [1, engines, b0, b1, 1 << crew, shop, first[0] * 6 + first[1]]
            free_mask &= ~(1 << crew)
            shops.remove(shop)
            stock -= needed
        return fleet_action, deferred

    def step(self, actions):
        fleet_action, deferred = self.compose_action(actions)
        self.central_observation, reward, terminated, truncated, info = self.env.step(fleet_action)
        self.coordination_deferrals += len(deferred)
        info['metrics']['coordination_deferrals'] = self.coordination_deferrals
        info.update(agent_ids=self.agent_ids, agent_proposals={key: np.asarray(value).tolist() for key, value in actions.items()},
                    fleet_action=fleet_action.tolist(), deferred_requests=deferred,
                    central_observation=self.central_observation.copy())
        # The cooperative reward is replicated, not divided by the number of agents.
        rewards = {agent: reward for agent in self.agent_ids}
        return self.observations(), rewards, terminated, truncated, info


class MultiAgentActorCritic(nn.Module):
    """Feed-forward MAPPO prototype with explicit per-agent policy statistics.

    Aircraft agents share actor weights; six specialist actors have separate
    weights. The centralized critic has one value head per agent and receives
    the full observed fleet belief, never the simulator's hidden health.
    Actors can execute through policy(agent_observations) without the critic.
    """

    def __init__(self, env, assignment='aircraft'):
        super().__init__()
        if assignment not in ('aircraft', 'component'):
            raise ValueError('assignment must be aircraft or component')
        self.assignment = assignment
        self.n = env.config.aircraft
        self.local = env.local_size
        self.public = env.public_size
        self.agent_ids = env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        self.num_agents = len(self.agent_ids)
        self.input_size = self.local + self.public if assignment == 'aircraft' else 15 * self.n + self.public
        self.fields = 2 if assignment == 'aircraft' else self.n
        self.categories = 64 if assignment == 'aircraft' else 4
        count = 1 if assignment == 'aircraft' else 6
        self.actors = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                                                  nn.Linear(64, 64), nn.Tanh(),
                                                  nn.Linear(64, self.fields * self.categories)) for _ in range(count)])
        self.critic = nn.Sequential(nn.Linear(env.observation_space.shape[0], 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.num_agents))

    def agent_inputs(self, obs):
        return mappo_agent_inputs(obs, self.n, self.local, self.public, self.assignment)

    def policy(self, agent_obs, actions=None, deterministic=False):
        """Execute only restricted actors; no central observation or critic needed."""
        x = torch.as_tensor(agent_obs, dtype=torch.float32)
        if x.ndim != 3 or x.shape[1:] != (self.num_agents, self.input_size):
            raise ValueError('incorrect per-agent observation shape')
        if self.assignment == 'aircraft':
            logits = self.actors[0](x).reshape(-1, self.num_agents, self.fields, self.categories)
            busy = (x[:, :, 37] > 0.5).unsqueeze(-1).expand(-1, -1, 2)
        else:
            logits = torch.stack([actor(x[:, i]) for i, actor in enumerate(self.actors)], dim=1)
            logits = logits.reshape(-1, 6, self.n, 4)
            busy = x[:, :, :15 * self.n].reshape(-1, 6, self.n, 15)[:, :, :, 11] > 0.5
        valid = (~busy).unsqueeze(-1) | (torch.arange(self.categories, device=x.device) == 0)
        distribution = Categorical(logits=logits.masked_fill(~valid, -1e9))
        if actions is None:
            selected = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        else:
            selected = torch.as_tensor(actions, device=x.device)
            if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
                raise ValueError('incorrect stored agent action shape or dtype')
            if torch.any((selected < 0) | (selected >= self.categories)) or torch.any(busy & (selected != 0)):
                raise ValueError('stored agent action violates a category or busy mask')
        # Sum fields WITHIN each agent; preserve the agent axis for MAPPO clipping.
        return selected, distribution.log_prob(selected).sum(-1), distribution.entropy().sum(-1)

    def active_agents(self, agent_obs):
        if self.assignment == 'aircraft':
            return (agent_obs[:, :, 37] <= 0.5).float()
        busy = agent_obs[:, :, :15 * self.n].reshape(-1, 6, self.n, 15)[:, :, :, 11] > 0.5
        return (~busy).any(-1).float()

    def value(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32)
        return self.critic(x.unsqueeze(0) if x.ndim == 1 else x)

    def act(self, obs, actions=None, deterministic=False):
        selected, logp, entropy = self.policy(self.agent_inputs(obs), actions, deterministic)
        return selected, logp, entropy, self.value(obs)

    def action_dict(self, selected):
        raw = torch.as_tensor(selected).detach().cpu().numpy()
        if raw.shape != (self.num_agents, self.fields):
            raise ValueError('incorrect unbatched agent action shape')
        return {agent: raw[i].copy() for i, agent in enumerate(self.agent_ids)}


DCMAC_METHODS = ('dcmac_aircraft', 'ddmac_aircraft', 'dcmac_component', 'ddmac_component')
REQUEST_METHODS = (*MAPPO_METHODS, *DCMAC_METHODS)


def request_assignment(method):
    """Resolve ownership for registered proposal methods; reject silent fallbacks."""
    if method not in REQUEST_METHODS:
        raise ValueError('unknown request policy method')
    return method.rsplit('_', 1)[1]


class DCMACActorCritic(MultiAgentActorCritic):
    """Fleet adaptation of DCMAC/DDMAC; global observed beliefs enter all actors.

    DCMAC has one actor with factorized control-unit outputs. DDMAC has an
    independently parameterized actor per control unit. Both have one scalar
    centralized value network and use replay with joint importance weighting.
    The fixed dispatcher owns resources; paper-specific inspection actions,
    budget augmentation and primal-dual risk constraints are not reproduced.
    Inherits only the proposal-to-dictionary conversion from the MAPPO class.
    """

    def __init__(self, env, assignment='aircraft', decentralized=False):
        nn.Module.__init__(self)
        if assignment not in ('aircraft', 'component'):
            raise ValueError('assignment must be aircraft or component')
        self.assignment = assignment
        self.decentralized = decentralized
        self.n = env.config.aircraft
        self.local = env.local_size
        self.public = env.public_size
        self.agent_ids = env.aircraft_plates if assignment == 'aircraft' else COMPONENTS
        self.num_agents = len(self.agent_ids)
        self.input_size = env.observation_space.shape[0]
        self.fields = 2 if assignment == 'aircraft' else self.n
        self.categories = 64 if assignment == 'aircraft' else 4
        count = self.num_agents if decentralized else 1
        output = self.fields * self.categories * (1 if decentralized else self.num_agents)
        self.actors = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                          nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, output))
            for _ in range(count)
        ])
        self.critic = nn.Sequential(nn.Linear(self.input_size, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def agent_inputs(self, obs):
        """Return global actor inputs; unlike MAPPO, execution needs fleet beliefs."""
        return _fleet_observation_tensor(obs, self.input_size,
                                         'incorrect global actor observation shape')

    def _busy_fields(self, obs):
        busy = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)[:, :, 37] > 0.5
        if self.assignment == 'aircraft':
            return busy.unsqueeze(-1).expand(-1, -1, 2)
        return busy.unsqueeze(1).expand(-1, 6, -1)

    def active_agents(self, obs):
        return (~self._busy_fields(self.agent_inputs(obs))).any(-1).float()

    def policy(self, obs, actions=None, deterministic=False):
        """Factorized categorical proposals; no hidden state or critic evaluation."""
        x = self.agent_inputs(obs)
        if self.decentralized:
            logits = torch.stack([actor(x) for actor in self.actors], dim=1)
        else:
            logits = self.actors[0](x)
        logits = logits.reshape(-1, self.num_agents, self.fields, self.categories)
        busy = self._busy_fields(x)
        valid = (~busy).unsqueeze(-1) | (torch.arange(self.categories, device=x.device) == 0)
        distribution = Categorical(logits=logits.masked_fill(~valid, -1e9))
        if actions is None:
            selected = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        else:
            selected = torch.as_tensor(actions, device=x.device)
            if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
                raise ValueError('incorrect stored agent action shape or dtype')
            if torch.any((selected < 0) | (selected >= self.categories)) or torch.any(busy & (selected != 0)):
                raise ValueError('stored agent action violates a category or busy mask')
        return selected, distribution.log_prob(selected).sum(-1), distribution.entropy().sum(-1)

    def value(self, obs):
        return self.critic(self.agent_inputs(obs)).squeeze(-1)

    def act(self, obs, actions=None, deterministic=False):
        selected, logp, entropy = self.policy(obs, actions, deterministic)
        return selected, logp, entropy, self.value(obs)


def dcmac_losses(logp, behavior_logp, values, next_values, rewards, dones,
                 gamma=0.99, importance_clip=2.0):
    """One-step replay update: joint truncated ratio and detached TD advantage.

    The ratio is pi(joint proposal|belief)/mu(joint proposal|belief), not the
    product of separately truncated ratios. Truncation introduces bias but
    bounds the high-variance replay weights. No PPO clipping or GAE is used.
    """
    if logp.ndim != 2 or behavior_logp.shape != logp.shape:
        raise ValueError('proposal log probabilities must have batch x agent shape')
    if any(t.shape != (len(logp),) for t in (values, next_values, rewards, dones)):
        raise ValueError('one scalar value, reward and terminal flag per replay transition required')
    if not np.isfinite(importance_clip) or importance_clip <= 0 or not 0 <= gamma <= 1:
        raise ValueError('positive finite importance cap and gamma in [0,1] required')
    if not all(bool(torch.isfinite(t).all()) for t in (logp, behavior_logp, values, next_values, rewards, dones)):
        raise ValueError('replay tensors must be finite')
    joint_logp = logp.sum(-1)
    log_ratio = (joint_logp - behavior_logp.sum(-1)).detach()
    weights = torch.exp(log_ratio.clamp(max=float(np.log(importance_clip))))
    target = rewards + gamma * (1 - dones.float()) * next_values.detach()
    delta = target - values
    actor_loss = -(weights * joint_logp * delta.detach()).mean()
    critic_loss = 0.5 * (weights * delta.square()).mean()
    return actor_loss, critic_loss, weights, target.detach()


def train_dcmac(config, method='dcmac_aircraft', steps=4096, seed=0,
                rollout_size=128, replay_capacity=4096, importance_clip=2.0):
    """Off-policy DCMAC/DDMAC adaptation with uniform bounded experience replay.

    Stochastic actor sampling is the behavior policy; its original proposal
    likelihood is stored even when the dispatcher defers that proposal.
    One replay minibatch is updated per day after two samples are available.
    rollout_size names minibatch size here, preserving the common train API.
    """
    if method not in DCMAC_METHODS or steps < 1 or rollout_size < 2 or replay_capacity < 2:
        raise ValueError('registered DCMAC/DDMAC method, positive steps and batch/capacity >= 2 required')
    if not np.isfinite(importance_clip) or importance_clip <= 0:
        raise ValueError('importance_clip must be positive and finite')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    assignment = request_assignment(method)
    env = MultiAgentFleetEnv(config, assignment)
    model = DCMACActorCritic(env.env, assignment, method.startswith('ddmac_'))
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    replay, logs = [], []
    _, info = env.reset(seed=seed)
    obs = info['central_observation']
    episode = 0
    for day in range(1, steps + 1):
        with torch.no_grad():
            action, behavior_logp, _ = model.policy(obs)
        _, rewards, done, truncated, info = env.step(model.action_dict(action[0]))
        next_obs = info['central_observation']
        terminal = done or truncated
        transition = (obs.copy(), action[0].numpy().copy(), behavior_logp[0].numpy().copy(),
                      rewards[model.agent_ids[0]], next_obs.copy(), terminal, day)
        if len(replay) < replay_capacity:
            replay.append(transition)
        else:
            replay[(day - 1) % replay_capacity] = transition
        if len(replay) >= 2:
            indices = rng.choice(len(replay), size=min(rollout_size, len(replay)), replace=False)
            batch = [replay[int(i)] for i in indices]
            states, actions, behavior, reward, following, terminal_flags, sampled_days = [
                torch.as_tensor(np.asarray(items)) for items in zip(*batch)
            ]
            _, logp, entropy = model.policy(states, actions)
            values = model.value(states)
            with torch.no_grad():
                next_values = model.value(following)
            actor_loss, value_loss, weights, _ = dcmac_losses(
                logp, behavior, values, next_values, reward.float(), terminal_flags,
                importance_clip=importance_clip
            )
            active = model.active_agents(states)
            entropy_bonus = (entropy * active).sum() / active.sum().clamp_min(1)
            policy_loss = actor_loss - 0.001 * entropy_bonus
            if not bool(torch.isfinite(policy_loss + value_loss)):
                raise RuntimeError('nonfinite DCMAC/DDMAC loss')
            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
            if bool(active.any()):
                actor_optimizer.step()
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            critic_optimizer.step()
            logs.append({'record_type': 'update', 'method': method, 'simulated_days': day,
                         'policy_loss': float(actor_loss.detach()), 'value_loss': float(value_loss.detach()),
                         'importance_mean': float(weights.mean()), 'importance_max': float(weights.max()),
                         'replay_size': len(replay), 'batch_size': len(batch),
                         'sampled_transition_days': sampled_days.tolist(),
                         'num_agents': model.num_agents})
        obs = next_obs
        if terminal:
            logs.append({'record_type': 'episode', 'episode': episode, 'simulated_days': day, **info['metrics']})
            episode += 1
            _, info = env.reset(seed=seed + episode)
            obs = info['central_observation']
    return model, logs, steps


def mappo_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Daily generalized advantages with a separate value baseline per agent."""
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if values.ndim != 2 or next_values.shape != values.shape or rewards.shape != (len(values),) or dones.shape != rewards.shape:
        raise ValueError('MAPPO rollout shapes must be time x agent, with one team reward and done per time')
    advantages = np.zeros_like(values)
    carry = np.zeros(values.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        live = 1 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * live - values[t]
        carry = delta + gamma * lam * live * carry
        advantages[t] = carry
    return advantages, advantages + values


def mappo_policy_loss(new_logp, old_logp, advantages, active, clip=0.2):
    """Clip each agent ratio BEFORE averaging; never form a fleet joint ratio."""
    if new_logp.ndim != 2 or not (new_logp.shape == old_logp.shape == advantages.shape == active.shape):
        raise ValueError('MAPPO policy tensors must have identical batch x agent shapes')
    ratio = torch.exp(new_logp - old_logp)
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -(surrogate * active).sum() / active.sum().clamp_min(1)


def train_mappo(config, method='mappo_aircraft', steps=4096, seed=0, rollout_size=128):
    """Train decentralized proposals with centralized observed-belief value heads.

    Educational adaptation of MAPPO, not an exact benchmark reproduction:
    feed-forward actors, four full-rollout epochs, clipped values, no recurrence,
    reward/value normalization, learned communication, or hierarchical options.
    """
    if method not in MAPPO_METHODS or steps < 1 or rollout_size < 2:
        raise ValueError('valid MAPPO method, positive steps, and rollout_size >= 2 required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    assignment = 'aircraft' if method == 'mappo_aircraft' else 'component'
    env = MultiAgentFleetEnv(config, assignment)
    model = MultiAgentActorCritic(env.env, assignment)
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    _, info = env.reset(seed=seed)
    obs = info['central_observation']
    days, episode = 0, 0
    logs = []
    while days < steps:
        data = {key: [] for key in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'done', 'active')}
        while len(data['obs']) < rollout_size and days < steps:
            with torch.no_grad():
                action, logp, _, values = model.act(torch.tensor(obs))
                active = model.active_agents(model.agent_inputs(obs))[0]
            _, rewards, done, truncated, info = env.step(model.action_dict(action[0]))
            next_obs = info['central_observation']
            with torch.no_grad():
                next_value = model.value(next_obs)[0].numpy()
            team_reward = rewards[model.agent_ids[0]]
            terminal = done or truncated
            items = (obs.copy(), action[0].numpy(), logp[0].numpy(), values[0].numpy(), next_value, team_reward, terminal, active.numpy())
            for key, item in zip(data, items):
                data[key].append(item)
            days += 1
            obs = next_obs
            if terminal:
                logs.append({'record_type': 'episode', 'episode': episode, 'simulated_days': days, **info['metrics']})
                episode += 1
                _, info = env.reset(seed=seed + episode)
                obs = info['central_observation']
        advantages, targets = mappo_gae(data['reward'], data['value'], data['next_value'], data['done'])
        tensors = {key: torch.as_tensor(np.asarray(value)) for key, value in data.items() if key in ('obs', 'action', 'logp', 'value', 'active')}
        advantages = torch.tensor(advantages)
        targets = torch.tensor(targets)
        active = tensors['active']
        counts = active.sum(0).clamp_min(1)
        mean = (advantages * active).sum(0) / counts
        variance = ((advantages - mean).square() * active).sum(0) / counts
        advantages = (advantages - mean) / torch.sqrt(variance + 1e-8) * active
        for _ in range(4):
            _, newlog, entropy, values = model.act(tensors['obs'], tensors['action'])
            policy_loss = mappo_policy_loss(newlog, tensors['logp'], advantages, active)
            entropy_bonus = (entropy * active).sum() / active.sum().clamp_min(1)
            clipped_values = tensors['value'] + (values - tensors['value']).clamp(-0.2, 0.2)
            value_loss = torch.maximum((values - targets).square(), (clipped_values - targets).square()).mean()
            actor_loss = policy_loss - 0.001 * entropy_bonus
            if not torch.isfinite(actor_loss + value_loss):
                raise RuntimeError('nonfinite MAPPO loss')
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
            if bool(active.any()):
                actor_optimizer.step()
            critic_optimizer.zero_grad()
            (0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            critic_optimizer.step()
        logs.append({'record_type': 'update', 'simulated_days': days, 'policy_loss': float(policy_loss.detach()),
                     'value_loss': float(value_loss.detach()), 'entropy': float(entropy_bonus.detach()),
                     'num_agents': model.num_agents, 'method': method})
    return model, logs, days

OPTION_NAMES = ('wait_one_day', 'targeted_maintenance', 'opportunistic_bundle', 'preventive_bundle')

HRL_METHODS = ('concurrent_options', 'planning_hierarchy', 'aggregate_targets',
               'maxq', 'option_critic', 'hierarchical_vdn', 'hrl_flat')
HRL_PPO_METHODS = ('concurrent_options', 'planning_hierarchy', 'aggregate_targets', 'hrl_flat')
HRL_BUNDLES = (1, 2, 4, 8, 16, 32, 63)
HRL_CATALOG = ((0, 0),) + tuple((b if e & 1 else 0, b if e & 2 else 0)
                                for e in (1, 2, 3) for b in HRL_BUNDLES)
HRL_BUDGETS = (0, 1, 2, 12)


def hrl_mlp(width, output):
    return nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, output))


def hrl_catalog_mask(observation, aircraft, local_size=70, mandatory=0.08, goals=None):
    """Structural masks only; the shared broker resolves stock/resource competition."""
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    rows = obs[:, :aircraft * local_size].reshape(-1, aircraft, local_size)
    busy = rows[:, :, 37] > 0.5
    urgent = rows[:, :, 24:36].amin(-1) <= mandatory
    valid = torch.ones((*busy.shape, len(HRL_CATALOG)), dtype=torch.bool, device=obs.device)
    valid[:, :, 0] = ~urgent | busy
    valid[:, :, 1:] &= ~busy.unsqueeze(-1)
    if goals is not None:
        goals = torch.as_tensor(goals, device=obs.device, dtype=torch.long)
        if goals.shape != busy.shape or torch.any((goals < 0) | (goals >= len(HRL_BUDGETS))):
            raise ValueError('one valid budget goal per aircraft required')
        counts = torch.tensor([sum(int(b).bit_count() for b in pair) for pair in HRL_CATALOG], device=obs.device)
        budgets = torch.tensor(HRL_BUDGETS, device=obs.device)[goals]
        valid &= counts <= budgets.unsqueeze(-1)
        valid[:, :, 0] |= busy
    return valid


def hrl_categorical(logits, valid, selected=None, deterministic=False):
    if logits.shape != valid.shape or not bool(valid.any(-1).all()):
        raise ValueError('categorical choices require matching masks with a feasible option')
    dist = Categorical(logits=logits.masked_fill(~valid, -1e9))
    if selected is None:
        selected = dist.probs.argmax(-1) if deterministic else dist.sample()
    else:
        selected = torch.as_tensor(selected, device=logits.device)
        if selected.shape != logits.shape[:-1] or selected.dtype not in (torch.int32, torch.int64):
            raise ValueError('incorrect hierarchical stored choice shape/dtype')
        if torch.any((selected < 0) | (selected >= logits.shape[-1])):
            raise ValueError('hierarchical choice outside action categories')
        if not bool(valid.gather(-1, selected.unsqueeze(-1)).all()):
            raise ValueError('hierarchical choice violates a feasibility mask')
    return selected, dist.log_prob(selected), dist.entropy()


class AircraftOptionEnv:
    """Daily fleet interface with independently committed aircraft repair options.

    Catalog 0 operates; other entries request one engine, the other, or both,
    with one component type or a full six-component bundle. Both engines use
    the same bundle if selected. This is a strict subset of the older action
    space. Every new method shares this catalog and the same fixed broker.
    A new decision for a free aircraft never waits for another aircraft's job.
    """

    def __init__(self, config=None):
        self.adapter = MultiAgentFleetEnv(config, 'aircraft')
        self.env = self.adapter.env
        self.n = self.env.config.aircraft
        self.active_options = {}
        self.option_history = []
        self.central_observation = None

    def reset(self, *, seed=None, options=None):
        _, info = self.adapter.reset(seed=seed, options=options)
        self.central_observation = info['central_observation']
        self.active_options.clear()
        self.option_history.clear()
        self.last_info = info
        return self.central_observation.copy(), info

    def decision_mask(self):
        if self.central_observation is None:
            raise RuntimeError('reset before reading aircraft options')
        rows, _ = self.env.split_observation(self.central_observation)
        return rows[:, 37] <= 0.5

    def step(self, choices):
        choices = np.asarray(choices)
        if choices.shape != (self.n,) or not np.issubdtype(choices.dtype, np.integer):
            raise ValueError('one integer catalog action per aircraft required')
        if np.any((choices < 0) | (choices >= len(HRL_CATALOG))):
            raise ValueError('unknown maintenance catalog action')
        free = self.decision_mask()
        if np.any((~free) & (choices != 0)):
            raise ValueError('committed aircraft options accept no new planning request')
        previous = {i: job.plan for i, job in self.env.jobs.items()}
        day = self.env.day
        proposals = {plate: np.asarray(HRL_CATALOG[int(choices[i])], dtype=np.int64)
                     for i, plate in enumerate(self.env.aircraft_plates)}
        _, rewards, done, truncated, info = self.adapter.step(proposals)
        for i, plan in previous.items():
            if i in self.env.jobs and self.env.jobs[i].plan != plan:
                raise RuntimeError('a committed aircraft plan was changed')
        started, finished = [], []
        for i, job in self.env.jobs.items():
            if i not in previous:
                record = {'aircraft': i, 'aircraft_plate': self.env.aircraft_plates[i],
                          'start_day': day, 'catalog_action': int(choices[i]),
                          'planned_duration': job.total_duration}
                self.active_options[i] = record
                started.append(dict(record))
        for i in info['completed']:
            record = dict(self.active_options.pop(i))
            record.update(end_day=self.env.day, duration=self.env.day - record['start_day'])
            self.option_history.append(record)
            finished.append(record)
        self.central_observation = info['central_observation']
        info.update(aircraft_option_started=started, aircraft_option_completed=finished,
                    decision_mask_before=free.tolist(), active_aircraft_options=sorted(self.active_options))
        info['metrics'].update(option_starts=len(self.option_history) + len(self.active_options),
                               option_completions=len(self.option_history))
        self.last_info = info
        return self.central_observation.copy(), rewards[self.env.aircraft_plates[0]], done, truncated, info


class AircraftPlanningHierarchy(nn.Module):
    """Independent aircraft options with learned planning stages or explicit targets.

    Ground/engine/bundle selections are computational planning stages, not
    three elapsed physical epochs. Temporal abstraction is the committed job.
    All actors use observed global context; this is not local-information CTDE.
    """

    def __init__(self, env, mode='planning_hierarchy'):
        super().__init__()
        if mode not in HRL_PPO_METHODS:
            raise ValueError('unknown aircraft planning hierarchy')
        self.mode = mode
        self.n = env.config.aircraft
        self.local = env.local_size
        self.width = env.observation_space.shape[0]
        self.mandatory = env.config.mandatory_health
        features = self.width + self.local
        if mode == 'hrl_flat':
            self.flat_policy = hrl_mlp(features, len(HRL_CATALOG))
        else:
            self.manager = hrl_mlp(features, 4 if mode == 'aggregate_targets' else 2)
            if mode == 'planning_hierarchy':
                self.engine_policy = hrl_mlp(features + 2, 3)
                self.component_policy = hrl_mlp(features + 5, 7)
            if mode == 'aggregate_targets':
                self.worker = hrl_mlp(features + 4, len(HRL_CATALOG))
        self.critic = hrl_mlp(self.width, self.n)

    def features(self, observation):
        obs = _fleet_observation_tensor(observation, self.width,
                                        'incorrect hierarchical fleet observation')
        local = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)
        return obs, local, torch.cat([obs[:, None].expand(-1, self.n, -1), local], -1)

    def value(self, observation):
        # A critic needs global inputs, not the repeated actor-context tensor.
        obs = _fleet_observation_tensor(observation, self.width,
                                        'incorrect hierarchical fleet observation')
        return self.critic(obs)

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('critic.')]

    def act(self, observation, actions=None, deterministic=False):
        obs, local, features = self.features(observation)
        if actions is not None:
            actions = torch.as_tensor(actions)
            if actions.shape != (len(obs), self.n, 3) or actions.dtype not in (torch.int32, torch.int64):
                raise ValueError('hierarchical actions must be integer batch x aircraft x 3')
        mask = hrl_catalog_mask(obs, self.n, self.local, self.mandatory)
        busy = local[:, :, 37] > 0.5
        urgent = local[:, :, 24:36].amin(-1) <= self.mandatory
        zero = torch.zeros_like(busy, dtype=torch.long)
        if self.mode == 'hrl_flat':
            choice, logp, entropy = hrl_categorical(self.flat_policy(features), mask,
                None if actions is None else actions[:, :, 0], deterministic)
            stored = torch.stack([choice, zero, zero], -1)
        else:
            count = 4 if self.mode == 'aggregate_targets' else 2
            valid = (~busy).unsqueeze(-1).expand(-1, -1, count).clone()
            valid[:, :, 0] = ~urgent | busy
            goal, logp, entropy = hrl_categorical(self.manager(features), valid,
                None if actions is None else actions[:, :, 0], deterministic)
            if self.mode == 'planning_hierarchy':
                active = (goal == 1) & ~busy
                engine_valid = active.unsqueeze(-1).expand(-1, -1, 3).clone()
                engine_valid[:, :, 0] |= ~active
                engine, lp, ent = hrl_categorical(self.engine_policy(torch.cat([
                    features, nn.functional.one_hot(goal, 2).float()], -1)), engine_valid,
                    None if actions is None else actions[:, :, 1], deterministic)
                logp, entropy = logp + lp, entropy + ent
                bundle_valid = active.unsqueeze(-1).expand(-1, -1, 7).clone()
                bundle_valid[:, :, 0] |= ~active
                bundle, lp, ent = hrl_categorical(self.component_policy(torch.cat([
                    features, nn.functional.one_hot(goal, 2).float(),
                    nn.functional.one_hot(engine, 3).float()], -1)), bundle_valid,
                    None if actions is None else actions[:, :, 2], deterministic)
                logp, entropy = logp + lp, entropy + ent
                stored = torch.stack([goal, engine, bundle], -1)
            elif self.mode == 'aggregate_targets':
                # A positive goal is a maximum replacement count, not a guaranteed improvement.
                worker_valid = hrl_catalog_mask(obs, self.n, self.local, self.mandatory, goal)
                worker_valid[:, :, 0] = (goal == 0) | busy
                choice, lp, ent = hrl_categorical(self.worker(torch.cat([
                    features, nn.functional.one_hot(goal, 4).float()], -1)), worker_valid,
                    None if actions is None else actions[:, :, 1], deterministic)
                logp, entropy = logp + lp, entropy + ent
                stored = torch.stack([goal, choice, zero], -1)
            else:
                stored = torch.stack([goal, zero, zero], -1)
        if actions is not None and not torch.equal(stored, actions):
            raise ValueError('unused hierarchical action fields must be zero')
        return stored, logp, entropy, self.value(obs)

    def catalog_actions(self, observation, actions):
        actions = torch.as_tensor(actions)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError('decode one fleet action at a time')
            actions = actions[0]
        if actions.shape != (self.n, 3):
            raise ValueError('incorrect hierarchical action for decoding')
        if self.mode == 'hrl_flat':
            return actions[:, 0].numpy().copy()
        if self.mode == 'aggregate_targets':
            return actions[:, 1].numpy().copy()
        if self.mode == 'planning_hierarchy':
            return torch.where(actions[:, 0] != 0, 1 + actions[:, 1] * 7 + actions[:, 2], 0).numpy()
        _, local, _ = self.features(observation)
        means = local[0, :, :12].reshape(self.n, 2, 6)
        engine = means.amin(-1).argmin(-1)
        selected = means[torch.arange(self.n), engine]
        component = selected.argmin(-1)
        bundle = torch.where((selected < .75).sum(-1) >= 3, 6, component)
        return torch.where(actions[:, 0] != 0, 1 + engine * 7 + bundle, 0).numpy()


def hrl_option_target(rewards, next_value, terminal, gamma=.99):
    """Discount each primitive day once, including bootstrap over the actual duration."""
    return sum(gamma ** t * float(r) for t, r in enumerate(rewards)) + (
        0. if terminal else gamma ** len(rewards) * float(next_value))


def train_aircraft_hrl(config, method='planning_hierarchy', steps=4096, seed=0, rollout_size=128):
    if method not in HRL_PPO_METHODS or steps < 1 or rollout_size < 2:
        raise ValueError('registered hierarchy, positive steps, rollout_size >= 2 required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = AircraftOptionEnv(config)
    model = AircraftPlanningHierarchy(env.env, method)
    actors = model.actor_parameters()
    actor_optimizer = torch.optim.Adam(actors, lr=3e-4)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    pending, ready, daily, logs = {}, [], [], []
    episode = 0
    for day in range(1, steps + 1):
        with torch.no_grad():
            action, logp, _, values = model.act(obs)
        free = env.decision_mask()
        for i in np.flatnonzero(free):
            pending[int(i)] = {'obs': obs.copy(), 'action': action[0].numpy().copy(),
                'logp': float(logp[0, i]), 'value': float(values[0, i]), 'aircraft': int(i), 'rewards': []}
        following, reward, done, truncated, info = env.step(model.catalog_actions(obs, action))
        terminal = done or truncated
        with torch.no_grad():
            next_values = model.value(following)[0].numpy()
        daily.append((obs.copy(), reward + .99 * (not terminal) * next_values))
        for i in list(pending):
            pending[i]['rewards'].append(reward)
            completed = terminal or bool(env.decision_mask()[i])
            # A budget cutoff bootstraps an unfinished option; it does not cancel its job.
            if completed or day == steps:
                item = pending.pop(i)
                item.update(target=hrl_option_target(item['rewards'], next_values[i], terminal),
                            duration=len(item['rewards']), option_complete=completed)
                ready.append(item)
        obs = following
        if terminal:
            logs.append({'record_type': 'episode', 'simulated_days': day, **info['metrics']})
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
        if day % rollout_size == 0 or day == steps:
            if ready:
                states = torch.tensor(np.stack([r['obs'] for r in ready]))
                actions = torch.tensor(np.stack([r['action'] for r in ready]))
                ids = torch.tensor([r['aircraft'] for r in ready])
                batch_ids = torch.arange(len(ready))
                old_logp = torch.tensor([r['logp'] for r in ready])
                advantages = torch.tensor([r['target'] - r['value'] for r in ready])
                if len(ready) > 1:
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
                for _ in range(4):
                    _, new_logp, entropy, _ = model.act(states, actions)
                    ratio = torch.exp(new_logp[batch_ids, ids] - old_logp)
                    loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
                    loss -= .001 * entropy[batch_ids, ids].mean()
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('nonfinite hierarchical policy loss')
                    actor_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors, .5)
                    actor_optimizer.step()
                logs.append({'record_type': 'option_update', 'method': method, 'simulated_days': day,
                    'decisions': len(ready), 'durations': [r['duration'] for r in ready],
                    'complete': [r['option_complete'] for r in ready], 'policy_loss': float(loss.detach())})
                ready.clear()
            states = torch.tensor(np.stack([row[0] for row in daily]))
            targets = torch.tensor(np.stack([row[1] for row in daily]))
            for _ in range(4):
                value_loss = .5 * (model.value(states) - targets).square().mean()
                if not bool(torch.isfinite(value_loss)):
                    raise RuntimeError('nonfinite hierarchical value loss')
                critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
                critic_optimizer.step()
            daily.clear()
    return model, logs, steps


class MAXQMaintenance(nn.Module):
    """Tabular MAXQ-0 baseline on one aircraft and the common repair catalog.

    Root chooses operate or Repair; Repair chooses an Engine task; each Engine
    chooses a component bundle then waits through its committed work. Primitive
    V values and composite completion C values are stored separately. No pseudo
    rewards or MAXQ-Q projected completion function is used. Beliefs are binned,
    so the original exact-state convergence guarantees are not claimed.
    """

    def __init__(self, env):
        super().__init__()
        if env.config.aircraft != 1:
            raise ValueError('MAXQ baseline currently requires Config(aircraft=1)')
        self.n = 1
        self.local = env.local_size
        self.mandatory = env.config.mandatory_health
        self.leaf_values = {}
        self.completion = {}

    def state(self, observation):
        # Explicit approximation: quantized observed beliefs, never true health.
        return tuple(np.rint(np.asarray(observation) * 8).astype(int).tolist())

    def task_children(self, task, observation):
        busy = observation[37] > .5
        if task == 'Root':
            if busy:
                return ('Wait',)
            urgent = np.min(observation[24:36]) <= self.mandatory
            return ('Repair',) if urgent else ('Operate', 'Repair')
        if task == 'Repair':
            return ('Engine0', 'Engine1', 'EngineBoth')
        if task.startswith('Engine'):
            if busy:
                return ('Wait',)
            e = ('Engine0', 'Engine1', 'EngineBoth').index(task)
            return tuple(f'Submit{1 + 7 * e + c}' for c in range(7))
        return ()

    def value(self, task, observation):
        children = self.task_children(task, observation)
        state = self.state(observation)
        if not children:
            return self.leaf_values.get((task, state), 0.)
        return max(self.q(task, observation, child) for child in children)

    def q(self, task, observation, child):
        return self.value(child, observation) + self.completion.get((task, self.state(observation), child), 0.)

    def update_completion(self, task, observation, child, next_value, duration, alpha=.15, gamma=.99):
        key = (task, self.state(observation), child)
        old = self.completion.get(key, 0.)
        self.completion[key] = old + alpha * (gamma ** duration * next_value - old)

    def execute(self, task, env, observation, done, budget, rng, learn=False, epsilon=.2):
        if budget < 1 or done:
            return observation, done, [], True
        children = self.task_children(task, observation)
        if not children:
            action = int(task[6:]) if task.startswith('Submit') else 0
            following, reward, done, truncated, info = env.step(np.array([action], dtype=np.int64))
            if learn:
                key = (task, self.state(observation))
                old = self.leaf_values.get(key, 0.)
                self.leaf_values[key] = old + .15 * (reward - old)
            return following, done or truncated, [(observation.copy(), reward)], True
        trajectory = []
        invoked = False
        while len(trajectory) < budget and not done:
            if invoked and task != 'Root' and observation[37] <= .5:
                break
            children = self.task_children(task, observation)
            scores = np.array([self.q(task, observation, child) for child in children])
            if learn and rng.random() < epsilon:
                selected = int(rng.integers(len(children)))
            else:
                selected = int(np.argmax(scores))
            child = children[selected]
            following, ended, trace, child_complete = self.execute(
                child, env, observation, done, budget - len(trajectory), rng, learn, epsilon)
            parent_terminal = ended or (task != 'Root' and following[37] <= .5)
            if learn and child_complete:
                next_value = 0. if parent_terminal else self.value(task, following)
                for offset, (state_obs, _) in enumerate(trace):
                    self.update_completion(task, state_obs, child, next_value, len(trace) - offset)
            trajectory.extend(trace)
            observation, done, invoked = following, ended, True
            if task == 'Repair' and child_complete:
                break
            if not trace:
                raise RuntimeError('MAXQ child made no physical progress')
        complete = done or (task != 'Root' and invoked and observation[37] <= .5)
        return observation, done, trajectory, complete

    def get_extra_state(self):
        return {'leaf_values': self.leaf_values, 'completion': self.completion}

    def set_extra_state(self, state):
        self.leaf_values = dict(state['leaf_values'])
        self.completion = dict(state['completion'])


def train_maxq(config, steps=4096, seed=0):
    if steps < 1:
        raise ValueError('positive MAXQ simulation budget required')
    env = AircraftOptionEnv(config)
    model = MAXQMaintenance(env.env)
    rng = np.random.default_rng(seed)
    days, episode, logs = 0, 0, []
    while days < steps:
        obs, _ = env.reset(seed=seed + episode)
        _, terminal, trace, _ = model.execute('Root', env, obs, False, steps - days, rng, learn=True)
        days += len(trace)
        logs.append({'record_type': 'episode' if terminal else 'budget_cutoff', 'simulated_days': days,
                     'return': sum(r for _, r in trace), 'primitive_values': len(model.leaf_values),
                     'completion_values': len(model.completion)})
        episode += 1
    return model, logs, days


class FleetOptionCritic(nn.Module):
    """Option-critic adaptation with safe learned planning-mode termination.

    Options condition learned maintenance policies. Beta is exactly zero while
    a repair is committed. Changing planning mode never interrupts physical work.
    The greedy/epsilon-greedy policy over options is derived from Q_Omega.
    """

    def __init__(self, env, options=3):
        super().__init__()
        if options < 2:
            raise ValueError('at least two planning options required')
        self.n = env.config.aircraft
        self.local = env.local_size
        self.width = env.observation_space.shape[0]
        self.mandatory = env.config.mandatory_health
        self.options = options
        self.worker = hrl_mlp(self.width + self.local + options, len(HRL_CATALOG))
        self.q_network = hrl_mlp(self.width, self.n * options)
        self.termination = hrl_mlp(self.width + self.local, options)

    def features(self, observation):
        obs = _fleet_observation_tensor(observation, self.width,
                                        'incorrect option-critic observation')
        local = obs[:, :self.n * self.local].reshape(-1, self.n, self.local)
        return obs, local, torch.cat([obs[:, None].expand(-1, self.n, -1), local], -1)

    def q_values(self, observation):
        obs = _fleet_observation_tensor(observation, self.width,
                                        'incorrect option-critic observation')
        return self.q_network(obs).reshape(-1, self.n, self.options)

    def beta(self, observation, option_ids):
        _, local, features = self.features(observation)
        ids = torch.as_tensor(option_ids, dtype=torch.long)
        if ids.shape != local.shape[:2] or torch.any((ids < 0) | (ids >= self.options)):
            raise ValueError('one valid option per aircraft required')
        raw = torch.sigmoid(self.termination(features)).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return raw * (local[:, :, 37] <= .5)

    def policy(self, observation, option_ids, actions=None, deterministic=False):
        obs, local, features = self.features(observation)
        ids = torch.as_tensor(option_ids, dtype=torch.long)
        if ids.shape != local.shape[:2] or torch.any((ids < 0) | (ids >= self.options)):
            raise ValueError('one valid option per aircraft required')
        logits = self.worker(torch.cat([features, nn.functional.one_hot(ids, self.options).float()], -1))
        valid = hrl_catalog_mask(obs, self.n, self.local, self.mandatory)
        return hrl_categorical(logits, valid, actions, deterministic)


def option_critic_target(reward, following_q, option_ids, beta, terminal, gamma=.99):
    current = following_q.gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
    continuation = (1 - beta) * current + beta * following_q.max(-1).values
    return reward + gamma * (1 - terminal.float()) * continuation


def hrl_epsilon_choice(values, valid, rng, epsilon=0.):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape != valid.shape or not valid.any(-1).all():
        raise ValueError('one nonempty choice mask per control unit required')
    choices = np.argmax(np.where(valid, values, -np.inf), axis=-1)
    for i in range(len(choices)):
        if rng.random() < epsilon:
            choices[i] = rng.choice(np.flatnonzero(valid[i]))
    return choices.astype(np.int64)


def train_option_critic(config, steps=4096, seed=0):
    if steps < 1:
        raise ValueError('positive option-critic simulation budget required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = AircraftOptionEnv(config)
    model = FleetOptionCritic(env.env)
    actor_opt = torch.optim.Adam(model.worker.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(model.q_network.parameters(), lr=3e-4)
    beta_opt = torch.optim.Adam(model.termination.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    def select(observation):
        with torch.no_grad():
            q = model.q_values(observation)[0].numpy()
        return hrl_epsilon_choice(q, np.ones_like(q, dtype=bool), rng, .1)
    ids = select(obs)
    episode, logs = 0, []
    for day in range(1, steps + 1):
        option_ids = torch.tensor(ids).unsqueeze(0)
        actions, logp, entropy = model.policy(obs, option_ids)
        active = torch.tensor(env.decision_mask()).unsqueeze(0).float()
        following, reward, done, truncated, info = env.step(actions[0].detach().numpy())
        terminal = done or truncated
        with torch.no_grad():
            following_q = model.q_values(following)
        beta = model.beta(following, option_ids)
        target = option_critic_target(reward, following_q, option_ids, beta.detach(), torch.tensor(terminal))
        q = model.q_values(obs).gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
        advantage = (target - q).detach()
        actor_loss = -((logp * advantage + .001 * entropy) * active).sum() / active.sum().clamp_min(1)
        critic_loss = .5 * (q - target.detach()).square().mean()
        safe = torch.tensor(env.decision_mask()).unsqueeze(0).float() * (not terminal)
        termination_advantage = (following_q.gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
                                  - following_q.max(-1).values + .01).detach()
        termination_loss = (beta * termination_advantage * safe).sum() / safe.sum().clamp_min(1)
        if not bool(torch.isfinite(actor_loss + critic_loss + termination_loss)):
            raise RuntimeError('nonfinite option-critic loss')
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.worker.parameters(), .5)
        if bool(active.any()):
            actor_opt.step()
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_network.parameters(), .5)
        critic_opt.step()
        beta_opt.zero_grad()
        termination_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.termination.parameters(), .5)
        if bool(safe.any()):
            beta_opt.step()
        terminate = (rng.random(config.aircraft) < beta.detach()[0].numpy()) & env.decision_mask()
        next_ids = select(following)
        ids = np.where(terminate, next_ids, ids)
        logs.append({'record_type': 'update', 'simulated_days': day,
                     'policy_loss': float(actor_loss.detach()), 'value_loss': float(critic_loss.detach()),
                     'termination_loss': float(termination_loss.detach()),
                     'safe_termination_samples': int(safe.sum()), 'option_terminations': int(terminate.sum())})
        obs = following
        if terminal:
            logs.append({'record_type': 'episode', 'simulated_days': day, **info['metrics']})
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            ids = select(obs)
    return model, logs, steps


class HierarchicalVDN(nn.Module):
    """Two-level value-decomposition adaptation, not an exact HAVEN reproduction.

    Per-aircraft high policies select replacement budgets; local low policies
    choose catalog requests. Both levels sum agent Q values. The low reward
    receives a detached share of the macro TD advantage (coefficient 0.1).
    Macro epochs are three days; busy aircraft retain goals and committed plans.
    """

    def __init__(self, env):
        super().__init__()
        self.n = env.config.aircraft
        self.local = env.local_size
        self.public = env.public_size
        self.mandatory = env.config.mandatory_health
        self.high_agents = nn.ModuleList([hrl_mlp(self.local + self.public, 4) for _ in range(self.n)])
        self.low_agents = nn.ModuleList([hrl_mlp(self.local + self.public + 4, len(HRL_CATALOG)) for _ in range(self.n)])

    def features(self, observation):
        return mappo_agent_inputs(observation, self.n, self.local, self.public, 'aircraft')

    def high_values(self, observation):
        x = self.features(observation)
        return torch.stack([agent(x[:, i]) for i, agent in enumerate(self.high_agents)], 1)

    def low_values(self, observation, goals):
        x = self.features(observation)
        goals = torch.as_tensor(goals, dtype=torch.long)
        if goals.shape != x.shape[:2] or torch.any((goals < 0) | (goals >= 4)):
            raise ValueError('one valid hierarchy goal per aircraft required')
        x = torch.cat([x, nn.functional.one_hot(goals, 4).float()], -1)
        return torch.stack([agent(x[:, i]) for i, agent in enumerate(self.low_agents)], 1)

    @staticmethod
    def mix(values):
        return values.sum(-1)

    def choose_goals(self, observation, previous, rng, epsilon=0.):
        local = self.features(observation)[0]
        busy = local[:, 37].numpy() > .5
        urgent = local[:, 24:36].amin(-1).numpy() <= self.mandatory
        valid = np.ones((self.n, 4), dtype=bool)
        valid[urgent & ~busy, 0] = False
        with torch.no_grad():
            choices = hrl_epsilon_choice(self.high_values(observation)[0].numpy(), valid, rng, epsilon)
        if previous is not None:
            choices[busy] = np.asarray(previous)[busy]
        else:
            choices[busy] = 0
        return choices

    def choose_actions(self, observation, goals, rng, epsilon=0.):
        goals = torch.tensor(np.asarray(goals)).unsqueeze(0)
        valid = hrl_catalog_mask(observation, self.n, self.local, self.mandatory, goals)
        # An urgent aircraft may outgrow an old zero-budget goal before the next macro epoch.
        if not bool(valid.any(-1).all()):
            raise ValueError('a stale hierarchy goal must be refreshed after an urgency alarm')
        with torch.no_grad():
            return hrl_epsilon_choice(self.low_values(observation, goals)[0].numpy(), valid[0].numpy(), rng, epsilon)


def train_hierarchical_vdn(config, steps=4096, seed=0, macro_days=3):
    if steps < 1 or macro_days < 1:
        raise ValueError('positive simulation budget and macro duration required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = AircraftOptionEnv(config)
    model = HierarchicalVDN(env.env)
    high_opt = torch.optim.Adam(model.high_agents.parameters(), lr=3e-4)
    low_opt = torch.optim.Adam(model.low_agents.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    goals, days, episode, logs = None, 0, 0, []
    while days < steps:
        goals = model.choose_goals(obs, goals, rng, .1)
        start, window = obs.copy(), []
        for _ in range(min(macro_days, steps - days)):
            actions = model.choose_actions(obs, goals, rng, .1)
            following, reward, done, truncated, info = env.step(actions)
            terminal = done or truncated
            window.append((obs.copy(), goals.copy(), actions.copy(), reward, following.copy(), terminal))
            days += 1
            obs = following
            # End the macro on a new urgency alarm so zero-budget goals never block safety.
            rows, _ = env.env.split_observation(obs)
            urgent = rows[:, 24:36].min(-1) <= config.mandatory_health
            if terminal or np.any(urgent & env.decision_mask() & (goals == 0)):
                break
        next_goals = model.choose_goals(obs, goals, rng)
        self_n = config.aircraft
        selected_goals = torch.tensor(goals).reshape(1, self_n, 1)
        high_value = model.mix(model.high_values(start).gather(-1, selected_goals).squeeze(-1))
        with torch.no_grad():
            next_high = model.mix(model.high_values(obs).gather(-1, torch.tensor(next_goals).reshape(1, self_n, 1)).squeeze(-1))
            target = hrl_option_target([r[3] for r in window], float(next_high[0]), terminal)
        high_advantage = (target - high_value).detach()
        high_loss = .5 * (high_value - target).square().mean()
        low_loss = torch.zeros(())
        for t, (state, old_goals, action, reward, following, ended) in enumerate(window):
            next_goal = next_goals if t == len(window) - 1 else old_goals
            q = model.mix(model.low_values(state, torch.tensor(old_goals).unsqueeze(0)).gather(
                -1, torch.tensor(action).reshape(1, self_n, 1)).squeeze(-1))
            with torch.no_grad():
                valid = hrl_catalog_mask(following, self_n, env.env.local_size, config.mandatory_health,
                                         torch.tensor(next_goal).unsqueeze(0))
                future = model.low_values(following, torch.tensor(next_goal).unsqueeze(0))
                next_low = model.mix(future.masked_fill(~valid, -1e9).max(-1).values)
                shaped_reward = reward + .1 * high_advantage / len(window)
                low_target = shaped_reward + .99 * (not ended) * next_low
            low_loss = low_loss + .5 * (q - low_target).square().mean() / len(window)
        if not bool(torch.isfinite(high_loss + low_loss)):
            raise RuntimeError('nonfinite hierarchical value-decomposition loss')
        high_opt.zero_grad()
        high_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.high_agents.parameters(), .5)
        high_opt.step()
        low_opt.zero_grad()
        low_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.low_agents.parameters(), .5)
        low_opt.step()
        logs.append({'record_type': 'macro_update', 'simulated_days': days, 'duration': len(window),
                     'high_loss': float(high_loss.detach()), 'low_loss': float(low_loss.detach()),
                     'high_advantage': float(high_advantage[0]), 'goals': goals.tolist()})
        if terminal:
            logs.append({'record_type': 'episode', 'simulated_days': days, **info['metrics']})
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            goals = None
    return model, logs, days


def evaluate_hrl(config, method, model, seeds):
    """Evaluate hierarchy controllers with a daily physical clock and fixed weights."""
    if method not in HRL_METHODS or model is None:
        raise ValueError('registered hierarchy and trained model required')
    records = []
    for seed in seeds:
        env = AircraftOptionEnv(config)
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        decisions, free_decisions, terminations = 0, 0, 0
        if method == 'maxq':
            _, _, trace, _ = model.execute('Root', env, obs, False, config.horizon, rng, learn=False)
            decisions = len(trace)
        else:
            goals, option_ids, macro_clock = None, None, 0
            while True:
                free_decisions += int(env.decision_mask().sum())
                with torch.no_grad():
                    if method in HRL_PPO_METHODS:
                        action = model.act(obs, deterministic=True)[0]
                        choices = model.catalog_actions(obs, action)
                    elif method == 'option_critic':
                        if option_ids is None:
                            option_ids = model.q_values(obs)[0].argmax(-1).numpy()
                        ids = torch.tensor(option_ids).unsqueeze(0)
                        choices = model.policy(obs, ids, deterministic=True)[0][0].numpy()
                    else:
                        if goals is None or macro_clock == 0:
                            goals = model.choose_goals(obs, goals, rng)
                        choices = model.choose_actions(obs, goals, rng)
                obs, _, done, truncated, info = env.step(choices)
                decisions += 1
                if method == 'option_critic' and not (done or truncated):
                    with torch.no_grad():
                        beta = model.beta(obs, torch.tensor(option_ids).unsqueeze(0))[0].numpy()
                        next_ids = model.q_values(obs)[0].argmax(-1).numpy()
                    change = (beta >= .5) & env.decision_mask()
                    option_ids = np.where(change, next_ids, option_ids)
                    terminations += int(change.sum())
                if method == 'hierarchical_vdn':
                    macro_clock = (macro_clock + 1) % 3
                    rows, _ = env.env.split_observation(obs)
                    if np.any((rows[:, 24:36].min(-1) <= config.mandatory_health) & env.decision_mask() & (goals == 0)):
                        macro_clock = 0
                if done or truncated:
                    break
        metrics = env.last_info['metrics']
        records.append({'seed': int(seed), 'decisions': decisions,
                        'learned_planning_opportunities': free_decisions,
                        'planning_mode_terminations': terminations,
                        'availability': metrics['flights'] / (config.aircraft * config.horizon), **metrics})
    return records


ALL_LEARNED_METHODS = ('flat', 'conditional', 'shared', 'options', 'learned_options',
                       *REQUEST_METHODS, *HRL_METHODS)


def execute_option(env, option, obs, gamma=0.99, max_days=14, planned_action=None):
    """Plan once, then keep all active jobs committed while the fleet advances.

    Manager returns at first task completion, a new observed urgency alarm,
    episode end, or max_days. It never consumes true event diagnostics.
    Waiting without any accepted job lasts one day.
    Other operating aircraft continue their modeled flights during this option.
    Longer manager intervals change decision opportunities, so this is a timing
    experiment as well as an architecture experiment, not a matched PPO ablation.
    """
    option = int(option)
    if option not in range(4):
        raise ValueError('unknown option')
    if max_days < 1:
        raise ValueError('max_days must be positive')
    thresholds = {1: (0.62, 0.62), 2: (0.65, 0.77), 3: (0.78, 0.85)}
    action = planned_action if planned_action is not None else env.idle_action() if option == 0 else threshold_action(env, obs, *thresholds[option])
    before, _ = env.split_observation(obs)
    before_alarm = before[:, 24:36].min(1) <= env.config.mandatory_health
    rewards = []
    while True:
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        rows, _ = env.split_observation(obs)
        new_alarm = np.any((rows[:, 24:36].min(1) <= env.config.mandatory_health) & ~before_alarm)
        if option == 0 or done or truncated or info['completed'] or new_alarm or (len(rewards) >= max_days) or (not env.jobs):
            break
        action = env.idle_action()
    discounted = sum((gamma ** i * r for i, r in enumerate(rewards)))
    return (obs, discounted, done, truncated, info, len(rewards), sum(rewards))


# ========================================================================
# fleet_maintenance/train.py
# ========================================================================


from dataclasses import asdict
from pathlib import Path
import argparse
import json

def make_model(env, method):
    if method in HRL_PPO_METHODS:
        return AircraftPlanningHierarchy(env, method)
    if method == 'maxq':
        return MAXQMaintenance(env)
    if method == 'option_critic':
        return FleetOptionCritic(env)
    if method == 'hierarchical_vdn':
        return HierarchicalVDN(env)
    if method in DCMAC_METHODS:
        return DCMACActorCritic(env, request_assignment(method), method.startswith('ddmac_'))
    if method in MAPPO_METHODS:
        return MultiAgentActorCritic(env, 'aircraft' if method == 'mappo_aircraft' else 'component')
    if method == 'learned_options':
        return LearnedOptionActorCritic(env)
    return OptionActorCritic(env) if method == 'options' else FleetActorCritic(env, method)

def load_checkpoint(path):
    """Reload one of this package's own checkpoints for independent evaluation."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('Checkpoint uses an incompatible observation/action layout. Retrain with individual crew masks and resource factors (version 3).')
    config = Config(**checkpoint['config'])
    model = make_model(FleetMaintenanceEnv(config), checkpoint['method'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return (config, checkpoint['method'], model)

def smdp_gae(rewards, values, next_values, dones, durations, gamma=0.99, lam=0.95):
    """Backward GAE with gamma**duration and lambda**duration per decision.

    rewards already contain the discounted primitive rewards inside each option.
    A one-day decision recovers ordinary GAE. Episode boundaries block bootstrap.
    """
    advantage = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        live = 1.0 - float(dones[t])
        delta = rewards[t] + gamma ** durations[t] * next_values[t] * live - values[t]
        carry = delta + (gamma * lam) ** durations[t] * live * carry
        advantage[t] = carry
    return (advantage, advantage + np.asarray(values))

def train(config, method='conditional', steps=4096, seed=0, rollout_size=128):
    """Interaction budget counts actual simulated days, not manager decisions."""
    if method in HRL_PPO_METHODS:
        return train_aircraft_hrl(config, method, steps, seed, rollout_size)
    if method == 'maxq':
        return train_maxq(config, steps, seed)
    if method == 'option_critic':
        return train_option_critic(config, steps, seed)
    if method == 'hierarchical_vdn':
        return train_hierarchical_vdn(config, steps, seed)
    if method in DCMAC_METHODS:
        return train_dcmac(config, method, steps, seed, rollout_size)
    if method in MAPPO_METHODS:
        return train_mappo(config, method, steps, seed, rollout_size)
    if steps < 1 or rollout_size < 2:
        raise ValueError('positive budget and rollout>=2 required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = FleetMaintenanceEnv(config)
    model = make_model(env, method)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    rng = np.random.default_rng(seed)
    episode = 0
    obs, _ = env.reset(seed=seed)
    simulated = 0
    logs = []
    while simulated < steps:
        data = {k: [] for k in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'done', 'duration')}
        while len(data['obs']) < rollout_size and simulated < steps:
            with torch.no_grad():
                action, logp, _, value = model.act(torch.tensor(obs))
            chosen = action[0].numpy()
            if method in ('options', 'learned_options'):
                macro = int(chosen) if method == 'options' else int(chosen[0])
                plan = None if method == 'options' else chosen[1:].reshape(config.aircraft, 8)
                next_obs, reward, done, truncated, info, duration, _ = execute_option(env, macro, obs, max_days=min(14, steps - simulated), planned_action=plan)
            else:
                next_obs, reward, done, truncated, info = env.step(chosen)
                duration = 1
            terminal = done or truncated
            with torch.no_grad():
                nv = float(model.value(torch.tensor(next_obs).unsqueeze(0))[0])
            for key, item in [('obs', obs.copy()), ('action', np.asarray(chosen).copy()), ('logp', float(logp[0])), ('value', float(value[0])), ('next_value', nv), ('reward', reward), ('done', terminal), ('duration', duration)]:
                data[key].append(item)
            simulated += duration
            obs = next_obs
            if terminal:
                logs.append({'episode': episode, 'simulated_days': simulated, **info['metrics']})
                episode += 1
                obs, _ = env.reset(seed=seed + episode)
        advantages, targets = smdp_gae(data['reward'], data['value'], data['next_value'], data['done'], data['duration'])
        tensors = {k: torch.tensor(np.asarray(v)) for k, v in data.items() if k in ('obs', 'action', 'logp')}
        advantages = torch.tensor(advantages)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
        targets = torch.tensor(targets, dtype=torch.float32)
        for _ in range(4):
            ids = rng.permutation(len(targets))
            for start in range(0, len(ids), 64):
                idx = ids[start:start + 64]
                _, newlog, entropy, values = model.act(tensors['obs'][idx], tensors['action'][idx])
                ratio = torch.exp(newlog - tensors['logp'][idx])
                surrogate = torch.minimum(ratio * advantages[idx], ratio.clamp(0.8, 1.2) * advantages[idx])
                loss = -surrogate.mean() + 0.5 * ((values - targets[idx]) ** 2).mean() - 0.001 * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite PPO loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
    return (model, logs, simulated)

def evaluate(config, method, model=None, seeds=range(10000, 10010)):
    """Evaluate fixed weights on held-out seeds; proposal methods use their common broker."""
    if method in HRL_METHODS:
        return evaluate_hrl(config, method, model, seeds)
    if method != 'threshold' and model is None:
        raise ValueError('evaluation requires a trained model')
    records = []
    for seed in seeds:
        if method in REQUEST_METHODS:
            env = MultiAgentFleetEnv(config, request_assignment(method))
            _, info = env.reset(seed=int(seed))
            obs = info['central_observation']
        else:
            env = FleetMaintenanceEnv(config)
            obs, _ = env.reset(seed=int(seed))
        decisions = 0
        while True:
            if method == 'threshold':
                obs, _, done, truncated, info = env.step(threshold_action(env, obs))
            else:
                with torch.no_grad():
                    if method in REQUEST_METHODS:
                        views = np.stack(list(env.observations().values())) if method in MAPPO_METHODS else obs
                        action, _, _ = model.policy(torch.tensor(views).unsqueeze(0) if method in MAPPO_METHODS else torch.tensor(views), deterministic=True)
                    else:
                        action, _, _, _ = model.act(torch.tensor(obs), deterministic=True)
                if method in REQUEST_METHODS:
                    _, _, done, truncated, info = env.step(model.action_dict(action[0]))
                    obs = info['central_observation']
                elif method in ('options', 'learned_options'):
                    chosen = action[0].numpy()
                    macro = int(chosen) if method == 'options' else int(chosen[0])
                    plan = None if method == 'options' else chosen[1:].reshape(config.aircraft, 8)
                    obs, _, done, truncated, info, _, _ = execute_option(env, macro, obs, planned_action=plan)
                else:
                    obs, _, done, truncated, info = env.step(action[0].numpy())
            decisions += 1
            if done or truncated:
                break
        records.append({'seed': int(seed), 'decisions': decisions, 'availability': info['metrics']['flights'] / (config.aircraft * config.horizon), **info['metrics']})
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=[*ALL_LEARNED_METHODS, 'all'], default='all')
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--aircraft', type=int, default=7)
    parser.add_argument('--crews', type=int, default=3)
    parser.add_argument('--workshops', type=int, default=2)
    parser.add_argument('--crew-time-factors', type=float, nargs='+', default=())
    parser.add_argument('--workshop-time-factors', type=float, nargs='+', default=())
    parser.add_argument('--horizon', type=int, default=90)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--output', default='runs/smoke')
    args = parser.parse_args()
    config = Config(aircraft=args.aircraft, horizon=args.horizon, crews=args.crews, workshops=args.workshops, crew_time_factors=tuple(args.crew_time_factors), workshop_time_factors=tuple(args.workshop_time_factors))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    result = {'config': asdict(config), 'training_seed': args.seed, 'requested_steps': args.steps, 'observation_version': OBSERVATION_VERSION, 'status': 'preliminary teaching experiment; not a paper reproduction or convergence claim', 'comparison_note': 'new HRL methods share a restricted 22-action catalog and broker; hrl_flat matches planning_hierarchy authority; concurrent_options has a fixed worker; target/option/VDN clocks and objectives differ; MAXQ is single-aircraft; legacy and MAPPO/DCMAC use broader action spaces', 'evaluation': {'threshold': evaluate(config, 'threshold', seeds=range(10000, 10000 + args.eval_episodes))}}
    methods = list(ALL_LEARNED_METHODS) if args.method == 'all' else [args.method]
    if args.method == 'all' and config.aircraft != 1:
        methods.remove('maxq')
        result['skipped_methods'] = {'maxq': 'requires aircraft=1'}
    for method in methods:
        model, logs, days = train(config, method, args.steps, args.seed)
        torch.save({'state_dict': model.state_dict(), 'config': asdict(config), 'method': method, 'training_seed': args.seed, 'simulated_days': days, 'observation_version': OBSERVATION_VERSION}, destination / f'{method}.pt')
        result['evaluation'][method] = evaluate(config, method, model, seeds=range(10000, 10000 + args.eval_episodes))
        result[method + '_training'] = logs
        print(method, 'days', days, 'mean test return', np.mean([r['return'] for r in result['evaluation'][method]]), flush=True)
    (destination / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    summary = ['# Preliminary comparison', '', result['status'], '', result['comparison_note'], '', '| Method | Return | Availability | Failed flights | Rejected plans | Decisions |', '|---|---:|---:|---:|---:|---:|']
    for method, records in result['evaluation'].items():
        mean = lambda k: np.mean([r[k] for r in records])
        summary.append(f"| {method} | {mean('return'):.2f} | {mean('availability'):.3f} | {mean('flight_failures'):.1f} | {mean('rejections'):.1f} | {mean('decisions'):.1f} |")
    (destination / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('Saved', destination.resolve())


OPERATE_UNTIL_LOW = 0
REPLACE_NOW = 1
