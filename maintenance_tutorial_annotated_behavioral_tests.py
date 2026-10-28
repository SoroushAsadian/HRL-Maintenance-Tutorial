# Consolidated behavioral test suite


# Explanation: Define and run the behavioral tests.
import unittest
from unittest.mock import patch
# Explanation: Create a modified copy of the frozen configuration in tests.
from dataclasses import replace
# Explanation: Use temporary files to verify checkpoint compatibility without altering saved experiment runs.
import tempfile
# Explanation: Use NumPy for arrays, vectorized simulation, random sampling, and numeric calculations.
import numpy as np
# Explanation: Use tensors, neural networks, optimizers, and automatic differentiation.
import torch
# Explanation: Check the adapted environment against Stable-Baselines3's contract.
from stable_baselines3.common.env_checker import check_env

# Explanation: Define class FleetTests: behavioral checks for environment contracts, masks, reproducibility, learning, and options.
class FleetTests(unittest.TestCase):

    # Explanation: Define config with the shown parameters/defaults; its purpose is to create a compact deterministic configuration for a test.
    def config(self, **kwargs):
        # Explanation: Return replace(Config(aircraft=3, horizon=40, bird_probability=0, hail_probability=0, weather_probability=0), **kwargs) to the caller; this ends the function and supplies its result.
        # Explanation: Keep existing timing tests on standard resources unless a test explicitly supplies heterogeneous profiles.
        kwargs.setdefault('crew_time_factors', (1.0,) * kwargs.get('crews', 3))
        # Explanation: Preserve standard workshop duration for existing physical-transition checks.
        kwargs.setdefault('workshop_time_factors', (1.0,) * kwargs.get('workshops', 2))
        # Explanation: Create the requested fleet with these validated resource profiles.
        return replace(Config(aircraft=3, horizon=40, bird_probability=0, hail_probability=0, weather_probability=0), **kwargs)

    # Explanation: Check that fixed plates identify the same aircraft across operation, maintenance, hazard diagnostics, and episode resets.
    def test_aircraft_plate_numbers(self):
        # Explanation: Use seven aircraft and guaranteed bird events so every flying aircraft produces a traceable diagnostic.
        env = FleetMaintenanceEnv(self.config(aircraft=7, bird_probability=1))
        # Explanation: Start an episode and capture the index-to-plate mapping returned to callers.
        _, reset_info = env.reset(seed=12)
        # Explanation: Verify one distinct synthetic plate for each aircraft in index order.
        self.assertEqual(env.aircraft_plates, ('AC-001', 'AC-002', 'AC-003', 'AC-004', 'AC-005', 'AC-006', 'AC-007'))
        # Explanation: Verify that reset diagnostics expose the same mapping used by the simulator.
        self.assertEqual(reset_info['aircraft_plates'], env.aircraft_plates)
        # Explanation: Begin a valid one-component maintenance plan on aircraft 2; the other aircraft continue flying.
        action = env.idle_action()
        # Explanation: Ground aircraft 2 and select engine 0, component 0, one crew, and workshop 0.
        action[2] = [1, 1, 1, 0, 1, 0, 0, 0]
        # Explanation: Advance one day to exercise identity during both maintenance and hazard generation.
        obs, _, _, _, info = env.step(action)
        # Explanation: Verify that the existing indexed job remains attached to the aircraft labeled AC-003.
        self.assertIn(2, env.jobs)
        # Explanation: Verify that all twelve expected engine events belong to the six flying aircraft.
        self.assertEqual(len(info['events']), 12)
        # Explanation: Check the label attached to every reported event against its aircraft index.
        for event in info['events']:
            # Explanation: The plate must match the row used by the health and exposure arrays.
            self.assertEqual(event['aircraft_plate'], env.aircraft_plates[event['aircraft']])
            # Explanation: The grounded aircraft must have no flight hazard event.
            self.assertNotEqual(event['aircraft'], 2)
        # Explanation: Step diagnostics must retain the full mapping for completed and rejected aircraft indices too.
        self.assertEqual(info['aircraft_plates'], reset_info['aircraft_plates'])
        # Explanation: Identity strings are metadata, so the numerical observation still satisfies its existing space.
        self.assertTrue(env.observation_space.contains(obs))
        # Explanation: Check that the human-readable display labels the busy aircraft by plate.
        self.assertIn("busy=['AC-003']", env.render())
        # Explanation: Start a new episode with a different seed to ensure identity does not depend on random exposure or health.
        _, next_info = env.reset(seed=99)
        # Explanation: A new episode changes simulated state but preserves aircraft identity.
        self.assertEqual(next_info['aircraft_plates'], reset_info['aircraft_plates'])

    # Explanation: Define test_gym_contract with the shown parameters/defaults; its purpose is to perform this test gym contract operation.
    def test_gym_contract(self):
        # Explanation: Unpack True) into named values check_env(FlattenFleetAction(FleetMaintenanceEnv(self.config, warn so each part can be used separately below.
        check_env(FlattenFleetAction(FleetMaintenanceEnv(self.config())), warn=True)

    # Explanation: Define test_sb3_adapter_shape with the shown parameters/defaults; its purpose is to perform this test sb3 adapter shape operation.
    def test_sb3_adapter_shape(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FlattenFleetAction(FleetMaintenanceEnv(self.config()))
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=2)
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        obs, *_ = env.step(env.reverse_action(env.unwrapped.idle_action()))
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertTrue(env.observation_space.contains(obs))
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(len(env.action_space.nvec.shape), 1)

    # Explanation: Define test_seed_reproducibility with the shown parameters/defaults; its purpose is to perform this test seed reproducibility operation.
    def test_seed_reproducibility(self):
        # Explanation: Compute and bind the right-hand value to e1, naming an intermediate for the surrounding algorithm.
        e1 = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to e2, naming an intermediate for the surrounding algorithm.
        e2 = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        o1, _ = e1.reset(seed=12)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        o2, _ = e2.reset(seed=12)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_array_equal(o1, o2)
        # Explanation: Visit each _ from range(20); the body applies the operation once per item.
        for _ in range(20):
            # Explanation: Compute and bind the right-hand value to r1, naming an intermediate for the surrounding algorithm.
            r1 = e1.step(threshold_action(e1, o1))
            # Explanation: Compute and bind the right-hand value to r2, naming an intermediate for the surrounding algorithm.
            r2 = e2.step(threshold_action(e2, o2))
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            np.testing.assert_array_equal(r1[0], r2[0])
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            self.assertEqual(r1[1:], r2[1:])
            # Explanation: Compute and bind the right-hand value to o1, naming an intermediate for the surrounding algorithm.
            o1 = r1[0]
            # Explanation: Compute and bind the right-hand value to o2, naming an intermediate for the surrounding algorithm.
            o2 = r2[0]

    # Explanation: Define test_commitment_concurrency_duration_and_effects with the shown parameters/defaults; its purpose is to perform this test commitment concurrency duration and effects operation.
    def test_commitment_concurrency_duration_and_effects(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=1)
        # Explanation: Compute and bind the right-hand value to old, naming an intermediate for the surrounding algorithm.
        old = env.health.copy()
        # Explanation: Compute and bind the right-hand value to a, naming an intermediate for the surrounding algorithm.
        a = env.idle_action()
        # Explanation: Compute and bind the right-hand value to a[0], naming an intermediate for the surrounding algorithm.
        a[0] = [1, 1, 4, 0, 1, 0, 2, 2]
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(a)
        # Explanation: Fetch aircraft 0's active Job so the test can retain and later compare its committed Plan.
        j = env.jobs[0]
        # Explanation: Compute the proposed or accepted maintenance plan and store it in plan; later checks/updates use this named intermediate.
        plan = j.plan
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(j.total_duration, 11)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertLess(env.health[1, 0, 0], old[1, 0, 0])
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.health[0, 0, 2], old[0, 0, 2])
        # Explanation: Compute and bind the right-hand value to changed, naming an intermediate for the surrounding algorithm.
        changed = env.idle_action()
        # Explanation: Compute and bind the right-hand value to changed[0], naming an intermediate for the surrounding algorithm.
        changed[0] = [1, 3, 63, 63, 3, 1, 11, 0]
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(changed)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.jobs[0].plan, plan)
        # Explanation: Visit each _ from range(9); the body applies the operation once per item.
        for _ in range(9):
            # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
            env.step(env.idle_action())
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertNotIn(0, env.jobs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.health[0, 0, 2], 1.0)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['replacements'], 1.0)

    # Explanation: Define test_rejection_is_atomic with the shown parameters/defaults; its purpose is to perform this test rejection is atomic operation.
    def test_rejection_is_atomic(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=2)
        # Explanation: Compute and bind the right-hand value to old_stock, naming an intermediate for the surrounding algorithm.
        old_stock = env.stock.copy()
        # Explanation: Compute and bind the right-hand value to a, naming an intermediate for the surrounding algorithm.
        a = env.idle_action()
        # Explanation: Compute and bind the right-hand value to a[0], naming an intermediate for the surrounding algorithm.
        a[0] = [1, 1, 0, 4, 1, 0, 0, 0]
        # Explanation: Compute and bind the right-hand value to info, naming an intermediate for the surrounding algorithm.
        info = env.step(a)[-1]
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(info['rejected'][0][1], 'inconsistent_engine_bundle')
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertFalse(env.jobs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_array_equal(env.stock, old_stock)

    # Explanation: Define test_resources_priority_and_spares with the shown parameters/defaults; its purpose is to perform this test resources priority and spares operation.
    def test_resources_priority_and_spares(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config(crews=1, workshops=1))
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=3)
        # Explanation: Compute and bind the right-hand value to a, naming an intermediate for the surrounding algorithm.
        a = env.idle_action()
        # Explanation: Compute and bind the right-hand value to a[0], naming an intermediate for the surrounding algorithm.
        a[0] = [1, 1, 4, 0, 1, 0, 0, 0]
        # Explanation: Compute and bind the right-hand value to a[1], naming an intermediate for the surrounding algorithm.
        a[1] = [1, 1, 4, 0, 1, 0, 0, 2]
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(a)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(set(env.jobs), {1})
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.resource_state()[0], 0)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.stock[2], 2)
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=3)
        # Explanation: Compute the spare-part inventory by component type and store it in env.stock[2]; later checks/updates use this named intermediate.
        env.stock[2] = 0
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(a)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertFalse(env.jobs)

    # Explanation: Define test_safety_hold_uses_estimate with the shown parameters/defaults; its purpose is to perform this test safety hold uses estimate operation.
    def test_safety_hold_uses_estimate(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=2)
        # Explanation: Unpack 0.01 into named values env.particles[0, 0, 2] so each part can be used separately below.
        env.particles[0, 0, 2] = 0.01
        # Explanation: Compute and bind the right-hand value to h, naming an intermediate for the surrounding algorithm.
        h = env.health[0].copy()
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(env.idle_action())
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_array_equal(env.health[0], h)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['downtime_aircraft_days'], 1)

    # Explanation: Define test_stochastic_events_and_no_weather_delay with the shown parameters/defaults; its purpose is to perform this test stochastic events and no weather delay operation.
    def test_stochastic_events_and_no_weather_delay(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config(bird_probability=1, hail_probability=1, weather_probability=1))
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=3)
        # Explanation: Use the simulator's env.step interface to reset, advance, or query the environment.
        env.step(env.idle_action())
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['bird_events'], 6)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['hail_events'], 6)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['weather_exposures'], 3)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.metrics['flights'], 3)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.day, 1)

    # Explanation: Define test_filter_is_not_true_health with the shown parameters/defaults; its purpose is to perform this test filter is not true health operation.
    def test_filter_is_not_true_health(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=4)
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        obs, *_ = env.step(env.idle_action())
        # Explanation: Unpack the return values from env.split_observation(obs) into rows; underscore entries are intentionally ignored.
        rows, _ = env.split_observation(obs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertFalse(np.array_equal(rows[:, :12], env.health.reshape(3, 12)))
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertTrue(env.observation_space.contains(obs))

    # Explanation: Define test_efficiency_temperature_mapping_and_decode with the shown parameters/defaults; its purpose is to perform this test efficiency temperature mapping and decode operation.
    def test_efficiency_temperature_mapping_and_decode(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=4)
        # Explanation: Compute and bind the right-hand value to healthy, naming an intermediate for the surrounding algorithm.
        healthy = env._sensor_model(np.ones((2, 6)), 1.0)
        # Explanation: Compute and bind the right-hand value to worn, naming an intermediate for the surrounding algorithm.
        worn = env._sensor_model(np.full((2, 6), 0.5), 1.0)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertTrue(np.all(worn[:, :, 0] < healthy[:, :, 0]))
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertTrue(np.all(worn[:, :, 1] > healthy[:, :, 1]))
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_allclose(healthy[:, :, 1], np.tile(TEMPERATURE_BASE_C, (2, 1)))
        # Explanation: Decode public sensor columns into efficiency, temperature, and reading-age values for comparison with the simulator's measurements; this is not the accumulated performance-metrics dictionary.
        metrics = env.observation_metrics(obs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_allclose(metrics['efficiency'], env.measurements[..., 0], atol=1e-06)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_allclose(metrics['temperature_c'], env.measurements[..., 1], atol=0.0001)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.local_size, 70)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(obs.shape, (3 * 70 + 10 + 2 * env.config.crews + 2 * env.config.workshops,))

    # Explanation: Define test_stale_grounded_sensor_readings with the shown parameters/defaults; its purpose is to perform this test stale grounded sensor readings operation.
    def test_stale_grounded_sensor_readings(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=4)
        # Explanation: Compute and bind the right-hand value to prior, naming an intermediate for the surrounding algorithm.
        prior = env.observation_metrics(obs)
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = env.idle_action()
        # Explanation: Compute the proposed or selected action and store it in action[0]; later checks/updates use this named intermediate.
        action[0] = [1, 1, 4, 0, 1, 0, 2, 2]
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        obs, *_ = env.step(action)
        # Explanation: Compute and bind the right-hand value to after, naming an intermediate for the surrounding algorithm.
        after = env.observation_metrics(obs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_allclose(after['efficiency'][0], prior['efficiency'][0])
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(after['reading_age_days'][0], 1)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(after['reading_age_days'][1], 0)

    # Explanation: Define test_filter_conditions_on_sensor_channels with the shown parameters/defaults; its purpose is to perform this test filter conditions on sensor channels operation.
    def test_filter_conditions_on_sensor_channels(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to env.reset(seed, naming an intermediate for the surrounding algorithm.
        env.reset(seed=5)
        # Explanation: Compute and bind the right-hand value to other, naming an intermediate for the surrounding algorithm.
        other = FleetMaintenanceEnv(self.config())
        # Explanation: Compute and bind the right-hand value to other.reset(seed, naming an intermediate for the surrounding algorithm.
        other.reset(seed=5)
        # Explanation: Create measurements as a fixed-shape array filled with the given starting value; it represents noisy sensor readings, not hidden true health.
        measurements = env._sensor_model(np.full((2, 6), 0.5), env.exposure[0])
        # Explanation: Compute health values held by this toy state or simulator and store it in other.health[0]; later checks/updates use this named intermediate.
        other.health[0] = 0.01
        # Explanation: Use the simulator's env._update_filter interface to reset, advance, or query the environment.
        env._update_filter(0, measurements)
        # Explanation: Call other._update_filter to perform this step of the surrounding algorithm.
        other._update_filter(0, measurements)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        np.testing.assert_array_equal(env.particles[0], other.particles[0])

    # Explanation: Define test_log_probability_reconstruction_and_gradients with the shown parameters/defaults; its purpose is to perform this test log probability reconstruction and gradients operation.
    def test_log_probability_reconstruction_and_gradients(self):
        # Explanation: Use torch.set_num_threads for the array/tensor operation required at this point.
        torch.set_num_threads(1)
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=4)
        # Explanation: Compute and bind the right-hand value to batch, naming an intermediate for the surrounding algorithm.
        batch = torch.tensor(np.stack([obs] * 5))
        # Explanation: Visit each mode from ('flat', 'conditional', 'shared'); the body applies the operation once per item.
        for mode in ('flat', 'conditional', 'shared'):
            # Explanation: Compute and bind the right-hand value to model, naming an intermediate for the surrounding algorithm.
            model = FleetActorCritic(env, mode)
            # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
            a, lp, entropy, v = model.act(batch)
            # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
            _, lp2, _, _ = model.act(batch, a)
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            torch.testing.assert_close(lp, lp2)
            # Explanation: Visit each row from a[0].tolist(); the body applies the operation once per item.
            for row in a[0].tolist():
                # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
                self.assertIsNone(env.validate_row(0, row))
            # Explanation: Call  to perform this step of the surrounding algorithm.
            (-lp.mean() + v.square().mean() - 0.001 * entropy.mean()).backward()
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            self.assertTrue(all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)))

    # Explanation: Define test_smdp_discount with the shown parameters/defaults; its purpose is to perform this test smdp discount operation.
    def test_smdp_discount(self):
        # Explanation: Unpack smdp_gae([5 + 0.9 * 5], [0], [4], [False], [2], gamma=0.9) into named values adv, target so each part can be used separately below.
        adv, target = smdp_gae([5 + 0.9 * 5], [0], [4], [False], [2], gamma=0.9)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertAlmostEqual(float(target[0]), 9.5 + 0.81 * 4, places=5)
        # Explanation: Unpack the return values from smdp_gae([9.5], [0], [100], [True], [2], gamma=0.9) into terminal; underscore entries are intentionally ignored.
        _, terminal = smdp_gae([9.5], [0], [100], [True], [2], gamma=0.9)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(float(terminal[0]), 9.5)

    # Explanation: Define test_learned_manager_worker_reconstructs_likelihood with the shown parameters/defaults; its purpose is to perform this test learned manager worker reconstructs likelihood operation.
    def test_learned_manager_worker_reconstructs_likelihood(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=4)
        # Explanation: Compute and bind the right-hand value to model, naming an intermediate for the surrounding algorithm.
        model = LearnedOptionActorCritic(env)
        # Explanation: Compute and bind the right-hand value to batch, naming an intermediate for the surrounding algorithm.
        batch = torch.tensor(np.stack([obs] * 3))
        # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
        action, lp, _, _ = model.act(batch)
        # Explanation: Unpack the policy result into its selected action, likelihood, entropy, and value outputs; underscore positions deliberately discard unused statistics.
        _, lp2, _, _ = model.act(batch, action)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        torch.testing.assert_close(lp, lp2)

    # Explanation: Define test_option_execution with the shown parameters/defaults; its purpose is to perform this test option execution operation.
    def test_option_execution(self):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=7)
        # Explanation: Unpack 0.4 into named values env.particles[0, 0, 2] so each part can be used separately below.
        env.particles[0, 0, 2] = 0.4
        # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
        obs = env._observation()
        # Explanation: Unpack the return values from execute_option(env, 1, obs) into end, discount, done, info, tau, raw; underscore entries are intentionally ignored.
        end, discount, done, _, info, tau, raw = execute_option(env, 1, obs)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertGreater(tau, 1)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertEqual(env.day, tau)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertGreater(info['metrics']['flights'], 0)
        # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
        self.assertTrue(env.observation_space.contains(end))

    # Explanation: Verify that crew expertise, team composition, and workshop quality change actual duration according to the specified model.
    def test_heterogeneous_resource_durations(self):
        # Explanation: Compare expert, standard, novice, combined-team, improved-shop, and slower-shop assignments on the same four-day component.
        for crew_mask, shop, expected in ((1, 0, 9), (2, 0, 11), (4, 0, 15), (3, 0, 9), (2, 1, 9), (2, 2, 15)):
            # Explanation: Isolate each assignment so reservations from another case cannot affect its duration.
            env = FleetMaintenanceEnv(self.config(crew_time_factors=(0.5, 1.0, 2.0), workshops=3, workshop_time_factors=(1.0, 0.5, 2.0)))
            # Explanation: Start every comparison from the same synthetic health and observations.
            env.reset(seed=21)
            # Explanation: Propose one HPC repair with the selected crew identities and workshop.
            action = env.idle_action()
            # Explanation: Component bit 2 selects HPC; crew_mask uses independent member bits.
            action[0] = [1, 1, 4, 0, crew_mask, shop, 2, 0]
            # Explanation: Accept the plan and advance its first day.
            env.step(action)
            # Explanation: Check the actual committed duration, not just a helper calculation.
            self.assertEqual(env.jobs[0].total_duration, expected)
            # Explanation: Retain the exact team identity in the committed plan.
            self.assertEqual(env.jobs[0].plan.crew_mask, crew_mask)
            # Explanation: Verify that the count-based reporting property agrees with selected identities.
            self.assertEqual(env.jobs[0].plan.crews, crew_mask.bit_count())
            # Explanation: Advance to the day immediately before completion.
            for _ in range(expected - 2):
                # Explanation: Let the fixed accepted job continue without replanning.
                env.step(env.idle_action())
            # Explanation: The job must still exist before its final day.
            self.assertIn(0, env.jobs)
            # Explanation: Execute its last day and perform the scheduled repair.
            env.step(env.idle_action())
            # Explanation: The job releases its resources exactly at the advertised completion time.
            self.assertNotIn(0, env.jobs)
            # Explanation: Confirm that the selected component was actually restored.
            self.assertEqual(env.health[0, 0, 2], 1.0)
            # Explanation: Every crew member becomes available again after the only job completes.
            self.assertEqual(env.available_crew_mask(), 7)

    # Explanation: Check intermediate component completion thresholds under faster resources, not only total downtime.
    def test_resource_adjusted_component_schedule(self):
        # Explanation: Expert member 0 and an improved workshop shorten a fan-plus-HPC sequence to completion days 8 and 10.
        env = FleetMaintenanceEnv(self.config(crew_time_factors=(0.5, 1.0, 2.0), workshop_time_factors=(0.75, 1.0)))
        # Explanation: Reset and retain initial health for checks before each repair.
        env.reset(seed=22)
        # Explanation: Copy the grounded aircraft's component health, which should not wear during maintenance.
        old = env.health[0].copy()
        # Explanation: Build an action selecting fan and HPC on engine 0 with expert member 0.
        action = env.idle_action()
        # Explanation: Bundle 5 is binary 000101, and crew mask 1 selects only member 0.
        action[0] = [1, 1, 5, 0, 1, 0, 0, 0]
        # Explanation: Start the committed maintenance plan.
        env.step(action)
        # Explanation: Verify the cumulative schedule uses the resource-adjusted task durations.
        self.assertEqual(env.jobs[0].completion_days, (8, 10))
        # Explanation: Advance through all seven setup days.
        for _ in range(6):
            # Explanation: Continue the existing job without changing the crew or workshop.
            env.step(env.idle_action())
        # Explanation: No component work is completed during fixed setup.
        np.testing.assert_array_equal(env.health[0], old)
        # Explanation: Day 8 completes the fan task only.
        env.step(env.idle_action())
        # Explanation: Verify the fan was repaired on its individual scheduled day.
        self.assertEqual(env.health[0, 0, 0], 1.0)
        # Explanation: The HPC remains unchanged until day 10.
        self.assertEqual(env.health[0, 0, 2], old[0, 2])
        # Explanation: Advance day 9 without prematurely completing the remaining component.
        env.step(env.idle_action())
        # Explanation: Day 10 completes the second component and the job.
        env.step(env.idle_action())
        # Explanation: Confirm that the second component also follows its stored threshold.
        self.assertEqual(env.health[0, 0, 2], 1.0)
        # Explanation: Both work and reservation end together.
        self.assertFalse(env.jobs)

    # Explanation: Ensure crew identity prevents double-booking even when the aggregate free count would permit another job.
    def test_individual_crew_conflict_and_observation(self):
        # Explanation: Provide three members and two standard workshops for two competing plans.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Initialize identical state for a deterministic conflict check.
        env.reset(seed=23)
        # Explanation: Retain inventory to verify the rejected plan commits no spares.
        stock = env.stock.copy()
        # Explanation: Propose two jobs on different workshops but the same crew member.
        action = env.idle_action()
        # Explanation: Higher priority reserves crew member 0 for aircraft 0 first.
        action[0] = [1, 1, 4, 0, 1, 0, 0, 2]
        # Explanation: Aircraft 1 requests the same identity despite two other members remaining free.
        action[1] = [1, 1, 4, 0, 1, 1, 0, 1]
        # Explanation: Resolve the joint action and expose the resulting resource flags.
        obs, _, _, _, info = env.step(action)
        # Explanation: The conflicting second job must be rejected atomically.
        self.assertEqual(info['rejected'], [(1, 'resource_conflict')])
        # Explanation: Only aircraft 0 can reserve the shared member.
        self.assertEqual(set(env.jobs), {0})
        # Explanation: Only the accepted HPC repair removes one spare.
        self.assertEqual(env.stock[2], stock[2] - 1)
        # Explanation: Free member bits 1 and 2 form binary 110.
        self.assertEqual(env.available_crew_mask(), 6)
        # Explanation: Decode the policy-visible resource data.
        rows, public = env.split_observation(obs)
        # Explanation: Each crew member's availability is individually visible to the actor and critic.
        np.testing.assert_array_equal(public[10:13], [0, 1, 1])
        # Explanation: Standard duration factors encode as 1 / (1 + 1).
        np.testing.assert_allclose(public[13:16], [0.5, 0.5, 0.5])
        # Explanation: Workshop 0 is reserved while workshop 1 remains available.
        np.testing.assert_array_equal(public[16:18], [0, 1])
        # Explanation: Diagnostics expose the same individual availability mask.
        self.assertEqual(info['resources']['available_crew_mask'], 6)
        # Explanation: Mark aircraft 1 as needing work to exercise the updated rule baseline.
        env.particles[1, 0, 2] = 0.4
        # Explanation: Build a feasible action from observations containing the earlier reservation.
        proposed = threshold_action(env, env._observation())
        # Explanation: The baseline must choose an available member rather than reuse crew 0.
        self.assertEqual(int(proposed[1, 4]), 2)
        # Explanation: It must also choose the remaining workshop.
        self.assertEqual(int(proposed[1, 5]), 1)
        # Explanation: Reconstruct an otherwise valid plan with an empty team.
        proposed[1, 4] = 0
        # Explanation: Empty teams are structurally invalid rather than silently interpreted as one member.
        self.assertEqual(env.validate_row(1, proposed[1]), 'invalid_crew_mask')

    # Explanation: Reject invalid profile lengths and nonphysical duration factors before simulation starts.
    def test_resource_profile_validation(self):
        # Explanation: Cover mismatched crew and workshop counts, zero, negative, nonfinite, Boolean, and textual factors.
        for changes in ({'crew_time_factors': (1.0,)}, {'workshop_time_factors': (1.0,)}, {'crew_time_factors': (0, 1, 1)}, {'crew_time_factors': (-1, 1, 1)}, {'crew_time_factors': (float('nan'), 1, 1)}, {'workshop_time_factors': (float('inf'), 1)}, {'crew_time_factors': (True, 1, 1)}, {'crew_time_factors': ('expert', 1, 1)}, {'crew_time_factors': None}, {'crew_time_factors': 1.0}):
            # Explanation: Every malformed profile should fail with a useful configuration error.
            with self.assertRaises(ValueError):
                # Explanation: Constructing the frozen configuration runs its validation immediately.
                Config(**changes)
        # Explanation: Default profiles are heterogeneous illustrative values in stable resource-ID order.
        env = FleetMaintenanceEnv(Config())
        # Explanation: Confirm the default crew factors match the documented teaching profile.
        self.assertEqual(env.crew_time_factors, (0.8, 1.0, 1.25))
        # Explanation: Confirm that the improved workshop has a smaller duration factor.
        self.assertEqual(env.workshop_time_factors, (0.8, 1.0))

    # Explanation: Exercise variable fleet sizes and larger crew bitmasks through every policy architecture.
    def test_configurable_fleet_size_and_resource_masks(self):
        # Explanation: Seven remains the default, but neither the simulator nor model factory is limited to it.
        for aircraft in (1, 7, 10):
            # Explanation: Eight individually selectable members exercise a 256-category crew-mask head.
            env = FleetMaintenanceEnv(Config(aircraft=aircraft, crews=8, workshops=3))
            # Explanation: Build a fleet-sized observation from the selected configuration.
            obs, _ = env.reset(seed=24)
            # Explanation: Identity labels scale with the configured fleet.
            self.assertEqual(len(env.aircraft_plates), aircraft)
            # Explanation: Each aircraft keeps eight action fields while crew masks cover every member subset.
            self.assertEqual(env.action_space.nvec.shape, (aircraft, 8))
            # Explanation: Verify all eight crew bits can be represented.
            self.assertEqual(env.row_dims[4], 256)
            # Explanation: Public observations grow with individual crews and workshops.
            self.assertEqual(obs.shape, (aircraft * 70 + 10 + 2 * 8 + 2 * 3,))
            # Explanation: Rebuild each architecture for this fleet rather than reusing incompatible fixed-size weights.
            for method in ('flat', 'conditional', 'shared', 'options', 'learned_options'):
                # Explanation: The existing factory sizes actor and critic inputs and outputs from the environment.
                model = make_model(env, method)
                # Explanation: Generate a policy action for the configured observation.
                action, logp, _, value = model.act(torch.tensor(obs))
                # Explanation: Re-score that action to verify the enlarged masks retain correct joint likelihoods.
                _, reconstructed, _, _ = model.act(torch.tensor(obs), action)
                # Explanation: Stored-action likelihood must match sampling-time likelihood before an update.
                torch.testing.assert_close(logp, reconstructed)
                # Explanation: The controller still returns one fleet-level value.
                self.assertEqual(value.shape, (1,))
                # Explanation: Daily policies and learned workers must select valid nonempty teams for new work.
                if method != 'options':
                    # Explanation: Extract the worker plan from the composite learned-option action when necessary.
                    plan = action[0, 1:].reshape(aircraft, 8) if method == 'learned_options' else action[0]
                    # Explanation: Validate each aircraft's new action row against the crew-mask semantics.
                    for row in plan.tolist():
                        # Explanation: All sampled rows should satisfy structural constraints.
                        self.assertIsNone(env.validate_row(0, row))

    # Explanation: Ensure new resource profiles survive checkpoint reload and old action layouts are rejected explicitly.
    def test_resource_checkpoint_roundtrip(self):
        # Explanation: Use JSON-style lists to verify profile normalization as well as model serialization.
        cfg = self.config(crew_time_factors=[0.5, 1.0, 2.0], workshop_time_factors=[0.75, 1.25])
        # Explanation: Build an observation and a conditional fleet model with the new layout.
        env = FleetMaintenanceEnv(cfg)
        # Explanation: Set a repeatable physical state for deterministic action comparison.
        obs, _ = env.reset(seed=25)
        # Explanation: Instantiate the custom model that the checkpoint loader must reconstruct.
        model = make_model(env, 'conditional')
        # Explanation: Keep test checkpoints outside the user's saved runs and clean them up automatically.
        with tempfile.TemporaryDirectory() as directory:
            # Explanation: Choose a temporary checkpoint filename.
            path = Path(directory) / 'resource_model.pt'
            # Explanation: Save the same configuration, weights, method, and layout metadata used by the runner.
            torch.save({'config': asdict(cfg), 'state_dict': model.state_dict(), 'method': 'conditional', 'observation_version': OBSERVATION_VERSION}, path)
            # Explanation: Rebuild the model through the public checkpoint API.
            loaded_cfg, method, loaded_model = load_checkpoint(path)
            # Explanation: Individual resource profiles and counts must remain identical.
            self.assertEqual(loaded_cfg, cfg)
            # Explanation: The selected policy architecture must be restored too.
            self.assertEqual(method, 'conditional')
            # Explanation: Check that both models choose the same action from the same observation after reload.
            torch.testing.assert_close(model.act(torch.tensor(obs), deterministic=True)[0], loaded_model.act(torch.tensor(obs), deterministic=True)[0])
            # Explanation: Replace the temporary file with an old-layout checkpoint marker.
            torch.save({'observation_version': 2}, path)
            # Explanation: Reject old crew-count actions before trying to reconstruct their network shapes.
            with self.assertRaisesRegex(ValueError, 'incompatible observation/action layout'):
                # Explanation: Exercise the actual user-facing loader error.
                load_checkpoint(path)

    # Explanation: Verify mappo views shapes and likelihood.
    def test_mappo_views_shapes_and_likelihood(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        for aircraft in (1, 3, 7):
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            cfg = self.config(aircraft=aircraft)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            for assignment in ('aircraft', 'component'):
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                env = MultiAgentFleetEnv(cfg, assignment)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                views, info = env.reset(seed=31)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                model = MultiAgentActorCritic(env.env, assignment)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(model.num_agents, aircraft if assignment == 'aircraft' else 6)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(tuple(views), model.agent_ids)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                for agent in env.agent_ids:
                    # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                    self.assertTrue(env.observation_spaces[agent].contains(views[agent]))
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                central = torch.tensor(np.stack([info['central_observation']] * 2))
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                actions, logp, entropy, values = model.act(central)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(logp.shape, (2, model.num_agents))
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(values.shape, logp.shape)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(entropy.shape, logp.shape)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                _, reconstructed, _, _ = model.act(central, actions)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                torch.testing.assert_close(logp, reconstructed)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                dictionary = model.action_dict(actions[0])
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                for agent in model.agent_ids:
                    # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                    self.assertTrue(env.action_spaces[agent].contains(dictionary[agent]))
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                _, rewards, _, _, next_info = env.step(dictionary)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(set(rewards), set(env.agent_ids))
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(len(set(rewards.values())), 1)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertTrue(env.env.observation_space.contains(next_info['central_observation']))

    # Explanation: Verify mappo component ownership and matched dispatch.
    def test_mappo_component_ownership_and_matched_dispatch(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        cfg = self.config()
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        aircraft_env = MultiAgentFleetEnv(cfg, 'aircraft')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_env = MultiAgentFleetEnv(cfg, 'component')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        aircraft_env.reset(seed=32)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_env.reset(seed=32)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        specialists = {name: np.zeros(3, dtype=np.int64) for name in COMPONENTS}
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        specialists['fan'][0] = 3
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        specialists['HPC'][0] = 1
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        merged, deferred = component_env.compose_action(specialists)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertFalse(deferred)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(merged[0, 1:4].tolist(), [3, 5, 1])
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(int(merged[:, 2].sum()), 5)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(int(merged[:, 3].sum()), 1)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        aircraft_actions = {plate: np.zeros(2, dtype=np.int64) for plate in aircraft_env.agent_ids}
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        aircraft_actions['AC-001'] = np.array([5, 1])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        matched, _ = aircraft_env.compose_action(aircraft_actions)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        np.testing.assert_array_equal(matched, merged)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        _, rewards, _, _, info = component_env.step(specialists)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        job = component_env.env.jobs[0]
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(set(job.tasks), {(0, 0), (0, 2), (1, 0)})
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(len(rewards), 6)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(info['agent_proposals']['HPC'], [1, 0, 0])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        plan = job.plan
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_env.step(specialists)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(component_env.env.jobs[0].plan, plan)

    # Explanation: Verify mappo actor information boundaries.
    def test_mappo_actor_information_boundaries(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        obs, _ = env.reset(seed=33)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        aircraft_model = MultiAgentActorCritic(env, 'aircraft')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        altered = obs.copy()
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        altered[70] = 1 - altered[70]
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        before = aircraft_model.agent_inputs(obs)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        after = aircraft_model.agent_inputs(altered)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        torch.testing.assert_close(before[:, 0], after[:, 0])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        action = torch.zeros((1, 3, 2), dtype=torch.long)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        torch.testing.assert_close(aircraft_model.policy(before, action)[1][:, 0], aircraft_model.policy(after, action)[1][:, 0])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_model = MultiAgentActorCritic(env, 'component')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        altered = obs.copy()
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        altered[2] = 1 - altered[2]
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        altered[47] = 1 - altered[47]
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        before = component_model.agent_inputs(obs)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        after = component_model.agent_inputs(altered)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        torch.testing.assert_close(before[:, 0], after[:, 0])
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertFalse(torch.equal(before[:, 2], after[:, 2]))
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        action = torch.zeros((1, 6, 3), dtype=torch.long)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        torch.testing.assert_close(component_model.policy(before, action)[1][:, 0], component_model.policy(after, action)[1][:, 0])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_model.critic.forward = lambda *args: (_ for _ in ()).throw(AssertionError('critic used in actor execution'))
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        component_model.policy(before, deterministic=True)

    # Explanation: Verify mappo busy masks and broker deferrals.
    def test_mappo_busy_masks_and_broker_deferrals(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        env = MultiAgentFleetEnv(self.config(crews=1, workshops=1), 'component')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        env.reset(seed=34)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        actions = {name: np.zeros(3, dtype=np.int64) for name in COMPONENTS}
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        actions['fan'][:] = 1
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        stock = env.env.stock.copy()
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        merged, deferred = env.compose_action(actions)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(int(merged[:, 0].sum()), 1)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(len(deferred), 2)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        np.testing.assert_array_equal(env.env.stock, stock)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        _, _, _, _, info = env.step(actions)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(len(env.env.jobs), 1)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertEqual(info['metrics']['coordination_deferrals'], 2)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertFalse(info['rejected'])
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        for assignment in ('aircraft', 'component'):
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            model = MultiAgentActorCritic(env.env, assignment)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            obs = env.central_observation.copy()
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            obs[:3 * 70].reshape(3, 70)[:, 37] = 1
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            selected, logp, entropy, _ = model.act(torch.tensor(obs))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertFalse(bool(selected.any()))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            torch.testing.assert_close(logp, torch.zeros_like(logp))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            torch.testing.assert_close(entropy, torch.zeros_like(entropy))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertFalse(bool(model.active_agents(model.agent_inputs(obs)).any()))
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            invalid = selected.clone()
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            invalid[0, 0, 0] = 1
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            with self.assertRaises(ValueError):
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                model.act(torch.tensor(obs), invalid)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        invalid = dict(actions)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        invalid.pop('fan')
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        with self.assertRaises(ValueError):
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            env.compose_action(invalid)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        actions['fan'][0] = 4
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        with self.assertRaises(ValueError):
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            env.compose_action(actions)

    # Explanation: Verify mappo per agent clipping and advantages.
    def test_mappo_per_agent_clipping_and_advantages(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        new = torch.log(torch.tensor([[2.0, 0.5]]))
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        old = torch.zeros_like(new)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        ones = torch.ones_like(new)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertAlmostEqual(float(mappo_policy_loss(new, old, ones, ones)), -0.85, places=6)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        self.assertAlmostEqual(float(mappo_policy_loss(new, old, ones, torch.tensor([[1.0, 0.0]]))), -1.2, places=6)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        with self.assertRaises(ValueError):
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            mappo_policy_loss(new.sum(-1), old.sum(-1), ones, ones)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        advantages, targets = mappo_gae([2], [[1, 3]], [[4, 6]], [False], gamma=0.9)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        np.testing.assert_allclose(targets, [[5.6, 7.4]], atol=1e-6)
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        np.testing.assert_allclose(advantages, [[4.6, 4.4]], atol=1e-6)
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        _, targets = mappo_gae([2], [[1, 3]], [[4, 6]], [True])
        # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
        np.testing.assert_array_equal(targets, [[2, 2]])

    # Explanation: Verify mappo training and checkpoint roundtrip.
    def test_mappo_training_and_checkpoint_roundtrip(self):
        # Explanation: Prepare or execute this step of the multi-agent behavioral check.
        for method in MAPPO_METHODS:
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            cfg = self.config(horizon=12)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            torch.manual_seed(35)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            initial = make_model(FleetMaintenanceEnv(cfg), method)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            actor_before = [p.detach().clone() for p in initial.actors.parameters()]
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            critic_before = [p.detach().clone() for p in initial.critic.parameters()]
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            model, logs, days = train(cfg, method, steps=32, seed=35, rollout_size=8)
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertEqual(days, 32)
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, model.actors.parameters())))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters())))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertTrue(any(record['record_type'] == 'update' for record in logs))
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertTrue(any(record['record_type'] == 'episode' for record in logs))
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            result = evaluate(cfg, method, model, seeds=[36])[0]
            # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
            self.assertIn('coordination_deferrals', result)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            env = FleetMaintenanceEnv(cfg)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            obs, _ = env.reset(seed=37)
            # Explanation: Prepare or execute this step of the multi-agent behavioral check.
            with tempfile.TemporaryDirectory() as directory:
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                path = Path(directory) / 'mappo.pt'
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                torch.save({'config': asdict(cfg), 'method': method, 'observation_version': OBSERVATION_VERSION, 'state_dict': model.state_dict()}, path)
                # Explanation: Prepare or execute this step of the multi-agent behavioral check.
                restored_cfg, restored_method, restored = load_checkpoint(path)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(restored_cfg, cfg)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                self.assertEqual(restored_method, method)
                # Explanation: Check the stated behavioral invariant; failure identifies an ownership, information, or learning error.
                torch.testing.assert_close(model.act(torch.tensor(obs), deterministic=True)[0], restored.act(torch.tensor(obs), deterministic=True)[0])

    # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
    def test_dcmac_ddmac_architectures_and_information(self):
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        cfg = self.config(aircraft=7)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        env = FleetMaintenanceEnv(cfg)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        obs, _ = env.reset(seed=51)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        for method in DCMAC_METHODS:
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model = make_model(env, method)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            units = 7 if method.endswith('aircraft') else 6
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            expected_networks = units if method.startswith('ddmac') else 1
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(len(model.actors), expected_networks)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(model.num_agents, units)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(model.agent_inputs(obs).shape, (1, len(obs)))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            selected, logp, entropy, value = model.act(obs)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(tuple(selected.shape), (1, units, model.fields))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(tuple(logp.shape), (1, units))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(tuple(value.shape), (1,))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            rescored = model.policy(obs, selected)[1]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            torch.testing.assert_close(rescored, logp)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            captured = []
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            hooks = [actor[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
                     # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                     for actor in model.actors]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model.policy(obs, deterministic=True)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(len(captured), expected_networks)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            for actor_input in captured:
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                torch.testing.assert_close(actor_input, torch.tensor(obs).unsqueeze(0))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            for hook in hooks:
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                hook.remove()
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            before = model.policy(obs, deterministic=True)[0]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            env.health[:] = 0
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            torch.testing.assert_close(before, model.policy(obs, deterministic=True)[0])
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model.critic.forward = lambda *args: (_ for _ in ()).throw(AssertionError('critic used in policy'))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model.policy(obs, deterministic=True)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            if model.decentralized:
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                parameter_sets = [{p.data_ptr() for p in actor.parameters()} for actor in model.actors]
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                self.assertTrue(all(not left.intersection(right) for i, left in enumerate(parameter_sets) for right in parameter_sets[i + 1:]))
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                fixed = selected.clone()
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                oldlog = model.policy(obs, fixed)[1].detach().clone()
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                with torch.no_grad():
                    # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                    model.actors[0][-1].bias[int(fixed[0, 0, 0])] += 2
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                changedlog = model.policy(obs, fixed)[1]
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                self.assertNotEqual(float(oldlog[0, 0]), float(changedlog[0, 0].detach()))
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                torch.testing.assert_close(oldlog[:, 1:], changedlog[:, 1:])

    # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
    def test_dcmac_joint_importance_and_terminal_targets(self):
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        logp = torch.log(torch.tensor([[4., .25], [4., 1.], [.5, .5]])).requires_grad_()
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        behavior = torch.zeros_like(logp, requires_grad=True)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        values = torch.tensor([1., 2., 3.], requires_grad=True)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        next_values = torch.tensor([10., 20., 30.], requires_grad=True)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        rewards = torch.tensor([2., 4., 6.])
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        dones = torch.tensor([False, True, False])
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        actor, critic, weights, targets = dcmac_losses(logp, behavior, values, next_values, rewards, dones, gamma=.9)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(weights, torch.tensor([1., 2., .25]))
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(targets, torch.tensor([11., 4., 33.]))
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        expected = -(weights * logp.detach().sum(-1) * (targets - values.detach())).mean()
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(actor.detach(), expected)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        actor.backward()
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(logp.grad, (-(weights * (targets - values.detach())) / 3).unsqueeze(-1).expand(-1, 2))
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        self.assertIsNone(values.grad)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        self.assertIsNone(next_values.grad)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        self.assertIsNone(behavior.grad)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        critic.backward()
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(values.grad, weights * (values.detach() - targets) / 3)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        self.assertIsNone(next_values.grad)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        extreme = dcmac_losses(torch.full((1, 2), 1000.), torch.full((1, 2), -1000.),
                               # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                               torch.zeros(1), torch.ones(1), torch.ones(1), torch.zeros(1))[2]
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        torch.testing.assert_close(extreme, torch.tensor([2.]))
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        with self.assertRaises(ValueError):
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            dcmac_losses(logp.sum(-1), behavior, values, next_values, rewards, dones)

    # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
    def test_dcmac_busy_masks_and_common_dispatch(self):
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        cfg = self.config()
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        env = FleetMaintenanceEnv(cfg)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        obs, _ = env.reset(seed=52)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        for method in DCMAC_METHODS:
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model = make_model(env, method)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            adapter = MultiAgentFleetEnv(cfg, request_assignment(method))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            adapter.reset(seed=52)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            proposal = model.act(obs)[0][0]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            merged, _ = adapter.compose_action(model.action_dict(proposal))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(adapter.env.action_space.contains(merged))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            busy_obs = obs.copy()
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            busy_obs[:cfg.aircraft * env.local_size].reshape(cfg.aircraft, env.local_size)[:, 37] = 1
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            selected, logp, entropy, _ = model.act(busy_obs)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertFalse(bool(selected.any()))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            torch.testing.assert_close(logp, torch.zeros_like(logp))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            torch.testing.assert_close(entropy, torch.zeros_like(entropy))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertFalse(bool(model.active_agents(busy_obs).any()))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            invalid = selected.clone()
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            invalid[0, 0, 0] = 1
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            with self.assertRaises(ValueError):
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                model.policy(busy_obs, invalid)
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        for n in (1, 10):
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            env = FleetMaintenanceEnv(self.config(aircraft=n))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            obs, _ = env.reset(seed=53)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model = make_model(env, 'ddmac_aircraft')
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(len(model.actors), n)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(tuple(model.act(obs)[0].shape), (1, n, 2))

    # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
    def test_dcmac_replay_training_and_checkpoint_roundtrip(self):
        # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
        for method in DCMAC_METHODS:
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            cfg = self.config(horizon=8)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            torch.manual_seed(54)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            initial = make_model(FleetMaintenanceEnv(cfg), method)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            actor_before = [p.detach().clone() for p in initial.actors.parameters()]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            critic_before = [p.detach().clone() for p in initial.critic.parameters()]
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            model, logs, days = train_dcmac(cfg, method, steps=24, seed=54, rollout_size=4, replay_capacity=6)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(days, 24)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, model.actors.parameters())))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters())))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            updates = [record for record in logs if record['record_type'] == 'update']
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertEqual(len(updates), 23)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(all(record['replay_size'] <= 6 and record['batch_size'] <= 4 for record in updates))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(all(0 <= record['importance_max'] <= 2.000001 for record in updates))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(any(min(record['sampled_transition_days']) < record['simulated_days'] for record in updates))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(all(min(record['sampled_transition_days']) >= max(1, record['simulated_days'] - 5) for record in updates))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertTrue(any(record['record_type'] == 'episode' for record in logs))
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            records = evaluate(cfg, method, model, seeds=[55])
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            self.assertIn('coordination_deferrals', records[0])
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            env = FleetMaintenanceEnv(cfg)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            obs, _ = env.reset(seed=56)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            with tempfile.TemporaryDirectory() as directory:
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                path = Path(directory) / 'dcmac.pt'
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                torch.save({'config': asdict(cfg), 'method': method, 'observation_version': OBSERVATION_VERSION,
                            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                            'state_dict': model.state_dict()}, path)
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                restored_cfg, restored_method, restored = load_checkpoint(path)
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                self.assertEqual(restored_cfg, cfg)
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                self.assertEqual(restored_method, method)
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                torch.testing.assert_close(model.act(obs, deterministic=True)[0], restored.act(obs, deterministic=True)[0])
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            again, _, _ = train_dcmac(cfg, method, steps=24, seed=54, rollout_size=4, replay_capacity=6)
            # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
            for a, b in zip(model.parameters(), again.parameters()):
                # Explanation: Verify this architecture, replay, gradient, or checkpoint invariant.
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_concurrent_aircraft_options_and_commitment(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        cfg = self.config(aircraft=2, horizon=30, crews=2, workshops=2)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = AircraftOptionEnv(cfg)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env.reset(seed=111)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        _, _, _, _, first = env.step(np.array([1, 0]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        committed = env.env.jobs[0].plan
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(first['aircraft_option_started'][0]['aircraft_plate'], 'AC-001')
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        np.testing.assert_array_equal(env.decision_mask(), [False, True])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        _, _, _, _, second = env.step(np.array([0, 3]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(len(env.env.jobs), 2)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(env.env.jobs[0].plan, committed)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        with self.assertRaises(ValueError):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            env.step(np.array([2, 0]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        while not env.decision_mask()[0]:
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            env.step(np.zeros(2, dtype=np.int64))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertFalse(env.decision_mask()[1])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(env.option_history[0]['duration'], 9)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        _, _, _, _, restarted = env.step(np.array([2, 0]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(restarted['aircraft_option_started'][0]['aircraft'], 0)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertIn(1, env.env.jobs)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        with self.assertRaises(ValueError):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            env.step(np.array([0.0, 0.0]))

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_planning_hierarchy_likelihood_and_learned_stages(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        obs, _ = env.reset(seed=112)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for method in HRL_PPO_METHODS:
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            model = make_model(env, method)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            action, logp, entropy, values = model.act(obs)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertEqual(tuple(action.shape), (1, 3, 3))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertEqual(tuple(logp.shape), (1, 3))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            torch.testing.assert_close(logp, model.act(obs, action)[1])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            choice = model.catalog_actions(obs, action)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertEqual(choice.shape, (3,))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            before = model.act(obs, deterministic=True)[0]
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            env.health[:] = 0
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            torch.testing.assert_close(before, model.act(obs, deterministic=True)[0])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            busy_obs = obs.copy()
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            busy_obs[:3 * 70].reshape(3, 70)[:, 37] = 1
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            busy_action, busy_lp, _, _ = model.act(busy_obs)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertFalse(bool(busy_action.any()))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            torch.testing.assert_close(busy_lp, torch.zeros_like(busy_lp))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model = make_model(env, 'planning_hierarchy')
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        specified = torch.tensor([[[1, 2, 2]] * 3])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        _, logp, _, _ = model.act(obs, specified)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        np.testing.assert_array_equal(model.catalog_actions(obs, specified), [17] * 3)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        (-logp.mean()).backward()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for policy in (model.manager, model.engine_policy, model.component_policy):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in policy.parameters()))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_hrl_matched_catalog_and_budget_goals(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = FleetMaintenanceEnv(self.config(aircraft=1))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        obs, _ = env.reset(seed=113)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        flat = make_model(env, 'hrl_flat')
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        hierarchy = make_model(env, 'planning_hierarchy')
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(len(HRL_CATALOG), 22)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for catalog in range(22):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            flat_action = torch.tensor([[[catalog, 0, 0]]])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            stage_action = torch.tensor([[[0, 0, 0]]]) if catalog == 0 else torch.tensor([[[1, (catalog - 1) // 7, (catalog - 1) % 7]]])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            flat.act(obs, flat_action)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            hierarchy.act(obs, stage_action)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            np.testing.assert_array_equal(flat.catalog_actions(obs, flat_action), hierarchy.catalog_actions(obs, stage_action))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        targets = make_model(env, 'aggregate_targets')
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for goal in range(4):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            valid = hrl_catalog_mask(obs, 1, goals=torch.tensor([[goal]]))[0, 0]
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            for index in torch.nonzero(valid).flatten().tolist():
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                count = sum(b.bit_count() for b in HRL_CATALOG[index])
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                self.assertLessEqual(count, HRL_BUDGETS[goal])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        with self.assertRaises(ValueError):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            targets.act(obs, torch.tensor([[[1, 15, 0]]]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        urgent = obs.copy()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        urgent[24:36] = 0
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for method in HRL_PPO_METHODS:
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            model = make_model(env, method)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            chosen = model.catalog_actions(urgent, model.act(urgent)[0])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertNotEqual(int(chosen[0]), 0)

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_hrl_duration_targets_and_budget_cutoff(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(hrl_option_target([2, 3], 4, False, .9), 7.94)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(hrl_option_target([2, 3], 999, True, .9), 4.7)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        _, logs, days = train(self.config(aircraft=2, horizon=30), 'concurrent_options', steps=32, seed=4, rollout_size=8)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        updates = [r for r in logs if r['record_type'] == 'option_update']
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(days, 32)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(any(duration > 1 for row in updates for duration in row['durations']))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(any(not complete for row in updates for complete in row['complete']))

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_maxq_value_completion_decomposition(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        cfg = self.config(aircraft=1)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = FleetMaintenanceEnv(cfg)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        obs, _ = env.reset(seed=114)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model = MAXQMaintenance(env)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        state = model.state(obs)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model.leaf_values[('Submit1', state)] = 2.
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model.completion[('Engine0', state, 'Submit1')] = 3.
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(model.q('Engine0', obs, 'Submit1'), 5.)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(model.value('Engine0', obs), 5.)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model.update_completion('Root', obs, 'Repair', 10., duration=3, alpha=1., gamma=.9)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(model.completion[('Root', state, 'Repair')], 7.29)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertAlmostEqual(model.q('Root', obs, 'Repair'), 12.29)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        with self.assertRaises(ValueError):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            MAXQMaintenance(FleetMaintenanceEnv(self.config(aircraft=2)))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        trained, logs, _ = train(cfg, 'maxq', steps=32, seed=4)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(trained.leaf_values)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(trained.completion)

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_option_critic_targets_and_safe_termination(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        following_q = torch.tensor([[[4., 10., 6.]]])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        ids = torch.tensor([[0]])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for beta, expected in [(0., 5.6), (1., 11.), (.25, 6.95)]:
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            target = option_critic_target(2., following_q, ids, torch.tensor([[beta]]), torch.tensor(False), gamma=.9)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertAlmostEqual(float(target[0, 0]), expected, places=5)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        terminal = option_critic_target(2., following_q, ids, torch.ones(1, 1), torch.tensor(True))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(terminal, torch.tensor([[2.]]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        obs, _ = env.reset(seed=115)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model = FleetOptionCritic(env)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        ids = torch.zeros((1, 3), dtype=torch.long)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        busy_obs = obs.copy()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        busy_obs[:3 * 70].reshape(3, 70)[:, 37] = 1
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(model.beta(busy_obs, ids), torch.zeros(1, 3))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertFalse(bool(model.policy(busy_obs, ids)[0].any()))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        (-2 * model.beta(obs, ids).mean()).backward()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.termination.parameters()))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertTrue(all(p.grad is None for p in model.q_network.parameters()))

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_hierarchical_vdn_mixing_and_local_information(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        values = torch.tensor([[2., 3., 5.]], requires_grad=True)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        mixed = HierarchicalVDN.mix(values)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(mixed, torch.tensor([10.]))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        mixed.sum().backward()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        env = FleetMaintenanceEnv(self.config())
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        obs, _ = env.reset(seed=116)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        model = HierarchicalVDN(env)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        goals = torch.ones((1, 3), dtype=torch.long)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        high_before = model.high_values(obs)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        low_before = model.low_values(obs, goals)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        changed = obs.copy()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        changed[70:82] *= .5
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(high_before[:, 0], model.high_values(changed)[:, 0])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        torch.testing.assert_close(low_before[:, 0], model.low_values(changed, goals)[:, 0])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        busy = obs.copy()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        busy[37] = 1
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        previous = np.array([2, 1, 0])
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        chosen = model.choose_goals(busy, previous, np.random.default_rng(1), epsilon=1.)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(chosen[0], 2)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertEqual(model.choose_actions(busy, chosen, np.random.default_rng(2))[0], 0)
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        urgent = obs.copy()
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        urgent[24:36] = 0
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        refreshed = model.choose_goals(urgent, np.zeros(3, dtype=int), np.random.default_rng(3))
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        self.assertNotEqual(refreshed[0], 0)

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_new_hrl_training_and_checkpoint_roundtrip(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for method in HRL_METHODS:
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            cfg = self.config(aircraft=1 if method == 'maxq' else 2, horizon=12)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            torch.manual_seed(4)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            initial = make_model(FleetMaintenanceEnv(cfg), method)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            before = [p.detach().clone() for p in initial.parameters()]
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            model, logs, days = train(cfg, method, steps=32, seed=4, rollout_size=8)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertEqual(days, 32)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertTrue(logs)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            if method != 'maxq':
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            if method == 'option_critic':
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                for old_module, new_module in ((initial.worker, model.worker), (initial.q_network, model.q_network),
                                                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                                                (initial.termination, model.termination)):
                    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old_module.parameters(), new_module.parameters())))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            if method == 'hierarchical_vdn':
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                for old_module, new_module in ((initial.high_agents, model.high_agents), (initial.low_agents, model.low_agents)):
                    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old_module.parameters(), new_module.parameters())))
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            expected = evaluate(cfg, method, model, seeds=[117])
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            with tempfile.TemporaryDirectory() as directory:
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                path = Path(directory) / 'hrl.pt'
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                torch.save({'config': asdict(cfg), 'method': method, 'observation_version': OBSERVATION_VERSION,
                            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                            'state_dict': model.state_dict()}, path)
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                restored_cfg, restored_method, restored = load_checkpoint(path)
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                self.assertEqual(restored_cfg, cfg)
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                self.assertEqual(restored_method, method)
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                self.assertEqual(expected, evaluate(cfg, method, restored, seeds=[117]))

    # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
    def test_new_hrl_reproducible_seeds(self):
        # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
        for method in ('planning_hierarchy', 'option_critic', 'hierarchical_vdn'):
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            cfg = self.config(aircraft=2, horizon=8)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            first, _, _ = train(cfg, method, steps=16, seed=118, rollout_size=8)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            second, _, _ = train(cfg, method, steps=16, seed=118, rollout_size=8)
            # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
            for a, b in zip(first.parameters(), second.parameters()):
                # Explanation: Check this hierarchy, timing, learning, or persistence invariant.
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Explanation: Define test_all_trainers_smoke with the shown parameters/defaults; its purpose is to perform this test all trainers smoke operation.
    def test_all_trainers_smoke(self):
        # Explanation: Visit each method from ('flat', 'conditional', 'shared', 'options', 'learned_options'); the body applies the operation once per item.
        for method in ('flat', 'conditional', 'shared', 'options', 'learned_options'):
            # Explanation: Unpack train(self.config(horizon=12), method, steps=32, rollout_size=8) into named values model, logs, days so each part can be used separately below.
            model, logs, days = train(self.config(horizon=12), method, steps=32, rollout_size=8)
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            self.assertEqual(days, 32)
            # Explanation: Verify the expected equality, shape, range, or numerical property; a mismatch fails the behavioral test.
            self.assertTrue(all((torch.isfinite(p).all() for p in model.parameters())))


    def test_critic_only_calls_skip_actor_context(self):
        env = FleetMaintenanceEnv(self.config())
        obs, _ = env.reset(seed=133)
        for model, value_function in (
            (AircraftPlanningHierarchy(env), 'value'),
            (FleetOptionCritic(env), 'q_values'),
        ):
            with patch.object(model, 'features', side_effect=AssertionError('actor context allocated')):
                value = getattr(model, value_function)(obs)
                self.assertTrue(torch.isfinite(value).all())
                with self.assertRaises(ValueError):
                    getattr(model, value_function)(obs[:-1])

    def test_reset_rejects_nonfinite_custom_health(self):
        env = FleetMaintenanceEnv(self.config())
        for invalid in (float('nan'), float('inf'), -float('inf')):
            health = np.full((env.config.aircraft, 2, 6), .8)
            health[0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                env.reset(seed=134, options={'health': health})
        obs, _ = env.reset(seed=134, options={'health': np.full_like(health, .8)})
        self.assertTrue(np.isfinite(obs).all())


# Explanation: Define run_checks with the shown parameters/defaults; its purpose is to load and execute the consolidated behavioral test case class.
def run_checks():
    # Explanation: This docstring states run_checks's purpose/contract: Run the fleet behavioral checks from the same consolidated module. It is documentation text, not an executable operation.
    """Run the fleet behavioral checks from the same consolidated module."""
    # Explanation: Define and run the behavioral tests.
    import unittest
    # Explanation: Compute and bind the right-hand value to suite, naming an intermediate for the surrounding algorithm.
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FleetTests)
    # Explanation: Compute and bind the right-hand value to result, naming an intermediate for the surrounding algorithm.
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # Explanation: Branch on not result.wasSuccessful(); the indented statements handle this case, and other cases fall through or use a later branch.
    if not result.wasSuccessful():
        # Explanation: Raise AssertionError('Fleet verification failed') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise AssertionError('Fleet verification failed')
    # Explanation: Return result to the caller; this ends the function and supplies its result.
    return result

# Explanation: Branch on __name__ == '__main__'; the indented statements handle this case, and other cases fall through or use a later branch.
if __name__ == '__main__':
    # Explanation: Call main to perform this step of the surrounding algorithm.
    main()
