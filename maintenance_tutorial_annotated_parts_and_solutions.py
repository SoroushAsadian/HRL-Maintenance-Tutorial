# Part 01 / predict_one_step
# Explanation: Define part01_predict_one_step with the shown parameters/defaults; its purpose is to implement one toy transition from health and a primitive action.
def part01_predict_one_step(health: int, action: int):
    # Explanation: Branch on health not in (0, 1, 2, 3); the indented statements handle this case, and other cases fall through or use a later branch.
    if health not in (0, 1, 2, 3):
        # Explanation: Raise ValueError('health must be in 0..3') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('health must be in 0..3')
    # Explanation: Branch on action == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if action == 0:
        # Explanation: Compute and bind the right-hand value to health_after, naming an intermediate for the surrounding algorithm.
        health_after = max(0, health - 1)
        # Explanation: Branch on health_after == 0; the indented statements handle this case, and other cases fall through or use a later branch.
        if health_after == 0:
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return (health_after, -12.0, 'failed_operation')
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (health_after, 5.0, 'successful_operation')
    # Explanation: Branch on action == 1; the indented statements handle this case, and other cases fall through or use a later branch.
    if action == 1:
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (3, -4.0, 'grounded_and_replaced')
    # Explanation: Raise ValueError('action must be 0 or 1') so invalid input/failure is explicit instead of continuing with inconsistent state.
    raise ValueError('action must be 0 or 1')



# Part 01 / student_policy
# Explanation: Define part01_student_policy with the shown parameters/defaults; its purpose is to choose a toy operation or replacement action from the observed state.
def part01_student_policy(observation: tuple[int, int]):
    # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
    health, remained_day = observation
    # Explanation: Branch on health > 1; the indented statements handle this case, and other cases fall through or use a later branch.
    if health > 1:
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = 0
    # Explanation: Handle the alternative case not selected by the preceding if/elif test.
    else:
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = 1
    # Explanation: Return action to the caller; this ends the function and supplies its result.
    return action



# Part 01 / wear_trace
# Explanation: Define part01_wear_trace with the shown parameters/defaults; its purpose is to record a seeded wear/reward sequence for inspecting the simulator.
def part01_wear_trace(seed, shock_prob=0.35):
    # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
    env = MiniMaintenanceEnv(horizon=8, shock_prob=shock_prob)
    # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
    obs, _ = env.reset(seed=seed)
    # Explanation: Compute and bind the right-hand value to trace, naming an intermediate for the surrounding algorithm.
    trace = []
    # Explanation: Visit each _ from range(8); the body applies the operation once per item.
    for _ in range(8):
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = threshold_policy(obs)
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        next_obs, reward, terminated, _truncated, info = env.step(action)
        # Explanation: Append the new record/value to trace so it remains in the result/history.
        trace.append((obs[0], action, info['wear'], next_obs[0], reward))
        # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
        obs = next_obs
        # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
        if terminated:
            # Explanation: Leave the nearest loop because its stopping condition has been reached.
            break
    # Explanation: Return trace to the caller; this ends the function and supplies its result.
    return trace



# Part 02 / evaluate
# Explanation: Define part02_evaluate with the shown parameters/defaults; its purpose is to compare a policy across seeded episodes using returns and failure counts.
def part02_evaluate(policy, *, episodes=200, shock_prob=0.0, horizon=12):
    # Explanation: Compute and bind the right-hand value to returns, naming an intermediate for the surrounding algorithm.
    returns = []
    # Explanation: Compute the number of failures in this transition or episode and store it in failures; later checks/updates use this named intermediate.
    failures = 0
    # Explanation: Visit each seed from range(episodes); the body applies the operation once per item.
    for seed in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        # Explanation: Compute and bind the right-hand value to rows, naming an intermediate for the surrounding algorithm.
        rows = rollout(env, policy, seed=seed)
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(sum((row['reward'] for row in rows)))
        # Explanation: Accumulate sum((row['event'] == 'failed_operation' for row in rows)) into failures; the previous value is retained and this amount is added.
        failures += sum((row['event'] == 'failed_operation' for row in rows))
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (float(np.mean(returns)), float(np.std(returns)), failures / episodes)



# Part 02 / student_td_target
# Explanation: Define part02_student_td_target with the shown parameters/defaults; its purpose is to build a one-step TD target and suppress bootstrap on termination.
def part02_student_td_target(reward: float, next_values, terminated: bool, gamma: float):
    # Explanation: Compute and bind the right-hand value to r, naming an intermediate for the surrounding algorithm.
    r = reward
    # Explanation: Check the episode/option stop condition terminated is True; when true, stop collecting transitions for this segment.
    if terminated is True:
        # Explanation: Compute and bind the right-hand value to y, naming an intermediate for the surrounding algorithm.
        y = reward
    # Explanation: Handle the alternative case not selected by the preceding if/elif test.
    else:
        # Explanation: Compute and bind the right-hand value to y, naming an intermediate for the surrounding algorithm.
        y = reward + gamma * max(next_values)
    # Explanation: Return y to the caller; this ends the function and supplies its result.
    return y



# Part 02 / student_epsilon_greedy
# Explanation: Define part02_student_epsilon_greedy with the shown parameters/defaults; its purpose is to explore randomly with probability epsilon; otherwise choose among maximum values.
def part02_student_epsilon_greedy(values, epsilon: float, rng: np.random.Generator):
    # Explanation: Convert the expression to a NumPy array and store it as values (critic estimates or table values, depending on context).
    values = np.asarray(values)
    # Explanation: Branch on rng.random() < epsilon; the indented statements handle this case, and other cases fall through or use a later branch.
    if rng.random() < epsilon:
        # Explanation: Return int(rng.integers(len(values))) to the caller; this ends the function and supplies its result.
        return int(rng.integers(len(values)))
    # Explanation: Compute and bind the right-hand value to best_actions, naming an intermediate for the surrounding algorithm.
    best_actions = np.flatnonzero(values == values.max())
    # Explanation: Return int(rng.choice(best_actions)) to the caller; this ends the function and supplies its result.
    return int(rng.choice(best_actions))



# Part 02 / train_q_learning
# Explanation: Define part02_train_q_learning with the shown parameters/defaults; its purpose is to learn tabular state-action values by repeatedly applying a TD update.
def part02_train_q_learning(*, episodes=3000, horizon=12, shock_prob=0.0, alpha=0.15, gamma=0.95, seed=23):
    # Explanation: Compute and bind the right-hand value to rng, naming an intermediate for the surrounding algorithm.
    rng = np.random.default_rng(seed)
    # Explanation: Initialize the primitive-action Q-table to zero with axes [days_left 0..horizon, health 0..3, action 0/1]. Each state/action estimate is learned through repeated visits.
    q = np.zeros((horizon + 1, 4, 2), dtype=float)
    # Explanation: Compute and bind the right-hand value to episode_returns, naming an intermediate for the surrounding algorithm.
    episode_returns = []
    # Explanation: Visit each episode from range(episodes); the body applies the operation once per item.
    for episode in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        # Explanation: Decay exploration from 0.4 toward a floor of 0.03 as training episodes progress; the floor keeps occasional exploration alive.
        epsilon = max(0.03, 0.4 * (1.0 - episode / episodes))
        # Explanation: Compute and bind the right-hand value to total_reward, naming an intermediate for the surrounding algorithm.
        total_reward = 0.0
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Unpack obs into named values health, days_left so each part can be used separately below.
            health, days_left = obs
            # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
            action = part02_student_epsilon_greedy(q[days_left, health], epsilon, rng)
            # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
            next_obs, reward, terminated, truncated, _info = env.step(action)
            # Explanation: Unpack next_obs into named values next_health, next_days_left so each part can be used separately below.
            next_health, next_days_left = next_obs
            # Explanation: Compute the return/TD target used for learning and store it in target; later checks/updates use this named intermediate.
            target = part02_student_td_target(reward, q[next_days_left, next_health], terminated, gamma)
            # Explanation: Accumulate alpha * (target - q[days_left, health, action]) into q[days_left, health, action]; the previous value is retained and this amount is added.
            q[days_left, health, action] += alpha * (target - q[days_left, health, action])
            # Explanation: Accumulate reward into total_reward; the previous value is retained and this amount is added.
            total_reward += reward
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = next_obs
            # Explanation: Check the episode/option stop condition terminated or truncated; when true, stop collecting transitions for this segment.
            if terminated or truncated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to episode_returns so it remains in the result/history.
        episode_returns.append(total_reward)
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (q, np.asarray(episode_returns))



# Part 03 / decode_observation
# Explanation: Define part03_decode_observation with the shown parameters/defaults; its purpose is to convert normalized observation coordinates back to interpretable state values.
def part03_decode_observation(obs, horizon):
    # Explanation: Unpack obs into named values health, days_left so each part can be used separately below.
    health, days_left = obs
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (health * 3, days_left * horizon)



# Part 03 / evaluate_policy_fn
# Explanation: Define part03_evaluate_policy_fn with the shown parameters/defaults; its purpose is to evaluate a callable policy through the Gym-style interface.
def part03_evaluate_policy_fn(choose_action, episodes=100, shock_prob=0.0):
    # Explanation: Unpack ([], []) into named values returns, failures so each part can be used separately below.
    returns, failures = ([], [])
    # Explanation: Visit each seed from range(1000, 1000 + episodes); the body applies the operation once per item.
    for seed in range(1000, 1000 + episodes):
        # Explanation: Compute and bind the right-hand value to test_env, naming an intermediate for the surrounding algorithm.
        test_env = GymMaintenanceEnv(horizon=12, shock_prob=shock_prob)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = test_env.reset(seed=seed)
        # Explanation: Unpack (0.0, 0) into named values total, failed so each part can be used separately below.
        total, failed = (0.0, 0)
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
            action = int(choose_action(obs))
            # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
            obs, reward, terminated, truncated, info = test_env.step(action)
            # Explanation: Accumulate reward into total; the previous value is retained and this amount is added.
            total += reward
            # Explanation: Accumulate int(info['failed']) into failed; the previous value is retained and this amount is added.
            failed += int(info['failed'])
            # Explanation: Check the episode/option stop condition terminated or truncated; when true, stop collecting transitions for this segment.
            if terminated or truncated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(total)
        # Explanation: Append the new record/value to failures so it remains in the result/history.
        failures.append(failed)
        # Explanation: Call test_env.close to perform this step of the surrounding algorithm.
        test_env.close()
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (np.asarray(returns), np.asarray(failures))



# Part 03 / ppo_action
# Explanation: Define part03_ppo_action with the shown parameters/defaults; its purpose is to adapt a model prediction to the scalar action expected by the toy environment.
def part03_ppo_action(obs, model):
    # Explanation: Unpack the external model's predicted action and its optional recurrent state; this wrapper only needs the action.
    action, _state = model.predict(obs, deterministic=True)
    # Explanation: Return int(action) to the caller; this ends the function and supplies its result.
    return int(action)



# Part 04 / engines_for_bundle
# Explanation: Define part04_engines_for_bundle with the shown parameters/defaults; its purpose is to convert a four-bit combined toy component mask to a two-bit engine mask.
def part04_engines_for_bundle(component_mask: int):
    # Explanation: Validate the input invariant not 0 <= component_mask <= 15; the guard rejects invalid values before the state-changing code runs.
    if not 0 <= component_mask <= 15:
        # Explanation: Raise ValueError('bundle must be a four-bit mask') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('bundle must be a four-bit mask')
    # Explanation: Compute the bitmask indicating selected engines and store it in engine_mask; later checks/updates use this named intermediate.
    engine_mask = 0
    # Explanation: 3 is 0011₂, so this AND keeps only the lowest two bits, assigned to engine 0 in the toy four-bit combined encoding. Nonzero means at least one engine-0 component was selected.
    if component_mask & 3:
        # Explanation: Set engine-mask bit 0 (value 1) for engine 0; OR-assignment preserves other flags already set.
        engine_mask |= 1
    # Explanation: 12 is 1100₂, so this AND keeps bits 2 and 3, assigned to engine 1 in the toy combined encoding. This checks the high pair independently.
    if component_mask & 12:
        # Explanation: Set engine-mask bit 1 (value 2, binary 10) for engine 1 while preserving engine 0's flag.
        engine_mask |= 2
    # Explanation: Return engine_mask to the caller; this ends the function and supplies its result.
    return engine_mask



# Part 04 / flat_decode
# Explanation: Define part04_flat_decode with the shown parameters/defaults; its purpose is to map a flat discrete ID back to its feasible action tuple.
def part04_flat_decode(action_id, actions=None):
    # Explanation: Branch on actions is None; the indented statements handle this case, and other cases fall through or use a later branch.
    if actions is None:
        # Explanation: Compute and bind the right-hand value to actions, naming an intermediate for the surrounding algorithm.
        actions = all_actions()
    # Explanation: Validate the input invariant not 0 <= action_id < len(actions); the guard rejects invalid values before the state-changing code runs.
    if not 0 <= action_id < len(actions):
        # Explanation: Raise ValueError('invalid flat action ID') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('invalid flat action ID')
    # Explanation: Return actions[action_id] to the caller; this ends the function and supplies its result.
    return actions[action_id]



# Part 04 / rule_policy
# Explanation: Define part04_rule_policy with the shown parameters/defaults; its purpose is to create an operate/replace action from health using the encoding helpers.
def part04_rule_policy(observation):
    # Explanation: Compute health values held by this toy state or simulator and store it in health; later checks/updates use this named intermediate.
    health = observation[:4]
    # Explanation: For every low-health component i, set bit i with 1 << i. Since indices are unique, summing these powers of two is equivalent to OR and records which components need work, not just their count.
    bundle = sum((1 << i for i, h in enumerate(health) if h <= 1))
    # Explanation: Branch on bundle == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if bundle == 0:
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (0, 0, 0)
    # Explanation: Branch on part04_engines_for_bundle(bundle) is None; the indented statements handle this case, and other cases fall through or use a later branch.
    if part04_engines_for_bundle(bundle) is None:
        # Explanation: Raise RuntimeError('Complete the exercise above first.') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise RuntimeError('Complete the exercise above first.')
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (1, part04_engines_for_bundle(bundle), bundle)



# Part 05 / available_options
# Explanation: Define part05_available_options with the shown parameters/defaults; its purpose is to return only high-level options legal in the current state.
def part05_available_options(observation):
    # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
    health, _days_left = observation
    # Explanation: Return [OPERATE_UNTIL_LOW, REPLACE_NOW] if health >= 2 else [REPLACE_NOW] to the caller; this ends the function and supplies its result.
    return [OPERATE_UNTIL_LOW, REPLACE_NOW] if health >= 2 else [REPLACE_NOW]



# Part 05 / execute_option
# Explanation: Define part05_execute_option with the shown parameters/defaults; its purpose is to run one option as primitive actions and collect its rewards.
def part05_execute_option(env, option_id):
    # Explanation: Compute the state visible to the policy and store it in observation; later checks/updates use this named intermediate.
    observation = (env.health, env.horizon - env.day)
    # Explanation: Branch on option_id not in part05_available_options(observation); the indented statements handle this case, and other cases fall through or use a later branch.
    if option_id not in part05_available_options(observation):
        # Explanation: Raise ValueError('option cannot start in this state') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('option cannot start in this state')
    # Explanation: Compute the primitive rewards collected during an option and store it in rewards; later checks/updates use this named intermediate.
    rewards = []
    # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
    while True:
        # Explanation: Compute the one-day action implementing the option and store it in primitive_action; later checks/updates use this named intermediate.
        primitive_action = 0 if option_id == OPERATE_UNTIL_LOW else 1
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        next_obs, reward, terminated, truncated, info = env.step(primitive_action)
        # Explanation: Append the new record/value to rewards so it remains in the result/history.
        rewards.append(reward)
        # Explanation: Compute and bind the right-hand value to option_finished, naming an intermediate for the surrounding algorithm.
        option_finished = option_id == REPLACE_NOW or next_obs[0] <= 1
        # Explanation: Check the episode/option stop condition option_finished or terminated or truncated; when true, stop collecting transitions for this segment.
        if option_finished or terminated or truncated:
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return (next_obs, rewards, terminated, info)



# Part 05 / student_smdp_target
# Explanation: Define part05_student_smdp_target with the shown parameters/defaults; its purpose is to discount a variable-duration option's rewards and bootstrap only if nonterminal.
def part05_student_smdp_target(rewards, next_values, terminated, gamma):
    # Explanation: Branch on len(rewards) == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if len(rewards) == 0:
        # Explanation: Raise ValueError('an executed option must have at least one reward') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('an executed option must have at least one reward')
    # Explanation: Compute the option's internal return: day 0 reward is unchanged, day 1 is multiplied by gamma, day 2 by gamma squared, and so on.
    discounted_rewards = sum((gamma ** j * r for j, r in enumerate(rewards)))
    # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
    if terminated:
        # Explanation: At termination, return only the rewards earned inside this option; there is no future state to bootstrap from.
        return discounted_rewards
    # Explanation: At a nonterminal option end, add the best next-option value discounted once per primitive day consumed.
    return discounted_rewards + gamma ** len(rewards) * max(next_values)



# Part 05 / train_option_manager
# Explanation: Define part05_train_option_manager with the shown parameters/defaults; its purpose is to train a table-based manager whose actions can last several days.
def part05_train_option_manager(episodes=2000, horizon=12, alpha=0.15, gamma=0.95, seed=29):
    # Explanation: Compute and bind the right-hand value to rng, naming an intermediate for the surrounding algorithm.
    rng = np.random.default_rng(seed)
    # Explanation: Initialize the option Q-table to zero with axes [days_left, health 0..3, option ID 0/1]. A value changes only when that state/option is selected and updated.
    q = np.zeros((horizon + 1, 4, 2))
    # Explanation: Compute and bind the right-hand value to returns, naming an intermediate for the surrounding algorithm.
    returns = []
    # Explanation: Visit each episode from range(episodes); the body applies the operation once per item.
    for episode in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=0.0)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        # Explanation: Decay exploration from 0.4 toward a floor of 0.03 as training episodes progress; the floor keeps occasional exploration alive.
        epsilon = max(0.03, 0.4 * (1 - episode / episodes))
        # Explanation: Compute the cumulative episode reward and store it in total; later checks/updates use this named intermediate.
        total = 0.0
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Unpack obs into named values health, days_left so each part can be used separately below.
            health, days_left = obs
            # Explanation: Compute the options legal in this state and store it in choices; later checks/updates use this named intermediate.
            choices = part05_available_options(obs)
            # Explanation: Branch on rng.random() < epsilon; the indented statements handle this case, and other cases fall through or use a later branch.
            if rng.random() < epsilon:
                # Explanation: Compute the selected high-level action and store it in option; later checks/updates use this named intermediate.
                option = int(rng.choice(choices))
            # Explanation: Handle the alternative case not selected by the preceding if/elif test.
            else:
                # Explanation: Gather Q-values for this state only at options currently legal; scores is a candidate vector, not the full Q-table.
                scores = q[days_left, health, choices]
                # Explanation: Compute indices tied for the highest value and store it in best; later checks/updates use this named intermediate.
                best = np.flatnonzero(scores == scores.max())
                # Explanation: Choose one tied-best index, map it back to the corresponding legal option ID, and convert the NumPy scalar to Python int.
                option = int(choices[int(rng.choice(best))])
            # Explanation: Unpack part05_execute_option(env, option) into named values next_obs, rewards, terminated, _info so each part can be used separately below.
            next_obs, rewards, terminated, _info = part05_execute_option(env, option)
            # Explanation: Unpack next_obs into named values next_health, next_days_left so each part can be used separately below.
            next_health, next_days_left = next_obs
            # Explanation: Compute options legal after the current option ends and store it in next_choices; later checks/updates use this named intermediate.
            next_choices = part05_available_options(next_obs)
            # Explanation: Compute the return/TD target used for learning and store it in target; later checks/updates use this named intermediate.
            target = part05_student_smdp_target(rewards, q[next_days_left, next_health, next_choices], terminated, gamma)
            # Explanation: Move this state/option estimate a fraction alpha toward target; target minus current Q is the TD error.
            q[days_left, health, option] += alpha * (target - q[days_left, health, option])
            # Explanation: Accumulate sum(rewards) into total; the previous value is retained and this amount is added.
            total += sum(rewards)
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = next_obs
            # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
            if terminated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(total)
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (q, np.asarray(returns))



# Solutions 01 / predict_one_step
# Explanation: Define solution01_predict_one_step with the shown parameters/defaults; its purpose is to perform the solution01 predict one step operation.
def solution01_predict_one_step(health: int, action: int):
    # Explanation: Branch on health not in (0, 1, 2, 3); the indented statements handle this case, and other cases fall through or use a later branch.
    if health not in (0, 1, 2, 3):
        # Explanation: Raise ValueError('health must be in 0..3') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('health must be in 0..3')
    # Explanation: Branch on action == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if action == 0:
        # Explanation: Compute and bind the right-hand value to next_health, naming an intermediate for the surrounding algorithm.
        next_health = max(0, health - 1)
        # Explanation: Branch on next_health == 0; the indented statements handle this case, and other cases fall through or use a later branch.
        if next_health == 0:
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return (next_health, -12.0, 'failed_operation')
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (next_health, 5.0, 'successful_operation')
    # Explanation: Branch on action == 1; the indented statements handle this case, and other cases fall through or use a later branch.
    if action == 1:
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (3, -4.0, 'grounded_and_replaced')
    # Explanation: Raise ValueError('action must be 0 or 1') so invalid input/failure is explicit instead of continuing with inconsistent state.
    raise ValueError('action must be 0 or 1')



# Solutions 01 / student_policy
# Explanation: Define solution01_student_policy with the shown parameters/defaults; its purpose is to perform the solution01 student policy operation.
def solution01_student_policy(observation: tuple[int, int]):
    # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
    health, _days_left = observation
    # Explanation: Return 1 if health <= 1 else 0 to the caller; this ends the function and supplies its result.
    return 1 if health <= 1 else 0



# Solutions 01 / wear_trace
# Explanation: Define solution01_wear_trace with the shown parameters/defaults; its purpose is to perform the solution01 wear trace operation.
def solution01_wear_trace(seed, shock_prob=0.35):
    # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
    env = MiniMaintenanceEnv(horizon=8, shock_prob=shock_prob)
    # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
    obs, _ = env.reset(seed=seed)
    # Explanation: Compute and bind the right-hand value to trace, naming an intermediate for the surrounding algorithm.
    trace = []
    # Explanation: Visit each _ from range(8); the body applies the operation once per item.
    for _ in range(8):
        # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
        action = threshold_policy(obs)
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        next_obs, reward, terminated, _truncated, info = env.step(action)
        # Explanation: Append the new record/value to trace so it remains in the result/history.
        trace.append((obs[0], action, info['wear'], next_obs[0], reward))
        # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
        obs = next_obs
        # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
        if terminated:
            # Explanation: Leave the nearest loop because its stopping condition has been reached.
            break
    # Explanation: Return trace to the caller; this ends the function and supplies its result.
    return trace



# Solutions 02 / evaluate
# Explanation: Define solution02_evaluate with the shown parameters/defaults; its purpose is to perform the solution02 evaluate operation.
def solution02_evaluate(policy, *, episodes=200, shock_prob=0.0, horizon=12):
    # Explanation: Compute and bind the right-hand value to returns, naming an intermediate for the surrounding algorithm.
    returns = []
    # Explanation: Compute the number of failures in this transition or episode and store it in failures; later checks/updates use this named intermediate.
    failures = 0
    # Explanation: Visit each seed from range(episodes); the body applies the operation once per item.
    for seed in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        # Explanation: Compute and bind the right-hand value to rows, naming an intermediate for the surrounding algorithm.
        rows = rollout(env, policy, seed=seed)
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(sum((row['reward'] for row in rows)))
        # Explanation: Accumulate sum((row['event'] == 'failed_operation' for row in rows)) into failures; the previous value is retained and this amount is added.
        failures += sum((row['event'] == 'failed_operation' for row in rows))
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (float(np.mean(returns)), float(np.std(returns)), failures / episodes)



# Solutions 02 / student_td_target
# Explanation: Define solution02_student_td_target with the shown parameters/defaults; its purpose is to perform the solution02 student td target operation.
def solution02_student_td_target(reward: float, next_values, terminated: bool, gamma: float):
    # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
    if terminated:
        # Explanation: Return float(reward) to the caller; this ends the function and supplies its result.
        return float(reward)
    # Explanation: Return float(reward + gamma * np.max(next_values)) to the caller; this ends the function and supplies its result.
    return float(reward + gamma * np.max(next_values))



# Solutions 02 / student_epsilon_greedy
# Explanation: Define solution02_student_epsilon_greedy with the shown parameters/defaults; its purpose is to perform the solution02 student epsilon greedy operation.
def solution02_student_epsilon_greedy(values, epsilon: float, rng: np.random.Generator):
    # Explanation: Branch on rng.random() < epsilon; the indented statements handle this case, and other cases fall through or use a later branch.
    if rng.random() < epsilon:
        # Explanation: Return int(rng.integers(len(values))) to the caller; this ends the function and supplies its result.
        return int(rng.integers(len(values)))
    # Explanation: Convert the expression to a NumPy array and store it as best (indices tied for the highest value).
    best = np.flatnonzero(np.asarray(values) == np.max(values))
    # Explanation: Return int(rng.choice(best)) to the caller; this ends the function and supplies its result.
    return int(rng.choice(best))



# Solutions 02 / train_q_learning
# Explanation: Define solution02_train_q_learning with the shown parameters/defaults; its purpose is to perform the solution02 train q learning operation.
def solution02_train_q_learning(*, episodes=3000, horizon=12, shock_prob=0.0, alpha=0.15, gamma=0.95, seed=23):
    # Explanation: Compute and bind the right-hand value to rng, naming an intermediate for the surrounding algorithm.
    rng = np.random.default_rng(seed)
    # Explanation: Initialize the primitive-action Q-table to zero with axes [days_left 0..horizon, health 0..3, action 0/1]. Each state/action estimate is learned through repeated visits.
    q = np.zeros((horizon + 1, 4, 2), dtype=float)
    # Explanation: Compute and bind the right-hand value to episode_returns, naming an intermediate for the surrounding algorithm.
    episode_returns = []
    # Explanation: Visit each episode from range(episodes); the body applies the operation once per item.
    for episode in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=shock_prob)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        # Explanation: Decay exploration from 0.4 toward a floor of 0.03 as training episodes progress; the floor keeps occasional exploration alive.
        epsilon = max(0.03, 0.4 * (1.0 - episode / episodes))
        # Explanation: Compute and bind the right-hand value to total_reward, naming an intermediate for the surrounding algorithm.
        total_reward = 0.0
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Unpack obs into named values health, days_left so each part can be used separately below.
            health, days_left = obs
            # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
            action = solution02_student_epsilon_greedy(q[days_left, health], epsilon, rng)
            # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
            next_obs, reward, terminated, truncated, _info = env.step(action)
            # Explanation: Unpack next_obs into named values next_health, next_days_left so each part can be used separately below.
            next_health, next_days_left = next_obs
            # Explanation: Use the completed one-step TD-target helper; it returns reward alone at termination and adds discounted best-next-action Q otherwise.
            target = solution02_student_td_target(reward, q[next_days_left, next_health], terminated, gamma)
            # Explanation: Accumulate alpha * (target - q[days_left, health, action]) into q[days_left, health, action]; the previous value is retained and this amount is added.
            q[days_left, health, action] += alpha * (target - q[days_left, health, action])
            # Explanation: Accumulate reward into total_reward; the previous value is retained and this amount is added.
            total_reward += reward
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = next_obs
            # Explanation: Check the episode/option stop condition terminated or truncated; when true, stop collecting transitions for this segment.
            if terminated or truncated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to episode_returns so it remains in the result/history.
        episode_returns.append(total_reward)
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (q, np.asarray(episode_returns))



# Solutions 03 / decode_observation
# Explanation: Define solution03_decode_observation with the shown parameters/defaults; its purpose is to perform the solution03 decode observation operation.
def solution03_decode_observation(obs, horizon):
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (int(round(float(obs[0]) * 3)), int(round(float(obs[1]) * horizon)))



# Solutions 03 / evaluate_policy_fn
# Explanation: Define solution03_evaluate_policy_fn with the shown parameters/defaults; its purpose is to perform the solution03 evaluate policy fn operation.
def solution03_evaluate_policy_fn(choose_action, episodes=100, shock_prob=0.0):
    # Explanation: Unpack ([], []) into named values returns, failures so each part can be used separately below.
    returns, failures = ([], [])
    # Explanation: Visit each seed from range(1000, 1000 + episodes); the body applies the operation once per item.
    for seed in range(1000, 1000 + episodes):
        # Explanation: Compute and bind the right-hand value to test_env, naming an intermediate for the surrounding algorithm.
        test_env = GymMaintenanceEnv(horizon=12, shock_prob=shock_prob)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = test_env.reset(seed=seed)
        # Explanation: Unpack (0.0, 0) into named values total, failed so each part can be used separately below.
        total, failed = (0.0, 0)
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Compute the proposed or selected action and store it in action; later checks/updates use this named intermediate.
            action = int(choose_action(obs))
            # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
            obs, reward, terminated, truncated, info = test_env.step(action)
            # Explanation: Accumulate reward into total; the previous value is retained and this amount is added.
            total += reward
            # Explanation: Accumulate int(info['failed']) into failed; the previous value is retained and this amount is added.
            failed += int(info['failed'])
            # Explanation: Check the episode/option stop condition terminated or truncated; when true, stop collecting transitions for this segment.
            if terminated or truncated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(total)
        # Explanation: Append the new record/value to failures so it remains in the result/history.
        failures.append(failed)
        # Explanation: Call test_env.close to perform this step of the surrounding algorithm.
        test_env.close()
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (np.asarray(returns), np.asarray(failures))



# Solutions 03 / ppo_action
# Explanation: Define solution03_ppo_action with the shown parameters/defaults; its purpose is to perform the solution03 ppo action operation.
def solution03_ppo_action(obs, model):
    # Explanation: Unpack the external model's predicted action and its optional recurrent state; this wrapper only needs the action.
    action, _state = model.predict(obs, deterministic=True)
    # Explanation: Return int(action) to the caller; this ends the function and supplies its result.
    return int(action)



# Solutions 04 / engines_for_bundle
# Explanation: Define solution04_engines_for_bundle with the shown parameters/defaults; its purpose is to provide the reference four-bit component-to-engine-mask conversion.
def solution04_engines_for_bundle(component_mask: int):
    # Explanation: Validate the input invariant not 0 <= component_mask <= 15; the guard rejects invalid values before the state-changing code runs.
    if not 0 <= component_mask <= 15:
        # Explanation: Raise ValueError('bundle must be a four-bit mask') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('bundle must be a four-bit mask')
    # Explanation: Turn the low-pair/high-pair tests into 0/1 flags, then shift engine 1's flag to bit 1. The result 0/1/2/3 means neither, engine 0, engine 1, or both.
    return int(bool(component_mask & 3)) | int(bool(component_mask & 12)) << 1



# Solutions 04 / flat_decode
# Explanation: Define solution04_flat_decode with the shown parameters/defaults; its purpose is to provide the reference lookup from a flat ID to a feasible action tuple.
def solution04_flat_decode(action_id, actions=None):
    # Explanation: Branch on actions is None; the indented statements handle this case, and other cases fall through or use a later branch.
    if actions is None:
        # Explanation: Compute and bind the right-hand value to actions, naming an intermediate for the surrounding algorithm.
        actions = all_actions()
    # Explanation: Validate the input invariant not 0 <= action_id < len(actions); the guard rejects invalid values before the state-changing code runs.
    if not 0 <= action_id < len(actions):
        # Explanation: Raise ValueError('invalid flat action ID') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('invalid flat action ID')
    # Explanation: Return actions[action_id] to the caller; this ends the function and supplies its result.
    return actions[action_id]



# Solutions 04 / rule_policy
# Explanation: Define solution04_rule_policy with the shown parameters/defaults; its purpose is to provide a reference rule that builds a consistent action from component health.
def solution04_rule_policy(observation):
    # Explanation: Compute health values held by this toy state or simulator and store it in health; later checks/updates use this named intermediate.
    health = observation[:4]
    # Explanation: For every low-health component i, set bit i with 1 << i. Since indices are unique, summing these powers of two is equivalent to OR and records which components need work, not just their count.
    bundle = sum((1 << i for i, h in enumerate(health) if h <= 1))
    # Explanation: Branch on bundle == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if bundle == 0:
        # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
        return (0, 0, 0)
    # Explanation: Branch on solution04_engines_for_bundle(bundle) is None; the indented statements handle this case, and other cases fall through or use a later branch.
    if solution04_engines_for_bundle(bundle) is None:
        # Explanation: Raise RuntimeError('Complete the exercise above first.') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise RuntimeError('Complete the exercise above first.')
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (1, solution04_engines_for_bundle(bundle), bundle)



# Solutions 05 / available_options
# Explanation: Define solution05_available_options with the shown parameters/defaults; its purpose is to perform the solution05 available options operation.
def solution05_available_options(observation):
    # Explanation: Read health from the observation and intentionally ignore the remaining-day field when this policy only needs health.
    health, _days_left = observation
    # Explanation: Return [OPERATE_UNTIL_LOW, REPLACE_NOW] if health >= 2 else [REPLACE_NOW] to the caller; this ends the function and supplies its result.
    return [OPERATE_UNTIL_LOW, REPLACE_NOW] if health >= 2 else [REPLACE_NOW]



# Solutions 05 / execute_option
# Explanation: Define solution05_execute_option with the shown parameters/defaults; its purpose is to perform the solution05 execute option operation.
def solution05_execute_option(env, option_id):
    # Explanation: Compute the state visible to the policy and store it in observation; later checks/updates use this named intermediate.
    observation = (env.health, env.horizon - env.day)
    # Explanation: Branch on option_id not in solution05_available_options(observation); the indented statements handle this case, and other cases fall through or use a later branch.
    if option_id not in solution05_available_options(observation):
        # Explanation: Raise ValueError('option cannot start in this state') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('option cannot start in this state')
    # Explanation: Compute the primitive rewards collected during an option and store it in rewards; later checks/updates use this named intermediate.
    rewards = []
    # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
    while True:
        # Explanation: Compute the one-day action implementing the option and store it in primitive_action; later checks/updates use this named intermediate.
        primitive_action = 0 if option_id == OPERATE_UNTIL_LOW else 1
        # Explanation: Unpack the environment step result in Gym order: next observation, reward, terminated, truncated, and info. The underscore field is intentionally discarded when present.
        next_obs, reward, terminated, truncated, info = env.step(primitive_action)
        # Explanation: Append the new record/value to rewards so it remains in the result/history.
        rewards.append(reward)
        # Explanation: Compute and bind the right-hand value to option_finished, naming an intermediate for the surrounding algorithm.
        option_finished = option_id == REPLACE_NOW or next_obs[0] <= 1
        # Explanation: Check the episode/option stop condition option_finished or terminated or truncated; when true, stop collecting transitions for this segment.
        if option_finished or terminated or truncated:
            # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
            return (next_obs, rewards, terminated, info)



# Solutions 05 / student_smdp_target
# Explanation: Define solution05_student_smdp_target with the shown parameters/defaults; its purpose is to provide the reference variable-duration target and terminal handling.
def solution05_student_smdp_target(rewards, next_values, terminated, gamma):
    # Explanation: Branch on len(rewards) == 0; the indented statements handle this case, and other cases fall through or use a later branch.
    if len(rewards) == 0:
        # Explanation: Raise ValueError('an executed option must have at least one reward') so invalid input/failure is explicit instead of continuing with inconsistent state.
        raise ValueError('an executed option must have at least one reward')
    # Explanation: Compute the option's internal return: day 0 reward is unchanged, day 1 is multiplied by gamma, day 2 by gamma squared, and so on.
    discounted_rewards = sum((gamma ** j * r for j, r in enumerate(rewards)))
    # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
    if terminated:
        # Explanation: At termination, return only the rewards earned inside this option; there is no future state to bootstrap from.
        return discounted_rewards
    # Explanation: At a nonterminal option end, add the best next-option value discounted once per primitive day consumed.
    return discounted_rewards + gamma ** len(rewards) * max(next_values)



# Solutions 05 / train_option_manager
# Explanation: Define solution05_train_option_manager with the shown parameters/defaults; its purpose is to provide the reference trainer for a tabular option manager.
def solution05_train_option_manager(episodes=2000, horizon=12, alpha=0.15, gamma=0.95, seed=29):
    # Explanation: Compute and bind the right-hand value to rng, naming an intermediate for the surrounding algorithm.
    rng = np.random.default_rng(seed)
    # Explanation: Initialize the option Q-table to zero with axes [days_left, health 0..3, option ID 0/1]. A value changes only when that state/option is selected and updated.
    q = np.zeros((horizon + 1, 4, 2))
    # Explanation: Compute and bind the right-hand value to returns, naming an intermediate for the surrounding algorithm.
    returns = []
    # Explanation: Visit each episode from range(episodes); the body applies the operation once per item.
    for episode in range(episodes):
        # Explanation: Compute and bind the right-hand value to env, naming an intermediate for the surrounding algorithm.
        env = MiniMaintenanceEnv(horizon=horizon, shock_prob=0.0)
        # Explanation: Unpack reset's pair into the starting observation and auxiliary info; the underscore means this caller does not need the info dictionary.
        obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        # Explanation: Decay exploration from 0.4 toward a floor of 0.03 as training episodes progress; the floor keeps occasional exploration alive.
        epsilon = max(0.03, 0.4 * (1 - episode / episodes))
        # Explanation: Compute the cumulative episode reward and store it in total; later checks/updates use this named intermediate.
        total = 0.0
        # Explanation: Repeat the body until an explicit break/return condition ends this episode, option, or training loop.
        while True:
            # Explanation: Unpack obs into named values health, days_left so each part can be used separately below.
            health, days_left = obs
            # Explanation: Compute the options legal in this state and store it in choices; later checks/updates use this named intermediate.
            choices = solution05_available_options(obs)
            # Explanation: Branch on rng.random() < epsilon; the indented statements handle this case, and other cases fall through or use a later branch.
            if rng.random() < epsilon:
                # Explanation: Compute the selected high-level action and store it in option; later checks/updates use this named intermediate.
                option = int(rng.choice(choices))
            # Explanation: Handle the alternative case not selected by the preceding if/elif test.
            else:
                # Explanation: Gather Q-values for this state only at options currently legal; scores is a candidate vector, not the full Q-table.
                scores = q[days_left, health, choices]
                # Explanation: Compute indices tied for the highest value and store it in best; later checks/updates use this named intermediate.
                best = np.flatnonzero(scores == scores.max())
                # Explanation: Choose one tied-best index, map it back to the corresponding legal option ID, and convert the NumPy scalar to Python int.
                option = int(choices[int(rng.choice(best))])
            # Explanation: Unpack solution05_execute_option(env, option) into named values next_obs, rewards, terminated, _info so each part can be used separately below.
            next_obs, rewards, terminated, _info = solution05_execute_option(env, option)
            # Explanation: Unpack next_obs into named values next_health, next_days_left so each part can be used separately below.
            next_health, next_days_left = next_obs
            # Explanation: Compute options legal after the current option ends and store it in next_choices; later checks/updates use this named intermediate.
            next_choices = solution05_available_options(next_obs)
            # Explanation: Compute the return/TD target used for learning and store it in target; later checks/updates use this named intermediate.
            target = solution05_student_smdp_target(rewards, q[next_days_left, next_health, next_choices], terminated, gamma)
            # Explanation: Move this state/option estimate a fraction alpha toward target; target minus current Q is the TD error.
            q[days_left, health, option] += alpha * (target - q[days_left, health, option])
            # Explanation: Accumulate sum(rewards) into total; the previous value is retained and this amount is added.
            total += sum(rewards)
            # Explanation: Compute the current encoded state and store it in obs; later checks/updates use this named intermediate.
            obs = next_obs
            # Explanation: Check the episode/option stop condition terminated; when true, stop collecting transitions for this segment.
            if terminated:
                # Explanation: Leave the nearest loop because its stopping condition has been reached.
                break
        # Explanation: Append the new record/value to returns so it remains in the result/history.
        returns.append(total)
    # Explanation: Return the grouped values as the function's output contract; tuple position or dictionary keys give each value its meaning.
    return (q, np.asarray(returns))



