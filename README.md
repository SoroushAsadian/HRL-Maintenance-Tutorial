# Comprehensive RL / HRL maintenance tutorial

Open **Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb** using the existing
**Python 3.13** kernel. All seven learner notebooks are merged, followed by the
five original solution notebooks in an optional appendix. Your completed learner
exercise answers are preserved.

The annotated Python implementation is split across **maintenance_tutorial_annotated_core.py**, **maintenance_tutorial_annotated_parts_and_solutions.py**, and **maintenance_tutorial_annotated_behavioral_tests.py**. The notebook setup cell reads these files in order and assembles their definitions as an in-memory `maintenance_tutorial` module, which notebook cells import. Keep all three files beside the notebook. There is no separate helper `.py` module to import from disk.

Learner helper names begin with `part01_`, `part02_`, etc.; solution versions
begin with `solution01_`, etc. These prefixes prevent collisions between functions
with the same original name. Exercise cells show their exact import. Edit a
function in the module, restart the kernel and rerun its notebook part to refresh
the imported definition.

## Contents

1. Build a small maintenance environment.
2. Tabular Q-learning and baselines.
3. Gymnasium and Stable-Baselines3 PPO.
4. Conditional aircraft/engine/component actions.
5. Temporal options and SMDP learning.
6. Fleet simulator: seven aircraft by default, two engines each, all six components.
7. Flat, conditional, shared-actor and manager/worker learning comparisons.
8. Optional solution appendix for parts 01–05.

The companion **Comprehensive_RL_HRL_Maintenance_Tutorial.md** contains the same
Markdown explanations and code for reading without Jupyter. The merge manifest
records the twelve source notebooks.

## Modeling assumptions

All deterioration, efficiency, temperature, fault probabilities, repair times and
costs are teaching assumptions. They are not validated C-MAPSS equations or real
aviation event statistics. Weather/bird/hail events affect health; there is no
separate weather-delay schedule. Estimates refresh after flights. Active maintenance
plans remain committed while other aircraft operate. Six components remain in the
action space; HPC is not the only maintainable item.

The current observation/action layout is version 3: 70 values per aircraft,
including estimated health, uncertainty, efficiency, temperature proxies, reading
age and committed resource selections. Public observations have
`10 + 2 * crews + 2 * workshops` values: the original ten public values, crew
availability flags, encoded crew duration factors, workshop availability flags,
and encoded workshop duration factors. Each factor is encoded as `1 / (1 + factor)`.
Version-1 and version-2 checkpoints require retraining. Historical runs remain
preserved in **runs/**, including **runs/sensor_v2_smoke/**.

## Aircraft identities and heterogeneous resources (25 October 2026)

Aircraft share the same plant/component model but have separate indexed health,
exposure and maintenance state. Fixed synthetic plates `AC-001`, `AC-002`, etc.
identify those indices across resets; a reset starts a new simulated episode,
not a continuation of the previous flight history. No full flight-history log is
implemented yet.

Crew members have fixed IDs `CR-001`, `CR-002`, etc. Each crew-mask bit identifies
one modeled member: `0b101` selects members 0 and 2. Workshops have fixed IDs
`WS-001`, `WS-002`, etc.; a job selects one workshop by its zero-based index.
Each selected member and workshop is reserved until the entire job finishes.
Two jobs cannot use the same member even if other members are available.

The eight fleet-action fields are now:

```text
[ground, engine_mask, bundle0, bundle1, crew_mask, workshop_index, order, priority]
```

Crew mask zero means no member selected and is invalid for new maintenance work.
This replaces the previous `crews - 1` action-field interpretation. Hand-written
fleet actions must be updated accordingly; this change does not alter the small
introductory environments. The current notebook and Markdown export have been revised for this layout,
including the manual fleet action examples. Historical copies still use the old
crew-count convention and should not be mixed with this implementation.

Resource duration factors below one speed up work; factors above one slow it down.
For selected crew set S and workshop w, the teaching model uses:

```text
combined_work_rate = sum(1 / crew_time_factors[i] for i in S)
component_days = max(1, ceil(LABOR_DAYS[component]
                             * workshop_time_factors[w] / combined_work_rate))
job_days = setup_days + sum(component_days for each ordered component task)
```

Setup remains fixed. Crew work rates add linearly, tasks execute sequentially,
and rounding to whole days can hide small factor differences. The accepted job
stores its component completion thresholds, so actual restoration and resource
release follow the same schedule used to calculate downtime. There is no separate
expertise reward: different durations affect the existing downtime costs and flight
revenue. No wage differences or skill-dependent repair quality are modeled.

Empty factor profiles use repeating illustrative defaults: crew factors
`(0.8, 1.0, 1.25)` and workshop factors `(0.8, 1.0, 1.2)`, truncated or repeated
to match the configured resource counts. These are assumptions for experiments,
not estimates from the papers. Supply explicit factors to define the experiment;
use all ones to recover standard homogeneous resource rates.

For example, from Python with this folder on the module search path:

```python
from maintenance_tutorial_annotated_core import Config, FleetMaintenanceEnv, train

cfg = Config(
    aircraft=10,
    crews=3,
    workshops=2,
    crew_time_factors=(0.8, 1.0, 1.25),
    workshop_time_factors=(0.8, 1.2),
)
env = FleetMaintenanceEnv(cfg)
observation, info = env.reset(seed=0)
action = env.idle_action()
action[0] = [1, 1, 4, 0, 0b101, 0, 2, 0]
observation, reward, terminated, truncated, info = env.step(action)
# Aircraft AC-001 requests engine 0's HPC, crew members CR-001 and CR-003,
# and workshop WS-001. IDs and resource factors are also in info['resources'].
model, logs, simulated_days = train(cfg, method='conditional', steps=128)
```

`Config(aircraft=N)` and the runner's existing `--aircraft N` select fleet size;
the runner also accepts `--crews`, `--workshops`, `--crew-time-factors` and
`--workshop-time-factors`. Every profile must have one finite positive factor
per resource. The configuration remains fixed during an episode and training run.

The original baseline controller selects one joint fleet action and learns from a fleet-level
reward. Daily modes share actor parameters across aircraft rows and use a full-fleet
critic. Option modes introduce a manager (and, for learned options, a worker);
they do not introduce independent aircraft agents. This remains a centralized
controller; the additional MAPPO methods below introduce explicit agent ownership and per-agent PPO updates. Model input/output sizes depend on fleet
and resource counts, including in shared mode, so changing these counts requires
constructing and training a new model. Configurability does not establish transfer
to an unseen fleet size or measured scalability. The categorical crew-mask head has
`2 ** crews` categories; large crew pools will need another action representation.

Vacation/holiday calendars, variable daily staffing, and the
paper-based bird-strike/hail upgrades are deferred. Existing hazard processes are
unchanged by this resource update.

## Multi-agent proximal policy optimization (MAPPO)

Two additional daily cooperative methods are implemented in the same core file:

- `mappo_aircraft`: one logical agent per aircraft plate. Homogeneous agents share
  actor weights, observe their own local condition/plan row and public resource
  information, and each select two six-bit replacement bundles.
- `mappo_component`: six agents, one per component type across the whole fleet.
  They have six separate actor networks. Each chooses a two-engine request mask
  for each aircraft and can request only its own component. Each specialist sees
  its component's health/performance estimates across the fleet, operational
  context, public resources, and a broadcast aircraft safety-criticality report.

Both assignments own the same twelve component-selection bits per aircraft.
Grounding and engine masks follow from nonempty requests. A common fixed
dispatcher merges requests and selects individual crews, workshop, task order,
and priority from observed state and known resource factors. It respects spares,
exclusive reservations and committed jobs. Full infeasible requests are deferred;
`coordination_deferrals` records those separately from simulator rejections.
The dispatcher does not consult hidden health, event tapes, or a learned critic.

This restricted action authority is matched between the two MAPPO assignments.
It differs from the legacy full-action PPO baselines, which also learn resource
and priority choices. Their scores are not controlled tests of agent count alone.
Proposal likelihoods are retained even when the dispatcher defers the request;
the trainer scores the original submitted proposal, never an edited final action.

`MultiAgentFleetEnv` provides dictionary observations, proposals and replicated
team rewards keyed by agent ID. It is a small project-specific adapter, not a
PettingZoo/RLlib implementation. `MultiAgentActorCritic` exposes restricted actors
through `policy(agent_observations)` and a centralized observed-belief critic with
one value head per agent. `train(..., method='mappo_aircraft')` or
`train(..., method='mappo_component')` routes to `train_mappo`.

The update keeps one log probability, entropy, advantage, active mask and value
estimate per agent. It clips each agent's likelihood ratio before averaging,
instead of clipping a single fleet-joint ratio. Actor weights and critic weights
have separate optimizers. The feed-forward prototype uses generalized advantage
estimation, clipped values, four full-rollout epochs and gradient clipping.
It omits recurrent policies, value normalization and counterfactual credit
assignment; it is not an exact benchmark reproduction or a convergence claim.

Centralized training with decentralized execution (CTDE) is an information pattern,
not a communication algorithm. Here actor proposal generation is decentralized,
but allocation uses a central rule-based dispatcher and a public request/safety
board. This is not fully decentralized scheduling. Learned peer-to-peer message
passing remains unimplemented. Deep centralized multi-agent actor–critic (DCMAC)
and deep decentralized multi-agent actor–critic (DDMAC) are distinct maintenance
methods; the MAPPO method names do not implement them. Separate off-policy
adaptations are now available as described below.

Primary method sources:

- Yu et al. (2022), https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html
- Author implementation, https://github.com/marlbenchmark/on-policy
- DCMAC, https://doi.org/10.1016/j.ress.2019.04.036
- DDMAC, https://doi.org/10.1016/j.ress.2021.107551

For a small Python run:

```python
from maintenance_tutorial_annotated_core import Config, train, evaluate
cfg = Config(aircraft=7, horizon=90)
model, logs, days = train(cfg, 'mappo_aircraft', steps=256, seed=0, rollout_size=32)
records = evaluate(cfg, 'mappo_aircraft', model, seeds=[10000, 10001])
```

The runner accepts both method names. The notebook's new Part 08 inspects both
assignments, demonstrates merging and per-agent clipping, trains two seeds per
method, evaluates on paired held-out seeds, and saves timestamped checkpoints
and settings under `runs/mappo_agents_*`. These short runs demonstrate execution,
not reliable policy rankings. It also exercises configuration at 1, 7 and 10
aircraft and runs the consolidated behavioral suite. The crew-mask examples and
version-3 observation descriptions in earlier sections have been updated, and
the Markdown notebook export has been refreshed.

The physical simulator, hazard process and post-flight sensor/filter rules have
not changed in this multi-agent update. The planned explicit synthetic flight
legs and common-reference post-landing assessment remain pending. Aircraft actor
weights are shared, but the full models/checkpoints still depend on configured
fleet/resource counts; six specialist agents do not by themselves establish
constant computational cost as fleet size grows.

The fleet options experiment changes decision frequency and planning freedom.
Its return does not isolate an HRL advantage. The shared-actor critic remains
fixed-size. Task order changes restoration timing but not total return in this
first model. Resource-conflict handling and longer, multi-seed experiments remain
necessary before research conclusions. Part 10 now adds a single-aircraft MAXQ-0 baseline, constrained option-critic,
replacement-target delegation, concurrent aircraft options and hierarchical VDN.
Graph architectures remain future work; paper-exact reproductions are not claimed.

## Verification and training

Open the notebook with the existing Python 3.13 kernel and run its setup cell first. To run the behavioral suite, execute `helpers.run_checks()` in a notebook cell. Training and evaluation are performed by the notebook's own cells; the old standalone command-line script has been removed. The test suite is in `maintenance_tutorial_annotated_behavioral_tests.py`.

Dependencies: NumPy, PyTorch, Gymnasium, Stable-Baselines3, Matplotlib, nbformat,
and a Jupyter interface. The existing Python 3.13 installation already contains
the tutorial dependencies; Python 3.14 is a different environment.

The original Desktop Code directory is removed only after verifying this new
folder. Its recovery location is recorded in the handoff message.


## DCMAC and DDMAC: global-belief off-policy methods

Four additional methods are available: dcmac_aircraft, ddmac_aircraft,
dcmac_component, and ddmac_component. DCMAC uses one global-input actor with
factorized output blocks; DDMAC uses a separate global-input actor per control
unit (seven networks for seven aircraft, or six for component-type agents).
Both have one centralized scalar critic, uniform bounded experience replay,
joint truncated importance weighting, and one-step TD targets. They share the
MAPPO request action space and fixed resource broker. Global actor inputs mean
that separate actor parameters do not imply local-information execution.

These are fleet adaptations, without the papers' inspection, budget-state and
primal-dual risk-constraint algorithms. See [DCMAC_DDMAC_changes.md](DCMAC_DDMAC_changes.md)
for equations, sources, limitations and usage. Notebook Part 09 runs both
architectures under both ownership choices with paired seeds, saves timestamped
checkpoints/settings/evaluations, and demonstrates seven aircraft actors.
The suite contains 43 behavioral checks. The physical simulator and observation
version 3 remain unchanged; flight-condition/duration and hazard upgrades are pending.

## Additional aircraft HRL methods

Part 10 adds `concurrent_options`, `planning_hierarchy`, `aggregate_targets`,
`maxq`, `option_critic`, `hierarchical_vdn`, and the matched `hrl_flat` comparator.
The shared `AircraftOptionEnv` advances one physical day at a time, preserves
per-aircraft committed plans, and lets free aircraft receive new decisions while
other repairs continue. New methods use a restricted 22-action replacement menu
and the same fixed resource broker; they do not learn crew/shop/order allocation.

The explicit planning model has separate learned grounding, engine and component
stages. Targets are replacement-count budgets. MAXQ-0 is a single-aircraft tabular
V+C decomposition. Option-critic learns internal policies and termination only
at safe free-aircraft boundaries. Hierarchical VDN sums independent aircraft
values at two levels with a disclosed inter-level shaped reward; it is not an
exact HAVEN reproduction. See [HRL_methods_changes.md](HRL_methods_changes.md)
for equations, interfaces, assumptions, comparison limits and primary references.

`ALL_LEARNED_METHODS` now contains 18 configurations: 17 neural variants plus
MAXQ. The threshold rule remains a fixed baseline. Common training, evaluation
and checkpoint helpers recognize all methods; the all-method runner records
MAXQ as skipped when the configured fleet is not one aircraft. Existing physical
observations stay at version 3. Flight-duration/condition and hazard upgrades
remain separate pending changes. Short notebook runs are software demonstrations,
not convergence or performance rankings.
