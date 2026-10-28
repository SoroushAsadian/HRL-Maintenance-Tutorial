# Aircraft maintenance with reinforcement learning

A teaching and research prototype for opportunistic aircraft maintenance using
single-agent reinforcement learning (RL), multi-agent reinforcement learning
(MARL), and hierarchical reinforcement learning (HRL). The project combines a
synthetic fleet simulator, uncertain condition estimates, constrained maintenance
actions, learning algorithms, and executable experiment notebooks.

**Main notebook for Emanuele and other readers:**
[Maintenance_Experiments_Presentation.ipynb](Maintenance_Experiments_Presentation.ipynb).
It runs training and execution, measures both durations per method, saves results and plots, and ends with an
automatically generated comparison and interpretation of the observed outcomes.

[Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb](Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb)
is Soroush's **personal learning notebook**, with detailed explanations, exercises,
and solutions. It is supplementary study material rather than the audience's main
entry point.

The current implementation contains **18 learned configurations** and a fixed
threshold baseline. Aircraft, crews, and workshops have individual identities;
fleet size and maintenance-resource profiles are configurable. All physical
parameters are synthetic teaching assumptions, rather than calibrated aviation
data. Short training runs demonstrate implementation behavior, not convergence
or a reliable performance ranking.

## Code reports

The reports and their historical versions are included in this repository.
**Start with the revised second report dated 28 October 2026.**

| Report | Scope | How to use it |
| --- | --- | --- |
| [Revised second code report, 28 October 2026](docs/reports/code_report_2026-10-28.pdf) · [TeX source](docs/reports/code_report_2026-10-28.tex) | 196 pages covering all 18 learned configurations, current source listings, 2,096 numbered explanations, cleanup changes, and the scope of compatibility checks. | Main implementation reference for the current code. |
| [Second code report, 26 October 2026](docs/reports/code_report_2026-10-26.pdf) | 205 pages covering the environment, agent and network differences, update equations, and the core before cleanup. | Historical implementation reference before the 28 October cleanup. |
| [Original code report — 22 October 2026](docs/reports/code_report_2026-10-22.pdf) | 74 pages explaining the earlier core implementation, small teaching environments, conditional actions, original fleet controller, temporal options, and the relationship between the library and custom trainers. | Historical foundation and explanation of the earlier design. |

The original report describes the code as it existed on 22 October. Its statements
that multi-agent methods were absent, and its older observation/resource layout,
are superseded by the updated report and current source. Source-line references
belong to each report's own version. The reports explain the core file line by
line; the separate exercises/solutions, tests, and notebook cells are not additional
line-by-line appendices.

The original report establishes the simulator/policy/trainer distinction and
the single-controller foundation. The updated report extends that explanation to
aircraft identities, heterogeneous maintenance resources, explicit agent ownership,
different actor information boundaries, independent versus shared parameters,
replay updates, and concurrent aircraft hierarchies.

## Repository files

| File | Role |
| --- | --- |
| [Maintenance_Experiments_Presentation.ipynb](Maintenance_Experiments_Presentation.ipynb) | **Main audience notebook:** experiment settings, all-method training, checkpointed execution, saved metrics/plots, and results interpretation. |
| [Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb](Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb) | **Personal learning notebook:** ten-part tutorial, exercises, and optional reference solutions. |
| [maintenance_tutorial_annotated_core.py](maintenance_tutorial_annotated_core.py) | Teaching environments, fleet simulator, policies, method registry, trainers, evaluation, checkpoints, and runner function. |
| [maintenance_tutorial_annotated_parts_and_solutions.py](maintenance_tutorial_annotated_parts_and_solutions.py) | Namespaced learner helpers and reference solutions. |
| [maintenance_tutorial_annotated_behavioral_tests.py](maintenance_tutorial_annotated_behavioral_tests.py) | Consolidated behavioral checks and their runner. |
| [docs/reports](docs/reports) | Original and revised PDF code reports, with TeX source for the current report. |

The main presentation notebook imports the core Python file directly. Keep both
in the repository root. The personal learning notebook needs all three Python
files together; its setup cell concatenates
the Python sources and registers an in-memory `maintenance_tutorial` module.
There is no separate `maintenance_tutorial.py` file to install. Learner functions
use `part01_`, `part02_`, etc.; reference solutions use `solution01_`, etc.
Restart the kernel and rerun setup after changing source definitions.

## Run the main presentation notebook

The recorded working environment uses **Python 3.13**. From the repository folder,
install the notebook dependencies into the Python environment you will use:

```bash
python -m pip install numpy torch gymnasium stable-baselines3 matplotlib pandas nbformat jupyterlab
python -m jupyterlab
```

Open **`Maintenance_Experiments_Presentation.ipynb`**, select that environment's
kernel, and run all cells in order. Its setup searches the current folder and its
parent for the core file. The notebook contains these presentation sessions:

1. Configure the experiment, seeds, output directory, and source provenance.
2. Identify controller groups and comparison assumptions.
3. Train every applicable method, save checkpoints/logs, verify reloading, and
   evaluate on held-out seeds.
4. Display ranked seven-aircraft and one-aircraft results separately.
5. Save evaluation, training-curve, and return-versus-training-time figures as PNG and SVG.
6. Compare training/execution duration per method and observed one-to-seven-aircraft runtime ratios.
7. Reload the selected fleet controller and execute additional timed episodes.
8. Generate and save performance and runtime interpretation from the actual tables.

Default presentation settings use **1,024 simulated training days per seed**, three
training seeds, 20 evaluation episodes per model, and a 90-day episode horizon.
The seven-aircraft panel runs 17 neural configurations plus the fixed threshold
rule. The separate one-aircraft panel includes all 18 learned configurations,
including MAXQ, plus the rule. This produces 105 trained models; execution can
take several minutes on a CPU. Change the settings in the first code cell for
longer experiments.

Each execution creates `results/presentation_<UTC timestamp>/` containing:

| Output | Contents | GitHub |
| --- | --- | --- |
| `manifest.json` | Configuration, seeds, source hash/commit, versions, CPU/thread information, timing boundaries, and completion status. | Included for the recorded presentation run. |
| `data/` | Raw timed evaluation episodes, training/execution blocks, summaries, episode curves, fleet-size runtime ratios, and selected-controller execution. | Included. |
| `plots/` | Performance comparisons, training curves, training/execution duration per method, and one-to-seven-aircraft timing ratios in PNG/SVG. | Included. |
| `results_interpretation.md`, `verification.json` | Generated conclusions and artifact checks. | Included. |
| `checkpoints/`, `training_logs/` | Every trained model and compressed detailed training log. | Saved locally; excluded from Git. Rerun the notebook to regenerate them. |

Results identify the **best observed** controllers under this fixed budget. They
do not claim convergence or statistical superiority. Performance/training error bars describe
training-seed variation; execution error bars describe measured episode-time
variation. Method authority, actor information, decision timing,
parameter counts, and objectives differ; the notebook explains these limits and
reports the closer `planning_hierarchy` versus `hrl_flat` comparison separately.

The [latest timed presentation run](results/presentation_20261026T154304Z) includes
the [performance results](results/presentation_20261026T154304Z/data/summary.csv),
[per-method runtime summary](results/presentation_20261026T154304Z/data/runtime_summary.csv),
[fleet-size timing ratios](results/presentation_20261026T154304Z/data/runtime_fleet_size_ratio.csv),
[saved plots](results/presentation_20261026T154304Z/plots), and
[generated interpretation](results/presentation_20261026T154304Z/results_interpretation.md).
The [original presentation run](results/presentation_20261026T143049Z) remains available.

### Timing protocol and large-model interpretation

Training time surrounds the entire `core.train()` call: model/environment
initialization, interaction, targets, and optimization. Checkpoint writing and
evaluation are outside that timer. One discarded 16-day warm-up initializes the
training framework before recorded runs.

Fixed-weight execution uses `time.perf_counter()` around each complete evaluation
episode, after one excluded warm-up episode per controller. It includes reset,
policy decisions, resource arbitration, sensors/filtering, and physical simulation;
it excludes learning, loading, saving and plotting. This is **end-to-end execution
time**, rather than isolated neural forward-pass latency. Every held-out episode
has its own recorded duration. The threshold rule has zero training cost and a
measured execution cost.

`runtime_summary.csv` reports training mean/standard deviation, execution mean,
standard deviation and episode percentiles, milliseconds per simulated day, and
milliseconds per aircraft-day. `execution_runs.csv` records each evaluation block
and its excluded warm-up duration. Training bars use variation across three
training seeds; execution bars use the 60 measured episode durations per learned
method (20 for the fixed rule). These spreads are not confidence intervals.

Both panels use one Torch CPU thread. Hardware, software versions and the timing
definitions are saved in the manifest. The observed one-to-seven-aircraft ratios
combine policy, broker and simulator work and depend on the policy's decisions.
They are a two-size measurement on this machine, not a prediction of large-fleet
complexity or a GPU benchmark. Repeat at the intended aircraft/resource counts
and hardware before using these numbers to size the large model.

### Recorded timing comparison (seven aircraft)

Means from the latest CPU rerun; training uses 1,024 simulated days per seed,
and execution uses fixed weights for one 90-day episode. Full tables and
variation plots are linked above. These are end-to-end implementation timings.

| Method | Training seconds | Execution milliseconds / episode |
| --- | ---: | ---: |
| `threshold` | 0.00 | 238.0 |
| `options` | 3.14 | 292.4 |
| `ddmac_aircraft` | 16.12 | 323.9 |
| `conditional` | 7.15 | 564.0 |
| `hierarchical_vdn` | 7.69 | 403.4 |

Fastest mean learned-method training: `options`; fastest mean learned
execution: `options`. The highest-return controller was `hierarchical_vdn`.
Training/execution leaders need not be the same as return leaders. Small
timing gaps should be interpreted with the episode/seed variation and
current policy decision frequencies, rather than as universal algorithm rankings.

### Personal learning notebook

Open `Comprehensive_RL_HRL_Maintenance_Tutorial.ipynb` for the detailed teaching
material and run its setup cell first. Its ten parts are:

| Part | Topic |
| --- | --- |
| 01 | Build and inspect a small maintenance environment. |
| 02 | Train and evaluate a tabular RL agent. |
| 03 | Gymnasium and Stable-Baselines3 (SB3). |
| 04 | Aircraft, engine, and component decisions within one time step. |
| 05 | Temporally extended options and duration-aware learning. |
| 06 | Fleet environment, condition estimates, actions, and resources. |
| 07 | Original flat, conditional, shared, and fleet-option policies. |
| 08 | Multi-agent assignments and cooperative policy learning. |
| 09 | Global-belief centralized and independent actor methods. |
| 10 | Concurrent aircraft options and additional HRL methods. |

SB3 proximal policy optimization (PPO) is used in the introductory library example.
The fleet controllers use their own custom trainers. The two paths construct and
train separate models; they do not apply two trainers to the same policy weights.

For a small fleet experiment without opening Jupyter:

```python
from maintenance_tutorial_annotated_core import Config, train, evaluate

config = Config(aircraft=7, horizon=90)
model, logs, simulated_days = train(
    config, method="mappo_aircraft", steps=256, seed=0, rollout_size=32
)
records = evaluate(config, "mappo_aircraft", model, seeds=[10000, 10001])
```

The core exposes a command-line runner as `main()`. Invoke it explicitly:

```bash
python -c "from maintenance_tutorial_annotated_core import main; main()" --method mappo_aircraft --aircraft 7 --steps 256 --eval-episodes 2 --output runs/mappo_demo
```

The runner saves model checkpoints, settings, training/evaluation records, and a
summary under the selected output directory. `--method all` runs all applicable
methods; it explicitly skips MAXQ when `--aircraft` is not 1. The local `runs/`
directory is ignored by Git and is not part of the uploaded example results.

## Fleet environment

### State, estimates, and daily time

The default fleet has seven aircraft, two engines per aircraft, and six component
types: fan, low-pressure compressor (LPC), high-pressure compressor (HPC), combustor,
high-pressure turbine (HPT), and low-pressure turbine (LPT). Hidden normalized
health has shape `(aircraft, 2, 6)`. Synthetic efficiency/temperature readings feed
a particle filter; policies receive estimated condition and uncertainty rather
than the hidden health array.

Each physical environment step advances **one day**. An eligible operating aircraft
performs one modeled flight; grounded aircraft do not. Wear depends on a fixed
synthetic aircraft exposure and a daily weather multiplier. Bird and hail events
are sampled per engine with illustrative probabilities, severities, and component
damage weights. Measurements and beliefs update after operating flights. Readings
remain old while an aircraft is grounded, and reading age is included in its
observation. No explicit flight legs, flight-hour durations, or route schedules
are implemented yet.

Observation version **3** contains 70 values per aircraft plus
`10 + 2 * crews + 2 * workshops` public values. With seven aircraft, three crew
members, and two workshops, this gives **510 values**. Local fields include health
means, standard deviations, lower quantiles, exposure, job status, committed plan,
sensor readings, and reading age. Public fields include resource availability,
duration-factor encodings, inventory, and replenishment timing. Older observation
versions are incompatible with the current model layout.

### Identities, binary encoding, and resources

Aircraft plates `AC-001`, `AC-002`, etc. map to stable fleet indices. Reset starts
a new simulated history while retaining that mapping. A plate is an identity
label; it is not a binary status flag or a persistent flight-history database.

The original full fleet action has eight fields per aircraft:

```text
[ground, engine_mask, bundle0, bundle1, crew_mask, workshop_index, order_start, priority]
category counts: [2, 4, 64, 64, 2**crews, workshops, 12, aircraft]
```

Each six-bit bundle selects component types on one engine. For example, `0b000101`
selects fan and HPC. Engine masks select the left engine, right engine, or both.
Crew bits identify individual members: `0b101` selects `CR-001` and `CR-003`.
Workshop selection is one index, and `order_start` selects a cyclic task order,
rather than an arbitrary permutation.

Crew and workshop duration factors below one speed up component work; factors
above one slow it down. For a selected crew set `S` and workshop `w`:

```text
combined_work_rate = sum(1 / crew_time_factors[i] for i in S)
component_days = max(1, ceil(component_labor * workshop_time_factors[w]
                             / combined_work_rate))
job_days = setup_days + sum(component_days for each ordered component task)
```

Setup remains fixed, work rates add, and component tasks execute sequentially.
Accepted jobs reserve individual crew members and a workshop, consume spares,
and store their component-completion times. A new proposal cannot overwrite a
busy aircraft's job. Components restore at their scheduled completion thresholds;
the aircraft stays grounded until the whole job ends. There is no separate
expertise reward, wage model, or skill-dependent repair quality.

Empty profiles use repeating synthetic crew factors `(0.8, 1.0, 1.25)` and workshop
factors `(0.8, 1.0, 1.2)`. Supply one positive factor per resource to override them:

```python
config = Config(
    aircraft=10,
    crews=3,
    workshops=2,
    crew_time_factors=(0.8, 1.0, 1.25),
    workshop_time_factors=(0.8, 1.2),
)
```

Daily reward is `(10 * flights - cost) / (10 * aircraft)`. Costs account for
maintenance, unused component life, grounding, failed flights, and rejected plans.
The conservative safety hold uses observed lower health quantiles across all
components; it does not wait for HPC failure alone.

## Implemented methods and their differences

Multi-agent proximal policy optimization (MAPPO), deep centralized multi-agent
actor–critic (DCMAC), and deep decentralized multi-agent actor–critic (DDMAC) have
distinct implementations. Semi-Markov decision process (SMDP) updates account for
variable action durations. Value-decomposition networks (VDN) represent team
values by adding agent values. Temporal-difference (TD) updates bootstrap from
following-state value estimates. In the table, `N` is aircraft count and a
scalar critic produces one fleet state value.

| Method | Actor ownership and information | Value and learning implementation |
| --- | --- | --- |
| `flat` | One reused full-action aircraft actor with global/local input; field logits are not explicitly conditioned on preceding field choices, although validity masks still depend on them. | Scalar global critic; fleet-joint PPO. |
| `conditional` | One reused full-action actor with global/local input; later fields also receive earlier choices. | Scalar global critic; fleet-joint PPO. |
| `shared` | One reused full-action actor using local/public inputs without the learned global actor encoding. | Scalar global critic; fleet-joint PPO. |
| `options` | Global fleet manager chooses among four fixed threshold routines. | Scalar manager critic; SMDP PPO with fleet-wide option execution. |
| `learned_options` | Learned global manager and option-conditioned full-action worker. | Manager critic and joint manager/worker likelihood; SMDP PPO. |
| `mappo_aircraft` | N aircraft roles sharing one local/public actor; each requests two 64-way component bundles. | One global critic network with N outputs; per-agent clipped PPO. |
| `mappo_component` | Six independent component-type actors; each sees component-restricted fleet context and chooses N four-way engine masks. | One global critic with six outputs; per-agent clipped PPO. |
| `dcmac_aircraft` | One global-input actor with distinct factorized output blocks for N aircraft roles. | Scalar global critic; replay and joint importance-weighted TD. |
| `ddmac_aircraft` | N independently parameterized actors, each receiving the global observed fleet vector. | Scalar global critic; replay and joint importance-weighted TD. |
| `dcmac_component` | One global-input actor with six component-type output blocks. | Scalar global critic; replay and joint importance-weighted TD. |
| `ddmac_component` | Six independent global-input component-type actors. | Scalar global critic; replay and joint importance-weighted TD. |
| `concurrent_options` | Shared aircraft operate/maintain manager with a fixed condition-based worker. | N-output global critic; actual aircraft-duration PPO targets. |
| `planning_hierarchy` | Separate shared grounding, engine, and component stage networks using global/local inputs and preceding choices. | N-output global critic; actual aircraft-duration PPO targets. |
| `aggregate_targets` | Shared upper budget policy and goal-conditioned catalog worker; goals are 0, 1, 2, or 12 component instances. | N-output global critic; joint stage learning with duration-aware PPO. |
| `maxq` | Single-aircraft recursive tabular hierarchy using quantized observations. | MAXQ-0 primitive reward estimates V plus parent completion estimates C. |
| `option_critic` | Three planning modes with a learned option-conditioned worker and safe termination. | Option values, internal-policy gradients, and termination updates; TD learning. |
| `hierarchical_vdn` | N local/public upper Q networks and N goal-conditioned lower Q networks. | Additive team values at both levels; macro/daily TD updates and disclosed reward shaping. |
| `hrl_flat` | Shared global/local actor choosing directly from the same 22-entry catalog used by the new planning methods. | N-output global critic; the same aircraft-duration PPO trainer as `planning_hierarchy`. |

The fixed `threshold` controller is an additional evaluation baseline. It has no
learned network. In total, the registry contains 17 neural configurations plus
the tabular MAXQ baseline.

### Sharing weights, agent ownership, and execution information

One network does not necessarily mean one agent. The original full-action
controllers combine aircraft/field log probabilities into **one fleet likelihood**
and learn resource selections as well as replacements. Aircraft MAPPO reuses one
actor for separate local/public observations, retains an advantage and likelihood
per aircraft agent, and **clips each agent's ratio before averaging**. It also
has a narrower replacement-request interface. These differences matter even
when both implementations have one shared actor network.

Centralized training with decentralized execution (CTDE) describes an information
pattern. MAPPO's actors can generate proposals from their restricted views while
its critic uses global observations during training. A fixed centralized broker
still handles resource allocation. There is no learned peer-to-peer messaging.

DCMAC and DDMAC here both require **global observed beliefs at actor execution**.
DCMAC factorizes outputs within one central actor. Aircraft DDMAC constructs N
independent actor networks; component DDMAC constructs six. Their replay trainer
uses a capped joint importance ratio and one-step TD targets, rather than MAPPO's
per-agent PPO update. Separate actor parameters do not imply local-information
execution or fully decentralized maintenance scheduling.

### Action authority, hierarchy, and decision timing

The six MAPPO/DCMAC/DDMAC configurations propose replacements. Their shared fixed
broker chooses crew identities, workshop, cyclic task order, and priority while
respecting stock and committed jobs. Deferred requests are recorded separately
from rejected simulator plans; learning retains the submitted proposal likelihood.

The new aircraft HRL methods use a **22-entry catalog**: operate, or one of three
engine masks combined with six single-component bundles or the full six-component
bundle. Selecting both engines applies the same bundle to both. The broker still
allocates resources. Budget masks impose structural constraints; they do not
guarantee joint stock/resource feasibility or target achievement.

`AircraftOptionEnv` advances one day and permits a free aircraft to start a new
decision while other aircraft remain under committed repair. The older fleet
`execute_option` call advances several days before its manager decides again.
The three learned calls in `planning_hierarchy` occur before one physical step;
they are computational stages, not three elapsed-time epochs. Option-critic
learns planning-mode termination only when safe; it cannot cancel an active job.
MAXQ implements an explicit recursive V+C decomposition for one aircraft.
Hierarchical VDN uses a three-day upper decision window, daily lower decisions,
and a detached upper-level advantage in the lower reward. It is an adaptation,
rather than the full HAVEN algorithm described in the report's references.

For a closer comparison of staged versus flat planning, compare `hrl_flat` with
`planning_hierarchy`: catalog, broker, information, decision opportunities, critic
shape, and duration accounting are matched, while parameterization and network
size differ. Comparisons across all 18 methods also change action authority,
information, timing, objectives, or update rules and cannot isolate agent count
or the benefit of hierarchy alone.

## Verification and interpretation

Recorded verification for the documented **26 October 2026** version:

- 43 behavioral checks passed, including observation/action contracts, identities,
  resource reservations, duration arithmetic, commitment, agent likelihoods,
  learning targets, parameter updates, reproducible seeds, and checkpoint loading.
- All 122 code cells in the 336-cell comprehensive notebook executed without errors.
- An all-method runner smoke test trained 17 neural configurations on a two-aircraft
  case and explicitly skipped MAXQ; its one-aircraft path was checked separately.

To run the consolidated suite, open the personal learning notebook, run its setup
cell, and then execute:

```python
helpers.run_checks()
```

Checkpoints store method/configuration, weights or MAXQ tables, and observation
version. They do not store the full optimizer/replay state for exact resumption.
Changing aircraft/resource counts generally changes model dimensions and requires
constructing and training a matching model. Shared weights alone do not establish
unchanged-weight transfer across fleet sizes. Crew action categories grow as
`2**crews`, which limits this representation for large resource pools.

## Core cleanup (28 October 2026)

The core retains all 18 method configurations and its existing public interfaces,
observation/action layouts, and checkpoint state keys. Statement-by-statement
generated comments and nine repeated imports were removed; docstrings and concise
model/safety notes remain. Full line-by-line explanations are in the versioned
PDF reports. The 28 October report uses current source line numbers; older
reports retain the line numbers of their own source snapshots.

Critic-only calls now validate global observations without constructing repeated
per-aircraft actor features. Flat action sampling skips unused conditioning-history
buffers. Shared observation validation preserves the previous shape checks and
error messages. Custom initial-health arrays now explicitly reject NaN and infinity.

Verification: 45 behavioral checks pass. All 18 methods matched the previous
implementation's weights, training logs and evaluation in a fixed-seed 64-day
regression run; a saved checkpoint per method loaded and evaluated identically.
For batch size 32 on one Torch CPU thread, isolated critic-call microbenchmarks
improved by about 1.9–2.0× at seven aircraft and 7.9–8.2× at 50 aircraft. At 50
aircraft, each such call avoids constructing a 22,976,000-byte actor-context
tensor. These are isolated-call measurements, not full training speedups.

Existing presentation results and timing tables predate this cleanup and retain
their original source hashes. Rerunning the main notebook records the new source
hash and produces new measurements. Unused registered network layers were kept
for checkpoint compatibility; the learning notebook's explanatory links were
updated without changing its code cells or saved experimental outputs.

## Current scope and pending work

The reports document educational implementations and explicitly scoped adaptations
of the research methods, rather than paper-exact reproductions. The original
single-agent report remains useful for fundamentals; the updated report is the
reference for current behavior and limitations.

Remaining work includes explicit synthetic flight legs and durations, richer
flight-condition inputs, standardized common-reference post-landing measurements,
and literature-based bird/hail encounter and damage upgrades. The current particle
filter provides condition/uncertainty summaries, not a remaining-useful-life
forecast. It resamples component marginals independently and does not represent
all shared-event dependencies.

Holiday/vacation staffing, learned communication, graph/attention architectures,
arbitrary repair permutations, procurement policies, and paper-level risk/budget
constraints are not implemented. Repair order can change restoration timing
without changing total return for a fixed package under whole-job grounding.
Longer controlled experiments, multiple seeds, uncertainty estimates, and physical
calibration are needed before drawing maintenance-policy performance conclusions.
