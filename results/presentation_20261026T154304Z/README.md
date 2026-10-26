# Presentation rerun with training and execution timing

The main `Maintenance_Experiments_Presentation.ipynb` reruns all controllers and
records wall-clock durations alongside their maintenance outcomes. Exact source,
seeds, hardware, software versions and timing boundaries are in
[manifest.json](manifest.json).

## Experiment

- Seven-aircraft panel: 17 learned methods and the fixed threshold rule.
- One-aircraft panel: all 18 learned methods, including MAXQ, and the rule.
- Each learned method: 1,024 training days for each of three seeds; 20 held-out
  90-day evaluation episodes per model.
- 105 recorded trained models, 107 evaluation blocks including the two rules,
  and 2,140 timed evaluation episodes. Three additional selected-controller
  episodes are saved separately from the ranking.

## Results and figures

- [Performance summary](data/summary.csv) and [interpretation](results_interpretation.md).
- [Training and execution summary per method](data/runtime_summary.csv).
- [One-to-seven-aircraft runtime ratios](data/runtime_fleet_size_ratio.csv).
- [Raw timed evaluation episodes](data/evaluation.csv).
- [Training runs](data/training_runs.csv) and [evaluation timing blocks](data/execution_runs.csv).
- [Figures](plots): performance and training curves, return-versus-training-time,
  training/execution duration for each panel, and measured fleet-size ratios.

## Timing boundaries

Training includes initialization, interaction and updates inside `core.train()`.
One discarded 16-day framework warm-up runs before recorded training. Fixed-weight
execution is timed separately for every episode after one excluded warm-up per
controller; it includes reset, policy, broker and physical simulation. Loading,
saving, plotting and learning are excluded from the execution timer. It measures
end-to-end execution, not pure neural forward-pass latency.

All times are measured on the recorded CPU with one Torch thread. Training
error bars show variation between three trained seeds. Execution error bars show
variation across 60 episode durations per learned method, or 20 for the rule.
Percentiles are descriptive episode-time percentiles. The rule has no training
cost. Its training fleet-size ratio is undefined and is left blank in the CSV.

These are two observed fleet sizes under a fixed simulation budget. They do not
establish asymptotic complexity, GPU speed, convergence, or runtime at the future
large-model scale. Policy choices, updates, decision frequency and simulator work
also influence durations. The saved verification checks timing arithmetic,
percentiles, checkpoint reloads and preservation of the previous physical results.

All model checkpoints and full compressed training logs are saved locally in
`checkpoints/` and `training_logs/`, which are excluded from Git. The compact
tables, figures and executed notebook are published. The original untimed
presentation run is retained in `../presentation_20261026T143049Z/`.
