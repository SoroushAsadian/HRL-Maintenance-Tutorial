# Recorded presentation experiment

Executed from `Maintenance_Experiments_Presentation.ipynb` on 26 October 2026.
See [manifest.json](manifest.json) for the exact source hash, configuration,
software versions, and seeds.

- Seven-aircraft panel: 17 learned configurations and the fixed threshold rule.
- One-aircraft panel: all 18 learned configurations, including MAXQ, and the rule.
- Training: 1,024 simulated days per seed, three seeds per learned configuration.
- Evaluation: 20 held-out 90-day episodes per trained model; 2,140 recorded episodes
  including the two threshold baselines.
- All 105 trained checkpoints passed a reload/evaluation consistency check.

Read the [results interpretation](results_interpretation.md), inspect the
[ranked summary](data/summary.csv), or open the [figures](plots).
The [raw evaluation rows](data/evaluation.csv) preserve panel, method, training
seed, evaluation seed, and the physical outcome metrics. Training curves and
timings are in [data](data).

Error bars show variation between training-seed means. They are not confidence
intervals. The threshold rule has no initialization variation. Rankings are
exploratory and depend on the short finite horizon, action authority, information
and update rules. No failed flights were observed, so this run does not distinguish
failure-risk performance. Sixteen methods tie at the one-aircraft return ceiling.

The notebook saved every model in `checkpoints/` and compressed detailed logs in
`training_logs/` locally. These large files are excluded from Git; rerun the
notebook to regenerate them. Published CSV, PNG and SVG files are the compact
results of the recorded execution. The core simulator and personal learning
notebook were unchanged by this experiment.
