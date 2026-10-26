### Observed outcome

- **Highest seven-aircraft mean return:** `hierarchical_vdn` (89.83).
- **Best observed MARL method:** `mappo_aircraft` (46.62; training-seed SD 0.42).
- **Best observed HRL method:** `hierarchical_vdn` (89.83; training-seed SD 0.00).
- The top learned controller is **+29.16 return units relative to the threshold rule**. Its lead over the second learned controller is 0.02; this is descriptive, not a significance test.
- **One-aircraft result:** 16 methods tie at the highest mean return (90.00); MAXQ ranks 1 of 19, with return 90.00. This panel cannot establish fleet scalability.

### Does explicit hierarchy help in the closer comparison?

`planning_hierarchy - hrl_flat` is **+1.06 mean return units**, positive in 1/3 paired training seeds. The two methods share the catalog, broker, information and duration accounting; network parameterization and size still differ. Other method comparisons also change decision authority, update rules or timing.

### What explains the scores?

Highest availability: **hierarchical_vdn, conditional**. Lowest cost per aircraft-day: **hierarchical_vdn**. Fewest failed flights: **all 18 methods (0)**. Inspect these alongside replacements and training curves: a return lead can arise from a different availability/maintenance trade-off, rather than a universally better maintenance policy.

The top learned controller's return difference from the threshold rule decomposes exactly into **+16.53 units from flown availability** and **+12.63 units from total cost**. This follows the simulator identity `episode return = horizon * availability - cost / (10 * aircraft)`; it is an accounting explanation rather than a causal architecture claim.

The leading learned controller averages **0.05 replacements and 99.9% availability**. The leading MARL controller averages **41.22 replacements and 71.4% availability**; the leading HRL controller averages **0.05 replacements and 99.9% availability**. These observed maintenance/grounding differences help explain the return differences. Different categorical action spaces and limited training are possible contributors that this experiment does not isolate.

The finite **90-day horizon**, initially high health, and absence of a terminal charge for remaining degradation can favor policies that defer maintenance beyond the evaluated episode. A score close to 90 approaches this environment's return ceiling. Inspect failures and replacement counts, and extend the horizon or test more degraded initial states consistently across methods before interpreting such a score as a strong lifetime maintenance policy.

### Limits and next decision

This run uses **1,024 simulated training days per seed, 3 training seeds and 20 held-out episodes per trained model**. Error bars show training-seed variation, not confidence intervals or all evaluation uncertainty. Only 0 failed-flight occurrences were recorded across the seven-aircraft evaluation rows; these rows reuse scenarios across methods, so that total is not an independent estimate of a real-world failure rate. Zero observed failures leave failure-risk performance unresolved in this pilot.

The current candidate for a longer fleet experiment is `hierarchical_vdn` by observed learned-policy return. Extend the budget and seed count and check performance on fresh evaluation scenarios before claiming a winner. All controllers use the current synthetic physical model; no hot-weather, dust, salt or flight-hour upgrade was introduced by this notebook. Results reflect the method implementations in this repository, not a ranking of the original papers' algorithms.
