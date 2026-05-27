# PostgreSQL JSON Plan Cost Summary

Derived from disabled-parallel `EXPLAIN (ANALYZE, FORMAT JSON)` traces.
`scaled_q_*` uses one log-mean multiplicative scale per workload to map planner cost units to milliseconds.

| workload | n | cost_runtime_pearson | cost_runtime_spearman | log_cost_runtime_pearson | median_cost_to_ms_scale | scaled_q_median | scaled_q_p90 | scaled_q_p95 | scaled_q_p99 | scaled_q_max | raw_cost_vs_ms_q_median | raw_cost_vs_ms_q_p95 | raw_cost_vs_ms_q_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| job_light | 70 | 0.937503 | 0.874453 | 0.910165 | 0.00310942 | 1.75965 | 5.71227 | 12.7463 | 23.3318 | 31.7527 | 235.784 | 4099.24 | 10211.8 |
| job | 113 | 0.0914864 | 0.159589 | 0.120258 | 0.0132421 | 3.05546 | 16.6297 | 136.901 | 469.1 | 2970.02 | 165.834 | 658.887 | 8427.93 |
| job_complex | 30 | 0.274274 | 0.670747 | 0.612042 | 0.0125079 | 2.76126 | 9.99715 | 24.2363 | 49.0293 | 57.0025 | 121.888 | 715.797 | 730.021 |
