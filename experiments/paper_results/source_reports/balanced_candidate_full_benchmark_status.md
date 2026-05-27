# Balanced DuckDB/PostgreSQL 200K Candidate Benchmark Status

Query root: `queries/`.

## Full-query cardinality
| Workload | Engine | Coverage | Median Q | P95 Q | Max Q |
|---|---|---:|---:|---:|---:|
| JOB | DuckDB | 113/113 | 267 | 2.09e+04 | 2.76e+05 |
| JOB | PostgreSQL | 113/113 | 28 | 2.37e+04 | 4.87e+05 |
| JOB-Complex | DuckDB | 30/30 | 190 | 6.4e+04 | 1.92e+06 |
| JOB-Complex | PostgreSQL | 30/30 | 20.5 | 3.07e+05 | 1.24e+06 |
| JOB-Light | DuckDB | 70/70 | 362 | 2.47e+04 | 3.89e+04 |
| JOB-Light | PostgreSQL | 70/70 | 1.98 | 153 | 260 |

## PostgreSQL selected-plan cost/runtime
| Workload | N | Scaled median Q | Scaled P95 Q | Scaled max Q |
|---|---:|---:|---:|---:|
| job_light | 70 | 1.76 | 12.7 | 31.8 |
| job | 113 | 3.06 | 137 | 2.97e+03 |
| job_complex | 30 | 2.76 | 24.2 | 57 |

## ZeroShot selected-plan cost/runtime
| Workload | Seed | Median Q | P95 Q | Max Q |
|---|---:|---:|---:|---:|
| job_light | 0 | 6.89 | 95 | 186 |
| job_light | 1 | 5.51 | 151 | 306 |
| job_light | 2 | 4 | 109 | 306 |
| job_light | pooled | 6.33 | 131 | 306 |
| job | 0 | 5.73 | 24.8 | 360 |
| job | 1 | 6.09 | 55.7 | 93.8 |
| job | 2 | 4.14 | 28.8 | 122 |
| job | pooled | 5.26 | 34.1 | 360 |
| job_complex | 0 | 4.29 | 23.7 | 24.4 |
| job_complex | 1 | 5.25 | 40.6 | 41.5 |
| job_complex | 2 | 4.28 | 63.3 | 64.4 |
| job_complex | pooled | 4.33 | 51.4 | 64.4 |

## Saved MSCN full-query cardinality, trained on 200K stablekw
| Workload | Seed | N | Median Q | P95 Q | Max Q |
|---|---:|---:|---:|---:|---:|
| job_light | 1 | 70 | 7.47 | 3.49e+05 | 3.13e+06 |
| job | 1 | 113 | 4.24 | 1.08e+03 | 4.85e+04 |
| job_complex | 1 | 30 | 5.21 | 6.23e+03 | 7.2e+04 |
| job_light | 2 | 70 | 6.82 | 7.96e+04 | 3.84e+05 |
| job | 2 | 113 | 5.89 | 2.16e+03 | 1.91e+04 |
| job_complex | 2 | 30 | 6.27 | 4.09e+03 | 1.06e+05 |
| job_light | 3 | 70 | 8.42 | 3.43e+04 | 4.62e+05 |
| job | 3 | 113 | 3.67 | 818 | 5.34e+04 |
| job_complex | 3 | 30 | 5.22 | 1.48e+04 | 5.7e+04 |

## Saved MSCN selected-plan cost/runtime, trained on 200K stablekw
| Workload | Seed | N | Median Q | P95 Q | Max Q |
|---|---:|---:|---:|---:|---:|
| job_light | 1 | 70 | 4.83 | 1.79e+03 | 6.47e+03 |
| job | 1 | 113 | 2.66 | 285 | 4.86e+03 |
| job_complex | 1 | 30 | 2.63 | 65 | 99 |
| job_light | 2 | 70 | 5.78 | 1.67e+03 | 1.34e+04 |
| job | 2 | 113 | 2.78 | 249 | 5.18e+03 |
| job_complex | 2 | 30 | 2.89 | 62.2 | 162 |
| job_light | 3 | 70 | 4.97 | 576 | 2.16e+03 |
| job | 3 | 113 | 2.52 | 238 | 6.48e+03 |
| job_complex | 3 | 30 | 3.07 | 64 | 134 |

## Verification
- present: DuckDB/PostgreSQL cardinality raw - `experiments/paper_results/raw_csv/full_query_cardinality_raw.csv`
- present: PostgreSQL selected-plan cost raw - `experiments/paper_results/raw_csv/selected_plan_cost_raw.csv`
- present: PostgreSQL cost summary - `experiments/paper_results/summaries/selected_plan_cost_summary.csv`
- present: MSCN cardinality source report - `experiments/paper_results/source_reports/mscn_cardinality_evaluation_manifest.json`
- present: ZeroShot source report - `experiments/paper_results/source_reports/zeroshot_summary_qerrors.csv`
- present: MSCN cost source report - `experiments/paper_results/source_reports/mscn_cost_evaluation_manifest.json`
