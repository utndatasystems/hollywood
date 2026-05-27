# Balanced DuckDB/PostgreSQL 200K Paper Bundle

Created: 2026-05-23T01:42:49

This folder packages the final balanced 200K Hollywood query candidate after
literal mining against both DuckDB and PostgreSQL.

## Contents

- `raw_csv/full_query_cardinality_raw.csv`: DuckDB, PostgreSQL, and saved MSCN full-query cardinality rows.
- `raw_csv/selected_plan_cost_raw.csv`: PostgreSQL selected-plan cost/runtime, pretrained ZeroShot, and saved MSCN cost rows.
- `raw_csv/base_selection_raw.csv`: base-selection rows if the extraction finished before bundle generation.
- `raw_csv/intermediate_postgres_cardinality_raw.csv`: old-style exact logical subplan COUNT(*) cardinality rows reconstructed from PostgreSQL selected plan alias subtrees.
- `summaries/*.csv`: grouped q-error summaries and completeness checks.
- `plots/*.png` and `plots/*.pdf`: paper-facing boxplots.
- `source_reports/*`: copied manifests and run summaries from the source runs.

## Notes

- This bundle is 200K-only; the supervisor discussion moved the final query
  comparison focus to the 200K benchmark.
- PostgreSQL selected-plan cost is reported with `total_cost` scaled to runtime
  by the log-mean cost/runtime ratio per workload. This matches the compact
  cost-summary table used for the balanced candidate.
- The ZeroShot run is checkpoint-statistics compatible and covers all 213
  selected plans per seed. Four long JOB plans required re-parsing with a
  higher LCM max-runtime cutoff; PostgreSQL `Memoize` nodes were mapped to
  `Materialize` for the pretrained LCM operator vocabulary.
- Intermediate cardinality is the old-style exact logical subplan view: each selected plan-node alias subtree is reconstructed as a COUNT(*) SQL query and counted exactly. The cheaper JSON plan-node rows are also preserved as `raw_csv/intermediate_postgres_plan_nodes_raw.csv`.
