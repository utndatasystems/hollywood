# Hollywood-200K DBMS-Guided Paper Results

Created: 2026-05-23T01:42:49

This folder packages the final Hollywood-200K paper workload after bounded,
DBMS-guided literal rebinding and validation against both DuckDB and PostgreSQL.

## Contents

- `raw_csv/full_query_cardinality_raw.csv`: DuckDB, PostgreSQL, and saved MSCN full-query cardinality rows.
- `raw_csv/selected_plan_cost_raw.csv`: PostgreSQL selected-plan cost/runtime, pretrained ZeroShot, and saved MSCN cost rows.
- `raw_csv/base_selection_raw.csv`: base-selection rows included in the compact paper bundle.
- `raw_csv/intermediate_postgres_cardinality_raw.csv`: exact logical-subplan COUNT(*) cardinality rows reconstructed from PostgreSQL selected-plan alias subtrees.
- `summaries/*.csv`: grouped q-error summaries and completeness checks.
- `plots/*.png` and `plots/*.pdf`: paper-facing boxplots.
- `source_reports/*`: copied manifests and run summaries from the source runs.

## Notes

- This bundle is 200K-only and is intended for the final Hollywood 200K
  benchmark comparison.
- PostgreSQL selected-plan cost is reported with `total_cost` scaled to runtime
  by a workload-specific geometric-mean runtime/cost scale. This matches the
  compact cost-summary table used for the released workload.
- The ZeroShot run is checkpoint-statistics compatible and covers all 213
  selected plans per seed. Four long JOB plans used a higher LCM max-runtime
  cutoff; PostgreSQL `Memoize` nodes were mapped to
  `Materialize` for the pretrained LCM operator vocabulary.
- Intermediate cardinality uses exact logical-subplan labels: each selected
  plan-node alias subtree is reconstructed as a COUNT(*) SQL query and counted
  exactly. The JSON plan-node rows are also preserved as
  `raw_csv/intermediate_postgres_plan_nodes_raw.csv`.
