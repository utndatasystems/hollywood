# MSCN Reproducibility

The final paper bundle includes MSCN result CSVs and summaries for the final
200K workload. Lightweight runner code is included under:

```text
experiments/mscn/
```

This repository includes:

- the MSCN runner and finalization scripts under `experiments/mscn/`,
- the final paper-facing MSCN CSV summaries under `experiments/paper_results/`,
- the Hollywood dataset, query files, and PostgreSQL workload collector needed
  to produce base labels.

This repository does not include:

- the external upstream MSCN project,
- large training corpora,
- trained checkpoints,
- precomputed training labels beyond the compact paper bundle.

To reproduce MSCN from scratch:

1. Generate or provide training query labels.
2. Build the same IMDb-compatible DuckDB/PostgreSQL database.
3. Use `experiments/mscn/paper_mscn_runner.py` for cardinality training and
   evaluation.
4. Use `experiments/mscn/mscn_selected_plan_cost_runner.py` for selected-plan
   cost/runtime evaluation.
5. Compare produced raw CSVs with `experiments/paper_results/raw_csv/`.

The normal label source for full-query cardinality experiments is:

```bash
python scripts/run_postgres_workload.py --workload all --conn "$PGDATABASE" --out-dir results/postgres_full_query
```

Record the external MSCN source repository, commit, and checkpoint identifier
used for a run. Those external versions are not pinned in this release folder.

The 100K-to-200K transfer setting means: train MSCN on the 100K Hollywood
dataset and evaluate the trained checkpoint on the 200K Hollywood dataset.
It is a scale-transfer diagnostic, not the main same-dataset training setup.
