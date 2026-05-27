# MSCN

This folder contains the MSCN runner and finalization scripts used for the
Hollywood paper experiments.

Typical same-dataset cardinality run:

```bash
python scripts/run_postgres_workload.py --workload all --conn "$PGDATABASE" --out-dir results/postgres_full_query

python experiments/mscn/paper_mscn_runner.py \
  --run-id hollywood_200k_mscn_cardinality \
  --db-name hollywood_200k \
  --base-run results/postgres_full_query \
  --exact-query-dir queries/job \
  --complex-query-dir queries/job_complex \
  --extra-eval-query-dir job_light=queries/job_light \
  --train-queries 100000 \
  --epochs 100 \
  --model-seeds 1,2,3 \
  --subplan-count-timeout-ms 60000
```

The runner expects PostgreSQL labels/traces. The small bridge files needed by
the runner are vendored in `experiments/mscn/vendor/`; large training corpora
and checkpoints are intentionally not committed.
