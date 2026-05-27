# Evaluation

## DuckDB

Run all three workloads:

```bash
python scripts/run_duckdb_workload.py --workload all
```

Outputs are written to `results/duckdb/`.

The default is read-only. Use `--analyze` only on a writable copy of the DB.

## PostgreSQL

Create and load the schema with `psql`:

```bash
python scripts/print_postgres_load_sql.py --csv-dir /absolute/path/to/data/imdb_csv > load_hollywood.sql
psql "$PGDATABASE" -f load_hollywood.sql
```

The generated script sets:

```sql
SET max_parallel_workers_per_gather = 0;
```

This matches the paper experiments, where parallel workers were disabled for
PostgreSQL planner comparability and compatibility with learned cost models.

After loading, collect full-query estimates and raw plan JSON:

```bash
python scripts/run_postgres_workload.py --workload all --conn "$PGDATABASE"
```

Outputs are written to `results/postgres_full_query/`.

## Paper Bundle

`experiments/paper_results/` contains raw rows, summaries, plots, and a manifest
for the final 200K balanced workload. Use those CSVs as the source for paper
tables and plot regeneration.
