# Artifact Locations

This note maps the files and directories that matter in the public release and
in a local generation run.

## Release Files

These paths are part of the public submission package.

- `data/imdb_csv/`: strict 21-table JOB/IMDb CSV export with headers.
- `data/hollywood_200k.duckdb`: DuckDB database built from the strict CSVs.
- `data/export_manifest.json`: row counts and export metadata for the released
  dataset.
- `data/genre_derivation_summary.json`: genre-row summary for the strict export.
- `data/source_export_coverage.csv`: source-to-export field mapping report.
- `data/checksums.sha256`: checksums for the bundled data artifacts.
- `queries/job_light/`, `queries/job/`, `queries/job_complex/`: final 213
  adapted SQL queries.
- `experiments/paper_results/`: final paper-facing CSVs, summaries, plots, and
  manifest.
- `third_party/JOB/`: copied original JOB schema/index/provenance files.

## Hollywood 200K Naming

`Hollywood 200K` refers to `200,000` primary movies, identified in the strict
IMDb schema as rows with `title.kind_id = 1`.

The full `title` table is larger than `200,000` rows because the same table
also stores non-primary title kinds such as TV series and episodes. The exact
released counts are recorded in `data/export_manifest.json`.

## Generator Workspace

If you run `python generator/run_pipeline.py` from this repository, runtime
outputs are created inside `generator/`. The same code can also be copied to a
separate workspace and run there. In both cases, the completed workspace is the
directory passed as `--base-dir` to `scripts/export_strict_imdb.py`.

The main generated locations are:

- `generator/entities/`: generated entity CSV/JSON files such as persons,
  companies, keywords, and title-bank artifacts.
- `generator/graph/`: graph/runtime artifacts created before movie assembly.
- `generator/_step100_resume/`: continuation state for step-100 movie assembly.
- `generator/*.arrow`: primary assembled tables used by the exporter, for
  example `movie.arrow`, `cast_info.arrow`, `movie_companies.arrow`, and
  related tables.
- `generator/imdb_schema/`: strict export written when the exporter is run from
  inside the generator workspace.

Static files that are already committed in `generator/`, such as
`world_policy.json`, `concept_packs.json`, and the various seed banks, are code
inputs and priors. They are not runtime outputs.

## Files Needed For Public Export

Do not assemble the public export by selecting individual workspace files by
hand. Use:

```bash
python scripts/export_strict_imdb.py --base-dir /path/to/workspace --out-dir exports/imdb_csv
```

That command converts a completed workspace into the strict public 21-table
bundle. The public release should contain the exported strict CSVs and any
derived packaged artifacts such as DuckDB and checksums, not the full generator
workspace.

## Evaluation Outputs

These directories are created only after you run the evaluation scripts.

- `reports/`: small connectivity or validation reports written by helper
  scripts. For example, `python generator/check_local_llm.py` writes
  `reports/local_llm_check.json` when run from the repository root with default
  arguments.
- `results/duckdb/`: CSV outputs from `scripts/run_duckdb_workload.py`.
- `results/postgres_full_query/`: CSV outputs and raw JSON plans from
  `scripts/run_postgres_workload.py`.
- `audit/export_integrity/`: integrity reports from
  `tools/check_export_integrity.py`.
