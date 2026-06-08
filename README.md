# Hollywood Benchmark

Hollywood is a synthetic IMDb-compatible benchmark for cardinality and cost
estimation experiments. This release contains:

- a public Hollywood 200K dataset in the strict 21-table JOB/IMDb schema,
- a DuckDB database built from the same CSV tables,
- 213 adapted SQL queries: JOB-Light, JOB, and JOB-Complex,
- the Python generator and IMDb exporter,
- small scripts for validation, DuckDB setup, PostgreSQL loading, and paper
  result reproduction.

The dataset and queries are under `DATASET_LICENSE`. The code is under `LICENSE`.

In this release, `Hollywood 200K` means `200,000` primary movies
(`title.kind_id = 1`). The full `title` table contains `351,455` rows because
the IMDb schema also stores other title kinds in the same table. Exact table
counts are listed in `data/export_manifest.json`.

## Layout

```text
data/                 Strict IMDb CSVs, DuckDB database, export reports
queries/              Final 213 Hollywood SQL queries
generator/            Synthetic generation and IMDb export code
scripts/              Public setup, validation, and workload helpers
experiments/          Paper result CSVs/plots and model-evaluation wrappers
docs/                 Setup, artifact-location, and reproducibility notes
```

## Choose A Starting Point

- Use `data/` and `queries/` if you only need the released benchmark.
- Use `scripts/` if you want to validate the release, rebuild DuckDB, or run
  workloads.
- Use `generator/` if you want to generate a new dataset or export a completed
  generator workspace.
- Use `experiments/paper_results/` if you want the final paper CSVs and plots.
- Use `docs/artifacts.md` if you need a path-by-path map of release files and
  generated outputs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -c constraints.txt
```

Use Python 3.10 or newer. If the repository was cloned with Git LFS pointers,
run:

```bash
git lfs pull
```

MSCN and ZeroShot reproduction need external code and ML dependencies. See
`requirements-mscn.txt`, `requirements-zeroshot.txt`, and the notes in `docs/`.

## Validate The Release

```bash
python scripts/validate_release.py
python scripts/smoke_test.py
```

Expected query counts are `JOB-Light: 70`, `JOB: 113`, `JOB-Complex: 30`.
`validate_release.py` also verifies `data/checksums.sha256`; use
`--skip-checksums` only for a quick layout check.
For a deeper FK/genre integrity check, run `tools/check_export_integrity.py`
as shown in `docs/schema.md`.

## DuckDB

The ready-to-use database is:

```text
data/hollywood_200k.duckdb
```

Run all workloads:

```bash
python scripts/run_duckdb_workload.py --workload all
```

The default DuckDB command is read-only. Add `--analyze` only when you want to
refresh DuckDB statistics and are willing to open the DB read-write.

Rebuild DuckDB from CSV:

```bash
python scripts/build_duckdb.py --csv-dir data/imdb_csv --out data/hollywood_200k.duckdb --overwrite
```

## PostgreSQL

Generate a `psql` load script:

```bash
python scripts/print_postgres_load_sql.py --csv-dir /absolute/path/to/data/imdb_csv > load_hollywood.sql
psql "$PGDATABASE" -f load_hollywood.sql
```

The generated script disables PostgreSQL parallel workers for planner
comparability and creates the JOB indexes from `generator/imdb_job_contract.py`.

Collect PostgreSQL full-query cardinality estimates and raw JSON plans:

```bash
python scripts/run_postgres_workload.py --workload all --conn "$PGDATABASE"
```

## Generate A New Dataset

Generation requires an LLM backend. The published Hollywood-200K dataset was
generated with `gemini-3.1-flash-lite-preview`. New runs in this release
default to `gemini-3.1-flash-lite`.

If you run the bundled generator in place, runtime outputs are written under
`generator/`. See `docs/artifacts.md` for the workspace layout and the expected
export outputs.

Google Gemini is the default supported API:

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY=...
python generator/smoke_gemini_provider.py
```

An OpenAI-compatible HTTP backend is also supported. The repository does not
execute model weights itself; start vLLM, Ollama, TGI, LiteLLM, the OpenAI
API, or another compatible service separately and point the generator at its
`/v1` endpoint:

```text
LLM_PROVIDER=local
LOCAL_LLM_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=<served model name>
LOCAL_LLM_API_KEY=not-needed
```

Small local run:

```bash
python generator/run_pipeline.py --profile tiny10_1y --fresh --benchmark-mode --from-step 4 --until-step 100
```

Full generation through the public IMDb export now produces both strict CSVs
and a DuckDB database at `generator/imdb_schema/imdb.duckdb`:

```bash
python generator/run_pipeline.py --profile tiny10_1y --fresh --benchmark-mode --from-step 4 --until-step 130
```

Export a completed generator workspace to strict IMDb CSVs:

```bash
python scripts/export_strict_imdb.py \
  --base-dir /path/to/generated_workspace \
  --out-dir exports/imdb_csv \
  --company-country-policy imdb-skewed
```

Then build DuckDB for that exported folder:

```bash
python scripts/build_duckdb.py --csv-dir exports/imdb_csv --out exports/hollywood.duckdb --overwrite
```

See `docs/generation.md` and `docs/api_setup.md`.

## Paper Results

The compact final 200K result bundle is under:

```text
experiments/paper_results/
```

It contains raw CSVs, grouped summaries, plots, and a manifest for the final
balanced 213-query workload. MSCN and ZeroShot reproduction notes are in
`docs/mscn.md` and `docs/zeroshot.md`.

## Large Files

The DuckDB file, dataset ZIP, and individual CSV tables larger than GitHub's
100 MB file limit are stored with Git LFS. Smaller CSV tables are committed as
regular repository files. `.gitattributes` is already configured for this split.

## Provenance

Adapted SQL workloads are derived from established JOB-family query sets. See
`docs/licensing.md` and `third_party/JOB/`. Generation model provenance and
current backend setup are documented in `docs/api_setup.md`.
