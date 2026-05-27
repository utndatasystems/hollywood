# Generation

The generator can run from tiny smoke profiles to larger benchmark profiles.
Profiles live in `generator/local_run_profiles/`.

If you run the generator from this public repository, runtime outputs are
written under `generator/`. See `artifacts.md` for the generated directory
layout and the public-export boundary.

## Choose An LLM Backend

Copy the example environment file and choose either Gemini or a local
OpenAI-compatible server:

```bash
cp .env.example .env
```

For Gemini, set:

```text
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<your key>
```

The released 200K dataset was generated with
`gemini-3.1-flash-lite-preview`. That preview name is no longer the public
default, so new runs use `gemini-3.1-flash-lite` unless you override the model
role variables.

For an OpenAI-compatible HTTP backend, set:

```text
LLM_PROVIDER=local
LOCAL_LLM_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=<served model name>
LOCAL_LLM_API_KEY=not-needed
```

Connectivity checks:

```bash
python generator/smoke_gemini_provider.py
python generator/check_local_llm.py --model <served model name>
```

## Tiny Smoke Run

```bash
python generator/run_pipeline.py --profile tiny10_1y --fresh --benchmark-mode --from-step 4 --until-step 100
```

## Candidate Runs

Examples:

```bash
python generator/run_pipeline.py --profile candidate20k --fresh --force-scalable-graph --benchmark-mode --from-step 4 --until-step 100
python generator/run_pipeline.py --profile candidate100k_2000_2050 --fresh --force-scalable-graph --benchmark-mode --from-step 4 --until-step 100
```

Large runs write a generator workspace with entity tables, graph artifacts, LLM
usage logs, and checkpoint directories. Keep that workspace outside `data/` and
export only the strict IMDb tables for public release.

Typical runtime locations inside an in-place run are:

- `generator/entities/` for generated entity tables and seed artifacts.
- `generator/graph/` for graph/runtime artifacts.
- `generator/_step100_resume/` for continuation state.
- `generator/*.arrow` for assembled tables consumed by the exporter.

## Export

Running the generator through step 130 now writes both the strict IMDb CSVs and
`generator/imdb_schema/imdb.duckdb`:

```bash
python generator/run_pipeline.py --profile tiny10_1y --fresh --benchmark-mode --from-step 4 --until-step 130
```

Export a completed workspace to strict IMDb CSVs without running the full
pipeline:

```bash
python scripts/export_strict_imdb.py \
  --base-dir /path/to/generated_workspace \
  --out-dir exports/imdb_csv \
  --company-country-policy imdb-skewed
```

The public wrapper always calls the underlying exporter in strict JOB mode and
omits research-only extra tables, so the output stays at the 21 public IMDb/JOB
tables. `--company-country-policy imdb-skewed` maps legacy-uniform generated
companies to an IMDb-like country distribution.

Build DuckDB for an exported CSV folder:

```bash
python scripts/build_duckdb.py --csv-dir exports/imdb_csv --out exports/hollywood.duckdb --overwrite
```
