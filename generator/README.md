# Generator And Exporter

This folder contains the Python generation pipeline and the strict IMDb exporter.

If you run the public generator in place, runtime outputs are created inside
this `generator/` directory. If you copy the generator snapshot elsewhere, that
copied directory becomes the generator workspace. In both cases, the completed
workspace is the directory passed as `--base-dir` to the public export script.

Main entry points:

- `run_pipeline.py`: end-to-end generation pipeline.
- `run_full_api_pipeline.py`: API-oriented orchestration wrapper.
- `llm_provider.py`: Gemini and OpenAI-compatible HTTP provider.
- `export_imdb_schema.py`: export a generated workspace to strict IMDb/JOB CSVs.
- `validate_imdb_schema.py`: validate strict CSV headers and core integrity.
- `build_duckdb_from_imdb_schema.py`: build DuckDB from exported CSVs.
- `run_job_queries_duckdb.py`: run a query directory against DuckDB.

Static JSON files in this folder are generation priors and seed banks. Runtime
outputs such as `entities/`, `graph/`, `_step100_resume/`, `imdb_schema/`, and
logs are generated locally and are not included in this repository.

For a path-by-path map of release files, generated outputs, and evaluation
results, see `../docs/artifacts.md`.

The default Gemini model for new runs is `gemini-3.1-flash-lite`. The released
Hollywood-200K dataset was generated with `gemini-3.1-flash-lite-preview`; see
`../docs/api_setup.md` for the current Google API and OpenAI-compatible
endpoint setup.
