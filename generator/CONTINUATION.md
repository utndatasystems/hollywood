# Mirage Continuation Runs

Mirage supports extending an existing Step100 workspace instead of starting a fresh movie run.

## Core idea

- Keep the existing `_step100_resume` plan, shards, manifest, and generated movie tables.
- Top up reusable entities to cumulative targets without renumbering existing rows.
- Enrich/generate latents only for new persons and companies.
- Rebuild graph/runtime artifacts against the larger entity pool.
- Append a new Step100 plan for the requested future year window.

## Example: 100K to 300K

Run from a folder that already contains the completed 100K workspace:

```bash
python run_pipeline.py \
  --n-movies 300000 \
  --start-year 2051 \
  --end-year 2100 \
  --n-persons 720000 \
  --n-companies 27000 \
  --n-keywords 36000 \
  --n-characters 5100000 \
  --n-titles 390000 \
  --mode research \
  --model gemini-3.1-flash-lite \
  --bootstrap-model gemini-3.1-flash-lite \
  --planning-model gemini-3.1-flash-lite \
  --bulk-artifact-model gemini-3.1-flash-lite \
  --force-scalable-graph \
  --skip-diagnostic-cold-edges \
  --benchmark-mode \
  --extend-step100 \
  --from-step 10 \
  --until-step 100
```

`--n-movies` is cumulative. In the example above, a 100K source run receives 200K additional movies in 2051-2100.

## Entity Top-Up Only

To top up one entity class manually:

```bash
python prepare_continuation_entities.py \
  --kind persons \
  --target-count 720000 \
  --extension-start-year 2051 \
  --extension-end-year 2100 \
  --mode research
```

Supported kinds: `persons`, `companies`, `keywords`, `characters`, `titles`, and `all`.

## Safety Notes

- Do not combine continuation with `--fresh`.
- Do not use `--reset-step100-resume` for continuation.
- Keep the source `_step100_resume` directory; it is the authoritative continuation state.
- The pipeline ignores stale completion checkpoints for continuation-sensitive steps and re-checks output counts before skipping.
