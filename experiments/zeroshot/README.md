# ZeroShot

This folder contains the public wrapper and summary scripts for evaluating the
external pretrained ZeroShot cost model on Hollywood PostgreSQL selected-plan
traces.

First collect lcm-eval compatible raw PostgreSQL traces:

```powershell
python experiments\zeroshot\collect_lcm_raw_traces.py `
  --all-final `
  --out-dir results\zeroshot_lcm_raw `
  --db-name hollywood_200k `
  --pg-container pg_bench `
  --pg-user postgres `
  --timeout-sec 600 `
  --disable-memoize
```

Then parse those traces with the upstream lcm-eval preprocessing pipeline.

Inputs expected by `run_pretrained_zeroshot_hollywood.ps1`:

- `LcmSrc`: checkout of the external lcm-eval/ZeroShot source tree.
- `InputRoot`: parsed Hollywood selected-plan inputs produced by lcm-eval.
- `ModelDir`: pretrained IMDb ZeroShot checkpoint directory.
- `OutRoot`: output directory for prediction CSVs.

Example:

```powershell
.\experiments\zeroshot\run_pretrained_zeroshot_hollywood.ps1 `
  -Python python `
  -LcmSrc C:\path\to\lcm-eval\src `
  -InputRoot C:\path\to\hollywood_lcm_inputs `
  -ModelDir C:\path\to\zeroshot\imdb `
  -OutRoot runs\zeroshot_hollywood_200k `
  -Device cpu
```

PyTorch checkpoint loading is pickle-like. Only load checkpoints from sources
you trust.
