# Experiments

- `paper_results/`: compact final 200K result bundle with raw CSVs, summaries,
  plots, and manifest.
- `mscn/`: MSCN runner and finalization scripts used for cardinality and
  selected-plan cost experiments.
- `zeroshot/`: wrappers and summarizers for pretrained ZeroShot cost evaluation.

Large training labels, checkpoints, and model outputs are not duplicated here.
Use the scripts and manifests to regenerate them when needed.

Use `paper_results/` if you only need the final paper tables, plots, and raw
summary CSVs. Use `mscn/` and `zeroshot/` only if you are reproducing the
learned-model experiments with external code and checkpoints.

The copied paper bundle is intentionally compact. It is suitable for regenerating
paper tables and plots without committing full PostgreSQL traces, MSCN
checkpoints, or ZeroShot intermediate tensors.
