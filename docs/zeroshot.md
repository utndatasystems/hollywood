# ZeroShot Cost Model

ZeroShot is an external pretrained cost-estimation model. This repository
includes only the wrapper and summarization scripts used for the Hollywood
experiments:

```text
experiments/zeroshot/
```

This repository includes:

- the PostgreSQL trace collector and summarization scripts under
  `experiments/zeroshot/`,
- the final paper-facing ZeroShot CSV summaries under
  `experiments/paper_results/`,
- the Hollywood dataset and query files used to produce PostgreSQL plans.

This repository does not include:

- the upstream ZeroShot or `lcm-eval` source tree,
- the pretrained model checkpoint,
- the parsed LCM input tensors produced by the upstream preprocessing pipeline.

To reproduce ZeroShot cost results:

1. Extract PostgreSQL selected-plan traces with
   `experiments/zeroshot/collect_lcm_raw_traces.py`.
2. Prepare inputs in the expected LCM/ZeroShot format with the upstream
   lcm-eval preprocessing pipeline.
3. Run `experiments/zeroshot/run_pretrained_zeroshot_hollywood.ps1` or port the
   same command to your shell.
4. Summarize with `experiments/zeroshot/summarize_pretrained_zeroshot_outputs.py`.

The pretrained model itself and upstream licensing should be obtained from the
corresponding ZeroShot project.

Record the external source repository, commit, checkpoint identifier, and
preprocessing configuration used for a run. Those external versions are not
pinned in this release folder.
