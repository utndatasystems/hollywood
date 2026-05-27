# Licensing And Provenance

## Repository Code

Code written for the Hollywood generator, exporter, loaders, and helper scripts
is released under the MIT license in `LICENSE`.

## Dataset And Queries

The Hollywood dataset and adapted SQL queries are released under the data license
in `DATASET_LICENSE`.

The adapted SQL workloads are derived from JOB, JOB-Light, and JOB-Complex query
families. Original JOB provenance files available locally are copied under:

```text
third_party/JOB/
```

The dataset license in this release is CC BY-NC 4.0.

## External Model Code

The MSCN and ZeroShot folders are reproducibility wrappers. Full reproduction
requires external upstream projects and checkpoints:

- learnedcardinalities / MSCN: check upstream MIT license.
- ZeroShot / lcm-eval: check upstream Apache-2.0 and project-specific terms.

Do not vendor external checkpoints into this repository unless their license and
redistribution terms are explicitly approved.
