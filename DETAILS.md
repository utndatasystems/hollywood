# Details

## Dataset

The bundled dataset is a strict IMDb/JOB-style export of the final Hollywood
200K generated workspace. It contains 21 core tables and is distributed as both
CSV and DuckDB.

## Query Workload

The release includes the final 213-query workload:

- 70 JOB-Light queries,
- 113 JOB queries,
- 30 JOB-Complex queries.

The workload preserves original query structure while rebinding literals to the
Hollywood data distribution.

## Exporter Fidelity

The exporter keeps the strict JOB schema as the canonical benchmark surface.
Generated richness that does not map naturally to IMDb/JOB is not forced into
core columns. Export audit files document preserved, derived, flattened, omitted,
and extra-only fields.

The genre repair exports primary and signal-backed secondary genres as separate
`movie_info` rows with `info_type_id = 3`, matching IMDb/JOB semantics.
