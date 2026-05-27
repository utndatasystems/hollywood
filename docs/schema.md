# Schema

The public dataset uses the strict JOB/IMDb 21-table schema:

```text
aka_name, aka_title, cast_info, char_name, comp_cast_type, company_name,
company_type, complete_cast, info_type, keyword, kind_type, link_type,
movie_companies, movie_info, movie_info_idx, movie_keyword, movie_link, name,
person_info, role_type, title
```

The column contract and DuckDB/PostgreSQL type mapping are defined in:

```text
generator/imdb_job_contract.py
```

Validate headers and core artifacts:

```bash
python scripts/validate_release.py
```

Run the deeper DuckDB integrity audit:

```bash
python tools/check_export_integrity.py --db data/hollywood_200k.duckdb --out-dir audit/export_integrity
```

The exporter also writes small audit files in `data/`:

- `source_export_coverage.csv`: preserved, derived, flattened, omitted, and
  extra-only fields.
- `genre_derivation_summary.json`: genre row generation summary.
- `export_manifest.json`: final export metadata.

## Optional TV And Episode Summaries

The generation pipeline can create TV-series summaries and episode descriptions,
and the strict exporter maps them to `movie_info` rows when they are present in a
generated workspace. The released Hollywood 200K artifact intentionally does not
add those optional rows after benchmark labeling, because extra unconstrained
`movie_info` rows would change cardinalities for some JOB-family queries. Movie
plot/tagline/trivia rows that were part of the labeled release are unchanged.
