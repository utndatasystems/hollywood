"""Shared JOB/IMDB core schema contract for v16."""

from __future__ import annotations

from collections import OrderedDict

JOB_TABLE_COLUMNS = OrderedDict([
    ("title", [
        "id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
        "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum",
    ]),
    ("name", [
        "id", "name", "imdb_index", "imdb_id", "gender",
        "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum",
    ]),
    ("cast_info", [
        "id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id",
    ]),
    ("char_name", [
        "id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum",
    ]),
    ("company_name", [
        "id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum",
    ]),
    ("movie_companies", [
        "id", "movie_id", "company_id", "company_type_id", "note",
    ]),
    ("movie_keyword", [
        "id", "movie_id", "keyword_id",
    ]),
    ("keyword", [
        "id", "keyword", "phonetic_code",
    ]),
    ("movie_link", [
        "id", "movie_id", "linked_movie_id", "link_type_id",
    ]),
    ("aka_title", [
        "id", "movie_id", "title", "imdb_index", "kind_id", "production_year",
        "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note", "md5sum",
    ]),
    ("aka_name", [
        "id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum",
    ]),
    ("movie_info", [
        "id", "movie_id", "info_type_id", "info", "note",
    ]),
    ("movie_info_idx", [
        "id", "movie_id", "info_type_id", "info", "note",
    ]),
    ("person_info", [
        "id", "person_id", "info_type_id", "info", "note",
    ]),
    ("info_type", ["id", "info"]),
    ("kind_type", ["id", "kind"]),
    ("role_type", ["id", "role"]),
    ("company_type", ["id", "kind"]),
    ("link_type", ["id", "link"]),
    ("complete_cast", ["id", "movie_id", "subject_id", "status_id"]),
    ("comp_cast_type", ["id", "kind"]),
])

JOB_CORE_TABLES = list(JOB_TABLE_COLUMNS.keys())

JOB_TABLE_TYPES = {
    "title": {
        "id": "INT", "title": "TEXT", "imdb_index": "TEXT", "kind_id": "INT", "production_year": "INT",
        "imdb_id": "INT", "phonetic_code": "TEXT", "episode_of_id": "INT", "season_nr": "INT",
        "episode_nr": "INT", "series_years": "TEXT", "md5sum": "TEXT",
    },
    "name": {
        "id": "INT", "name": "TEXT", "imdb_index": "TEXT", "imdb_id": "INT", "gender": "TEXT",
        "name_pcode_cf": "TEXT", "name_pcode_nf": "TEXT", "surname_pcode": "TEXT", "md5sum": "TEXT",
    },
    "cast_info": {
        "id": "INT", "person_id": "INT", "movie_id": "INT", "person_role_id": "INT", "note": "TEXT",
        "nr_order": "INT", "role_id": "INT",
    },
    "char_name": {
        "id": "INT", "name": "TEXT", "imdb_index": "TEXT", "imdb_id": "INT", "name_pcode_nf": "TEXT",
        "surname_pcode": "TEXT", "md5sum": "TEXT",
    },
    "company_name": {
        "id": "INT", "name": "TEXT", "country_code": "TEXT", "imdb_id": "INT", "name_pcode_nf": "TEXT",
        "name_pcode_sf": "TEXT", "md5sum": "TEXT",
    },
    "movie_companies": {
        "id": "INT", "movie_id": "INT", "company_id": "INT", "company_type_id": "INT", "note": "TEXT",
    },
    "movie_keyword": {"id": "INT", "movie_id": "INT", "keyword_id": "INT"},
    "keyword": {"id": "INT", "keyword": "TEXT", "phonetic_code": "TEXT"},
    "movie_link": {"id": "INT", "movie_id": "INT", "linked_movie_id": "INT", "link_type_id": "INT"},
    "aka_title": {
        "id": "INT", "movie_id": "INT", "title": "TEXT", "imdb_index": "TEXT", "kind_id": "INT",
        "production_year": "INT", "phonetic_code": "TEXT", "episode_of_id": "INT", "season_nr": "INT",
        "episode_nr": "INT", "note": "TEXT", "md5sum": "TEXT",
    },
    "aka_name": {
        "id": "INT", "person_id": "INT", "name": "TEXT", "imdb_index": "TEXT", "name_pcode_cf": "TEXT",
        "name_pcode_nf": "TEXT", "surname_pcode": "TEXT", "md5sum": "TEXT",
    },
    "movie_info": {"id": "INT", "movie_id": "INT", "info_type_id": "INT", "info": "TEXT", "note": "TEXT"},
    "movie_info_idx": {"id": "INT", "movie_id": "INT", "info_type_id": "INT", "info": "TEXT", "note": "TEXT"},
    "person_info": {"id": "INT", "person_id": "INT", "info_type_id": "INT", "info": "TEXT", "note": "TEXT"},
    "info_type": {"id": "INT", "info": "TEXT"},
    "kind_type": {"id": "INT", "kind": "TEXT"},
    "role_type": {"id": "INT", "role": "TEXT"},
    "company_type": {"id": "INT", "kind": "TEXT"},
    "link_type": {"id": "INT", "link": "TEXT"},
    "complete_cast": {"id": "INT", "movie_id": "INT", "subject_id": "INT", "status_id": "INT"},
    "comp_cast_type": {"id": "INT", "kind": "TEXT"},
}

JOB_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_title_kind_year ON title(kind_id, production_year)",
    "CREATE INDEX IF NOT EXISTS idx_cast_info_movie ON cast_info(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_cast_info_person ON cast_info(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_cast_info_role ON cast_info(role_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_companies_movie ON movie_companies(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_companies_company ON movie_companies(company_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_companies_type ON movie_companies(company_type_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_keyword_movie ON movie_keyword(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_keyword_keyword ON movie_keyword(keyword_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_movie ON movie_info(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_type ON movie_info(info_type_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_idx_movie ON movie_info_idx(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_idx_type ON movie_info_idx(info_type_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_info_person ON person_info(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_info_type ON person_info(info_type_id)",
    "CREATE INDEX IF NOT EXISTS idx_aka_title_movie ON aka_title(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_aka_name_person ON aka_name(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_link_movie ON movie_link(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_link_linked ON movie_link(linked_movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_link_type ON movie_link(link_type_id)",
    "CREATE INDEX IF NOT EXISTS idx_complete_cast_movie ON complete_cast(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_complete_cast_subject ON complete_cast(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_complete_cast_status ON complete_cast(status_id)",
]
