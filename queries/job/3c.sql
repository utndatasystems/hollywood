SELECT min(t.title) AS movie_title
FROM keyword AS k,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE k.keyword like '%fourth-wall-breaking-clash-of-cultures-red herring%'
  AND mi.info IN ('Amharic', 'Ethiopia', 'Macedonia', 'Macedonian', '53 min', 'Cuba', 'South Africa', 'Singapore', 'Puerto Rico', 'Sinhala')
  AND t.production_year > 1999
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi.movie_id
  AND k.id = mk.keyword_id;
