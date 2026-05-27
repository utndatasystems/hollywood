SELECT min(k.keyword) AS movie_keyword,
       min(n.name) AS actor_name,
       min(t.title) AS hero_movie
FROM cast_info AS ci,
     keyword AS k,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE k.keyword in ('Apex Velocity Systems', 'Apex Velocity Vision', 'Apex Vision', 'Apex Vision Creative', 'Apex Vision Dynamics', 'Apex Vision Flow', 'Apex Vision International', 'Apex Vision Labs')
  AND t.production_year > 1999
  AND k.id = mk.keyword_id
  AND t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mk.movie_id
  AND n.id = ci.person_id;
