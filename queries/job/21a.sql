SELECT min(cn.name) AS company_name,
       min(lt.link) AS link_type,
       min(t.title) AS western_follow_up
FROM company_name AS cn,
     company_type AS ct,
     keyword AS k,
     link_type AS lt,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     movie_link AS ml,
     title AS t
WHERE cn.country_code !='[pl]'
  AND (cn.name LIKE '%Omni Group%'
       OR cn.name LIKE '%Omni Group%')
  AND ct.kind ='production companies'
  AND k.keyword ='arthouse-sensibility-adventurous-hidden treasure'
  AND lt.link LIKE '%follows%'
  AND mc.note IS NULL
  AND mi.info IN ('Action', 'R', 'Comedy', 'Romance', 'Thriller', 'TV-MA', 'Spanish', 'Several scenes were improvised by the cast during filming.')
  AND t.production_year BETWEEN 1950 AND 2000
  AND lt.id = ml.link_type_id
  AND ml.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND t.id = mc.movie_id
  AND mc.company_type_id = ct.id
  AND mc.company_id = cn.id
  AND mi.movie_id = t.id
  AND ml.movie_id = mk.movie_id
  AND ml.movie_id = mc.movie_id
  AND mk.movie_id = mc.movie_id
  AND ml.movie_id = mi.movie_id
  AND mk.movie_id = mi.movie_id
  AND mc.movie_id = mi.movie_id;
