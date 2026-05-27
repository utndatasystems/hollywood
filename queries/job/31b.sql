SELECT min(mi.info) AS movie_budget,
       min(mi_idx.info) AS movie_votes,
       min(n.name) AS writer,
       min(t.title) AS violent_liongate_movie
FROM cast_info AS ci,
     company_name AS cn,
     info_type AS it1,
     info_type AS it2,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE ci.note in ('(as Dmitry J. Stepanov)',
                  '(as Dmitry J. Stepanov)',
                  '(as Dmitry J. Stepanov)',
                  '(as Dmitry J. Stepanov)',
                  '(as Dmitry J. Stepanov)')
  AND cn.name like '%Stu%'
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND k.keyword in ('vulnerable-period-piece',
                    'breakout-performance-earnest-psychological-study',
                    'addiction-recovery',
                    'introspective-isolated-community-period-piece',
                    'tragedy',
                    'melancholic-isolated-community-social-realism',
                    'love triangle-metropolitan')
  AND mc.note like '%(VHS)%'
  AND mi.info in ('Drama',
                  'Comedy')
  AND n.gender = 'm'
  AND t.production_year > 2000
  AND (t.title like '%Golden%'
       OR t.title like '%Blade%'
       OR t.title like 'The Golden%')
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND ci.movie_id = mi.movie_id
  AND ci.movie_id = mi_idx.movie_id
  AND ci.movie_id = mk.movie_id
  AND ci.movie_id = mc.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi_idx.movie_id = mk.movie_id
  AND mi_idx.movie_id = mc.movie_id
  AND mk.movie_id = mc.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND k.id = mk.keyword_id
  AND cn.id = mc.company_id;
