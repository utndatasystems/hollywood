SELECT min(an.name) AS acress_pseudonym,
       min(t.title) AS japanese_anime_movie
FROM aka_name AS an,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note ='(editor)'
  AND cn.country_code ='[us]'
  AND mc.note like '%ide%'
  AND mc.note not like '%(USA)%'
  AND (mc.note like '%(2000) (Germany) (video)%'
       OR mc.note like '%(2000) (Germany) (video)%')
  AND n.name like '%Hala Mona El-Sayed%'
  AND n.name not like '%Yu%'
  AND rt.role ='editor'
  AND t.production_year BETWEEN 1999 AND 2000
  AND (t.title like 'Silence at Zone: Endgame%'
       OR t.title like 'Silence at Zone: Endgame%')
  AND an.person_id = n.id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id;
