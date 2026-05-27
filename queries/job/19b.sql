SELECT min(n.name) AS voicing_actress,
       min(t.title) AS kung_fu_panda
FROM aka_name AS an,
     char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note = '(voice: English version)'
  AND cn.country_code ='[us]'
  AND it.info = 'release dates'
  AND mc.note like '%(co-production)%'
  AND (mc.note like '%(co-production)%'
       OR mc.note like '%(co-production)%')
  AND mi.info IS NOT NULL
  AND (mi.info like '%to%'
       OR mi.info like 'Australia:17 August 2006%')
  AND n.gender ='m'
  AND n.name like '%Eun-woo Shin%'
  AND rt.role ='actor'
  AND t.production_year BETWEEN 2005 AND 2006
  AND t.title like '%The Faded Flame of Kingdom%'
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND mc.movie_id = mi.movie_id
  AND mi.movie_id = ci.movie_id
  AND cn.id = mc.company_id
  AND it.id = mi.info_type_id
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND n.id = an.person_id
  AND ci.person_id = an.person_id
  AND chn.id = ci.person_role_id;
