// 1. 查询影响全社会用电量的候选机制
MATCH (c:Entity)-[r:MAY_AFFECT]->(i:Entity {id:'ind_consumption'}) RETURN c.name,r.description;

// 2. 查询某个月的全部观测
MATCH (o:Observation)-[:AT_TIME]->(m:Entity {id:'month_2025-08'})
MATCH (o)-[:MEASURES]->(i:Entity) RETURN i.name,o.value,o.unit,o.yoy_pct;

// 3. 查询春节错位相关异常
MATCH (a:Entity)-[:CANDIDATE_CAUSE]->(c:Entity {id:'concept_holiday_shift'}) RETURN a.name,a.description;
