// 打开一个小规模的交互式知识图谱
MATCH p = (n:Entity)-[r]->(m:Entity)
RETURN p
LIMIT 100;

// 查询2024年7月的用电、天气与日历观测
MATCH (o:Observation)-[:AT_TIME]->(p:Entity {id: 'period-2024-07'})
MATCH (o)-[:MEASURES]->(i:Entity)
RETURN i.name, o.value, o.unit, o.quality_flag
ORDER BY i.name;

// 查询最近的三次产业季度累计增加值
MATCH (o:Observation)-[:MEASURES]->(i:Entity)
WHERE i.id STARTS WITH 'indicator-gdp-'
MATCH (o)-[:AT_TIME]->(p:Entity)
RETURN p.name, i.name, o.value, o.yoy_pct
ORDER BY p.name DESC
LIMIT 30;

// 查看温度特征对用电的先验关系
MATCH p = (w:Entity)-[r:AFFECTS]->(e:Entity {id: 'indicator-power-total'})
RETURN p;
