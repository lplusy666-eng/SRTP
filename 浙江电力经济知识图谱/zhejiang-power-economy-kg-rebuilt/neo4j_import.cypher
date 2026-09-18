// 将 nodes.csv、observations.csv、relationships.csv 放入 Neo4j 的 import 目录后执行。
// 第三段使用 APOC 创建动态关系类型，请先安装并启用 APOC。

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {id: row.id})
SET n.label = row.label,
    n.name = row.name,
    n.category = row.category,
    n.description = row.description,
    n.unit = row.unit,
    n.frequency = row.frequency,
    n.source_id = row.source_id,
    n.quality_flag = row.quality_flag,
    n.notes = row.notes;

LOAD CSV WITH HEADERS FROM 'file:///observations.csv' AS row
MERGE (o:Entity:Observation {id: row.id})
SET o.label = row.label,
    o.name = row.name,
    o.category = row.category,
    o.value = toFloat(row.value),
    o.unit = row.unit,
    o.yoy_pct = CASE WHEN row.yoy_pct = '' THEN null ELSE toFloat(row.yoy_pct) END,
    o.frequency = row.frequency,
    o.quality_flag = row.quality_flag,
    o.notes = row.notes;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (s:Entity {id: row.source_id}), (t:Entity {id: row.target_id})
CALL apoc.create.relationship(
  s,
  row.type,
  {
    id: row.id,
    description: row.description,
    direction: row.direction,
    lag: row.lag,
    confidence: row.confidence,
    provenance: row.provenance
  },
  t
) YIELD rel
RETURN count(rel) AS imported_relationships;
