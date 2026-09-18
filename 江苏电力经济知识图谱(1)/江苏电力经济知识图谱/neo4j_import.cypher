// 将CSV复制到Neo4j import目录后执行
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {id: row.node_id})
SET n.name=row.name, n.type=row.label, n.category=row.category,
    n.description=row.description, n.source_url=row.source_url, n.quality_flag=row.quality_flag;

LOAD CSV WITH HEADERS FROM 'file:///observations.csv' AS row
MERGE (o:Observation {id: row.observation_id})
SET o.period=row.period, o.region=row.region, o.value=toFloat(row.value),
    o.unit=row.unit, o.yoy_pct=CASE WHEN row.yoy_pct='' THEN null ELSE toFloat(row.yoy_pct) END,
    o.scope=row.scope, o.source_url=row.source_url, o.quality_flag=row.quality_flag;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (a {id: row.source_id}), (b {id: row.target_id})
CALL apoc.create.relationship(a,row.relationship_type,
 {description:row.description,evidence_level:row.evidence_level,weight:row.weight},b) YIELD rel
RETURN count(rel);
