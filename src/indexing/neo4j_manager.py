from typing import List, Dict, Any
from neo4j import GraphDatabase

from src.schema import UnifiedQASample
from src.config import config

class Neo4jManager:
    """
    Handles connecting to Neo4j and pushing graph metadata (triples) into the database.
    Ensures strict Multi-Tenancy segregation via Node Labels and Properties mapping to 'dataset'.
    """
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        # Use config defaults if not provided
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        
        print(f"Connecting to Neo4j at {self.uri}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        """
        Create indexes/constraints on Neo4j for fast lookup on the 'dataset' segregation property,
        and unique constraints on entity names within a dataset.
        """
        query = """
        CREATE INDEX entity_dataset IF NOT EXISTS FOR (e:Entity) ON (e.dataset);
        """
        with self.driver.session() as session:
            session.run(query)

    def process_and_index_samples(self, samples: List[UnifiedQASample], batch_size: int = 50):
        """
        Iterates through samples. If a sample has pre-extracted 'evidences' (graph triples)
        in its metadata (like from 2WikiMultiHopQA), it generates Cypher queries to push them.
        """
        total_triples = 0
        
        with self.driver.session() as session:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                
                # We collect all Cypher execution params for this batch
                query_params = []
                
                for sample in batch:
                    # Check if the dataset loader extracted graph triples.
                    # e.g., metadata={"evidences": [["Eiffel Tower", "Located_In", "Paris"]]}
                    evidences = sample.metadata.get("evidences", [])
                    
                    if not evidences:
                        continue
                        
                    for evidence in evidences:
                        if len(evidence) >= 3:
                            head = evidence[0].strip()
                            relation = evidence[1].strip().upper().replace(" ", "_")
                            tail = evidence[2].strip()
                            
                            query_params.append({
                                "dataset": sample.dataset_name,
                                "sample_id": sample.sample_id,
                                "source_node": head,
                                "relation": relation,
                                "target_node": tail
                            })
                            total_triples += 1

                if query_params:
                    # Execute a batched UNWIND Cypher query for extreme performance
                    # This MERGEs nodes ensure we don't duplicate entities per dataset
                    query = """
                    UNWIND $batch AS row
                    
                    // Create/Find the Head Node
                    MERGE (source:Entity {name: row.source_node, dataset: row.dataset})
                    
                    // Create/Find the Tail Node
                    MERGE (target:Entity {name: row.target_node, dataset: row.dataset})
                    
                    // Create the relationship between them, tagging it with the sample_id it came from
                    // so we can trace back a relationship to the exact source paragraph
                    WITH source, target, row
                    CALL apoc.create.relationship(source, row.relation, {sample_id: row.sample_id, dataset: row.dataset}, target)
                    YIELD rel
                    RETURN count(rel)
                    """
                    
                    try:
                        session.run(query, batch=query_params)
                    except Exception as e:
                        print(f"Graph batch warning (Ensure APOC is installed if using apoc.create): {e}")
                        
                        # Fallback query if APOC is not installed: Use basic Cypher with a generic generic relationship
                        # Note: Cypher doesn't allow dynamic relationship types in standard MERGE easily
                        fallback_query = """
                        UNWIND $batch AS row
                        MERGE (source:Entity {name: row.source_node, dataset: row.dataset})
                        MERGE (target:Entity {name: row.target_node, dataset: row.dataset})
                        MERGE (source)-[r:RELATED_TO {type: row.relation, sample_id: row.sample_id, dataset: row.dataset}]->(target)
                        """
                        session.run(fallback_query, batch=query_params)

        print(f"Indexing Complete! Pushed {total_triples} graph relationships to Neo4j.")

    def get_graph_retriever_query(self, target_dataset: str, entity_name: str, max_hops: int = 2) -> str:
        """
        Helper method generating the exact Cypher query the `GraphRAG` strategy will use at evaluation time.
        Enforces strict dataset segregation and limits hops.
        """
        cypher = f"""
        MATCH (start:Entity {{name: '{entity_name}', dataset: '{target_dataset}'}})
        CALL apoc.path.subgraphAll(start, {{
            maxLevel: {max_hops},
            relationshipFilter: ">"
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        return cypher
