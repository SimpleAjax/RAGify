import pytest
from unittest.mock import MagicMock, patch

from src.schema import UnifiedQASample
from src.indexing.neo4j_manager import Neo4jManager

@patch('src.indexing.neo4j_manager.GraphDatabase')
def test_neo4j_manager_initialization(mock_gdb):
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver
    
    manager = Neo4jManager(uri="bolt://test:7687")
    
    # Verify driver is created and constraints are ensured
    mock_gdb.driver.assert_called_once_with("bolt://test:7687", auth=("neo4j", "password"))
    
    # Verify the session was started to run the constraint query
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_session.run.assert_called_once()
    args, _ = mock_session.run.call_args
    assert "CREATE INDEX entity_dataset" in args[0]


@patch('src.indexing.neo4j_manager.GraphDatabase')
def test_neo4j_manager_process_and_index(mock_gdb):
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    manager = Neo4jManager()
    
    # Mock dataset sample WITH graph evidences (e.g. 2WikiMultiHopQA format)
    sample = UnifiedQASample(
        sample_id="wiki_123",
        dataset_name="2wiki",
        query="Who is the director of Titanic?",
        ground_truth_answer="James Cameron",
        supporting_contexts=[],
        corpus=[],
        metadata={
            "evidences": [
                ["Titanic", "directed_by", "James Cameron"]
            ]
        }
    )
    
    # Run method
    manager.process_and_index_samples([sample])
    
    # Ensure it ran a batch query
    # Initialization ran 1 query, process ran 1 more
    assert mock_session.run.call_count == 2
    
    # Check the args of the last run (the ingestion)
    args, kwargs = mock_session.run.call_args
    query = args[0]
    params = kwargs.get("batch")
    
    assert "UNWIND $batch AS row" in query
    
    # Verify the parameters were mapped correctly
    assert len(params) == 1
    p = params[0]
    
    # Strict metadata tagging checks
    assert p["dataset"] == "2wiki"
    assert p["sample_id"] == "wiki_123"
    assert p["source_node"] == "Titanic"
    assert p["relation"] == "DIRECTED_BY"
    assert p["target_node"] == "James Cameron"


@patch('src.indexing.neo4j_manager.GraphDatabase')
def test_neo4j_get_retriever_query(mock_gdb):
    manager = Neo4jManager()
    
    cypher = manager.get_graph_retriever_query(target_dataset="test_data", entity_name="Apple Inc", max_hops=3)
    
    assert "Apple Inc" in cypher
    assert "test_data" in cypher
    assert "maxLevel: 3" in cypher
    assert "apoc.path.subgraphAll" in cypher
