import duckdb
import pytest
from unittest.mock import MagicMock, patch
from agent import SQLAgent, HybridAgent, QueryIntent

@pytest.fixture
def agent(tmp_path):
    """Fixture to provide a SQLAgent with a temporary test database."""
    db_path = tmp_path / "test_election.db"
    # Using DuckDB as per your get_answer implementation
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE turnout (id INTEGER, region TEXT)")
        conn.execute("INSERT INTO turnout VALUES (1, 'AGNEBY')")
    
    return SQLAgent(db_path=str(db_path))

def test_relevance_blocker_invalid_intent(agent):
    """Test that non-election queries are caught by the intent classifier."""
    # Mock _get_intent to return INVALID
    with patch.object(agent, '_get_intent', return_value=QueryIntent.INVALID):
        response = agent.get_answer("What is the weather in Paris?")
        
        # Check if the standardized 'not found' format is triggered
        assert "Invalid Intent" in response["content"] or "not found" in str(response).lower()

def test_sql_injection_guardrail(agent):
    """Ensure the sealed forbidden list blocks destructive commands."""
    # Test keywords from your sealed __dict__ list
    malicious_queries = [
        "DELETE FROM turnout",
        "DROP TABLE turnout",
        "UPDATE turnout SET region='PARIS'",
        "INSERT INTO turnout (region) VALUES ('NEW')"
    ]
    
    for query in malicious_queries:
        assert agent.validate_sql(query) is False, f"Failed to block: {query}"

def test_read_only_enforcement(agent):
    """Verify DuckDB read_only=True enforcement works at the connection level."""
    # This mimics the 'with duckdb.connect(..., read_only=True)' in your get_answer
    with duckdb.connect(agent.db_path, read_only=True) as conn:
        with pytest.raises(duckdb.PermissionException):
            conn.execute("INSERT INTO turnout (region) VALUES ('SUD-COMOE')")

def test_sealed_attribute_persistence(agent):
    """Verify that even a clever dev cannot modify the forbidden list at runtime."""
    with pytest.raises(AttributeError):
        agent.forbidden = ["SELECT"] # Should fail via __setattr__ override
    
    with pytest.raises(AttributeError):
        agent._forbidden = ("DROP",) # Should also fail


# Setup a temporary test database
@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test_election.db")
    with duckdb.connect(db_path) as conn:
        conn.execute("CREATE TABLE results (candidate TEXT, votes INTEGER, region TEXT)")
        conn.execute("INSERT INTO results VALUES ('Alice', 1200, 'Abidjan'), ('Bob', 800, 'Abidjan')")
    return db_path

# Security Test: Is the forbidden list actually sealed?
def test_agent_security_lock():
    agent = SQLAgent()
    with pytest.raises(AttributeError):
        agent.forbidden = ["SELECT"]  # Attempted override should fail
    assert "DROP" in agent.forbidden

# Logic Test: Does the SQLAgent catch forbidden keywords in strings?
def test_sql_validation_logic():
    agent = SQLAgent()
    malicious_sql = "SELECT * FROM results; DROP TABLE results;"
    assert agent.validate_sql(malicious_sql) is False
    assert agent.validate_sql("SELECT * FROM results") is True

# Routing Test: Does HybridAgent send analytics to SQL?
def test_hybrid_routing_logic(test_db):
    hybrid = HybridAgent()
    # We mock the _get_intent to avoid hitting the LLM for every unit test
    hybrid._get_intent = lambda x: QueryIntent.AGGREGATION
    
    # Test if it routes to the sql_expert
    result = hybrid.route_query("Total votes in Abidjan")
    assert result["intent"] == QueryIntent.AGGREGATION

def test_rag_path_isolation(rag_agent):
    """Ensure RAGAgent uses its retrieval tool, not SQL tools."""
    with patch.object(rag_agent, 'search_election_documents') as mock_search:
        mock_search.return_value = "Retrieved law snippet."
        rag_agent.process_query("What is the turnout in San Pedro?", QueryIntent.GENERAL)
        mock_search.assert_called_once()
