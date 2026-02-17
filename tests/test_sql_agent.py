import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from src.db.sql_agent import SQLAgent

@pytest.fixture
def agent(tmp_path):
    # Create a temporary test database
    db_path = tmp_path / "test_election.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE turnout (id INTEGER, region TEXT)")
    conn.close()
    return SQLAgent(str(db_path))

def test_relevance_blocker_weather(agent):
    """Test that generic non-election queries are blocked early."""
    with patch.object(agent.client.chat.completions, 'create') as mock_create:
        # Mock the intent classifier returning 'NO'
        mock_create.return_value.choices[0].message.content = "NO"
        
        response = agent.run_query("What is the weather in Paris?")
        
        assert "Not found in the provided PDF dataset" in response
        assert "Topic not related" in agent._format_not_found("test", "Topic not related to election data.")

def test_sql_injection_guardrail(agent):
    """Ensure destructive SQL commands are blocked by validate_sql."""
    malicious_sql = "DELETE FROM turnout; SELECT * FROM turnout"
    assert agent.validate_sql(malicious_sql) is False

def test_read_only_enforcement(agent):
    """Verify the database connection is strictly Read-Only."""
    # Attempting to write to a RO connection should raise a sqlite3.OperationalError
    conn = sqlite3.connect(f"file:{agent.db_path}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO turnout (region) VALUES ('AGNEBY')")
