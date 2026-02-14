from src.rag.prompt import build_prompt

class DummyChunk:
    def __init__(self, text):
        self.text = text

def test_build_prompt_contains_query_and_context():
    query = "How do I open a sqlite3 connection?"
    chunks = [DummyChunk("sqlite3.connect(...) docs text")]
    p = build_prompt(query, chunks)
    assert query in p
    assert "sqlite3.connect" in p
