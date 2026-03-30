from pathlib import Path

from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievedChunk


class _FakeCrossEncoder:
    def predict(self, pairs):
        out = []
        for _, text in pairs:
            t = text.lower()
            if "sqlite3" in t:
                out.append(2.0)
            elif "argparse" in t:
                out.append(0.1)
            else:
                out.append(1.0)
        return out


def _hit(chunk_id: str, module: str, score: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{module}",
        module=module,
        score=score,
        text=text,
        source_path=f"data/raw/python_stdlib/{module}.rst",
        heading=None,
        meta={"source_path": f"data/raw/python_stdlib/{module}.rst"},
        start_char=0,
        end_char=100,
        chunk_index=0,
        vector_id=0,
    )


def test_pair_text_includes_module_and_text():
    h = _hit("c1", "sqlite3", 0.5, "Connection object docs")
    txt = CrossEncoderReranker._pair_text(h)
    assert "sqlite3" in txt
    assert "Connection object docs" in txt


def test_reranker_reorders_using_model_but_preserves_original_scores():
    repo_root = Path(__file__).resolve().parents[1]
    r = CrossEncoderReranker(repo_root, model=_FakeCrossEncoder())
    r.enabled = True
    r.candidate_k = 5

    hits = [
        _hit("a", "argparse", 0.95, "argparse usage"),
        _hit("s", "sqlite3", 0.60, "sqlite3 connection details"),
        _hit("z", "zipfile", 0.80, "zip files"),
    ]

    out = r.rerank("sqlite connection", hits, top_k=3)

    assert out[0].chunk_id == "s"
    by_id = {h.chunk_id: h.score for h in out}
    assert by_id["s"] == 0.60
    assert by_id["a"] == 0.95
