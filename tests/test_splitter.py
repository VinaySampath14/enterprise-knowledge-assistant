from src.chunking.splitter import SplitConfig, split_text_with_offsets


def test_splitter_respects_size_and_offsets():
    text = "a" * 2000
    cfg = SplitConfig(chunk_size=800, overlap=150)
    chunks = split_text_with_offsets(text, cfg)

    assert len(chunks) >= 2

    # size bounds and offsets valid
    for c in chunks:
        assert 0 <= c["start_char"] < c["end_char"] <= len(text)
        assert len(c["text"]) == c["end_char"] - c["start_char"]
        assert len(c["text"]) <= cfg.chunk_size

    # overlap check between consecutive chunks
    # overlap = previous_end - next_start  (except maybe last edge cases)
    for prev, nxt in zip(chunks, chunks[1:]):
        overlap = prev["end_char"] - nxt["start_char"]
        assert overlap == cfg.overlap
