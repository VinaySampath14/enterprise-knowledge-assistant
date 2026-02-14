from src.rag.confidence import decide_confidence


def test_confidence_answer():
    d = decide_confidence([0.6, 0.4], threshold_high=0.40, threshold_low=0.25)
    assert d.decision == "answer"


def test_confidence_clarify():
    d = decide_confidence([0.30, 0.20], threshold_high=0.40, threshold_low=0.25)
    assert d.decision == "clarify"


def test_confidence_refuse():
    d = decide_confidence([0.10], threshold_high=0.40, threshold_low=0.25)
    assert d.decision == "refuse"


def test_confidence_no_scores():
    d = decide_confidence([], threshold_high=0.40, threshold_low=0.25)
    assert d.decision == "refuse"
