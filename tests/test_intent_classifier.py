from src.rag.intent import classify_query_intent, should_refuse_upstream


def test_in_domain_when_stdlib_module_anchor_present() -> None:
    out = classify_query_intent("In argparse, how do I define an optional argument?")

    assert out.label == "in_domain"
    assert out.confidence >= 0.9


def test_python_general_out_of_scope_for_unanchored_python_concept() -> None:
    out = classify_query_intent("What is a Python decorator and how do I write one?")

    assert out.label == "python_general_out_of_scope"
    assert should_refuse_upstream(out) is True


def test_out_of_domain_for_non_python_request() -> None:
    out = classify_query_intent("What is the capital of France?")

    assert out.label == "out_of_domain"
    assert should_refuse_upstream(out) is True


def test_ambiguous_when_no_strong_signals() -> None:
    out = classify_query_intent("Can you help me with this?")

    assert out.label == "ambiguous"
    assert should_refuse_upstream(out) is False


def test_threshold_guard_prevents_low_confidence_refusal() -> None:
    out = classify_query_intent("I use pandas for analytics")

    assert out.label == "python_general_out_of_scope"
    assert out.confidence < 0.85
    assert should_refuse_upstream(out) is False
