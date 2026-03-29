from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Literal

IntentLabel = Literal[
    "in_domain",
    "python_general_out_of_scope",
    "out_of_domain",
    "ambiguous",
]

REFUSE_INTENT_LABELS = {"python_general_out_of_scope", "out_of_domain"}


@dataclass(frozen=True)
class IntentClassification:
    label: IntentLabel
    confidence: float
    rationale: str
    signals: List[str]


_STDLIB_MODULES = {
    "argparse",
    "array",
    "asyncio",
    "bisect",
    "collections",
    "concurrent",
    "contextlib",
    "csv",
    "datetime",
    "functools",
    "heapq",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "queue",
    "re",
    "sqlite3",
    "statistics",
    "subprocess",
    "sys",
    "threading",
    "time",
    "typing",
    "unittest",
}


_OUT_OF_DOMAIN_PATTERNS = [
    r"\bcapital of\b",
    r"\bweather\b",
    r"\bmovie\b",
    r"\bplot of\b",
    r"\brecipe\b",
    r"\bstock price\b",
    r"\bfootball\b",
    r"\bcricket\b",
    r"\bpolitics\b",
    r"\bcelebrity\b",
]


_PYTHON_GENERAL_PATTERNS = [
    r"\bdecorator\b",
    r"\bgil\b",
    r"\blist comprehension\b",
    r"\bclass\b",
    r"\binheritance\b",
    r"\bvirtualenv\b",
    r"\bvenv\b",
    r"\bpip install\b",
    r"\bnumpy\b",
    r"\bpandas\b",
    r"\bdjango\b",
    r"\bflask\b",
    r"\bpython internals\b",
    r"\bcache results in python\b",
]


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _has_stdlib_anchor(q: str) -> tuple[bool, List[str]]:
    signals: List[str] = []
    if not q:
        return False, signals

    if "python standard library" in q or "stdlib" in q:
        signals.append("mentions stdlib directly")
        return True, signals

    for module in _STDLIB_MODULES:
        if re.search(rf"\b{re.escape(module)}\b", q):
            signals.append(f"mentions stdlib module '{module}'")
            return True, signals

    dotted = re.findall(r"\b([a-z_][\w]*)\.([a-z_][\w]*)\b", q)
    for mod, _ in dotted:
        if mod in _STDLIB_MODULES:
            signals.append(f"mentions module.member anchor '{mod}.*'")
            return True, signals

    framed = re.findall(r"\b(?:in|from)\s+([a-z_][\w]*)\b", q)
    for token in framed:
        if token in _STDLIB_MODULES:
            signals.append(f"uses explicit module framing 'in/from {token}'")
            return True, signals

    return False, signals


def classify_query_intent(query: str) -> IntentClassification:
    q = _normalize(query)
    if not q:
        return IntentClassification(
            label="ambiguous",
            confidence=0.0,
            rationale="empty query",
            signals=[],
        )

    for pat in _OUT_OF_DOMAIN_PATTERNS:
        if re.search(pat, q):
            return IntentClassification(
                label="out_of_domain",
                confidence=0.98,
                rationale="strong explicit out-of-domain lexical signal",
                signals=[f"matched pattern: {pat}"],
            )

    has_anchor, anchor_signals = _has_stdlib_anchor(q)
    if has_anchor:
        return IntentClassification(
            label="in_domain",
            confidence=0.92,
            rationale="query contains explicit stdlib/module anchor",
            signals=anchor_signals,
        )

    has_python_word = bool(re.search(r"\bpython\b", q))
    general_matches = [pat for pat in _PYTHON_GENERAL_PATTERNS if re.search(pat, q)]
    if has_python_word and general_matches:
        return IntentClassification(
            label="python_general_out_of_scope",
            confidence=0.90,
            rationale="python-general question without stdlib anchor",
            signals=[f"matched pattern: {m}" for m in general_matches],
        )

    if general_matches:
        return IntentClassification(
            label="python_general_out_of_scope",
            confidence=0.82,
            rationale="general Python ecosystem signals without stdlib anchor",
            signals=[f"matched pattern: {m}" for m in general_matches],
        )

    return IntentClassification(
        label="ambiguous",
        confidence=0.45,
        rationale="insufficient lexical evidence for safe intent routing",
        signals=[],
    )


def should_refuse_upstream(intent: IntentClassification, *, threshold: float = 0.85) -> bool:
    return intent.label in REFUSE_INTENT_LABELS and intent.confidence >= threshold
