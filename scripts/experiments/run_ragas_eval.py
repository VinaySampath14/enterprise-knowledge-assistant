from __future__ import annotations

import argparse
import ast
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

from src.rag.pipeline import RAGPipeline
from src.utils.jsonl import iter_jsonl


def _sample_chunks(chunks: List[Dict[str, Any]], per_module: int = 5, max_total: int = 80, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        module = str(c.get("module", "")).strip()
        if not module:
            continue
        by_module[module].append(c)

    sampled: List[Dict[str, Any]] = []
    for module in sorted(by_module.keys()):
        rows = by_module[module]
        rng.shuffle(rows)
        sampled.extend(rows[:per_module])

    rng.shuffle(sampled)
    return sampled[:max_total]


def _generate_question(client: OpenAI, model: str, module: str, chunk_text: str) -> str:
    prompt = (
        f"Given this Python stdlib documentation chunk from module {module}, write one question a developer would ask that this chunk answers. One question only, no preamble.\n\n"
        f"Chunk:\n{chunk_text}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _load_eval_questions(path: Path, source: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for row in iter_jsonl(path):
        q = str(row.get("query", "")).strip()
        if not q:
            continue
        items.append({"question": q, "module": source, "source": source})
    return items


def _coerce_scores(result_obj: Any) -> Dict[str, float]:
    if hasattr(result_obj, "to_dict"):
        raw = result_obj.to_dict()
    elif hasattr(result_obj, "dict"):
        raw = result_obj.dict()
    elif hasattr(result_obj, "model_dump"):
        raw = result_obj.model_dump()
    else:
        raw = {"result": str(result_obj)}

    # Handle ragas versions that return {"result": "{'faithfulness': ...}"}
    if isinstance(raw, dict) and "result" in raw and isinstance(raw["result"], str):
        text = raw["result"].strip()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                raw = parsed
        except Exception:
            pass

    out: Dict[str, float] = {}
    for k in ("faithfulness", "answer_relevancy", "context_precision"):
        try:
            out[k] = float(raw.get(k))
        except Exception:
            out[k] = 0.0
    return out


def _retrieve_context_texts(pipeline: RAGPipeline, query: str) -> List[str]:
    retrieval_k = pipeline.reranker.candidate_k if pipeline.reranker.enabled else None
    hits = pipeline.retriever.retrieve(query, top_k=retrieval_k)
    reranker_applied = pipeline.reranker.should_rerank(hits)
    if reranker_applied:
        hits = pipeline.reranker.rerank(query, hits, top_k=int(pipeline.cfg.retrieval.top_k))
    else:
        hits = list(hits)[: int(pipeline.cfg.retrieval.top_k)]

    decision = pipeline.gate.decide(hits, query=query)
    return [str(getattr(c, "text", "") or "") for c in decision.used_chunks if str(getattr(c, "text", "") or "").strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS eval on generated + manual/holdout queries.")
    parser.add_argument("--generated-max", type=int, default=80, help="Max generated chunk-based questions.")
    parser.add_argument("--generated-per-module", type=int, default=5, help="Generated questions per module.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--include-manual", action="store_true", help="Include eval/manual.jsonl queries.")
    parser.add_argument("--include-holdout", action="store_true", help="Include eval/holdout_paraphrases.jsonl queries.")
    parser.add_argument(
        "--out-json",
        default="artifacts/ragas/ragas_scores.json",
        help="Output path for scores JSON.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")

    # 1) Load chunks
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    chunks = list(iter_jsonl(chunks_path))

    # 2) Sample 5 chunks per module, max 80 total
    sampled_chunks = _sample_chunks(
        chunks,
        per_module=int(args.generated_per_module),
        max_total=int(args.generated_max),
        seed=int(args.seed),
    )

    # 3) Generate one question per sampled chunk with gpt-4o-mini
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)

    qa_items: List[Dict[str, str]] = []
    for c in sampled_chunks:
        module = str(c.get("module", "")).strip()
        text = str(c.get("text", "")).strip()
        if not module or not text:
            continue
        q = _generate_question(client, "gpt-4o-mini", module, text)
        if q:
            qa_items.append({"question": q, "module": module, "source": "generated"})

    # Include interview-defensible fixed sets when requested.
    if args.include_manual:
        qa_items.extend(_load_eval_questions(repo_root / "eval" / "manual.jsonl", source="manual"))
    if args.include_holdout:
        qa_items.extend(_load_eval_questions(repo_root / "eval" / "holdout_paraphrases.jsonl", source="holdout"))

    # Deduplicate by normalized question text while preserving first source occurrence.
    seen_questions: set[str] = set()
    deduped_items: List[Dict[str, str]] = []
    for item in qa_items:
        key = item["question"].strip().lower()
        if not key or key in seen_questions:
            continue
        seen_questions.add(key)
        deduped_items.append(item)
    qa_items = deduped_items

    # 4-6) Run pipeline query, keep type=answer, build RAGAS rows
    pipeline = RAGPipeline(repo_root)
    ragas_rows: List[Dict[str, Any]] = []
    total_questions = len(qa_items)
    answer_kept = 0
    source_counts: Dict[str, int] = defaultdict(int)
    source_kept_counts: Dict[str, int] = defaultdict(int)

    for item in qa_items:
        question = item["question"]
        src = item.get("source", "generated")
        source_counts[src] += 1

        out = pipeline.run(question)
        if str(out.get("type", "")) != "answer":
            continue

        answer = str(out.get("answer", "")).strip()
        contexts = _retrieve_context_texts(pipeline, question)
        if not answer or not contexts:
            continue

        ragas_rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            # context_precision requires a reference column in current RAGAS versions.
            "reference": contexts[0],
        })
        answer_kept += 1
        source_kept_counts[src] += 1

    if not ragas_rows:
        raise RuntimeError("No answer rows were produced for RAGAS evaluation.")

    # 7) Run ragas.evaluate()
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import llm_factory
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except Exception as e:
        raise RuntimeError(
            "RAGAS dependencies are missing. Install with: pip install ragas datasets"
        ) from e

    dataset = Dataset.from_list(ragas_rows)
    ragas_llm = llm_factory(model="gpt-4o-mini", provider="openai", client=client)
    ragas_embeddings = LCOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    metric_scores = _coerce_scores(result)
    scores = {
        **metric_scores,
        "total_questions": total_questions,
        "answer_rows_used": answer_kept,
        "answer_keep_rate": (answer_kept / total_questions) if total_questions else 0.0,
        "source_counts": dict(source_counts),
        "source_answer_rows_used": dict(source_kept_counts),
    }

    # 8) Print the scores
    print(json.dumps(scores, indent=2))

    # 9) Save scores
    out_path = repo_root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"[OK] Wrote RAGAS scores: {out_path}")


if __name__ == "__main__":
    main()
