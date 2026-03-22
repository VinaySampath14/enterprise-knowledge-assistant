from typing import Any, Dict, List, Tuple
from src.retrieval.retriever import RetrievedChunk


def format_retrieved_chunks(chunks: List[RetrievedChunk]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build citation-ready numbered context text and a structured source map.
    """
    context_blocks: List[str] = []
    source_map: List[Dict[str, Any]] = []

    for i, c in enumerate(chunks, start=1):
        heading_text = c.heading if c.heading is not None else "None"
        source_text = c.source_path if c.source_path is not None else "None"

        context_blocks.append(
            "\n".join(
                [
                    f"[{i}]",
                    f"Module: {c.module}",
                    f"Source: {source_text}",
                    f"Heading: {heading_text}",
                    f"Chunk ID: {c.chunk_id}",
                    "Text:",
                    c.text,
                ]
            )
        )

        source_map.append(
            {
                "id": i,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "module": c.module,
                "source_path": c.source_path,
                "heading": c.heading,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "score": float(c.score),
            }
        )

    return "\n\n".join(context_blocks), source_map


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context_text, _ = format_retrieved_chunks(chunks)

    prompt = f"""
You are a documentation assistant.

Use ONLY the provided documentation context.

Citation rules:
- Every factual statement, function/module claim, and code usage guidance must include one or more inline citations.
- Inline citations must use ONLY the numbered context blocks, like [1], [2], [1][2], or [1, 2].
- Never invent citation numbers.
- Do not cite statements that are purely connective language.
- Place citations naturally, usually at the end of the sentence.

Safety rule:
- If the context does not clearly support the answer, respond exactly with:
I don't have enough information from the provided documentation.

Documentation Context:
{context_text}

User Question:
{query}

Provide a concise, natural answer.
""".strip()

    return prompt
