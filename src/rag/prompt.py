from typing import Any, Dict, List, Tuple
from src.retrieval.retriever import RetrievedChunk


def format_retrieved_chunks(chunks: List[RetrievedChunk]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build citation-ready numbered context text and a structured source map.
    """
    context_blocks: List[str] = []
    source_map: List[Dict[str, Any]] = []

    for i, c in enumerate(chunks, start=1):
        module = getattr(c, "module", "unknown")
        source_path = getattr(c, "source_path", None)
        heading = getattr(c, "heading", None)
        chunk_id = getattr(c, "chunk_id", f"chunk_{i}")
        text = getattr(c, "text", "")
        doc_id = getattr(c, "doc_id", None)
        start_char = getattr(c, "start_char", None)
        end_char = getattr(c, "end_char", None)
        score = float(getattr(c, "score", 0.0))

        heading_text = heading if heading is not None else "None"
        source_text = source_path if source_path is not None else "None"

        context_blocks.append(
            "\n".join(
                [
                    f"[{i}]",
                    f"Module: {module}",
                    f"Source: {source_text}",
                    f"Heading: {heading_text}",
                    f"Chunk ID: {chunk_id}",
                    "Text:",
                    text,
                ]
            )
        )

        source_map.append(
            {
                "id": i,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "module": module,
                "source_path": source_path,
                "heading": heading,
                "start_char": start_char,
                "end_char": end_char,
                "score": score,
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
