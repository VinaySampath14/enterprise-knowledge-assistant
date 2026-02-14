from typing import List
from src.retrieval.retriever import RetrievedChunk


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] {c.text}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You must answer using ONLY the provided documentation context.
If the answer is not explicitly supported by the context, say:
"I don't have enough information from the provided documentation."

Documentation Context:
{context_text}

User Question:
{query}

Provide a clear, concise answer.
""".strip()

    return prompt
