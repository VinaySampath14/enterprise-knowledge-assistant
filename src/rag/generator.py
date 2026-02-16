from __future__ import annotations

import os
from pathlib import Path
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.retriever import RetrievedChunk


class Generator:
    def __init__(self, repo_root: Path):
        load_dotenv(repo_root / ".env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I do not have enough information in the provided documentation."

        context_blocks = []
        for c in chunks:
            context_blocks.append(
                f"[{c.chunk_id}]\n{c.text.strip()}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a documentation assistant.

Answer the user question using ONLY the provided documentation context.
If the answer is not clearly supported by the context, say you do not have enough information.

Question:
{query}

Documentation:
{context}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()
