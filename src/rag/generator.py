from __future__ import annotations

import os
from pathlib import Path
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from src.config import load_app_config
from src.retrieval.retriever import RetrievedChunk
from src.rag.prompt import build_prompt


class Generator:
    def __init__(self, repo_root: Path):
        load_dotenv(repo_root / ".env")
        cfg, _ = load_app_config(repo_root)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = cfg.generation.model
        self.temperature = float(cfg.generation.temperature)

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I do not have enough information in the provided documentation."
        prompt = build_prompt(query, chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError("generation provider request failed") from e

        return response.choices[0].message.content.strip()
