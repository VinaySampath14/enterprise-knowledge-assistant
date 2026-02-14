from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import os

from openai import OpenAI
from dotenv import load_dotenv

from src.rag.prompt import build_prompt
from src.rag.confidence import ConfidenceResult
from src.retrieval.retriever import RetrievedChunk


class Generator:
    def __init__(self, repo_root: Path):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # cost-efficient + strong

    def _call_llm(self, prompt: str) -> str:
        print("[GEN] Calling OpenAI")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python Standard Library documentation assistant. "
                               "Answer only using provided context. "
                               "Do not hallucinate."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )
        

        return response.choices[0].message.content.strip()
    

    

    def generate(self, query: str, confidence: ConfidenceResult) -> Dict[str, Any]:

        if confidence.decision == "refuse":
            return {
                "type": "refuse",
                "answer": "I do not have enough information in the Python standard library documentation to answer that.",
                "confidence": confidence.confidence,
            }

        if confidence.decision == "clarify":
            return {
                "type": "clarify",
                "answer": "Could you clarify your question or specify which Python module you are referring to?",
                "confidence": confidence.confidence,
            }

        # ANSWER case
        prompt = build_prompt(query, confidence.used_chunks)

        llm_answer = self._call_llm(prompt)

        return {
            "type": "answer",
            "answer": llm_answer,
            "confidence": confidence.confidence,
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "module": c.module,
                    "score": c.score,
                }
                for c in confidence.used_chunks
            ],
        }
