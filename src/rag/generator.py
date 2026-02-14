from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml

from dotenv import load_dotenv

from src.rag.prompt import build_prompt
from src.rag.confidence import ConfidenceResult


class Generator:
    def __init__(self, repo_root: Path, *, config_path: Optional[Path] = None):
        self.repo_root = repo_root
        self.config_path = config_path or (repo_root / "config.yaml")

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        gen_cfg = cfg.get("generation", {})
        self.enabled = bool(gen_cfg.get("enabled", True))
        self.model = str(gen_cfg.get("model", "gpt-4o-mini"))
        self.temperature = float(gen_cfg.get("temperature", 0.0))

        self._client = None

        # Only require OpenAI if generation is enabled
        if self.enabled:
            load_dotenv(dotenv_path=self.repo_root / ".env")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment (.env).")

            from openai import OpenAI  # import only when needed
            self._client = OpenAI(api_key=api_key)

    def _call_llm(self, prompt: str) -> str:
        assert self._client is not None, "OpenAI client not initialized"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python Standard Library documentation assistant. "
                        "Answer only using the provided context. "
                        "If the answer is not supported by the context, say you don't have enough information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

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
                "answer": "Could you clarify your question (e.g., which module/function you mean) so I can look it up in the documentation?",
                "confidence": confidence.confidence,
            }

        # answer
        if not self.enabled:
            # Fallback: return a short evidence snippet (keeps API usable without OpenAI)
            snippet = confidence.used_chunks[0].text[:500] if confidence.used_chunks else ""
            return {
                "type": "answer",
                "answer": snippet,
                "confidence": confidence.confidence,
            }

        prompt = build_prompt(query, confidence.used_chunks)
        llm_answer = self._call_llm(prompt)

        return {
            "type": "answer",
            "answer": llm_answer,
            "confidence": confidence.confidence,
        }
