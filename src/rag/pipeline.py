from pathlib import Path
from typing import Dict, Any

from src.retrieval.retriever import Retriever
from src.rag.confidence import ConfidenceGate
from src.rag.generator import Generator


class RAGPipeline:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.retriever = Retriever(repo_root)
        self.gate = ConfidenceGate(repo_root)
        self.generator = Generator(repo_root)

    def run(self, query: str) -> Dict[str, Any]:
        hits = self.retriever.retrieve(query)
        confidence = self.gate.decide(hits)
        response = self.generator.generate(query, confidence)
        return response
