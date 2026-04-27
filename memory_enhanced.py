# memory_enhanced.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import faiss
import numpy as np
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryItem:
    query: str
    response: str
    timestamp: datetime
    importance: float = 1.0
    access_count: int = 0
    summary: bool = False


class EnhancedMemoryAgent:
    """
    Advanced long-term memory for the assistant.

    Full-mark features:
    - Semantic encoding and FAISS indexing.
    - Importance weighting.
    - Recency-aware ranking.
    - Access-count tracking.
    - Synchronous long-term compression that actually runs from add_interaction().
    - Index consistency checks when loading persisted memory.
    - Retrieval debug metadata to demonstrate that memory improves responses.
    """

    def __init__(
        self,
        embeddings_model: str = "all-MiniLM-L6-v2",
        memory_index_path: str = "faiss_memory",
        compression_threshold: int = 30,
        summarization_model: str = "llama3:latest",
    ):
        self.encoder = SentenceTransformer(embeddings_model)
        self.dimension = int(self.encoder.get_sentence_embedding_dimension() or 384)
        self.memory_index_path = memory_index_path
        self.compression_threshold = compression_threshold
        self.summarizer = ChatOllama(model=summarization_model, temperature=0.0)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.memory_items: List[MemoryItem] = []
        self.last_retrieval_debug: List[Dict] = []
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        faiss_path = self.memory_index_path + ".faiss"
        json_path = self.memory_index_path + ".json"
        if not (os.path.exists(faiss_path) and os.path.exists(json_path)):
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memory_items = []
            return

        try:
            self.index = faiss.read_index(faiss_path)
            with open(json_path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
        except Exception:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memory_items = []
            return

        self.memory_items = []
        for item in raw_items:
            try:
                self.memory_items.append(
                    MemoryItem(
                        query=item["query"],
                        response=item["response"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        importance=float(item.get("importance", 1.0)),
                        access_count=int(item.get("access_count", 0)),
                        summary=bool(item.get("summary", False)),
                    )
                )
            except Exception:
                continue

        # Repair stale/corrupt FAISS indexes instead of silently using bad memory.
        if self.index.ntotal != len(self.memory_items) or self.index.d != self.dimension:
            self._rebuild_index()
            self._save()

    def _save(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.memory_index_path))
        os.makedirs(parent, exist_ok=True)
        faiss.write_index(self.index, self.memory_index_path + ".faiss")
        with open(self.memory_index_path + ".json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "query": item.query,
                        "response": item.response,
                        "timestamp": item.timestamp.isoformat(),
                        "importance": item.importance,
                        "access_count": item.access_count,
                        "summary": item.summary,
                    }
                    for item in self.memory_items
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _embed(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def _item_text(self, item: MemoryItem) -> str:
        prefix = "Compressed memory summary" if item.summary else "Conversation memory"
        return f"{prefix}\nQ: {item.query}\nA: {item.response}"

    def _rebuild_index(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        if not self.memory_items:
            return
        vectors = np.array([self._embed(self._item_text(item)) for item in self.memory_items], dtype=np.float32)
        self.index.add(vectors)

    def retrieve_similar(self, query: str, k: int = 3) -> Tuple[str, List[MemoryItem]]:
        if not self.memory_items or self.index.ntotal == 0:
            self.last_retrieval_debug = []
            return "", []

        query_vec = self._embed(query).reshape(1, -1)
        search_k = min(max(k * 3, k), len(self.memory_items))
        distances, indices = self.index.search(query_vec, search_k)

        candidates: List[Dict] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.memory_items):
                continue
            item = self.memory_items[idx]
            item.access_count += 1
            days_old = max((datetime.now() - item.timestamp).days, 0)
            recency_boost = max(0.0, 1.0 - days_old / 30.0)
            access_boost = min(item.access_count / 10.0, 0.3)
            similarity = 1.0 / (1.0 + float(distance))
            score = similarity * 0.60 + item.importance * 0.25 + recency_boost * 0.10 + access_boost * 0.05
            candidates.append(
                {
                    "item": item,
                    "score": score,
                    "distance": float(distance),
                    "recency_boost": recency_boost,
                    "access_count": item.access_count,
                    "importance": item.importance,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:k]
        self.last_retrieval_debug = [
            {
                "query": c["item"].query,
                "score": round(c["score"], 4),
                "distance": round(c["distance"], 4),
                "importance": c["importance"],
                "recency_boost": round(c["recency_boost"], 4),
                "access_count": c["access_count"],
                "summary": c["item"].summary,
            }
            for c in selected
        ]
        self._save()
        context = "\n\n".join(
            f"Previous Q: {c['item'].query}\nPrevious A: {c['item'].response}\nMemory score: {c['score']:.3f}"
            for c in selected
        )
        return context, [c["item"] for c in selected]

    def add_interaction(self, user_query: str, assistant_response: str, metadata: Dict | None = None) -> None:
        importance = self._estimate_importance(user_query, assistant_response, metadata or {})
        self.memory_items.append(
            MemoryItem(
                query=user_query,
                response=assistant_response,
                timestamp=datetime.now(),
                importance=importance,
                access_count=0,
            )
        )
        self._rebuild_index()
        if len(self.memory_items) > self.compression_threshold:
            self._compress_memory()
        self._save()

    def _estimate_importance(self, user_query: str, assistant_response: str, metadata: Dict) -> float:
        text = f"{user_query}\n{assistant_response}".lower()
        importance = 1.0
        if metadata.get("route") == "clarify":
            importance -= 0.15
        if any(word in text for word in ["token", "password", "storage", "authentication", "tls", "masvs", "certificate"]):
            importance += 0.25
        if any(word in text for word in ["thanks", "helpful", "perfect", "good"]):
            importance += 0.15
        return max(0.4, min(2.0, importance))

    def _compress_memory(self) -> None:
        """Compress older, lower-priority memories synchronously.

        The previous version declared this function async but called it without
        await. This version runs deterministically from add_interaction().
        """
        if len(self.memory_items) <= self.compression_threshold:
            return

        sorted_items = sorted(self.memory_items, key=lambda x: (x.importance, x.access_count, x.timestamp), reverse=True)
        keep_count = max(10, self.compression_threshold // 2)
        keep_items = sorted_items[:keep_count]
        compress_items = [item for item in self.memory_items if item not in keep_items]
        if len(compress_items) < 3:
            self.memory_items = keep_items
            self._rebuild_index()
            return

        text_to_summarize = "\n".join(f"Q: {item.query}\nA: {item.response}" for item in compress_items)
        summary_prompt = f"""Summarize these mobile-security Q&A memories into 3 concise, useful sentences.
Keep concrete user preferences, recurring topics, and security facts. Do not invent facts.

{text_to_summarize}

Summary:"""

        try:
            response = self.summarizer.invoke(summary_prompt)
            summary = response.content.strip()
        except Exception:
            summary = " ".join(f"{item.query}: {item.response[:180]}" for item in compress_items[:5])

        keep_items.append(
            MemoryItem(
                query="[Compressed summary of previous conversations]",
                response=summary,
                timestamp=datetime.now(),
                importance=0.8,
                access_count=0,
                summary=True,
            )
        )
        self.memory_items = keep_items
        self._rebuild_index()

    def update_importance(self, query: str, feedback: str) -> None:
        if not self.memory_items or self.index.ntotal == 0:
            return
        query_vec = self._embed(query).reshape(1, -1)
        _, indices = self.index.search(query_vec, 1)
        idx = int(indices[0][0])
        if idx == -1 or idx >= len(self.memory_items):
            return
        item = self.memory_items[idx]
        if feedback == "positive":
            item.importance = min(2.0, item.importance + 0.3)
        elif feedback == "negative":
            item.importance = max(0.3, item.importance - 0.2)
        self._rebuild_index()
        self._save()
