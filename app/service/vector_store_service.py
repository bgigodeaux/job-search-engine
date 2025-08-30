from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from app.config import llm_config
from app.model.schemas import RawJob, ProcessedCandidate
from app.service.llm_manager import LLMManager


def asyncio_run_safe(coro):
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)


class VectorStoreService:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "candidates",
        llm_manager: Optional[LLMManager] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings_model = llm_config.MODEL_NAME_EMBED
        self.llm_manager = llm_manager

        self._embeddings = MistralAIEmbeddings(model=self.embeddings_model)

        self._vs: Optional[Chroma] = None

        self._candidate_by_id: Dict[int, Dict[str, Any]] = {}

    def _get_store(self) -> Chroma:
        if self._vs is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vs

    def build_from_file(self, processed_candidates_path: Path) -> int:
        data: List[Dict[str, Any]]
        with open(processed_candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._candidate_by_id.clear()

        docs: List[Document] = []
        ids: List[str] = []

        for item in data:
            cand_id = item.get("id")
            feats = item.get("engineered_features", {}) or {}
            summary = feats.get("candidate_summary", "")

            if not isinstance(cand_id, int):
                continue
            if not summary:
                continue

            self._candidate_by_id[cand_id] = item

            docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": "processed_candidates",
                        "candidate_id": cand_id,
                        "recent_title": feats.get("recent_job_title", ""),
                        "recent_company": feats.get("recent_company", ""),
                    },
                )
            )
            ids.append(str(cand_id))

        vs = self._get_store()
        if ids:
            try:
                vs.delete(ids=ids)
            except Exception:
                pass

            vs.add_documents(documents=docs, ids=ids)
            vs.persist()

        return len(docs)

    def search_by_text(
        self, query: str, k: int = 10, score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        vs = self._get_store()

        results = vs.similarity_search_with_score(
            query=query, k=k, filter={"source": "processed_candidates"}
        )

        out: List[Tuple[Dict[str, Any], float]] = []
        for doc, distance in results:
            cand_id = doc.metadata.get("candidate_id")
            if cand_id is None:
                continue
            candidate_dict = self._candidate_by_id.get(cand_id)
            if candidate_dict is None:
                candidate_dict = {"id": cand_id, "engineered_features": {}, "original_data": {}}

            if score_threshold is not None and distance > score_threshold:
                continue

            out.append((candidate_dict, distance))

        return out

    def search_by_job(
        self, raw_job: RawJob, k: int = 10, score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Use LLMManager to turn a RawJob into a summary string for querying the vector store.
        Falls back to concatenating title/description if LLM is not provided.
        """
        if self.llm_manager:
            feats = asyncio_run_safe(self.llm_manager.generate_job_features(raw_job))
            if feats and hasattr(feats, "job_summary_for_embedding"):
                query = feats.job_summary_for_embedding
            else:
                query = f"{raw_job.job_title}. {raw_job.job_description}"
        else:
            query = f"{raw_job.job_title}. {raw_job.job_description}"

        return self.search_by_text(query=query, k=k, score_threshold=score_threshold)
