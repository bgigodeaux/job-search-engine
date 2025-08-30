from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.service.llm_manager import LLMManager
from app.model.schemas import (
    RawJob,
    ProcessedCandidate,
    EngineeredJobFeatures,
    RankedCandidate,
)

logger = logging.getLogger(__name__)

SKILL_MATCH_THRESHOLD = 0.6


class SearchService:
    def __init__(self, llm_manager: LLMManager, processed_candidates_path: Path):
        self.llm_manager = llm_manager
        self.candidates: List[ProcessedCandidate] = []
        self.candidate_embeddings: np.ndarray | None = None
        self._load_candidates(processed_candidates_path)

    def _load_candidates(self, file_path: Path) -> None:
        logger.info("Loading pre-processed candidates from %s ...", file_path)
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("processed_candidates.json must be a JSON list.")

            self.candidates = [ProcessedCandidate(**c) for c in raw]
            try:
                embeddings = [c["embedding"] for c in raw]
            except (KeyError, TypeError):
                embeddings = [pc.embedding for pc in self.candidates]

            self.candidate_embeddings = np.asarray(embeddings, dtype=np.float32)
            logger.info("Successfully loaded %d candidates.", len(self.candidates))

        except Exception as e:
            logger.critical("Fatal: could not load or parse candidate data: %s", e)
            raise

    async def _process_job(
        self, raw_job: RawJob
    ) -> Tuple[EngineeredJobFeatures | None, List[float] | None]:
        job_features = await self.llm_manager.generate_job_features(raw_job)
        if not job_features:
            return None, None
        try:
            job_embedding = await asyncio.to_thread(
                self.llm_manager.get_embedding, job_features.job_summary_for_embedding
            )
        except Exception as e:
            logger.error("Failed to compute job embedding: %s", e, exc_info=True)
            job_embedding = None

        return job_features, job_embedding

    def _filter_candidates(
        self, job_features: EngineeredJobFeatures
    ) -> List[tuple[int, ProcessedCandidate]]:
        """
        Hard filters before semantic ranking:
          - experience: candidate years >= required years
          - skills: at least SKILL_MATCH_THRESHOLD overlap with required skills
        Returns pairs (index_in_embeddings, candidate)
        """
        filtered: List[tuple[int, ProcessedCandidate]] = []
        req_years = float(job_features.required_experience_years or 0.0)

        required_skills = {s.strip().lower() for s in (job_features.extracted_skills or []) if s}
        check_skills = len(required_skills) > 0

        for i, cand in enumerate(self.candidates):
            feats = cand.engineered_features

            # Experience rule
            cand_years = float(feats.total_years_of_experience or 0.0)
            if req_years > cand_years:
                continue

            # Skill coverage rule (if the job has explicit skills)
            if check_skills:
                cand_sk = {s.strip().lower() for s in (feats.skill_keywords or []) if s}
                if not cand_sk:
                    continue
                matched = required_skills.intersection(cand_sk)
                if (len(matched) / len(required_skills)) < SKILL_MATCH_THRESHOLD:
                    continue

            filtered.append((i, cand))

        return filtered

    def _rank_candidates(
        self,
        job_embedding: np.ndarray,
        filtered_candidates_with_indices: List[tuple[int, ProcessedCandidate]],
    ) -> List[RankedCandidate]:

        if not filtered_candidates_with_indices or self.candidate_embeddings is None:
            return []

        indices = [idx for idx, _ in filtered_candidates_with_indices]
        filtered_embs = self.candidate_embeddings[indices]

        scores = cosine_similarity(job_embedding.reshape(1, -1), filtered_embs)[0]

        ranked: List[RankedCandidate] = []
        for score, (_, cand) in zip(scores, filtered_candidates_with_indices):
            ranked.append(RankedCandidate(candidate=cand, score=float(score)))

        ranked.sort(key=lambda rc: rc.score, reverse=True)
        return ranked

    async def find_top_candidates(self, raw_job: RawJob, top_n: int = 100) -> List[RankedCandidate]:
        logger.info("Starting search for job: %s", raw_job.job_title)

        job_feats, job_emb_list = await self._process_job(raw_job)
        if not job_feats or not job_emb_list:
            logger.error("Failed to process job; cannot perform search.")
            return []

        job_emb = np.asarray(job_emb_list, dtype=np.float32)

        filtered = self._filter_candidates(job_feats)
        ranked = self._rank_candidates(job_emb, filtered)

        return ranked[:top_n]
