import json
import logging
import asyncio
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

from app.service.llm_manager import LLMManager
from app.model.schemas import (
    RawJob,
    ProcessedCandidate,
    EngineeredJobFeatures,
)


# Define the missing data model for a ranked result
class RankedCandidate(BaseModel):
    candidate: ProcessedCandidate
    score: float


logger = logging.getLogger(__name__)

# Constants for less strict filtering
SKILL_MATCH_THRESHOLD = 0.6  # Candidate must have at least 60% of required skills


class SearchService:
    def __init__(self, llm_manager: LLMManager, processed_candidates_path: Path):
        self.llm_manager = llm_manager
        self.candidates: List[ProcessedCandidate] = []
        self.candidate_embeddings: np.ndarray | None = None
        self._load_candidates(processed_candidates_path)

    def _load_candidates(self, file_path: Path):
        logger.info(f"Loading pre-processed candidates from {file_path}...")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self.candidates = [ProcessedCandidate(**c) for c in data]

            embeddings = [c["embedding"] for c in data]
            self.candidate_embeddings = np.array(embeddings, dtype=np.float32)

            logger.info(f"Successfully loaded {len(self.candidates)} candidates.")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.critical(f"Fatal: Could not load or parse candidate data: {e}")
            raise

    async def _process_job(
        self, raw_job: RawJob
    ) -> tuple[EngineeredJobFeatures | None, List[float] | None]:
        job_features = await self.llm_manager.generate_job_features(raw_job)
        if not job_features:
            return None, None

        # NOTE: Run sync embedding in executor to not block the event loop
        loop = asyncio.get_running_loop()
        job_embedding = await loop.run_in_executor(
            None,
            self.llm_manager.get_embedding,
            job_features.job_summary_for_embedding,
        )
        return job_features, job_embedding

    def _filter_candidates(
        self, job_features: EngineeredJobFeatures
    ) -> List[tuple[int, ProcessedCandidate]]:
        filtered_indices = []
        for i, candidate in enumerate(self.candidates):
            features = candidate.engineered_features

            # Rule 1: Candidate experience must be sufficient
            if (
                job_features.required_experience_years
                > features.total_years_of_experience
            ):
                continue

            # Rule 2: Candidate must have a minimum percentage of skills
            required_skills = set(s.lower() for s in job_features.extracted_skills)
            if required_skills:  # Only check if there are skills required
                candidate_skills = set(s.lower() for s in features.skill_keywords)
                matched_skills = required_skills.intersection(candidate_skills)

                if (len(matched_skills) / len(required_skills)) < SKILL_MATCH_THRESHOLD:
                    continue

            filtered_indices.append((i, candidate))

        return filtered_indices

    def _rank_candidates(
        self,
        job_embedding: np.ndarray,
        filtered_candidates_with_indices: List[tuple[int, ProcessedCandidate]],
    ) -> List[RankedCandidate]:
        if not filtered_candidates_with_indices or self.candidate_embeddings is None:
            return []

        indices = [idx for idx, _ in filtered_candidates_with_indices]
        filtered_embeddings = self.candidate_embeddings[indices]

        similarity_scores = cosine_similarity(
            job_embedding.reshape(1, -1), filtered_embeddings
        )[0]

        ranked_results = []
        for score, (original_index, candidate) in zip(
            similarity_scores, filtered_candidates_with_indices
        ):
            ranked_results.append(RankedCandidate(candidate=candidate, score=score))

        ranked_results.sort(key=lambda x: x.score, reverse=True)
        return ranked_results

    async def find_top_candidates(
        self, raw_job: RawJob, top_n: int = 100
    ) -> List[RankedCandidate]:
        logger.info(f"Starting search for job: '{raw_job.job_title}'")

        # Step 1: Process job to get features and embedding
        job_features, job_embedding_list = await self._process_job(raw_job)
        print(job_features)
        if not job_features or not job_embedding_list:
            logger.error("Failed to process job, cannot perform search.")
            return []

        job_embedding = np.array(job_embedding_list, dtype=np.float32)

        # Step 2: Filter candidates based on hard rules
        filtered_candidates = self._filter_candidates(job_features)
        logger.info(
            f"Filtering phase: {len(self.candidates)} -> {len(filtered_candidates)} candidates."
        )

        # Step 3: Rank the filtered candidates by semantic similarity
        ranked_candidates = self._rank_candidates(job_embedding, filtered_candidates)
        logger.info(
            f"Ranking phase complete. Found {len(ranked_candidates)} relevant candidates."
        )

        return ranked_candidates[:top_n]
