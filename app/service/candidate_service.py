import json
import logging
import asyncio
from pathlib import Path
from typing import List
from tqdm.asyncio import tqdm

from app.service.llm_manager import LLMManager
from app.model.schemas import (
    RawCandidate,
    ProcessedCandidate,
    EngineeredCandidateFeatures,
)

logger = logging.getLogger(__name__)


class CandidateService:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.semaphore = asyncio.Semaphore(10)  # Limits concurrent API calls to 10

    async def _engineer_features(
        self, raw_candidate: RawCandidate
    ) -> EngineeredCandidateFeatures | None:
        engineered_features = await self.llm_manager.generate_candidate_features(
            raw_candidate
        )
        if not engineered_features:
            logger.warning(
                f"Skipping candidate due to feature generation failure: {raw_candidate.email}"
            )
            return None
        return engineered_features

    async def _generate_embedding(
        self, engineered_features: EngineeredCandidateFeatures
    ) -> List[float] | None:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            self.llm_manager.get_embedding,
            engineered_features.candidate_summary,
        )
        if not embedding:
            logger.warning(
                f"Skipping candidate due to embedding failure for summary: '{engineered_features.candidate_summary[:50]}...'"
            )
            return None
        return embedding

    async def _process_single_candidate(
        self, candidate_data: dict
    ) -> ProcessedCandidate | None:
        async with self.semaphore:
            try:
                raw_candidate = RawCandidate(**candidate_data)

                # Step 1: Generate structured features
                engineered_features = await self._engineer_features(raw_candidate)
                if not engineered_features:
                    return None

                # Step 2: Generate embedding
                embedding = await self._generate_embedding(engineered_features)
                if not embedding:
                    return None

                # Step 3: Combine into final object
                return ProcessedCandidate(
                    original_data=raw_candidate,
                    engineered_features=engineered_features,
                    embedding=embedding,
                )
            except Exception as e:
                logger.error(
                    f"Unhandled error processing candidate {candidate_data.get('email')}: {e}",
                    exc_info=True,
                )
                return None

    async def process_candidates_from_file(self, input_path: Path, output_path: Path):
        try:
            with open(input_path, "r") as f:
                raw_candidates_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(
                f"Fatal: Error loading candidate data from {input_path}: {e}"
            )
            return

        tasks = [self._process_single_candidate(c) for c in raw_candidates_data]

        results = await tqdm.gather(*tasks, desc="Processing Candidates")

        processed_results = [
            result.model_dump() for result in results if result is not None
        ]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(processed_results, f, indent=2)
            logger.info(
                f"\nSuccessfully processed {len(processed_results)}/{len(raw_candidates_data)} candidates and saved to {output_path}"
            )
        except IOError as e:
            logger.critical(f"Fatal: Error saving results to {output_path}: {e}")
