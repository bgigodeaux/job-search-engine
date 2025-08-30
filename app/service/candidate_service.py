from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as tqdm_asyncio

from app.model.schemas import (
    RawCandidate,
    EngineeredCandidateFeatures,
    ProcessedCandidate,
)
from app.service.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class CandidateService:
    def __init__(self, llm_manager: LLMManager, max_concurrency: int = 10) -> None:
        self.llm_manager = llm_manager
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _engineer_features(self, raw: RawCandidate) -> Optional[EngineeredCandidateFeatures]:
        feats = await self.llm_manager.generate_candidate_features(raw)
        if not feats:
            logger.warning("Feature generation failed for: %s", raw.email)
            return None
        return feats

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            logger.warning("Empty text passed for embedding; skipping.")
            return None
        try:
            return await asyncio.to_thread(self.llm_manager.get_embedding, text)
        except Exception as e:
            logger.warning("Embedding failure for text: %.50s… | %s", text, e)
            return None

    async def _process_single_candidate(
        self, cand_data: Dict[str, Any]
    ) -> Optional[ProcessedCandidate]:
        async with self.semaphore:
            try:
                raw_id = cand_data.get("id")
                if raw_id is None:
                    logger.warning(
                        "Candidate missing 'id'; skipping. email=%s", cand_data.get("email")
                    )
                    return None

                raw = RawCandidate(**cand_data)

                # get the engineered features
                feats = await self._engineer_features(raw)
                if not feats:
                    return None

                # generate the embedding
                summary = feats.candidate_summary or " ".join(raw.skills)
                embedding = await self._generate_embedding(summary)
                if not embedding:
                    return None

                return ProcessedCandidate(
                    id=int(raw_id),
                    original_data=raw,
                    engineered_features=feats,
                    embedding=embedding,
                )

            except Exception as e:
                logger.error(
                    "Unhandled error processing candidate %s: %s",
                    cand_data.get("email"),
                    e,
                    exc_info=True,
                )
                return None

    async def process_candidates_from_file(self, input_path: Path, output_path: Path) -> None:
        """
        Read ALL raw candidates from input_path, process concurrently,
        and OVERWRITE output_path with a JSON list of ProcessedCandidate dicts.
        """
        try:
            with input_path.open("r", encoding="utf-8") as f:
                raw_list = json.load(f)
            if not isinstance(raw_list, list):
                raise ValueError("Input must be a JSON list of candidate dicts.")
        except Exception as e:
            logger.critical("Fatal: error loading %s: %s", input_path, e)
            return

        tasks = [self._process_single_candidate(c) for c in raw_list]
        results = await asyncio.gather(*tasks)

        processed = [r.model_dump() for r in results if r is not None]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Processed %d/%d candidates → %s", len(processed), len(raw_list), output_path
            )
        except Exception as e:
            logger.critical("Fatal: error saving %s: %s", output_path, e)
