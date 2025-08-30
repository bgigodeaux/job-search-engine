# app/service/candidate_service.py
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
    """
    Processes raw candidates into ProcessedCandidate objects:
      - id: int             (taken from the raw dict)
      - original_data: RawCandidate
      - engineered_features: EngineeredCandidateFeatures
      - embedding: List[float]
    Writes a JSON list of these to output_path (OVERWRITE).
    """

    def __init__(self, llm_manager: LLMManager, max_concurrency: int = 10) -> None:
        self.llm_manager = llm_manager
        self.semaphore = asyncio.Semaphore(max_concurrency)

    # ---------- internals ----------

    async def _engineer_features(self, raw: RawCandidate) -> Optional[EngineeredCandidateFeatures]:
        """
        Call LLM to generate engineered features; coerce to EngineeredCandidateFeatures.
        """
        feats = await self.llm_manager.generate_candidate_features(raw)
        if not feats:
            logger.warning("Skipping candidate due to feature generation failure: %s", raw.email)
            return None

        if isinstance(feats, EngineeredCandidateFeatures):
            return feats
        if hasattr(feats, "model_dump"):
            return EngineeredCandidateFeatures(**feats.model_dump())  # type: ignore[arg-type]
        if hasattr(feats, "dict"):
            return EngineeredCandidateFeatures(**feats.dict())  # type: ignore[arg-type]
        if isinstance(feats, dict):
            return EngineeredCandidateFeatures(**feats)

        logger.warning("Unexpected features type for %s: %r", raw.email, type(feats))
        return None

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute embedding for a text. Runs in a thread to avoid blocking the event loop.
        """
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
        """
        Build a ProcessedCandidate (or return None if any step fails).
        Expects an 'id' key in cand_data; RawCandidate schema does not include it.
        """
        async with self.semaphore:
            try:
                # Ensure we have an id for the processed object
                raw_id = cand_data.get("id")
                if raw_id is None:
                    logger.warning(
                        "Candidate missing 'id'; skipping. email=%s", cand_data.get("email")
                    )
                    return None

                # Validate and normalize the raw portion with Pydantic
                raw = RawCandidate(**cand_data)

                # 1) engineered features
                feats = await self._engineer_features(raw)
                if not feats:
                    return None

                # 2) embedding (prefer summary, fallback to simple join of skills)
                summary = feats.candidate_summary or " ".join(raw.skills)
                embedding = await self._generate_embedding(summary)
                if not embedding:
                    return None

                # 3) pack -> ProcessedCandidate
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

    # ---------- public API ----------

    async def process_candidates_from_file(self, input_path: Path, output_path: Path) -> None:
        """
        Reads ALL raw candidates from input_path, processes them concurrently,
        and OVERWRITES output_path with a JSON list of ProcessedCandidate dicts.
        """
        # Load raw list
        try:
            raw_list = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(raw_list, list):
                raise ValueError("Input must be a JSON list of candidate dicts.")
        except Exception as e:
            logger.critical("Fatal: error loading %s: %s", input_path, e)
            return

        # Process with progress
        tasks = [self._process_single_candidate(c) for c in raw_list]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing Candidates")

        # Keep only successful; serialize to plain dicts (Pydantic v2/v1)
        processed: List[Dict[str, Any]] = []
        for r in results:
            if r is None:
                continue
            if hasattr(r, "model_dump"):
                processed.append(r.model_dump())  # Pydantic v2
            elif hasattr(r, "dict"):
                processed.append(r.dict())  # Pydantic v1
            else:
                processed.append(r)  # last-resort fallback (shouldn't happen)

        # Save (overwrite)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Processed %d/%d candidates → %s",
                len(processed),
                len(raw_list),
                output_path,
            )
        except Exception as e:
            logger.critical("Fatal: error saving %s: %s", output_path, e)
