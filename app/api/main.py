from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.model.schemas import RawJob, RankedCandidate
from app.service.llm_manager import LLMManager
from app.service.search_service import SearchService
from app.service.candidate_service import CandidateService


logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=False)


APP_DIR = Path(__file__).resolve().parents[1]  # .../app
DATA_DIR = APP_DIR / "data"
RAW_CANDIDATES_PATH = DATA_DIR / "raw_candidates.json"
PROCESSED_CANDIDATES_PATH = DATA_DIR / "processed_candidates.json"


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


app = FastAPI(
    title="Candidate Recommendation API",
    description="An API that recommends top candidates for a given job and processes candidates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    """
    Initialize shared services once and keep them on app.state.
    """
    try:
        logger.info("Initializing services...")
        llm = LLMManager()
        app.state.llm_manager = llm

        app.state.search_service = SearchService(
            llm_manager=llm,
            processed_candidates_path=PROCESSED_CANDIDATES_PATH,
        )

        app.state.candidate_service = CandidateService(llm_manager=llm)

        logger.info("Services initialized.")
    except Exception:
        app.state.llm_manager = None
        app.state.search_service = None
        app.state.candidate_service = None
        logger.critical("Failed to initialize services", exc_info=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    pass


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check (reports if core services are up)."""
    ok = getattr(app.state, "search_service", None) is not None
    return {"status": "ok" if ok else "degraded"}


@app.post("/recommend", response_model=List[RankedCandidate])
async def recommend_candidates(
    job: RawJob,
    top_n: int = Query(100, gt=0, le=500, description="How many top candidates to return"),
) -> List[RankedCandidate]:
    """
    Accepts a raw job description and returns a ranked list of the most suitable candidates.
    """
    search_service: SearchService | None = getattr(app.state, "search_service", None)
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service is not available.")

    try:
        logger.info("Recommendation request for job: %s", job.job_title)
        ranked = await search_service.find_top_candidates(raw_job=job, top_n=top_n)
        logger.info("Returning %d recommendations.", len(ranked))
        return ranked
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during recommendation: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/process-candidates")
async def process_all_candidates() -> Dict[str, Any]:
    """
    Process ALL raw candidates and overwrite processed_candidates.json.
    Simple, predictable batch run (blocking until done).
    """
    candidate_service: CandidateService | None = getattr(app.state, "candidate_service", None)
    if candidate_service is None:
        raise HTTPException(status_code=503, detail="Candidate service is not available.")

    try:
        await candidate_service.process_candidates_from_file(
            input_path=RAW_CANDIDATES_PATH,
            output_path=PROCESSED_CANDIDATES_PATH,
        )

        processed = _read_json_list(PROCESSED_CANDIDATES_PATH)
        return {
            "status": "ok",
            "message": "Processed all candidates.",
            "processed_count": len(processed),
            "output_path": str(PROCESSED_CANDIDATES_PATH),
        }
    except Exception as e:
        logger.error("Candidate processing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Candidate processing failed.")


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
