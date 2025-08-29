import logging
from pathlib import Path
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from model.schemas import RawJob, RankedCandidate
from app.service.llm_manager import LLMManager
from app.service.search_service import SearchService

# --- Application Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Candidate Recommendation API",
    description="An API that recommends top candidates for a given job.",
    version="1.0.0",
)


# --- Service Initialization (Singleton Pattern) ---
# These services are loaded once when the API starts up.
try:
    PROCESSED_CANDIDATES_PATH = Path("data/processed_candidates.json")

    logger.info("Initializing AI and Search services...")
    llm_manager = LLMManager()
    search_service = SearchService(
        llm_manager=llm_manager,
        processed_candidates_path=PROCESSED_CANDIDATES_PATH,
    )
    logger.info("Services initialized successfully.")
except Exception as e:
    logger.critical(f"Fatal: Could not initialize services. Error: {e}", exc_info=True)
    # In a real app, you might exit or prevent the app from starting.
    search_service = None


# --- API Endpoints ---
@app.post("/recommend", response_model=List[RankedCandidate])
async def recommend_candidates(job: RawJob, top_n: int = 100):
    """
    Accepts a raw job description and returns a ranked list of the most suitable candidates.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service is not available.")

    try:
        logger.info(f"Received recommendation request for job: {job.job_title}")
        ranked_candidates = await search_service.find_top_candidates(
            raw_job=job, top_n=top_n
        )
        logger.info(f"Returning {len(ranked_candidates)} recommendations.")
        return ranked_candidates
    except Exception as e:
        logger.error(f"An error occurred during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {"status": "ok"}


# --- Uvicorn Runner ---
# This allows you to run the API directly for development: `python api/main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
