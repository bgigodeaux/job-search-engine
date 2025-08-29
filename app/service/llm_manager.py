import logging
from typing import List, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from mistralai import Mistral

from app.config import llm_config
from app.prompts.prompts import (
    CANDIDATE_FEATURE_ENGINEERING_PROMPT,
    JOB_FEATURE_ENGINEERING_PROMPT,
)
from app.model.schemas import (
    RawCandidate,
    RawJob,
    EngineeredCandidateFeatures,
    EngineeredJobFeatures,
)

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self):
        self.chat_model = ChatMistralAI(
            api_key=llm_config.MISTRAL_API_KEY,
            model_name=llm_config.MODEL_NAME_FE,
            temperature=0,
        )
        self.embedding_client = Mistral(api_key=llm_config.MISTRAL_API_KEY)

        job_prompt = ChatPromptTemplate.from_template(JOB_FEATURE_ENGINEERING_PROMPT)
        candidate_prompt = ChatPromptTemplate.from_template(
            CANDIDATE_FEATURE_ENGINEERING_PROMPT
        )
        self.candidate_feature_chain = (
            candidate_prompt | self.chat_model | JsonOutputParser()
        )
        self.job_feature_chain = job_prompt | self.chat_model | JsonOutputParser()

    async def generate_candidate_features(
        self, candidate: RawCandidate
    ) -> EngineeredCandidateFeatures | None:
        try:
            candidate_json_str = candidate.model_dump_json(indent=2)
            raw_features = await self.candidate_feature_chain.ainvoke(
                {"candidate_json": candidate_json_str}
            )
            return EngineeredCandidateFeatures(**raw_features)
        except Exception as e:
            logging.error(
                f"Error generating features for candidate: {e}", exc_info=True
            )
            return None

    async def generate_job_features(self, job: RawJob) -> EngineeredJobFeatures | None:
        try:
            job_json_str = job.model_dump_json(indent=2)
            raw_features = await self.job_feature_chain.ainvoke(
                {"job_json": job_json_str}
            )
            return EngineeredJobFeatures(**raw_features)
        except Exception as e:
            logging.error(f"Error generating features for job: {e}", exc_info=True)
            return None

    def get_embedding(self, text: str) -> List[float] | None:
        if not text:
            logging.warning("Attempted to generate embedding for empty text.")
            return None
        try:
            response = self.embedding_client.embeddings.create(
                model=llm_config.MODEL_NAME_EMBED, inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(
                f"An error occurred during embedding generation: {e}", exc_info=True
            )
            return None
