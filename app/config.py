import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

logger = logging.getLogger(__name__)

# Constants for model names
MODEL_NAME_SMALL = "mistral-small-latest"
MODEL_NAME_MEDIUM = "mistral-medium-latest"
MODEL_NAME_EMBED = "mistral-embed"


@dataclass
class LLMConfig:
    """
    Manages configuration for the LLM services, loading from environment variables.
    """

    MISTRAL_API_KEY: str
    MODEL_NAME_FE: str  # Model for Feature Engineering
    MODEL_NAME_EMBED: str  # Model for Embeddings


def get_llm_config() -> LLMConfig:
    """
    Initializes and validates the LLM configuration.
    Raises:
        KeyError: If the MISTRAL_API_KEY is not found in the environment.
    """
    try:
        api_key = os.environ["MISTRAL_API_KEY"]
    except KeyError:
        logger.error("FATAL: MISTRAL_API_KEY environment variable not set.")
        raise

    return LLMConfig(
        MISTRAL_API_KEY=api_key,
        MODEL_NAME_FE=MODEL_NAME_SMALL,
        MODEL_NAME_EMBED=MODEL_NAME_EMBED,
    )


# Create a single, importable instance of the configuration
llm_config = get_llm_config()
