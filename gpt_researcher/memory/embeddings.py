from __future__ import annotations

import logging
import os
import traceback

from typing import Any

OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

logger: logging.Logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS: set[str] = {
    "azure_openai",
    "aimlapi",
    "bedrock",
    "cohere",
    "custom",
    "dashscope",
    "fireworks",
    "gigachat",
    "google_genai",
    "google_vertexai",
    "huggingface",
    "mistralai",
    "nomic",
    "ollama",
    "openai",
    "together",
    "voyageai",
}

# Priority order for automatic fallbacks (providers that work without external API keys first)
_FALLBACK_PRIORITY: list[str] = [
    "huggingface",  # Works locally without API key
    "ollama",       # Works if local Ollama is running
    "custom",       # Custom provider
    "openai",       # Then try API-based providers
    "aimlapi",
    "together",
    "mistralai",
    "cohere",
    "google_genai",
    "google_vertexai",
    "fireworks",
    "voyageai",
    "dashscope",
    "bedrock",
    "azure_openai",
    "gigachat",
]


def _try_init_provider(
    embedding_provider: str,
    model: str,
    **embedding_kwargs: Any,
) -> Any | None:
    """Try to initialize a single embedding provider.
    
    Returns the embeddings object if successful, None if failed.
    """
    try:
        if embedding_provider == "custom":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=model,
                openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                openai_api_base=os.getenv(
                    "OPENAI_BASE_URL",
                    "http://localhost:1234/v1",
                ),
                check_embedding_ctx_length=False,
                **embedding_kwargs,
            )
        elif embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            if "api_key" in embedding_kwargs:
                return OpenAIEmbeddings(model=model, **embedding_kwargs)
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning(f"Skipping {embedding_provider}: OPENAI_API_KEY not set")
                    return None
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key=openai_api_key,
                    openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    **embedding_kwargs,
                )
        elif embedding_provider == "azure_openai":
            from langchain_openai import AzureOpenAIEmbeddings

            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

            if not all([azure_endpoint, azure_api_key, azure_api_version]):
                logger.warning(f"Skipping {embedding_provider}: Missing Azure OpenAI credentials")
                return None

            return AzureOpenAIEmbeddings(
                model=model,
                azure_endpoint=azure_endpoint,
                openai_api_key=azure_api_key,
                openai_api_version=azure_api_version,
                **embedding_kwargs,
            )
        elif embedding_provider == "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "google_vertexai":
            from langchain_google_vertexai import VertexAIEmbeddings

            return VertexAIEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "fireworks":
            from langchain_fireworks import FireworksEmbeddings

            return FireworksEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "gigachat":
            from langchain_gigachat import GigaChatEmbeddings

            return GigaChatEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            return OllamaEmbeddings(
                model=model,
                base_url=ollama_base_url,
                **embedding_kwargs,
            )
        elif embedding_provider == "together":
            from langchain_together import TogetherEmbeddings

            return TogetherEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "mistralai":
            from langchain_mistralai import MistralAIEmbeddings

            return MistralAIEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=model, **embedding_kwargs)
        elif embedding_provider == "nomic":
            from langchain_nomic import NomicEmbeddings

            return NomicEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "voyageai":
            from langchain_voyageai import VoyageAIEmbeddings

            voyage_api_key = os.getenv("VOYAGE_API_KEY")
            if not voyage_api_key:
                logger.warning(f"Skipping {embedding_provider}: VOYAGE_API_KEY not set")
                return None

            return VoyageAIEmbeddings(
                voyage_api_key=voyage_api_key,
                model=model,
                **embedding_kwargs,
            )
        elif embedding_provider == "dashscope":
            from langchain_community.embeddings import DashScopeEmbeddings

            return DashScopeEmbeddings(model=model, **embedding_kwargs)
        elif embedding_provider == "bedrock":
            from langchain_aws.embeddings import BedrockEmbeddings

            return BedrockEmbeddings(model_id=model, **embedding_kwargs)
        elif embedding_provider == "aimlapi":
            from langchain_openai import OpenAIEmbeddings

            aimlapi_key = os.getenv("AIMLAPI_API_KEY")
            if not aimlapi_key:
                logger.warning(f"Skipping {embedding_provider}: AIMLAPI_API_KEY not set")
                return None

            return OpenAIEmbeddings(
                model=model,
                openai_api_key=aimlapi_key,
                openai_api_base=os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1"),
                **embedding_kwargs,
            )
        else:
            logger.warning(f"Unknown embedding provider: {embedding_provider}")
            return None

    except Exception as e:
        logger.warning(
            f"Failed to initialize {embedding_provider} embedding provider: {e.__class__.__name__}: {e}",
        )
        return None


class Memory:
    def __init__(
        self,
        embedding_provider: str,
        model: str,
        **embedding_kwargs: Any,
    ):
        _embeddings: Any | None = None
        failed_providers: dict[str, Exception] = {}

        # Try the requested provider first
        logger.info(f"Attempting to initialize embedding provider: {embedding_provider}")
        _embeddings = _try_init_provider(embedding_provider, model, **embedding_kwargs)
        
        if _embeddings is not None:
            logger.info(f"Successfully initialized {embedding_provider} embedding provider")
            self._embeddings = _embeddings
            self._active_provider = embedding_provider
            return

        logger.warning(f"Primary embedding provider {embedding_provider} failed, trying fallbacks...")

        # Try fallback providers in priority order
        for fallback_provider in _FALLBACK_PRIORITY:
            if fallback_provider == embedding_provider:
                continue  # Already tried this one
            
            if fallback_provider not in _SUPPORTED_PROVIDERS:
                continue

            logger.info(f"Trying fallback embedding provider: {fallback_provider}")
            _embeddings = _try_init_provider(fallback_provider, model, **embedding_kwargs)
            
            if _embeddings is not None:
                logger.info(
                    f"Successfully initialized fallback embedding provider: {fallback_provider} "
                    f"(primary {embedding_provider} was unavailable)"
                )
                self._embeddings = _embeddings
                self._active_provider = fallback_provider
                return

        # All providers failed
        error_msg = (
            f"Failed to initialize any embedding provider. "
            f"Primary provider '{embedding_provider}' and all fallbacks failed. "
            f"Please configure at least one embedding provider with valid credentials. "
            f"Tried providers in order: {embedding_provider}, {', '.join(_FALLBACK_PRIORITY)}"
        )
        logger.error(error_msg)
        raise Exception(error_msg)

    def get_embeddings(self) -> Any:
        return self._embeddings


class FallbackMemory(Memory):
    """Memory with fallback support for embeddings."""

    def __init__(
        self,
        embedding_provider: str,
        model: str,
        fallback_models: list[str] | None = None,
        **embedding_kwargs: Any,
    ) -> None:
        """Initialize the Memory with fallback support.

        Args:
            embedding_provider: Primary embedding provider
            model: Primary embedding model
            fallback_models: List of fallback models (format: 'provider:model')
            **embedding_kwargs: Additional embedding arguments
        """
        # Initialize primary embedding provider
        try:
            super().__init__(embedding_provider, model, **embedding_kwargs)
            self._primary_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize primary embedding provider {embedding_provider}:{model}: {e.__class__.__name__}: {e}")
            self._primary_initialized = False
            self._embeddings = None
            self._primary_error: Exception = e

        self.fallback_memories: list[Memory] = []
        self.fallback_configs: list[tuple[str, str, dict[str, Any]]] = []

        # Store fallback configurations but don't initialize them yet
        if fallback_models:
            for fallback_model in fallback_models:
                try:
                    # Standard format "provider:model"
                    if ":" not in fallback_model:
                        logger.warning(f"Invalid fallback model format {fallback_model}: Expected format 'provider:model'")
                        continue

                    fallback_provider, fallback_model_name = fallback_model.split(":", 1)

                    # Copy kwargs and update model name
                    fallback_kwargs: dict[str, Any] = embedding_kwargs.copy()
                    # Store configuration for lazy initialization
                    self.fallback_configs.append((fallback_provider, fallback_model_name, fallback_kwargs))
                    logger.info(f"Added fallback embedding model configuration: {fallback_provider}:{fallback_model_name}")
                except Exception as e:
                    logger.warning(f"Invalid fallback model format {fallback_model}: {e.__class__.__name__}: {e}")

    def get_embeddings(self) -> Any:
        """Get embeddings with fallback support.

        Attempts to use the primary embeddings, and if it fails, tries each fallback in order.

        Returns:
            The embeddings provider

        Raises:
            Exception: If all providers fail
        """
        # Try primary provider first
        if self._primary_initialized:
            try:
                return self._embeddings
            except Exception as e:
                logger.warning(f"Primary embedding provider failed: {e.__class__.__name__}: {e}. Trying fallbacks...")

        # If we get here, either the primary wasn't initialized or failed
        if not self.fallback_configs and not self.fallback_memories:
            # No fallbacks available, raise an error about the primary failing
            if hasattr(self, "_primary_error"):
                raise Exception(f"Primary embedding provider failed and no fallbacks available: {self._primary_error}")
            else:
                raise Exception("Primary embedding provider failed and no fallbacks available")

        # Initialize any remaining fallbacks that haven't been tried yet
        while self.fallback_configs:
            fallback_provider, fallback_model_name, fallback_kwargs = self.fallback_configs.pop(0)
            try:
                fallback_memory = Memory(fallback_provider, fallback_model_name, **fallback_kwargs)
                self.fallback_memories.append(fallback_memory)
                logger.info(f"Successfully initialized fallback embedding model: {fallback_provider}:{fallback_model_name}")
                return fallback_memory.get_embeddings()
            except Exception as e:
                logger.warning(f"Failed to initialize fallback embedding provider {fallback_provider}:{fallback_model_name}: {e.__class__.__name__}: {e}")
                # Continue to the next fallback

        # Try any already initialized fallbacks
        errors: list[Exception] = []
        for i, fallback_memory in enumerate(self.fallback_memories):
            try:
                logger.warning(f"Trying fallback embedding provider {i+1}/{len(self.fallback_memories)}")
                return fallback_memory.get_embeddings()
            except Exception as fallback_error:
                logger.warning(f"Fallback embedding provider {i+1} failed: {fallback_error.__class__.__name__}: {fallback_error}")
                errors.append(fallback_error)

        # All fallbacks failed
        if errors:
            last_error: Exception = errors[-1]
            raise Exception(f"All embedding providers failed. Last error: {last_error.__class__.__name__}: {last_error}") from last_error
        else:
            raise Exception("All embedding providers failed to initialize")
