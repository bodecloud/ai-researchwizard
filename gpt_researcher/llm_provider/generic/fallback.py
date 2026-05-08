"""LLM Provider with Fallback Support."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TypeVar

from gpt_researcher.llm_provider.generic.base import GenericLLMProvider
from gpt_researcher.utils.llm_debug_logger import LLMDebugLogger, get_llm_debug_logger

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_auth_failure(error: Exception) -> bool:
    """Return True when an exception indicates provider credential/auth failure."""
    error_name: str = error.__class__.__name__.lower()
    error_msg: str = str(error).lower()

    if "auth" in error_name or "unauthorized" in error_name:
        return True

    auth_markers: tuple[str, ...] = (
        "incorrect api key",
        "invalid api key",
        "authentication",
        "unauthorized",
        "user not found",
        "invalid x-api-key",
    )
    return any(marker in error_msg for marker in auth_markers)


def _is_openrouter_rate_limit_failure(error: Exception) -> bool:
    """Return True when OpenRouter responds with key/rate limit errors (typically 403)."""
    error_msg: str = str(error).lower()
    return (
        "openrouter" in error_msg
        and "403" in error_msg
        and (
            "key limit exceeded" in error_msg
            or "rate limit" in error_msg
            or "total limit" in error_msg
        )
    )


def _provider_base_url(provider_obj: Any) -> str:
    llm_obj: Any = getattr(provider_obj, "llm", provider_obj)
    return str(getattr(llm_obj, "openai_api_base", None) or getattr(llm_obj, "base_url", "") or "").strip().lower()


def _provider_cache_key(provider_obj: Any) -> str:
    """Build a stable provider key for temporary runtime disablement."""
    llm_obj: Any = getattr(provider_obj, "llm", provider_obj)
    model_name: str = str(
        getattr(llm_obj, "model_name", None)
        or getattr(llm_obj, "model", None)
        or "unknown"
    )
    module_name: str = str(getattr(llm_obj.__class__, "__module__", "unknown"))
    base_url: str = str(getattr(llm_obj, "openai_api_base", None) or getattr(llm_obj, "base_url", ""))
    return f"{module_name}|{model_name}|{base_url}"


class FallbackGenericLLMProvider(GenericLLMProvider):
    """LLM Provider with fallback support."""

    _auth_failed_provider_cache_keys: set[str] = set()
    _rate_limited_provider_bases: set[str] = set()

    def __init__(
        self,
        llm: Any,
        fallback_providers: list[GenericLLMProvider] | None = None,
        chat_log: str | None = None,
        verbose: bool = True,
    ):
        """Initialize the fallback LLM provider.

        Args:
            llm: Primary LLM provider
            fallback_providers: List of fallback LLM providers
            chat_log: Path to chat log
            verbose: Verbosity level
        """
        super().__init__(llm, chat_log=chat_log, verbose=verbose)
        self.fallback_providers: list[GenericLLMProvider] = (
            []
            if fallback_providers is None
            else fallback_providers
        )

        # Ensure all fallback providers share the same debug logger instance
        if self.debug_logger is not None:
            for fallback_provider in self.fallback_providers:
                fallback_provider.debug_logger = self.debug_logger

    @classmethod
    def from_provider_with_fallbacks(
        cls,
        provider: str,
        fallback_models: list[str] = None,
        chat_log: str | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> FallbackGenericLLMProvider:
        """Create a fallback LLM provider from a provider name and fallback model list.

        Args:
            provider: Primary provider name
            fallback_models: List of fallback model strings in format 'provider:model'
            chat_log: Path to chat log
            verbose: Verbosity level
            **kwargs: Additional keyword arguments for the primary provider

        Returns:
            FallbackGenericLLMProvider instance
        """
        # Handle OpenRouter models
        primary_kwargs: dict[str, Any] = kwargs.copy()
        if provider == "openrouter":
            if not os.environ.get("OPENROUTER_API_KEY"):
                logger.info("OPENROUTER_API_KEY not set. OpenRouter models will not be used.")

            # OpenRouter uses the OpenAI API format with a different base URL
            # Pass through to the OpenAI provider with proper configuration
            provider = "openai"
            primary_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            primary_kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]

            # Set up model_kwargs (removed transforms as it's not supported by OpenAI client)
            primary_kwargs["model_kwargs"] = primary_kwargs.get("model_kwargs", {})

            # Log the configuration
            logger.debug(f"Using OpenRouter model: '{primary_kwargs.get('model')}'")

        # Create fallback providers if specified
        fallback_providers: list[GenericLLMProvider] = []
        if fallback_models:
            for fallback_model in fallback_models:
                try:
                    # Standard format "provider:model"
                    if ":" not in fallback_model:
                        logger.warning(f"Skipping invalid fallback model format: {fallback_model}. Expected format: 'provider:model'")
                        continue

                    provider_name, model_name = fallback_model.split(":", 1)

                    # Skip vertexai models if not configured properly
                    if provider_name in ["google_vertexai", "vertexai"]:
                        if not (
                            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                            or os.environ.get("GOOGLE_CLOUD_PROJECT")
                        ):
                            logger.debug(f"Skipping Google VertexAI model '{model_name}' because Google Cloud is not configured")
                            continue

                    # Handle provider name mappings and aliases
                    if provider_name == "gemini":
                        provider_name = "google_genai"
                        logger.debug(f"Mapping 'gemini' to 'google_genai' provider for '{model_name}'")

                    # Special handling for OpenRouter models
                    fallback_kwargs: dict[str, Any] = kwargs.copy()
                    if provider_name == "openrouter":
                        if not os.environ.get("OPENROUTER_API_KEY"):
                            logger.debug(f"Skipping OpenRouter model '{model_name}' because OPENROUTER_API_KEY is not set")
                            continue

                        # OpenRouter models use OpenAI's API with a different base URL
                        provider_name = "openai"
                        fallback_kwargs["base_url"] = "https://openrouter.ai/api/v1"
                        fallback_kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]

                        # Set up model_kwargs (removed transforms as it's not supported by OpenAI client)
                        fallback_kwargs["model_kwargs"] = fallback_kwargs.get("model_kwargs", {})

                        logger.debug(f"Using OpenRouter model '{model_name}'")

                    # Set the model name and create the provider
                    fallback_kwargs["model"] = model_name

                    # Log the fallback configuration
                    logger.debug(f"Initializing fallback provider '{provider_name}' with model '{model_name}'")

                    fallback_provider: GenericLLMProvider = (
                        GenericLLMProvider.from_provider(
                            provider_name,
                            chat_log=chat_log,
                            verbose=verbose,
                            **fallback_kwargs,
                        )
                    )
                    fallback_providers.append(fallback_provider)
                except (ValueError, ImportError) as e:
                    logger.warning(f"Failed to initialize fallback provider '{fallback_model}': {e.__class__.__name__}: {e}")

        # Initialize the primary provider
        try:
            primary_provider: GenericLLMProvider = GenericLLMProvider.from_provider(
                provider,
                chat_log=chat_log,
                verbose=verbose,
                **primary_kwargs,
            )

            return cls(
                primary_provider.llm,
                fallback_providers=fallback_providers,
                chat_log=chat_log,
                verbose=verbose,
            )
        except Exception as e:
            logger.error(f"Failed to initialize primary provider '{provider}': {e.__class__.__name__}: {e}")

            # If we have fallbacks, try to use the first one as primary
            if fallback_providers:
                logger.warning("Using first fallback provider as primary due to initialization failure")
                primary: GenericLLMProvider = fallback_providers.pop(0)
                return cls(
                    primary.llm,
                    fallback_providers=fallback_providers,
                    chat_log=chat_log,
                    verbose=verbose,
                )
            else:
                # No fallbacks available, re-raise the exception
                raise

    async def get_chat_response(
        self,
        messages: list[dict[str, str]],
        stream: bool,
        websocket: Any | None = None,
    ) -> str:
        """Generate LLM response with fallback support.

        Attempts to generate using the primary LLM, and if it fails, tries each fallback in order.

        Args:
            messages: The messages to send to the chat completion
            stream: Whether to stream the response
            websocket: The websocket to use for streaming responses

        Returns:
            The response from the chat completion

        Raises:
            Exception: If all providers fail
        """
        primary_model_name: str = getattr(self.llm, "model_name", None) or getattr(
            self.llm,
            "model",
            "unknown",
        )

        if self.verbose:
            logger.debug(f"[FALLBACK-DEBUG] Starting fallback-enabled request with primary model: '{primary_model_name}'")

        # Get debug logger and start a single interaction for the entire fallback process
        debug_logger: LLMDebugLogger = get_llm_debug_logger()
        interaction_id: str | None = None
        start_time: float = time.time()

        # Start debug logging for the entire fallback process
        if debug_logger is not None:
            try:
                # Extract model and provider info
                provider_name: str = getattr(self.llm.__class__, "__module__", "unknown").split(".")[-1]

                # Start logging interaction for the entire fallback process
                interaction_id = debug_logger.start_interaction(
                    step_name="fallback_chat_response",
                    model=primary_model_name,
                    provider=provider_name,
                    context_info={
                        "stream": stream,
                        "has_websocket": websocket is not None,
                        "message_count": len(messages),
                        "fallback_providers_count": len(self.fallback_providers),
                    },
                )

                # Log request details
                system_msg: str = ""
                user_msg: str = ""
                if messages:
                    for msg in messages:
                        if msg.get("role") == "system":
                            system_msg += msg.get("content", "") + "\n"
                        elif msg.get("role") == "user":
                            user_msg += msg.get("content", "") + "\n"

                # Get model parameters
                temperature: float | None = getattr(self.llm, "temperature", None)
                max_tokens: int | None = getattr(self.llm, "max_tokens", None)
                model_kwargs: dict[str, Any] = getattr(self.llm, "model_kwargs", {})

                debug_logger.log_request(
                    system_message=system_msg.strip(),
                    user_message=user_msg.strip(),
                    full_messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **model_kwargs,
                )
            except Exception as e:
                logger.warning(f"Debug logging start failed: {e.__class__.__name__}: {e}")

        # Try primary provider first
        primary_cache_key: str = _provider_cache_key(self)
        e = None
        primary_was_blacklisted = False
        
        # Only attempt primary provider if it hasn't been marked as failed
        primary_base_url: str = _provider_base_url(self)
        primary_rate_limited = bool(primary_base_url and primary_base_url in self._rate_limited_provider_bases)

        if primary_cache_key in self._auth_failed_provider_cache_keys or primary_rate_limited:
            # Primary provider already marked as failed, skip it and go to fallbacks
            logger.debug(
                f"[FALLBACK-DEBUG] Primary provider {primary_model_name} already marked unavailable from previous auth failure. Skipping to fallbacks..."
            )
            primary_was_blacklisted = True
        else:
            try:
                if self.verbose:
                    logger.debug(f"[FALLBACK-DEBUG] Attempting primary provider: '{primary_model_name}'")

                # Temporarily disable debug logging for the primary provider to avoid duplicate interactions
                original_debug_logger: LLMDebugLogger | None = self.debug_logger
                self.debug_logger = None

                try:
                    response: str = await super().get_chat_response(
                        messages,
                        stream,
                        websocket,
                    )
                finally:
                    # Restore debug logger
                    self.debug_logger = original_debug_logger

                if self.verbose:
                    logger.debug(f"[FALLBACK-DEBUG] Primary provider succeeded: '{primary_model_name}'")

                # Log successful response
                if interaction_id and interaction_id.strip():
                    try:
                        duration: float = time.time() - start_time
                        debug_logger.log_response(
                            response=response,
                            success=True,
                            duration_seconds=duration,
                        )
                        debug_logger.finish_interaction()
                    except Exception as log_e:
                        if self.verbose:
                            logger.warning(f"Debug logging response failed: {log_e.__class__.__name__}: {log_e}")

                return response

            except Exception as primary_e:
                e = primary_e
        
        # Handle primary failure (e will be None if primary was skipped OR if primary succeeded and returned above)
        if e is not None or primary_was_blacklisted:
            if e is not None:
                if _is_auth_failure(e):
                    self._auth_failed_provider_cache_keys.add(_provider_cache_key(self))
                if _is_openrouter_rate_limit_failure(e):
                    blocked_base = _provider_base_url(self)
                    if blocked_base:
                        self._rate_limited_provider_bases.add(blocked_base)

                logger.warning(f"[FALLBACK-DEBUG] Primary provider failed: '{primary_model_name}'")
                logger.warning(f"[FALLBACK-DEBUG] Primary error: '{e.__class__.__name__}: {e}...'")
            else:
                # Primary was blacklisted, so no real exception to report
                logger.debug(f"[FALLBACK-DEBUG] Primary provider {primary_model_name} is blacklisted, trying fallbacks")

            if not self.fallback_providers:
                logger.warning("[FALLBACK-DEBUG] No fallback providers available. Re-raising primary error.")

                # Log final failure (no retry attempts since no fallbacks available)
                if interaction_id and interaction_id.strip():
                    try:
                        duration = time.time() - start_time
                        error_msg = str(e) if e else "Primary provider unavailable"
                        error_type = e.__class__.__name__ if e else "Unavailable"
                        debug_logger.log_response(
                            response="",
                            success=False,
                            error_message=error_msg,
                            error_type=error_type,
                            duration_seconds=duration,
                        )
                        debug_logger.finish_interaction()
                    except Exception as log_error:
                        if self.verbose:
                            logger.warning(f"Debug logging error failed: {log_error.__class__.__name__}: {log_error}")

                raise

            # Log the primary failure only when fallbacks are available
            if debug_logger.current_interaction is not None:
                primary_error_msg = f"{e.__class__.__name__}: {e}" if e else "Primary provider already marked unavailable from previous auth failure"
                debug_logger.log_retry_attempt(
                    attempt_number=1,
                    reason=f"Primary provider {primary_model_name} failed",
                    details={
                        "primary_error": primary_error_msg,
                        "primary_model": primary_model_name,
                        "total_fallbacks_available": len(self.fallback_providers),
                    },
                )

            if self.verbose:
                logger.debug(f"[FALLBACK-DEBUG] Trying {len(self.fallback_providers)} fallback providers...")

            # Try each fallback in order
            # Initialize last_error to either the primary exception or a generic message if primary was just skipped
            if e is not None:
                last_error: Exception = e
            else:
                last_error: Exception = Exception("Primary provider unavailable, attempting fallbacks")
            for i, fallback_provider in enumerate(self.fallback_providers):
                try:
                    fallback_cache_key: str = _provider_cache_key(fallback_provider)
                    fallback_base_url: str = _provider_base_url(fallback_provider)
                    if fallback_cache_key in self._auth_failed_provider_cache_keys:
                        if self.verbose:
                            logger.debug(
                                "[FALLBACK-DEBUG] Skipping fallback %s due to previous auth failure",
                                fallback_cache_key,
                            )
                        continue
                    if fallback_base_url and fallback_base_url in self._rate_limited_provider_bases:
                        if self.verbose:
                            logger.debug(
                                "[FALLBACK-DEBUG] Skipping fallback %s due to provider-wide rate-limit blacklist",
                                fallback_cache_key,
                            )
                        continue

                    fallback_model: str = getattr(
                        fallback_provider.llm,
                        "model_name",
                        None,
                    ) or getattr(fallback_provider.llm, "model", "unknown")

                    if self.verbose:
                        logger.debug(f"[FALLBACK-DEBUG] Trying fallback provider {i + 1}/{len(self.fallback_providers)}: '{fallback_model}'")

                    # Update the current interaction to reflect fallback attempt
                    if debug_logger.current_interaction is not None:
                        debug_logger.current_interaction.is_fallback = True
                        debug_logger.current_interaction.fallback_attempt = i + 1
                        debug_logger.current_interaction.primary_provider = primary_model_name
                        debug_logger.current_interaction.model = fallback_model
                        debug_logger.current_interaction.provider = getattr(
                            fallback_provider,
                            "provider",
                            "unknown",
                        )

                        debug_logger.log_retry_attempt(
                            # +2 because attempt 1 was the primary
                            attempt_number=i + 2,
                            reason=f"Trying fallback provider {fallback_model}",
                            details={
                                "fallback_model": fallback_model,
                                "fallback_attempt": i + 1,
                                "total_fallbacks": len(self.fallback_providers),
                                "remaining_fallbacks": len(self.fallback_providers)
                                - i
                                - 1,
                            },
                        )

                    # Ensure fallback provider shares the same debug logger but disable its own interaction creation
                    original_fallback_debug_logger = fallback_provider.debug_logger
                    fallback_provider.debug_logger = None

                    try:
                        response: str = await fallback_provider.get_chat_response(
                            messages,
                            stream,
                            websocket,
                        )
                    finally:
                        # Restore fallback provider's debug logger
                        fallback_provider.debug_logger = original_fallback_debug_logger

                    if self.verbose:
                        logger.debug(f"[FALLBACK-DEBUG] Fallback provider {i + 1} succeeded: '{fallback_model}'")

                    # Log successful fallback response
                    if interaction_id and interaction_id.strip():
                        try:
                            duration = time.time() - start_time
                            debug_logger.log_response(
                                response=response,
                                success=True,
                                duration_seconds=duration,
                            )
                            debug_logger.finish_interaction()
                        except Exception as log_error:
                            if self.verbose:
                                logger.warning(f"Debug logging response failed: {log_error.__class__.__name__}: {log_error}")

                    return response

                except Exception as fallback_error:
                    if _is_auth_failure(fallback_error):
                        self._auth_failed_provider_cache_keys.add(
                            _provider_cache_key(fallback_provider)
                        )
                    if _is_openrouter_rate_limit_failure(fallback_error):
                        blocked_base = _provider_base_url(fallback_provider)
                        if blocked_base:
                            self._rate_limited_provider_bases.add(blocked_base)

                    if self.verbose:
                        logger.warning(f"[FALLBACK-DEBUG] Fallback provider {i + 1} failed: '{fallback_model}' Fallback error: '{fallback_error.__class__.__name__}: {str(fallback_error)[:300] if len(str(fallback_error)) > 300 else str(fallback_error)}'")

                    # Log the fallback failure
                    if debug_logger.current_interaction is not None:
                        debug_logger.log_retry_attempt(
                            # +2 because attempt 1 was the primary
                            attempt_number=i + 2,
                            reason=f"Fallback provider {fallback_model} also failed",
                            details={
                                "fallback_error": f"{fallback_error.__class__.__name__}: {fallback_error}",
                                "fallback_model": fallback_model,
                                "fallback_attempt": i + 1,
                                "remaining_fallbacks": len(self.fallback_providers)
                                - i
                                - 1,
                            },
                        )

                    last_error = fallback_error

            # All fallbacks failed
            if self.verbose:
                logger.error(f"[FALLBACK-DEBUG] All providers failed. Primary: '{primary_model_name}', Fallbacks: {len(self.fallback_providers)}")
                logger.error(f"[FALLBACK-DEBUG] Last error: '{last_error.__class__.__name__}: {last_error}'")

            # Log final failure
            if debug_logger.current_interaction is not None:
                debug_logger.log_retry_attempt(
                    attempt_number=len(self.fallback_providers) + 2,
                    reason="All fallback providers exhausted",
                    details={
                        "primary_model": primary_model_name,
                        "total_fallbacks_tried": len(self.fallback_providers),
                        "final_error": f"{last_error.__class__.__name__}: {last_error}",
                    },
                )

            # Log final error response
            if interaction_id and interaction_id.strip():
                try:
                    duration = time.time() - start_time
                    debug_logger.log_response(
                        response="",
                        success=False,
                        error_message=str(last_error),
                        error_type=last_error.__class__.__name__,
                        duration_seconds=duration,
                    )
                    debug_logger.finish_interaction()
                except Exception as log_error:
                    if self.verbose:
                        logger.warning(f"Debug logging error failed: {log_error.__class__.__name__}: {log_error}")
            lst_err_msg_str: str = f"{last_error.__class__.__name__}: {last_error}"
            raise Exception(f"All providers failed. Primary: '{primary_model_name}', Fallbacks: {len(self.fallback_providers)}. Last error: '{lst_err_msg_str}'") from last_error
