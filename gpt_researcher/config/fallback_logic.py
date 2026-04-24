from __future__ import annotations

import logging
import os
import re
import traceback
from typing import TYPE_CHECKING, Any


from gpt_researcher.llm_provider import GenericLLMProvider

if TYPE_CHECKING:
    from typing_extensions import Literal

    from gpt_researcher.config import Config
    from llm_fallbacks.config import LiteLLMBaseModelSpec


logger: logging.Logger = logging.getLogger(__name__)


MAX_FALLBACKS: int = 25

_LITELLM_PROVIDER_ALIASES: dict[str, str] = {
    "google": "gemini",
    "google_genai": "gemini",
    "google-generativeai": "gemini",
    "vertexai": "vertex_ai",
    "together": "together_ai",
}

_LITELLM_KNOWN_PROVIDER_PREFIXES: set[str] = {
    "anthropic",
    "azure",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "groq",
    "huggingface",
    "ollama",
    "openai",
    "openrouter",
    "replicate",
    "together_ai",
    "vertex_ai",
    "xai",
}
def _infer_litellm_provider_from_model(model_id: str) -> str | None:
    normalized = (model_id or "").strip().lower()
    if not normalized or "/" not in normalized:
        return None
    return normalized.split("/", 1)[0]

_OPENROUTER_VENDOR_PREFIXES: set[str] = {
    "anthropic",
    "deepseek",
    "google",
    "meta",
    "meta-llama",
    "microsoft",
    "mistralai",
    "openai",
    "qwen",
}


def _env_value_from_var_or_file(env_name: str) -> str | None:
    """Return secret value from ENV or matching *_FILE, if available."""
    direct_value: str | None = os.environ.get(env_name)
    if direct_value and direct_value.strip():
        return direct_value.strip()

    file_var: str = f"{env_name}_FILE"
    file_path: str | None = os.environ.get(file_var)
    if not file_path or not file_path.strip():
        return None

    try:
        with open(file_path.strip(), "r", encoding="utf-8") as secret_file:
            value: str = secret_file.read().strip()
    except OSError:
        return None

    return value or None


def _looks_placeholder_secret(value: str | None) -> bool:
    """Best-effort filter for unset/demo placeholder secret values."""
    normalized: str = (value or "").strip().lower()
    if not normalized:
        return True

    placeholder_fragments: tuple[str, ...] = (
        "your_",
        "replace",
        "changeme",
        "example",
        "dummy",
        "placeholder",
        "test-key",
        "linux_env",
    )
    return any(fragment in normalized for fragment in placeholder_fragments)


def _is_provider_available(provider_name: str | None) -> bool:
    provider = (provider_name or "").strip().lower()
    if not provider:
        return False

    provider = _LITELLM_PROVIDER_ALIASES.get(provider, provider)

    key_requirements: dict[str, tuple[str, ...]] = {
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
        "groq": ("GROQ_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "mistralai": ("MISTRAL_API_KEY", "MISTRALAI_API_KEY"),
        "together": ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "google_genai": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "google_vertexai": ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"),
        "huggingface": ("HUGGINGFACE_API_TOKEN", "HF_TOKEN"),
        "xai": ("XAI_API_KEY",),
        "cohere": ("COHERE_API_KEY",),
        "fireworks": ("FIREWORKS_API_KEY",),
        "dashscope": ("DASHSCOPE_API_KEY",),
        "aimlapi": ("AIMLAPI_API_KEY",),
        "forge": ("FORGE_API_KEY",),
        "gigachat": ("GIGACHAT_API_KEY",),
        "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
    }

    if provider in {"litellm", "ollama"}:
        return True

    required_keys = key_requirements.get(provider)
    if not required_keys:
        return True

    has_key = any(
        bool(secret_value)
        and not _looks_placeholder_secret(secret_value)
        for secret_value in (_env_value_from_var_or_file(key) for key in required_keys)
    )
    if not has_key:
        return False

    # Azure models also require an endpoint/base URL. A key alone can still
    # cause LiteLLM Azure initialization to fail hard at runtime.
    if provider == "azure":
        has_azure_endpoint = any(
            bool((os.environ.get(key) or "").strip())
            for key in ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT")
        )
        if not has_azure_endpoint:
            return False

    return True


def _normalize_litellm_model_identifier(
    model_id: str,
    provider_hint: str | None = None,
) -> str | None:
    """Return a LiteLLM-compatible model id, or None when it cannot be inferred safely."""
    normalized_model = (model_id or "").strip().lower()
    if not normalized_model:
        return None

    hint = (provider_hint or "").strip().lower()
    hint = _LITELLM_PROVIDER_ALIASES.get(hint, hint)

    if "/" in normalized_model:
        prefix = normalized_model.split("/", 1)[0]
        if prefix in _LITELLM_KNOWN_PROVIDER_PREFIXES:
            return normalized_model

        if hint in _LITELLM_KNOWN_PROVIDER_PREFIXES:
            if hint == "openrouter":
                return f"openrouter/{normalized_model}"
            return f"{hint}/{normalized_model}"

        if prefix in _OPENROUTER_VENDOR_PREFIXES:
            return f"openrouter/{normalized_model}"

        return None

    if hint in _LITELLM_KNOWN_PROVIDER_PREFIXES:
        return f"{hint}/{normalized_model}"

    if normalized_model.startswith("gemini"):
        return f"gemini/{normalized_model}"

    return None


def _get_model_provider_family(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower()
    if ":" in model_id:
        return model_id.split(":", 1)[0].lower()
    return model_id.lower()


def _get_runtime_fallback_priority(model_id: str) -> int:
    provider_family = _get_model_provider_family(model_id)

    if provider_family == "openrouter":
        return 0
    if provider_family == "openai" and os.environ.get("OPENAI_API_KEY"):
        return 1
    if provider_family == "groq" and os.environ.get("GROQ_API_KEY"):
        return 2

    return 10


def _extend_chat_models_with_runtime_candidates(
    free_models: list[tuple[str, "LiteLLMBaseModelSpec"]],
) -> list[tuple[str, "LiteLLMBaseModelSpec"]]:
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        return free_models

    try:
        from llm_fallbacks.config import ALL_CHAT_MODELS
    except Exception:
        return free_models

    seen_model_ids: set[str] = {model_id for model_id, _ in free_models}
    runtime_candidates: list[tuple[int, int, tuple[str, LiteLLMBaseModelSpec]]] = []

    for index, (model_id, spec) in enumerate(ALL_CHAT_MODELS):
        if model_id in seen_model_ids:
            continue

        provider_family = _get_model_provider_family(model_id)
        litellm_provider: str | None = spec.get("litellm_provider") or spec.get("provider")

        if provider_family == "openai":
            if not os.environ.get("OPENAI_API_KEY") or litellm_provider != "openai":
                continue
        elif provider_family == "groq":
            if not os.environ.get("GROQ_API_KEY") or litellm_provider != "groq":
                continue
        else:
            continue

        runtime_candidates.append(
            (_get_runtime_fallback_priority(model_id), index, (model_id, spec))
        )

    if not runtime_candidates:
        return free_models

    sorted_runtime_candidates = [
        candidate
        for _, _, candidate in sorted(runtime_candidates, key=lambda item: (item[0], item[1]))
    ]
    return [*free_models, *sorted_runtime_candidates]

def map_litellm_provider_to_gptr_provider(
    litellm_provider_name: str | None,
) -> str | None:
    """Maps LiteLLM provider names to GPT Researcher expected provider names.

    Key Requirements:
    - OpenRouter models: Keep provider as "openrouter"
    - All other models: Provider should be "litellm" (normalized for llm_fallbacks system)
    - This is ONLY for llm_fallbacks models, not user-configured models
    """
    if not litellm_provider_name or not litellm_provider_name.strip():
        return None

    name_lower: str = litellm_provider_name.lower()

    # Prefer native GPT-Researcher providers whenever they are supported.
    # This avoids sending provider-less model identifiers to ChatLiteLLM.
    provider_aliases: dict[str, str] = {
        "gemini": "google_genai",
        "google": "google_genai",
        "vertex_ai": "google_vertexai",
        "vertexai": "google_vertexai",
        "mistral": "mistralai",
        "together_ai": "together",
        "vllm": "vllm_openai",
    }

    mapped_name: str = provider_aliases.get(name_lower, name_lower)

    # Keep known providers as-is so each client receives the model format it expects.
    supported_native_providers: set[str] = {
        "aimlapi",
        "anthropic",
        "azure_openai",
        "bedrock",
        "cohere",
        "dashscope",
        "deepseek",
        "fireworks",
        "gigachat",
        "google_genai",
        "google_vertexai",
        "groq",
        "huggingface",
        "mistralai",
        "netmind",
        "ollama",
        "openai",
        "openrouter",
        "together",
        "vllm_openai",
        "xai",
        "forge",
    }

    if mapped_name in supported_native_providers:
        return mapped_name

    # Fallback to LiteLLM only when we cannot confidently map the provider.
    return "litellm"


def generate_auto_fallbacks_from_free_models(
    free_models: list[tuple[str, LiteLLMBaseModelSpec]],
    model_type: Literal["embedding", "chat", "strategic_chat", "fast_chat"],
    clean_primary_model_id: str,
    fast_token_limit: int,
    smart_token_limit: int,
    strategic_token_limit: int,
    verbose_logging: bool = False,
) -> list[str]:
    """Extract auto-generation logic for reuse in manual fallback mode."""
    # Get required token limit based on model type
    required_token_limit: int = 0
    if model_type == "fast_chat":
        required_token_limit = fast_token_limit
    elif model_type == "strategic_chat":
        required_token_limit = strategic_token_limit
    elif model_type == "chat":
        required_token_limit = smart_token_limit

    # Filter models based on token limit and other criteria
    final_candidates: list[str] = []
    blacklisted_patterns: list[str] = GenericLLMProvider.MODEL_BLACKLIST
    filtered_counts: dict[str, int] = {
        "blacklisted": 0,
        "wrong_mode": 0,
        "insufficient_tokens": 0,
    }

    # Process free models
    for model_id, spec in free_models:
        # Skip blacklisted models
        blacklist_match: str | None = None
        for pattern in blacklisted_patterns:
            if re.search(pattern, model_id, re.IGNORECASE):
                blacklist_match = pattern
                break

        if blacklist_match and blacklist_match.strip():
            filtered_counts["blacklisted"] += 1
            continue

        # Skip non-matching model types
        model_mode: str = spec.get("mode", "unknown")
        if model_type == "embedding" and model_mode != "embedding":
            filtered_counts["wrong_mode"] += 1
            continue
        elif (
            model_type in {"chat", "strategic_chat", "fast_chat"}
            and model_mode != "chat"
        ):
            filtered_counts["wrong_mode"] += 1
            continue

        # Check token capacity for chat models
        if model_type in {"chat", "strategic_chat", "fast_chat"}:
            max_output_tokens: int = spec.get("max_output_tokens", 0)
            # Include models with 0 tokens (likely means unknown/not configured)
            # Only filter out models that have a known token limit that's insufficient
            if max_output_tokens > 0 and max_output_tokens < required_token_limit:
                filtered_counts["insufficient_tokens"] += 1
                continue

        # Model passed all filters
        final_candidates.append(model_id)

    # Summary debug logging
    logger.debug(
        f"Filtering summary for {model_type}: "
        f"blacklisted={filtered_counts['blacklisted']}, "
        f"wrong_mode={filtered_counts['wrong_mode']}, "
        f"insufficient_tokens={filtered_counts['insufficient_tokens']}, "
        f"accepted={len(final_candidates)}"
    )

    # Remove the primary model from fallbacks if present
    if clean_primary_model_id and clean_primary_model_id.strip():
        final_candidates: list[str] = [
            model_id
            for model_id in final_candidates
            if model_id != clean_primary_model_id
        ]

    # Convert model IDs to correct fallback format
    formatted_fallbacks: list[str] = []
    free_models_dict: dict[str, LiteLLMBaseModelSpec] = dict(free_models)

    # Separate OpenRouter models for prioritization (to match manual config pattern)
    openrouter_candidates: list[str] = []
    other_candidates: list[tuple[int, int, str]] = []

    for index, model_id_key in enumerate(final_candidates):
        spec: LiteLLMBaseModelSpec = free_models_dict.get(model_id_key, {})
        litellm_provider: str | None = spec.get("litellm_provider") or spec.get("provider")

        # Map litellm provider to GPT-Researcher supported provider
        mapped_provider: str | None = map_litellm_provider_to_gptr_provider(litellm_provider)

        # Check if this is an OpenRouter model (prioritize like manual config)
        # OpenRouter models can have various litellm_provider values but all map to "openrouter"
        if (
            mapped_provider == "openrouter"
            or litellm_provider == "openrouter"
            or model_id_key.startswith("openrouter/")
        ):
            openrouter_candidates.append(model_id_key)
        else:
            other_candidates.append(
                (_get_runtime_fallback_priority(model_id_key), index, model_id_key)
            )

    # Process OpenRouter models first (to match manual priority)
    prioritized_other_candidates: list[str] = [
        model_id_key
        for _, _, model_id_key in sorted(other_candidates, key=lambda item: (item[0], item[1]))
    ]
    prioritized_candidates: list[str] = openrouter_candidates + prioritized_other_candidates

    for model_id_key in prioritized_candidates:
        original_model_id_lower: str = model_id_key.lower()
        model_id_key_lower: str = original_model_id_lower
        spec: LiteLLMBaseModelSpec = free_models_dict.get(model_id_key, {})
        litellm_provider: str | None = spec.get("litellm_provider") or spec.get("provider")

        # Map litellm provider to GPT-Researcher supported provider
        mapped_provider: str | None = map_litellm_provider_to_gptr_provider(litellm_provider)

        # Use mapped provider if available, fallback to original
        final_provider: str = mapped_provider or litellm_provider or "unknown"

        if not _is_provider_available(final_provider):
            continue

        # Determine actual model name and format according to manual patterns
        if "/" in model_id_key_lower:
            # e.g. "openrouter/meta-llama/llama-3-8b-instruct" -> "openrouter:meta-llama/llama-3-8b-instruct:free"
            split: list[str] = model_id_key_lower.split("/", 1)
            provider_from_key, model_id_key_lower = split[0], split[1]

            # If the key starts with a provider name, use the mapped version of that provider
            if provider_from_key in ["openrouter", "litellm"]:
                final_provider = provider_from_key
            # For models from llm_fallbacks, most should map to openrouter or litellm
            elif mapped_provider in ["openrouter", "litellm"]:
                final_provider = mapped_provider

            # Some generated model IDs are shaped like "meta/..." for OpenRouter.
            # Preserve the original qualified ID for OpenRouter to avoid dropping vendor prefixes.
            if final_provider == "openrouter" and provider_from_key not in ["openrouter"]:
                model_id_key_lower = original_model_id_lower

        elif (
            model_id_key_lower.count(":") == 1 and not model_id_key_lower.endswith(":free")
            or model_id_key_lower.count(":") == 2 and model_id_key_lower.endswith(":free")
        ):
            # Already in provider:model_name format
            formatted_fallbacks.append(model_id_key)
            continue

        # LiteLLM-backed models need to keep their provider-qualified model identifier.
        if final_provider == "litellm":
            formatted_model_id = _normalize_litellm_model_identifier(
                original_model_id_lower,
                litellm_provider,
            )
            if not formatted_model_id:
                continue
            litellm_provider_family = _infer_litellm_provider_from_model(formatted_model_id)
            if litellm_provider_family and not _is_provider_available(litellm_provider_family):
                continue
        else:
            formatted_model_id = model_id_key_lower

        # Format according to manual configuration patterns using mapped providers
        final_format: str = f"{final_provider}:{formatted_model_id}"
        formatted_fallbacks.append(final_format)

    # Final check for environment requirements
    if (
        not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or not os.environ.get("GOOGLE_CLOUD_PROJECT")
    ):
        formatted_fallbacks = [
            fb
            for fb in formatted_fallbacks
            if "google_vertexai" not in fb and "vertex_ai" not in fb
        ]

    # Apply MAX_FALLBACKS limit
    final_result: list[str] = formatted_fallbacks[:MAX_FALLBACKS]

    return final_result


def parse_model_fallbacks(
    fallbacks_str: str,
    model_type: Literal["embedding", "chat", "strategic_chat", "fast_chat"],
    primary_model_id: str,
    fast_token_limit: int = 3000,
    smart_token_limit: int = 6000,
    strategic_token_limit: int = 4000,
) -> list[str]:
    """Parse model fallbacks string into a list of model names."""
    if not fallbacks_str:
        return []

    # Clean the primary model ID to handle empty or "auto" values
    clean_primary_model_id: str = (
        ""
        if not primary_model_id or primary_model_id.strip().lower() == "auto"
        else primary_model_id
    )

    # For manual fallbacks, honor only the explicitly configured models.
    # Auto-generated free-model fallbacks are only used when the value is "auto".
    if fallbacks_str.strip().lower() != "auto":
        raw_fallbacks: list[str] = [
            fb.strip() for fb in fallbacks_str.split(",") if fb.strip()
        ]
        parsed_fallbacks: list[str] = [
            fb
            for fb in raw_fallbacks
            if not clean_primary_model_id or fb != clean_primary_model_id
        ]

        # Deduplicate manual fallbacks
        seen: set[str] = set()
        manual_fallbacks: list[str] = [
            x for x in parsed_fallbacks if not (x in seen or seen.add(x))
        ]

        # Manual mode must stay explicit: do not append auto-generated fallbacks.
        return manual_fallbacks[:MAX_FALLBACKS]

    # Auto-generate fallbacks
    try:
        # Load model specs from llm_fallbacks package
        if model_type == "embedding":
            from llm_fallbacks.config import FREE_EMBEDDING_MODELS

            free_models: list[tuple[str, LiteLLMBaseModelSpec]] = FREE_EMBEDDING_MODELS
        elif model_type == "chat":
            from llm_fallbacks.config import FREE_CHAT_MODELS

            free_models: list[tuple[str, LiteLLMBaseModelSpec]] = FREE_CHAT_MODELS
        else:
            from llm_fallbacks.config import FREE_MODELS

            free_models = FREE_MODELS

        if model_type in {"chat", "strategic_chat", "fast_chat"}:
            free_models = _extend_chat_models_with_runtime_candidates(free_models)

        # Use the enhanced method
        final_fallbacks_list: list[str] = generate_auto_fallbacks_from_free_models(
            free_models,
            model_type,
            clean_primary_model_id,
            fast_token_limit,
            smart_token_limit,
            strategic_token_limit,
            verbose_logging=False,
        )

        return final_fallbacks_list

    except Exception:
        if os.getenv("VERBOSE", "").lower() in ("true", "1"):
            traceback.print_exc()
        return []


def initialize_fallback_providers_for_type(
    fallback_str_list: list[str] | None = None,
    llm_type_name: Literal["smart", "fast", "strategic", "embedding"] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    chat_log: str | None = None,
    verbose: bool = True,
) -> list[GenericLLMProvider]:
    """Initialize fallback providers for a specific LLM type."""
    initialized_providers: list[GenericLLMProvider] = []
    if not fallback_str_list:
        return initialized_providers

    if llm_kwargs is None:
        llm_kwargs = {}

    for i, fallback_spec_str in enumerate(fallback_str_list):
        try:
            # Split into provider and model name
            if ":" not in fallback_spec_str:
                continue

            provider_name, model_name = fallback_spec_str.split(":", 1)

            # Fix OpenRouter model names to ensure they have the proper provider prefix
            if provider_name == "openrouter":
                # Check if model name needs a provider prefix
                if "/" not in model_name:
                    # This is likely a model that needs a provider prefix
                    # Common OpenRouter model patterns that need prefixes
                    if any(pattern in model_name.lower() for pattern in [
                        "gemini", "claude", "gpt", "llama", "mistral", "qwen", "deepseek"
                    ]):
                        # Try to determine the correct prefix based on model name
                        if "gemini" in model_name.lower():
                            model_name = f"google/{model_name}"
                        elif "claude" in model_name.lower():
                            model_name = f"anthropic/{model_name}"
                        elif "gpt" in model_name.lower():
                            model_name = f"openai/{model_name}"
                        elif "llama" in model_name.lower():
                            model_name = f"meta-llama/{model_name}"
                        elif "mistral" in model_name.lower():
                            model_name = f"mistralai/{model_name}"
                        elif "qwen" in model_name.lower():
                            model_name = f"qwen/{model_name}"
                        elif "deepseek" in model_name.lower():
                            model_name = f"deepseek/{model_name}"

            # Construct kwargs for the provider
            # Start with general llm_kwargs and add the specific model
            if not _is_provider_available(provider_name):
                continue

            if provider_name == "litellm":
                model_name = _normalize_litellm_model_identifier(model_name)
                if not model_name:
                    continue
                litellm_provider_family = _infer_litellm_provider_from_model(model_name)
                if litellm_provider_family and not _is_provider_available(litellm_provider_family):
                    continue

            provider_kwargs: dict[str, Any] = llm_kwargs.copy()
            provider_kwargs["model"] = model_name
            provider_kwargs["verbose"] = verbose

            fallback_instance: GenericLLMProvider = GenericLLMProvider.from_provider(
                provider_name,
                chat_log=chat_log,
                **provider_kwargs,
            )
            initialized_providers.append(fallback_instance)

        except Exception:
            # Always show the traceback info if DEBUG_FALLBACKS is set
            if (
                os.getenv("DEBUG_FALLBACKS", "").lower() in ("true", "1")
                or os.getenv("VERBOSE", "").lower() in ("true", "1")
            ):
                tb_str: str = traceback.format_exc()
                print(f"Traceback for '{fallback_spec_str}':\n{tb_str}")

    return initialized_providers


def set_llm_attributes(config_instance: Config) -> None:
    """Set LLM attributes for a config instance.

    Args:
        config_instance: The Config instance to set attributes on
    """
    # Parse fallbacks for each LLM type first (list of strings)
    fast_llm_fallback_str_list: list[str] = parse_model_fallbacks(
        config_instance.fast_llm_fallbacks,
        "fast_chat",
        config_instance.fast_llm or "",
        config_instance.fast_token_limit,
        config_instance.smart_token_limit,
        config_instance.strategic_token_limit,
    )
    smart_llm_fallback_str_list: list[str] = parse_model_fallbacks(
        config_instance.smart_llm_fallbacks,
        "chat",
        config_instance.smart_llm or "",
        config_instance.fast_token_limit,
        config_instance.smart_token_limit,
        config_instance.strategic_token_limit,
    )
    strategic_llm_fallback_str_list: list[str] = parse_model_fallbacks(
        config_instance.strategic_llm_fallbacks,
        "strategic_chat",
        config_instance.strategic_llm or "",
        config_instance.fast_token_limit,
        config_instance.smart_token_limit,
        config_instance.strategic_token_limit,
    )

    # If main LLM is empty or "auto", use the first fallback
    if not config_instance.fast_llm or config_instance.fast_llm.strip().lower() == "auto":
        if fast_llm_fallback_str_list:
            config_instance.fast_llm = fast_llm_fallback_str_list[0]

    if not config_instance.smart_llm or config_instance.smart_llm.strip().lower() == "auto":
        if smart_llm_fallback_str_list:
            config_instance.smart_llm = smart_llm_fallback_str_list[0]

    if not config_instance.strategic_llm or config_instance.strategic_llm.strip().lower() == "auto":
        if strategic_llm_fallback_str_list:
            config_instance.strategic_llm = strategic_llm_fallback_str_list[0]

    # Now parse the main LLMs (which might have been updated from fallbacks)
    # Import the Config class to access the static parse_llm method
    from gpt_researcher.config.config import Config
    config_instance.fast_llm_provider, config_instance.fast_llm_model = Config.parse_llm(config_instance.fast_llm)
    config_instance.smart_llm_provider, config_instance.smart_llm_model = Config.parse_llm(config_instance.smart_llm)
    config_instance.strategic_llm_provider, config_instance.strategic_llm_model = Config.parse_llm(config_instance.strategic_llm)

    # Remove the main model from fallbacks to avoid duplication
    if config_instance.fast_llm and config_instance.fast_llm in fast_llm_fallback_str_list:
        fast_llm_fallback_str_list.remove(config_instance.fast_llm)
    if config_instance.smart_llm and config_instance.smart_llm in smart_llm_fallback_str_list:
        smart_llm_fallback_str_list.remove(config_instance.smart_llm)
    if config_instance.strategic_llm and config_instance.strategic_llm in strategic_llm_fallback_str_list:
        strategic_llm_fallback_str_list.remove(config_instance.strategic_llm)

    # Initialize fallback providers (use cached values if available)
    # Import the Config class to access class variables
    if not Config._fallbacks_initialized:
        Config._cached_fast_llm_fallback_providers = initialize_fallback_providers_for_type(fast_llm_fallback_str_list, "fast", config_instance.llm_kwargs, getattr(config_instance, "chat_log", None), getattr(config_instance, "verbose", True))
        Config._cached_smart_llm_fallback_providers = initialize_fallback_providers_for_type(smart_llm_fallback_str_list, "smart", config_instance.llm_kwargs, getattr(config_instance, "chat_log", None), getattr(config_instance, "verbose", True))
        Config._cached_strategic_llm_fallback_providers = initialize_fallback_providers_for_type(strategic_llm_fallback_str_list, "strategic", config_instance.llm_kwargs, getattr(config_instance, "chat_log", None), getattr(config_instance, "verbose", True))
        Config._fallbacks_initialized = True

    # Assign the cached providers to this instance
    config_instance.fast_llm_fallback_providers = Config._cached_fast_llm_fallback_providers
    config_instance.smart_llm_fallback_providers = Config._cached_smart_llm_fallback_providers
    config_instance.strategic_llm_fallback_providers = Config._cached_strategic_llm_fallback_providers
