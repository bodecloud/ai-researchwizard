"""Agent creation and selection utilities for GPT Researcher.

This module provides functions to automatically select and configure
the appropriate research agent based on the query type.
"""

import json
import logging
import re
from typing import Any, Callable

import json_repair

from ..config import Config
from ..prompts import PromptFamily
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)

async def choose_agent(
    query: str,
    cfg: Config,
    parent_query: str | None = None,
    cost_callback: Callable[[float], None] | None = None,
    headers: dict[str, str] | None = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs,
) -> tuple[str, str]:
    """Chooses the agent automatically.

    Args:
        query: original query
        cfg: Config object
        parent_query: In some cases the research is conducted on a subtopic from the main query.
            The parent query allows the agent to know the main context for better reasoning.
        cost_callback: callback for calculating llm costs
        headers: headers for the llm request
        prompt_family: Family of prompts

    Returns:
        agent: Agent name
        agent_role_prompt: Agent role prompt
    """
    parent_query = "" if parent_query is None else parent_query
    query = f"{parent_query} - {query}" if parent_query else f"{query}"
    response: str | None = None

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[
                {"role": "system", "content": f"{prompt_family.auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"},
            ],
            llm_provider=cfg.strategic_llm_provider,
#            max_tokens=cfg.strategic_token_limit,  # pyright: ignore[reportAttributeAccessIssue]
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            cfg=cfg,
            **kwargs,
        )
    except Exception as e:
        print(f"Error with strategic LLM, falling back to smart LLM: {e.__class__.__name__}: {e}")
        import sys

        sys.stderr.write(f"Error with strategic LLM, falling back to smart LLM: {e.__class__.__name__}: {e}\n")
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{prompt_family.auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"},
            ],
            temperature=0.15,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            cfg=cfg,
            **kwargs
        )

    try:
        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception:
        return await handle_json_error(response)


async def handle_json_error(response: str | None) -> tuple[str, str]:
    try:
        agent_dict: dict[str, Any] = json_repair.loads(response)
        if agent_dict.get("server") and agent_dict.get("agent_role_prompt"):  # pyright: ignore[reportAttributeAccessIssue]
            return agent_dict["server"], agent_dict["agent_role_prompt"]  # pyright: ignore[reportCallIssue, reportIndexIssue, reportOptionalSubscript]
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(
            f"Failed to parse agent JSON with json_repair: {error_type}: {error_msg}",
            exc_info=True
        )
        if response:
            logger.debug(f"LLM response that failed to parse: {response[:500]}...")

    json_string: str | None = extract_json_with_regex(response)
    if json_string is not None:
        try:
            json_data: dict[str, Any] = json.loads(json_string)
            return json_data["server"], json_data["agent_role_prompt"]
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to decode JSON from regex extraction: {str(e)}",
                exc_info=True
            )

    logger.info("No valid JSON found in LLM response. Falling back to default agent.")
    return "Default Agent", (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )


def extract_json_with_regex(response: str | None) -> str | None:
    """Extract JSON object from a string using regex.

    Attempts to find the first JSON object pattern in the response string.

    Args:
        response: The string to search for JSON content.

    Returns:
        The extracted JSON string if found, None otherwise.
    """
    if not response:
        return None
    json_match = re.search(r"{.*?}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None
