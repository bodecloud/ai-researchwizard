#!/usr/bin/env python3
"""
Headless GPT Researcher runner for Trask (`@openkotor/trask`).

Reads one JSON object from stdin (UTF-8), matching `HeadlessGptResearcherRequestPayload`:
  - query (required)
  - custom_prompt (optional)
  - source_urls (optional list[str])
  - query_domains (optional list[str])
  - report_type (optional, default research_report)
  - report_source (optional, default web)

Writes a single JSON object to stdout:
  { "report": "<markdown>", "research_information": { "source_urls": [...], "visited_urls": [...] } }

Stderr may contain library logs; stdout must remain valid JSON only.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "Missing dependency: python-dotenv.\n"
        "Install GPT Researcher dependencies from the monorepo:\n"
        "  pip install -r vendor/ai-researchwizard/requirements.txt\n"
        "Or quick fix:\n"
        "  pip install python-dotenv",
        file=sys.stderr,
    )
    sys.exit(2)


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _normalize_model_for_ai_researchwizard(model: str) -> str:
    value = model.strip()
    if not value:
        return "openrouter:openrouter/auto"
    if ":" in value:
        return value
    return f"litellm:{value}"


def _model_id_for_completion(normalized_model: str) -> str:
    """Convert provider:model into a litellm completion model id."""
    if ":" not in normalized_model:
        return normalized_model
    _, model = normalized_model.split(":", 1)
    return model


def _is_openrouter_limit_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "openrouter" in lowered
        and "403" in lowered
        and (
            "key limit exceeded" in lowered
            or "rate limit" in lowered
            or "total limit" in lowered
        )
    )


def _probe_model(normalized_model: str) -> tuple[bool, str]:
    """Return (ok, error_text). Used to quickly verify a model is currently usable."""
    try:
        from litellm import completion
    except Exception as exc:
        return False, f"litellm import failed: {exc}"

    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            completion(
                model=_model_id_for_completion(normalized_model),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
                timeout=12,
            )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _resolve_fallback_chat_model() -> str:
    script_dir = Path(__file__).resolve().parent
    llm_fallbacks_src = script_dir.parent / "llm_fallbacks" / "src"
    if llm_fallbacks_src.is_dir():
        sys.path.insert(0, str(llm_fallbacks_src))

    try:
        from llm_fallbacks import get_fallback_list
    except Exception:
        return "openrouter:openrouter/auto"

    try:
        models = list(get_fallback_list("chat"))
    except Exception:
        models = []

    if not models:
        return "openrouter:openrouter/auto"

    return _normalize_model_for_ai_researchwizard(models[0])


def _resolve_next_working_free_chat_model(exclude: set[str] | None = None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    llm_fallbacks_src = script_dir.parent / "llm_fallbacks" / "src"
    if llm_fallbacks_src.is_dir():
        sys.path.insert(0, str(llm_fallbacks_src))

    try:
        from llm_fallbacks import get_fallback_list
    except Exception:
        return None

    blocked = exclude or set()
    tried = 0
    # Keep probing bounded to avoid long startup delays.
    max_probes = 12

    for raw in list(get_fallback_list("chat")):
        normalized = _normalize_model_for_ai_researchwizard(raw)
        if normalized in blocked:
            continue
        if normalized.startswith("openrouter:"):
            # On OpenRouter 403, prefer non-OpenRouter free candidates next.
            continue
        tried += 1
        ok, _ = _probe_model(normalized)
        if ok:
            return normalized
        if tried >= max_probes:
            break

    return None


def _ensure_runtime_defaults() -> None:
    # Keep OpenRouter as primary implicit default when its key is available.
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        default_llm = "openrouter:openrouter/auto"
    else:
        default_llm = _resolve_fallback_chat_model()

    for key in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
        if not os.environ.get(key, "").strip():
            os.environ[key] = default_llm

    # If OpenRouter is currently rate-limited (403 key limit), rotate to the
    # next working free model immediately to avoid long retry chains.
    current_fast = _normalize_model_for_ai_researchwizard(os.environ.get("FAST_LLM", ""))
    if current_fast.startswith("openrouter:"):
        ok, err = _probe_model(current_fast)
        if not ok and _is_openrouter_limit_error(err):
            replacement = _resolve_next_working_free_chat_model(
                exclude={current_fast, "openrouter:openrouter/auto"},
            )
            if replacement:
                for key in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
                    configured = _normalize_model_for_ai_researchwizard(os.environ.get(key, ""))
                    if not configured or configured.startswith("openrouter:"):
                        os.environ[key] = replacement

    # Tavily remains optional; default to duckduckgo retriever when unset.
    if not os.environ.get("RETRIEVER", "").strip():
        os.environ["RETRIEVER"] = "duckduckgo"


def _payload_from_stdin() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        _die("stdin: expected JSON object", 1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _die(f"stdin: invalid JSON: {exc}", 1)
    if not isinstance(data, dict):
        _die("stdin: JSON root must be an object", 1)
    return data


async def _run_standard(payload: dict[str, Any]) -> dict[str, Any]:
    from gpt_researcher import GPTResearcher
    from gpt_researcher.utils.enum import ReportSource, ReportType, Tone

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        _die('payload: string field "query" is required', 1)

    custom_prompt = payload.get("custom_prompt")
    if custom_prompt is not None and not isinstance(custom_prompt, str):
        _die('payload: "custom_prompt" must be a string when set', 1)
    custom_prompt = custom_prompt or ""

    source_urls = payload.get("source_urls") or []
    if not isinstance(source_urls, list) or not all(isinstance(u, str) for u in source_urls):
        _die('payload: "source_urls" must be a list of strings when set', 1)

    query_domains = payload.get("query_domains") or []
    if not isinstance(query_domains, list) or not all(isinstance(d, str) for d in query_domains):
        _die('payload: "query_domains" must be a list of strings when set', 1)

    report_type = payload.get("report_type") or ReportType.ResearchReport.value
    report_source = payload.get("report_source") or ReportSource.Web.value
    if not isinstance(report_type, str):
        _die('payload: "report_type" must be a string', 1)
    if not isinstance(report_source, str):
        _die('payload: "report_source" must be a string', 1)

    # Wrap the constructor AND the async calls in redirect_stdout so that
    # any debug loggers initialised during GPTResearcher.__init__ don't
    # corrupt the JSON-only stdout contract.
    capture = io.StringIO()
    with redirect_stdout(capture):
        researcher = GPTResearcher(
            query=query.strip(),
            report_type=report_type,
            report_source=report_source,
            tone=Tone.Objective,
            source_urls=source_urls,
            query_domains=query_domains,
            complement_source_urls=False,
            verbose=False,
            encoding="utf-8",
        )
        await researcher.conduct_research()
        report = await researcher.write_report(custom_prompt=custom_prompt)

    visited = list(researcher.visited_urls)
    return {
        "report": report if isinstance(report, str) else str(report),
        "research_information": {
            "source_urls": visited,
            "visited_urls": visited,
        },
    }


async def _run_detailed(payload: dict[str, Any]) -> dict[str, Any]:
    """Matches `cli.py` detailed_report path (no custom_prompt hook in DetailedReport)."""
    from backend.report_type import DetailedReport

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        _die('payload: string field "query" is required', 1)

    source_urls_raw = payload.get("source_urls") or []
    query_domains_raw = payload.get("query_domains") or []
    if not isinstance(source_urls_raw, list):
        _die('payload: "source_urls" invalid', 1)
    if not isinstance(query_domains_raw, list):
        _die('payload: "query_domains" invalid', 1)

    source_urls = [u for u in source_urls_raw if isinstance(u, str)]
    query_domains = [d for d in query_domains_raw if isinstance(d, str)]

    capture = io.StringIO()
    with redirect_stdout(capture):
        detailed = DetailedReport(
            query=query.strip(),
            query_domains=query_domains,
            report_type="research_report",
            report_source="web_search",
            source_urls=source_urls,
        )
        report = await detailed.run()

    visited = list(detailed.gpt_researcher.visited_urls)
    return {
        "report": report if isinstance(report, str) else str(report),
        "research_information": {
            "source_urls": visited,
            "visited_urls": visited,
        },
    }


async def _async_main(payload: dict[str, Any]) -> dict[str, Any]:
    report_type = payload.get("report_type") or "research_report"
    if report_type == "detailed_report":
        return await _run_detailed(payload)
    return await _run_standard(payload)


def _ensure_ai_researchwizard_importable() -> None:
    try:
        from gpt_researcher import GPTResearcher  # noqa: F401
    except ImportError as exc:
        print(
            "Failed to import gpt_researcher (ai-researchwizard dependencies missing).\n"
            "From the monorepo root, install:\n"
            "  pip install -r vendor/ai-researchwizard/requirements.txt\n"
            "Or run: scripts/bootstrap_trask_ai_researchwizard.ps1 (Windows) or\n"
            "        scripts/bootstrap_trask_ai_researchwizard.sh (Unix)\n"
            f"Detail: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    load_dotenv()
    _ensure_runtime_defaults()
    _ensure_ai_researchwizard_importable()
    payload = _payload_from_stdin()
    try:
        result = asyncio.run(_async_main(payload))
    except SystemExit:
        raise
    except Exception as exc:
        _die(f"{type(exc).__name__}: {exc}", 1)

    report = result.get("report")
    if not isinstance(report, str) or not report.strip():
        _die("ai-researchwizard produced an empty report.", 1)

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
