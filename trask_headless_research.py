#!/usr/bin/env python3
"""
Headless GPT Researcher runner for Trask (`@bodencrouch/trask`).

Reads one JSON object from stdin (UTF-8), matching `HeadlessGptResearcherRequestPayload`:
  - query (required)
  - custom_prompt (optional)
  - source_urls (optional list[str])
  - query_domains (optional list[str])
    - model (optional provider-qualified model id, e.g. openrouter:openrouter/auto)
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
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


TRASK_FREE_CHAT_MODELS = [
    "gemini/gemini-2.5-flash-lite",
    "groq/llama-3.1-8b-instant",
]


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _normalize_model_for_ai_researchwizard(model: str) -> str:
    value = model.strip()
    if not value:
        return "openrouter:openrouter/free"
    if ":" in value:
        return value
    return f"litellm:{value}"


def _resolve_payload_model(payload: dict[str, Any]) -> str | None:
    raw = payload.get("model")
    if raw is None:
        return None
    if not isinstance(raw, str):
        _die('payload: "model" must be a string when set', 1)
    if raw.strip().casefold() == "auto":
        return None
    normalized = _normalize_model_for_ai_researchwizard(raw)
    if len(normalized) > 180:
        _die('payload: "model" is too long', 1)
    return normalized


def _apply_payload_model_override(payload: dict[str, Any]) -> None:
    model = _resolve_payload_model(payload)
    if not model:
        return
    for key in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
        os.environ[key] = model


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
                timeout=6,
            )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _free_chat_fallback_models() -> list[str]:
    if os.environ.get("TRASK_DYNAMIC_FREE_MODELS", "").strip().lower() not in {"1", "true", "yes"}:
        return list(TRASK_FREE_CHAT_MODELS)

    script_dir = Path(__file__).resolve().parent
    llm_fallbacks_src = script_dir.parent / "llm_fallbacks" / "src"
    if llm_fallbacks_src.is_dir():
        sys.path.insert(0, str(llm_fallbacks_src))

    try:
        from llm_fallbacks.config import FREE_CHAT_MODELS
    except Exception:
        try:
            from llm_fallbacks import filter_models
        except Exception:
            return []
        try:
            return list(filter_models(model_type="chat", free_only=True))
        except Exception:
            return []

    try:
        return [name for name, _ in FREE_CHAT_MODELS]
    except Exception:
        return []


def _resolve_fallback_chat_model() -> str:
    models = _free_chat_fallback_models()

    for raw in models:
        normalized = _normalize_model_for_ai_researchwizard(raw)
        if not normalized.startswith("openrouter:"):
            return normalized

    if models:
        return _normalize_model_for_ai_researchwizard(models[0])

    return "openrouter:openrouter/free"


def _resolve_next_working_free_chat_model(exclude: set[str] | None = None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    llm_fallbacks_src = script_dir.parent / "llm_fallbacks" / "src"
    if llm_fallbacks_src.is_dir():
        sys.path.insert(0, str(llm_fallbacks_src))

    blocked = exclude or set()
    tried = 0
    # Keep probing bounded to avoid long startup delays.
    max_probes = 5

    for raw in _free_chat_fallback_models():
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
    default_llm = _resolve_fallback_chat_model()
    os.environ.setdefault("TRASK_FAST_QA", "1")
    os.environ.setdefault("TRASK_DISABLE_EMERGENCY_FALLBACKS", "1")

    for key in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
        current = _normalize_model_for_ai_researchwizard(os.environ.get(key, ""))
        if not current or current.startswith("openrouter:"):
            os.environ[key] = default_llm

    # Cap fallback provider initialization. The upstream "auto" setting
    # instantiates many providers and can dominate cold start. Repeating the
    # primary model in the manual fallback list makes upstream parsing remove
    # it as a duplicate, leaving an explicit empty fallback chain.
    primary = os.environ.get("FAST_LLM", "")
    fallback_env_value = primary
    for key in ("FAST_LLM_FALLBACKS", "SMART_LLM_FALLBACKS", "STRATEGIC_LLM_FALLBACKS"):
        if not os.environ.get(key, "").strip() or os.environ[key].strip().lower() == "auto":
            os.environ[key] = fallback_env_value

    # Prefer Tavily when an API key is set: it returns content snippets,
    # which dramatically reduces scrape volume vs. duckduckgo + per-URL fetches.
    if not os.environ.get("RETRIEVER", "").strip():
        if os.environ.get("TAVILY_API_KEY", "").strip():
            os.environ["RETRIEVER"] = "tavily"
        else:
            os.environ["RETRIEVER"] = "duckduckgo"

    # Performance knobs for Trask: keep wall time under 90s on free models.
    # Upstream config.py mishandles `from __future__ import annotations` when
    # reading typed env overrides, so we ship a JSON config file instead and
    # point CONFIG_PATH at it (only when the operator hasn't pinned one).
    if not os.environ.get("CONFIG_PATH", "").strip():
        import tempfile
        perf_overrides = {
            "MAX_ITERATIONS": 1,
            "MAX_SEARCH_RESULTS_PER_QUERY": 3,
            "MAX_SCRAPER_WORKERS": 6,
            "TOTAL_WORDS": 180,
            "SUMMARY_TOKEN_LIMIT": 240,
            "CURATE_SOURCES": False,
        }
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "trask-research"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = tmp_dir / "perf-overrides.json"
            cfg_path.write_text(json.dumps(perf_overrides), encoding="utf-8")
            os.environ["CONFIG_PATH"] = str(cfg_path)
        except OSError:
            # If we can't write the config file, fall back to defaults.
            pass


def _payload_string_list(payload: dict[str, Any], field: str) -> list[str]:
    raw = payload.get(field) or []
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        _die(f'payload: "{field}" must be a list of strings when set', 1)
    return [value.strip().rstrip("/") for value in raw if value.strip()]


def _set_allowlist_env_from_payload(payload: dict[str, Any]) -> None:
    """Keep direct CLI/script execution as restricted as the Node spawn path."""
    query_domains = _payload_string_list(payload, "query_domains")
    allowed_url_prefixes = _payload_string_list(payload, "allowed_url_prefixes")
    if query_domains:
        os.environ["TRASK_ALLOWED_QUERY_DOMAINS"] = "\n".join(query_domains)
    if allowed_url_prefixes:
        os.environ["TRASK_ALLOWED_URL_PREFIXES"] = "\n".join(allowed_url_prefixes)


def _allowlist_values(env_name: str) -> list[str]:
    raw = os.environ.get(env_name, "")
    return [part.strip().rstrip("/") for part in raw.replace(",", "\n").splitlines() if part.strip()]


def _hostname_matches(hostname: str, allowed_domain: str) -> bool:
    host = hostname.lower().removeprefix("www.")
    domain = allowed_domain.lower().removeprefix("www.")
    return host == domain or host.endswith(f".{domain}")


def _url_allowed(url: str) -> bool:
    prefixes = _allowlist_values("TRASK_ALLOWED_URL_PREFIXES")
    domains = _allowlist_values("TRASK_ALLOWED_QUERY_DOMAINS")
    if not prefixes and not domains:
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    normalized = url.strip().rstrip("/")
    if prefixes:
        return any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in prefixes)
    return bool(domains and any(_hostname_matches(parsed.hostname or "", domain) for domain in domains))


def _partition_allowed_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = raw.strip().rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        if _url_allowed(url):
            accepted.append(url)
        else:
            rejected.append(url)
    return accepted, rejected


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


def _context_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n\n".join(str(part) for part in context if str(part).strip())
    return str(context)


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_.+-]{4,}", query)
        if token.lower() not in {"what", "where", "when", "which", "with", "from", "kotor", "kotor1", "kotor2"}
    }


def _parse_context_blocks(context: Any) -> list[dict[str, str]]:
    text = _context_text(context)
    blocks: list[dict[str, str]] = []
    pattern = re.compile(
        r"Source:\s*(?P<url>\S+)\s*\nTitle:\s*(?P<title>.*?)\s*\nContent:\s*(?P<content>.*?)(?=\nSource:\s*\S+\s*\nTitle:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        url = match.group("url").strip()
        title = re.sub(r"\s+", " ", match.group("title")).strip() or url
        content = re.sub(r"\s+", " ", match.group("content")).strip()
        if url and content:
            blocks.append({"url": url, "title": title, "content": content})

    if blocks:
        return blocks

    compact = re.sub(r"\s+", " ", text).strip()
    return [{"url": "", "title": "Research context", "content": compact}] if compact else []


def _sentence_score(sentence: str, terms: set[str]) -> int:
    lowered = sentence.lower()
    score = sum(2 for term in terms if term in lowered)
    if re.search(r"\b(?:tool|tools|model|models|mdl|mdx|blender|3ds\s*max|import|export|edit|convert|compile|decompile)\b", lowered):
        score += 3
    if re.search(r"\b(?:pykotor|mdlops|mdledit|kotormax|kotorblender|tga2tpc|xoreos)\b", lowered):
        score += 4
    return score


def _trim_sentence(sentence: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", sentence).strip(" -•\t\n")
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _build_fast_report_from_context(query: str, context: Any) -> str:
    blocks = _parse_context_blocks(context)
    if not blocks:
        return "I could not complete live archive synthesis for this question right now."

    terms = _query_terms(query)
    source_numbers: dict[str, int] = {}
    source_rows: list[dict[str, str]] = []
    bullets: list[str] = []

    for block in blocks[:8]:
        url = block["url"]
        if url and url not in source_numbers:
            source_numbers[url] = len(source_rows) + 1
            source_rows.append({"title": block["title"], "url": url})

        sentences = [
            _trim_sentence(part)
            for part in re.split(r"(?<=[.!?])\s+|\s+[•*-]\s+", block["content"])
            if len(part.strip()) >= 40
        ]
        ranked = sorted(sentences, key=lambda item: _sentence_score(item, terms), reverse=True)
        for sentence in ranked[:2]:
            if len(bullets) >= 5:
                break
            if any(sentence.lower() == existing.lower().rsplit(" [", 1)[0] for existing in bullets):
                continue
            citation = f" [{source_numbers[url]}]" if url in source_numbers else ""
            bullets.append(f"- {sentence}{citation}")
        if len(bullets) >= 5:
            break

    if not bullets:
        first = blocks[0]
        url = first["url"]
        if url and url not in source_numbers:
            source_numbers[url] = len(source_rows) + 1
            source_rows.append({"title": first["title"], "url": url})
        citation = f" [{source_numbers[url]}]" if url in source_numbers else ""
        bullets.append(f"- {_trim_sentence(first['content'])}{citation}")

    source_lines = ["Sources"]
    for index, row in enumerate(source_rows[:6], start=1):
        source_lines.append(f"{index}. {row['title']} - {row['url']}")
    return "\n".join([*bullets, "", *source_lines])


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

    _set_allowlist_env_from_payload(payload)
    source_urls_raw = _payload_string_list(payload, "source_urls")
    source_urls, rejected_source_urls = _partition_allowed_urls(source_urls_raw)
    query_domains = _payload_string_list(payload, "query_domains")
    allowed_url_prefixes = _payload_string_list(payload, "allowed_url_prefixes")

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
        t_init = time.perf_counter()
        researcher = GPTResearcher(
            query=query.strip(),
            report_type=report_type,
            report_source=report_source,
            tone=Tone.Objective,
            source_urls=source_urls,
            query_domains=query_domains,
            complement_source_urls=False,
            verbose=False,
            agent="KOTOR archive research assistant",
            role="Answer KOTOR modding questions from the provided approved source context.",
            encoding="utf-8",
        )
        print(f"[TRASK_TIMING] init={time.perf_counter() - t_init:.2f}s", file=sys.stderr, flush=True)

        t_research = time.perf_counter()
        await researcher.conduct_research()
        print(
            f"[TRASK_TIMING] conduct_research={time.perf_counter() - t_research:.2f}s "
            f"visited={len(researcher.visited_urls)} retriever={os.environ.get('RETRIEVER', '?')}",
            file=sys.stderr,
            flush=True,
        )

        if os.environ.get("TRASK_FAST_QA", "").strip().lower() in {"1", "true", "yes"}:
            t_report = time.perf_counter()
            report = _build_fast_report_from_context(query.strip(), researcher.context)
            print(
                f"[TRASK_TIMING] fast_report={time.perf_counter() - t_report:.2f}s "
                f"report_chars={len(report) if isinstance(report, str) else 0}",
                file=sys.stderr,
                flush=True,
            )
        else:
            t_report = time.perf_counter()
            report = await researcher.write_report(custom_prompt=custom_prompt)
            print(
                f"[TRASK_TIMING] write_report={time.perf_counter() - t_report:.2f}s "
                f"report_chars={len(report) if isinstance(report, str) else 0}",
                file=sys.stderr,
                flush=True,
            )

    raw_visited = list(researcher.visited_urls)
    visited, rejected_visited = _partition_allowed_urls(raw_visited)
    rejected_by_pipeline = getattr(researcher, "trask_rejected_source_urls", [])
    if not isinstance(rejected_by_pipeline, list):
        rejected_by_pipeline = []
    return {
        "report": report if isinstance(report, str) else str(report),
        "research_information": {
            "source_urls": visited,
            "visited_urls": visited,
            "query_domains": query_domains,
            "allowed_url_prefixes": allowed_url_prefixes,
            "rejected_source_urls": [*rejected_source_urls, *rejected_by_pipeline, *rejected_visited],
        },
    }


async def _run_detailed(payload: dict[str, Any]) -> dict[str, Any]:
    """Matches `cli.py` detailed_report path (no custom_prompt hook in DetailedReport)."""
    from backend.report_type import DetailedReport

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        _die('payload: string field "query" is required', 1)

    _set_allowlist_env_from_payload(payload)
    source_urls_raw = _payload_string_list(payload, "source_urls")
    source_urls, rejected_source_urls = _partition_allowed_urls(source_urls_raw)
    query_domains = _payload_string_list(payload, "query_domains")
    allowed_url_prefixes = _payload_string_list(payload, "allowed_url_prefixes")

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

    raw_visited = list(detailed.gpt_researcher.visited_urls)
    visited, rejected_visited = _partition_allowed_urls(raw_visited)
    rejected_by_pipeline = getattr(detailed.gpt_researcher, "trask_rejected_source_urls", [])
    if not isinstance(rejected_by_pipeline, list):
        rejected_by_pipeline = []
    return {
        "report": report if isinstance(report, str) else str(report),
        "research_information": {
            "source_urls": visited,
            "visited_urls": visited,
            "query_domains": query_domains,
            "allowed_url_prefixes": allowed_url_prefixes,
            "rejected_source_urls": [*rejected_source_urls, *rejected_by_pipeline, *rejected_visited],
        },
    }


async def _async_main(payload: dict[str, Any]) -> dict[str, Any]:
    _apply_payload_model_override(payload)
    report_type = payload.get("report_type") or "research_report"
    t_total = time.perf_counter()
    try:
        if report_type == "detailed_report":
            return await _run_detailed(payload)
        return await _run_standard(payload)
    finally:
        print(
            f"[TRASK_TIMING] total={time.perf_counter() - t_total:.2f}s",
            file=sys.stderr,
            flush=True,
        )


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
    t_boot = time.perf_counter()
    load_dotenv()
    print(f"[TRASK_TIMING] load_dotenv={time.perf_counter() - t_boot:.2f}s", file=sys.stderr, flush=True)

    t_defaults = time.perf_counter()
    _ensure_runtime_defaults()
    print(f"[TRASK_TIMING] runtime_defaults={time.perf_counter() - t_defaults:.2f}s", file=sys.stderr, flush=True)

    payload = _payload_from_stdin()
    try:
        result = asyncio.run(_async_main(payload))
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _die(f"{type(exc).__name__}: {exc}", 1)

    report = result.get("report")
    if not isinstance(report, str) or not report.strip():
        _die("ai-researchwizard produced an empty report.", 1)

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
