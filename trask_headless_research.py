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

TRASK_LLM_CREDENTIAL_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
)

TRASK_QUERY_STOPWORDS = {
    "what",
    "where",
    "when",
    "which",
    "with",
    "from",
    "about",
    "into",
    "this",
    "that",
    "used",
    "does",
    "help",
    "work",
    "works",
    "used",
    "kotor",
    "kotor1",
    "kotor2",
    "knights",
    "republic",
    "toolchain",
}


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


def _credential_backed_default_llm() -> str | None:
    """Pick the first free-chat model whose provider credentials are present."""
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter:openrouter/free"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai:gpt-4o-mini"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq/llama-3.1-8b-instant"
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if gemini_key:
        return "gemini/gemini-2.5-flash-lite"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic/claude-3-5-haiku-latest"
    return None


def _resolve_fallback_chat_model() -> str:
    backed = _credential_backed_default_llm()
    if backed:
        return backed

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
    has_credentials = _has_llm_credentials()
    default_llm = _resolve_fallback_chat_model() if has_credentials else "openrouter:openrouter/free"
    os.environ.setdefault("TRASK_FAST_QA", "0")
    os.environ.setdefault("TRASK_DISABLE_EMERGENCY_FALLBACKS", "1")

    if not has_credentials:
        # Avoid probing Gemini/OpenRouter without keys — retriever-only path handles research.
        return

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
            "MAX_ITERATIONS": 3,
            "MAX_SEARCH_RESULTS_PER_QUERY": 8,
            "MAX_SCRAPER_WORKERS": 6,
            "TOTAL_WORDS": 720,
            "SUMMARY_TOKEN_LIMIT": 720,
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
        if token.lower() not in TRASK_QUERY_STOPWORDS
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


_GARBAGE_PATTERNS = re.compile(
    r"enable javascript|cookies to continue|please enable|cloudflare|"
    r"checking your browser|just a moment|ray id:\s*[0-9a-f]+|"
    r"ddos protection|access denied|403 forbidden|error\s+403|"
    r"this page requires javascript|javascript is required",
    re.IGNORECASE,
)


def _is_garbage_sentence(sentence: str) -> bool:
    return bool(_GARBAGE_PATTERNS.search(sentence))


def _sentence_score(sentence: str, terms: set[str]) -> int:
    lowered = sentence.lower()
    # Base score: relevance to query terms.
    score = sum(2 for term in terms if term in lowered)
    # Boost sentences that cite a URL (more informative).
    if re.search(r"https?://", lowered):
        score += 1
    return score


def _trim_sentence(sentence: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", sentence).strip(" -•\t\n")
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _has_llm_credentials() -> bool:
    return any(os.environ.get(name, "").strip() for name in TRASK_LLM_CREDENTIAL_ENV_NAMES)


def _extract_cited_urls_from_report(report: str) -> list[str]:
    if not report.strip():
        return []
    normalized = report.replace("\r\n", "\n")
    match = re.search(r"\n(?:#{1,6}\s*)?(?:Sources|References)\s*\n", normalized, re.IGNORECASE)
    source_text = normalized[match.end() :] if match else normalized
    urls = re.findall(r"https?://[^\s)\]]+", source_text)
    return _partition_allowed_urls(urls)[0]


def _block_relevance(block: dict[str, str], terms: set[str]) -> tuple[int, int]:
    title = block["title"].lower()
    content = block["content"].lower()
    url = block["url"].lower()
    title_hits = sum(1 for term in terms if term in title)
    url_hits = sum(1 for term in terms if term in url)
    content_hits = sum(1 for term in terms if term in content)
    matched = title_hits + url_hits + content_hits
    score = title_hits * 6 + url_hits * 5 + content_hits * 2
    if title_hits > 0 or url_hits > 0:
        score += 3
    return score, matched


def _extract_context_urls(context: Any) -> list[str]:
    urls = [block["url"] for block in _parse_context_blocks(context) if block.get("url")]
    return _partition_allowed_urls(urls)[0]


def _title_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        slug = parts[-1] if parts else parsed.hostname or url
        cleaned = re.sub(r"^\d+-?", "", slug)
        cleaned = re.sub(r"[-_]+", " ", cleaned).strip()
        return cleaned.title() if cleaned else (parsed.hostname or url)
    except Exception:
        return url


def _load_trask_dotenv() -> None:
    """Load repo + vendor dotenv files so subprocess cwd does not hide API keys."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    for candidate in (
        repo_root / ".env",
        repo_root / ".env.local",
        script_dir / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _context_from_scraped_rows(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or url or "Source").strip()
        content = str(row.get("raw_content") or "").strip()
        if not content:
            continue
        parts.append(f"Source: {url}\nTitle: {title}\nContent: {content}")
    return "\n\n".join(parts)


def _search_hit_score(query: str, row: dict[str, Any]) -> int:
    terms = _query_terms(query)
    if not terms:
        return 1
    title = str(row.get("title") or "").lower()
    body = str(row.get("body") or "").lower()
    href = str(row.get("href") or row.get("url") or row.get("link") or "").lower()
    score = 0
    for term in terms:
        if term in title:
            score += 6
        if term in href:
            score += 5
        if term in body:
            score += 2
    return score


def _collect_search_urls(query: str, query_domains: list[str] | None, *, max_results: int = 12) -> list[str]:
    rows: list[dict[str, Any]] = []
    try:
        retriever_name = os.environ.get("RETRIEVER", "duckduckgo").strip().lower() or "duckduckgo"
        if retriever_name == "tavily" and os.environ.get("TAVILY_API_KEY", "").strip():
            from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

            rows.extend(TavilySearch(query, query_domains=query_domains or None).search(max_results=max_results) or [])
        else:
            from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo

            rows.extend(Duckduckgo(query, query_domains=query_domains or None).search(max_results=max_results) or [])
    except Exception as exc:
        print(
            f"[TRASK_WARN] retriever search failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    ranked = sorted(rows, key=lambda row: _search_hit_score(query, row), reverse=True)
    urls: list[str] = []
    for row in ranked:
        if not isinstance(row, dict):
            continue
        href = str(row.get("href") or row.get("url") or row.get("link") or "").strip()
        if href:
            urls.append(href)

    if len(urls) < 2:
        try:
            from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo

            supplemental = Duckduckgo(query, query_domains=None).search(max_results=max_results) or []
            for row in sorted(supplemental, key=lambda r: _search_hit_score(query, r), reverse=True):
                if isinstance(row, dict):
                    href = str(row.get("href") or row.get("url") or row.get("link") or "").strip()
                    if href:
                        urls.append(href)
        except Exception:
            pass

    return urls


def _run_retriever_only_fallback(
    query: str,
    query_domains: list[str],
    source_urls: list[str],
) -> dict[str, Any]:
    """
    DuckDuckGo/Tavily search + lightweight scrape when GPTResearcher cannot run
    (missing/invalid LLM credentials). No LLM calls — grounded fast report only.
    """
    allowed_seed, _ = _partition_allowed_urls(source_urls)
    visited: list[str] = list(allowed_seed)
    visited.extend(_collect_search_urls(query, query_domains or None))

    allowed_urls, _ = _partition_allowed_urls(visited)
    terms = _query_terms(query)
    if terms:
        scored_allowed: list[tuple[int, str]] = []
        for url in allowed_urls:
            lowered = url.lower()
            score = sum(4 if term in lowered else 0 for term in terms)
            scored_allowed.append((score, url))
        scored_allowed.sort(key=lambda item: item[0], reverse=True)
        allowed_urls = [url for score, url in scored_allowed if score > 0] or [url for _, url in scored_allowed]
    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in allowed_urls:
        if url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)
        if len(unique_urls) >= 6:
            break

    context_text = ""
    if unique_urls:
        try:
            from gpt_researcher.scraper.scraper import Scraper

            user_agent = os.environ.get(
                "USER_AGENT",
                "Mozilla/5.0 (compatible; OpenKotor-Trask/1.0; +https://github.com/bodencrouch/community-bots)",
            )
            scraper_backend = os.environ.get("SCRAPER", "bs").strip() or "bs"
            scraped = Scraper(unique_urls, user_agent, scraper_backend).run()
            context_text = _context_from_scraped_rows(scraped)
        except Exception as exc:
            print(
                f"[TRASK_WARN] retriever-only scrape failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if context_text.strip():
        return _build_fast_report_from_context(query, context_text, unique_urls)

    return _build_report_from_urls(query, unique_urls)


def _build_report_from_urls(query: str, urls: list[str]) -> dict[str, Any]:
    """When scraped context is empty but allowlisted URLs were visited, return candidate sources only."""
    unique: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = raw.strip().rstrip("/")
        if not url or url in seen or not _url_allowed(url):
            continue
        seen.add(url)
        unique.append(url)
    if not unique:
        return {
            "report": "I could not complete live archive synthesis for this question right now.",
            "retrieved_urls": [],
            "cited_urls": [],
        }

    rows = [{"title": _title_from_url(url), "url": url} for url in unique[:6]]
    topic = query.strip().rstrip("?")
    bullets: list[str] = []
    for index, row in enumerate(rows, start=1):
        line = _trim_sentence(
            f"{row['title']} is an approved archive page that may answer questions about {topic}.",
            220,
        )
        bullets.append(f"- {line} [{index}]")
    source_lines = ["Sources"]
    for index, row in enumerate(rows, start=1):
        source_lines.append(f"{index}. {row['title']} - {row['url']}")
    report = "\n".join([*bullets, "", *source_lines])
    return {
        "report": report,
        "retrieved_urls": [row["url"] for row in rows],
        "cited_urls": [row["url"] for row in rows],
    }


def _build_fast_report_from_context(
    query: str,
    context: Any,
    visited_urls: list[str] | None = None,
) -> dict[str, Any]:
    raw_blocks = _parse_context_blocks(context)
    # Pre-filter whole blocks whose content is entirely a scraper-block response
    # (e.g. Cloudflare "Enable JavaScript and cookies to continue" pages).
    blocks = [b for b in raw_blocks if not _is_garbage_sentence(b["content"])]

    if not blocks:
        if visited_urls:
            return _build_report_from_urls(query, visited_urls)
        return {
            "report": "I could not complete live archive synthesis for this question right now.",
            "retrieved_urls": [],
            "cited_urls": [],
        }

    terms = _query_terms(query)
    ranked_blocks: list[tuple[int, int, dict[str, str]]] = []
    for block in blocks[:12]:
        score, matched = _block_relevance(block, terms)
        if terms and matched == 0:
            continue
        if score < 4:
            continue
        ranked_blocks.append((score, matched, block))

    if not ranked_blocks:
        if visited_urls:
            return _build_report_from_urls(query, visited_urls)
        return {
            "report": "I could not complete live archive synthesis for this question right now.",
            "retrieved_urls": [],
            "cited_urls": [],
        }

    ranked_blocks.sort(key=lambda item: (item[0], item[1]), reverse=True)

    source_numbers: dict[str, int] = {}
    source_rows: list[dict[str, str]] = []
    bullets: list[str] = []
    retrieved_rows: list[dict[str, str]] = []
    cited_urls: list[str] = []
    top_blocks = ranked_blocks[:5]

    for _score, matched_count, block in top_blocks:
        url = block["url"]
        if url and url not in source_numbers:
            source_numbers[url] = len(source_rows) + 1
            source_rows.append({"title": block["title"], "url": url})
        if url:
            retrieved_rows.append({"title": block["title"], "url": url})

        sentences = [
            _trim_sentence(part)
            for part in re.split(r"(?<=[.!?])\s+|\s+[•*-]\s+", block["content"])
            if len(part.strip()) >= 40 and not _is_garbage_sentence(part)
        ]
        ranked_sentences: list[tuple[int, int, str]] = []
        for index, sentence in enumerate(sentences[:10]):
            term_hits = sum(1 for term in terms if term in sentence.lower())
            if terms and term_hits == 0 and matched_count < 2:
                continue
            sentence_score = _sentence_score(sentence, terms) + (2 if index == 0 else 0)
            if term_hits == 0:
                sentence_score -= 1
            ranked_sentences.append((sentence_score, term_hits, sentence))
        ranked_sentences.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _sentence_score_value, term_hits, sentence in ranked_sentences[:2]:
            if len(bullets) >= 5:
                break
            if terms and term_hits == 0 and matched_count < 2:
                continue
            if any(sentence.lower() == existing.lower().rsplit(" [", 1)[0] for existing in bullets):
                continue
            citation = f" [{source_numbers[url]}]" if url in source_numbers else ""
            bullets.append(f"- {sentence}{citation}")
            if url:
                cited_urls.append(url)
        if len(bullets) >= 5:
            break

    if not bullets:
        first = top_blocks[0][2]
        url = first["url"]
        if url and url not in source_numbers:
            source_numbers[url] = len(source_rows) + 1
            source_rows.append({"title": first["title"], "url": url})
        citation = f" [{source_numbers[url]}]" if url in source_numbers else ""
        if terms and top_blocks[0][1] == 0:
            return _build_report_from_urls(query, visited_urls or [])
        bullets.append(f"- {_trim_sentence(first['content'])}{citation}")
        if url:
            cited_urls.append(url)

    if not bullets and visited_urls:
        return _build_report_from_urls(query, visited_urls)

    source_lines = ["Sources"]
    for index, row in enumerate(source_rows[:6], start=1):
        source_lines.append(f"{index}. {row['title']} - {row['url']}")
    report = "\n".join([*bullets, "", *source_lines])
    retrieved_urls = [row["url"] for row in retrieved_rows[:6] if row.get("url")]
    return {
        "report": report,
        "retrieved_urls": retrieved_urls,
        "cited_urls": _partition_allowed_urls(cited_urls)[0],
    }


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
    has_llm_credentials = _has_llm_credentials()

    if not has_llm_credentials:
        print(
            "[TRASK_WARN] No LLM API keys detected; using retriever-only web research.",
            file=sys.stderr,
            flush=True,
        )
        report_meta = _run_retriever_only_fallback(query.strip(), query_domains, source_urls)
        visited, rejected_visited = _partition_allowed_urls(
            [*source_urls, *report_meta.get("cited_urls", [])],
        )
        return {
            "report": report_meta["report"],
            "research_information": {
                "source_urls": report_meta["cited_urls"],
                "cited_urls": report_meta["cited_urls"],
                "retrieved_urls": report_meta["retrieved_urls"],
                "visited_urls": visited,
                "query_domains": query_domains,
                "allowed_url_prefixes": allowed_url_prefixes,
                "rejected_source_urls": [*rejected_source_urls, *rejected_visited],
            },
        }

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
        try:
            await researcher.conduct_research()
            print(
                f"[TRASK_TIMING] conduct_research={time.perf_counter() - t_research:.2f}s "
                f"visited={len(researcher.visited_urls)} retriever={os.environ.get('RETRIEVER', '?')}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            if has_llm_credentials:
                print(
                    f"[TRASK_WARN] conduct_research failed with LLM credentials; trying retriever-only fallback: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[TRASK_WARN] conduct_research failed without LLM credentials; trying retriever-only fallback: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            partial_visited = list(getattr(researcher, "visited_urls", []) or [])
            report_meta = _run_retriever_only_fallback(
                query.strip(),
                query_domains,
                [*source_urls, *partial_visited],
            )
            return {
                "report": report_meta["report"],
                "research_information": {
                    "source_urls": report_meta["cited_urls"],
                    "cited_urls": report_meta["cited_urls"],
                    "retrieved_urls": report_meta["retrieved_urls"],
                    "visited_urls": _partition_allowed_urls(
                        [*partial_visited, *report_meta.get("cited_urls", [])],
                    )[0],
                    "query_domains": query_domains,
                    "allowed_url_prefixes": allowed_url_prefixes,
                    "rejected_source_urls": rejected_source_urls,
                },
            }

        if os.environ.get("TRASK_FAST_QA", "").strip().lower() in {"1", "true", "yes"}:
            t_report = time.perf_counter()
            report_meta = _build_fast_report_from_context(
                query.strip(),
                researcher.context,
                list(researcher.visited_urls),
            )
            print(
                f"[TRASK_TIMING] fast_report={time.perf_counter() - t_report:.2f}s "
                f"report_chars={len(report_meta['report']) if isinstance(report_meta.get('report'), str) else 0}",
                file=sys.stderr,
                flush=True,
            )
        else:
            t_report = time.perf_counter()
            if not has_llm_credentials:
                report_meta = _build_fast_report_from_context(
                    query.strip(),
                    researcher.context,
                    list(researcher.visited_urls),
                )
                print(
                    f"[TRASK_WARN] write_report skipped without LLM credentials; using grounded fast report fallback",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                report = await researcher.write_report(custom_prompt=custom_prompt)
                report_meta = {
                    "report": report if isinstance(report, str) else str(report),
                    "retrieved_urls": _extract_context_urls(researcher.context),
                    "cited_urls": _extract_cited_urls_from_report(report if isinstance(report, str) else str(report)),
                }
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
        "report": report_meta["report"],
        "research_information": {
            "source_urls": report_meta["cited_urls"],
            "cited_urls": report_meta["cited_urls"],
            "retrieved_urls": report_meta["retrieved_urls"],
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
    report_text = report if isinstance(report, str) else str(report)
    return {
        "report": report_text,
        "research_information": {
            "source_urls": _extract_cited_urls_from_report(report_text),
            "cited_urls": _extract_cited_urls_from_report(report_text),
            "retrieved_urls": _extract_context_urls(detailed.gpt_researcher.context),
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
    _load_trask_dotenv()
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
