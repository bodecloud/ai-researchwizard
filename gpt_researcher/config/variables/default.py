from __future__ import annotations

from gpt_researcher.config.variables.base import BaseConfig

DEFAULT_CONFIG: BaseConfig = {
    "RETRIEVER": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "EMBEDDING_FALLBACKS": "auto",  # Comma-separated list of model names or "auto" for automatic free models
    "SIMILARITY_THRESHOLD": 0.42,
    #    "FAST_LLM": "openrouter:mistralai/mistral-small-3.1-24b-instruct:free",
    #    "SMART_LLM": "openrouter:google/gemini-2.0-flash-exp:free",
    #    "STRATEGIC_LLM": "openrouter:moonshotai/kimi-vl-a3b-thinking:free",
    #    "FAST_LLM": "openai:gpt-5-mini",
    #    "SMART_LLM": "openai:gpt-5",  # Has support for long responses (2k+ words).
    #    "STRATEGIC_LLM": "openai:o4-mini",  # Can be used with o1 or o3, please note it will make tasks slower.
    "FAST_LLM": "auto",  # Will use first fallback when empty or "auto"
    "SMART_LLM": "auto",
    "STRATEGIC_LLM": "auto",
    "FAST_LLM_FALLBACKS": "auto",  # Comma-separated list of model names or "auto" for automatic free models
    "SMART_LLM_FALLBACKS": "auto",  # Comma-separated list of model names or "auto" for automatic free models
    "STRATEGIC_LLM_FALLBACKS": "auto",  # Comma-separated list of model names or "auto" for automatic free models
    "FAST_TOKEN_LIMIT": 2000,
    "SMART_TOKEN_LIMIT": 4000,
    "STRATEGIC_TOKEN_LIMIT": 4000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.55,
    "LLM_TEMPERATURE": 0.55,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None,
    "SCRAPER": "bs",
    "MAX_SCRAPER_WORKERS": 15,
    "MAX_SUBTOPICS": 3,
    "LANGUAGE": "english",
    "REPORT_SOURCE": "web",
    "DOC_PATH": "./my-docs",
    "PROMPT_FAMILY": "default",
    "LLM_KWARGS": {},
    "EMBEDDING_KWARGS": {},
    "VERBOSE": True,
    # Deep research specific settings
    "DEEP_RESEARCH_BREADTH": 3,
    "DEEP_RESEARCH_DEPTH": 2,
    "DEEP_RESEARCH_CONCURRENCY": 4,
    # MCP retriever specific settings
    "MCP_SERVERS": [],  # List of predefined MCP server configurations
    "MCP_AUTO_TOOL_SELECTION": True,  # Whether to automatically select the best tool for a query
    "MCP_ALLOWED_ROOT_PATHS": [],  # List of allowed root paths for local file access
    "MCP_STRATEGY": "fast",  # MCP execution strategy: "fast", "deep", "disabled"
    "REASONING_EFFORT": "medium",
    # RAG (Retrieval-Augmented Generation) settings
    "ENABLE_RAG_REPORT_GENERATION": False,  # Enable RAG-based report generation for large contexts
    "RAG_CHUNK_SIZE": 2000,  # Size of text chunks for vector storage
    "RAG_CHUNK_OVERLAP": 200,  # Overlap between chunks
    "RAG_MAX_CHUNKS_PER_SECTION": 10,  # Maximum chunks to retrieve per report section
}
