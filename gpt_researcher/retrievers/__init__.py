from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .arxiv.arxiv import ArxivSearch
    from .bing.bing import BingSearch
    from .bocha.bocha import BoChaSearch
    from .custom.custom import CustomRetriever
    from .duckduckgo.duckduckgo import Duckduckgo
    from .exa.exa import ExaSearch
    from .google.google import GoogleSearch
    from .mcp import MCPRetriever
    from .pubmed_central.pubmed_central import PubMedCentralSearch
    from .searchapi.searchapi import SearchApiSearch
    from .searx.searx import SearxSearch
    from .semantic_scholar.semantic_scholar import SemanticScholarSearch
    from .serpapi.serpapi import SerpApiSearch
    from .serper.serper import SerperSearch
    from .tavily.tavily_search import TavilySearch


_RETRIEVER_EXPORTS: dict[str, tuple[str, str]] = {
    "ArxivSearch": (".arxiv.arxiv", "ArxivSearch"),
    "BingSearch": (".bing.bing", "BingSearch"),
    "BoChaSearch": (".bocha.bocha", "BoChaSearch"),
    "CustomRetriever": (".custom.custom", "CustomRetriever"),
    "Duckduckgo": (".duckduckgo.duckduckgo", "Duckduckgo"),
    "ExaSearch": (".exa.exa", "ExaSearch"),
    "GoogleSearch": (".google.google", "GoogleSearch"),
    "MCPRetriever": (".mcp", "MCPRetriever"),
    "PubMedCentralSearch": (".pubmed_central.pubmed_central", "PubMedCentralSearch"),
    "SearchApiSearch": (".searchapi.searchapi", "SearchApiSearch"),
    "SearxSearch": (".searx.searx", "SearxSearch"),
    "SemanticScholarSearch": (".semantic_scholar.semantic_scholar", "SemanticScholarSearch"),
    "SerpApiSearch": (".serpapi.serpapi", "SerpApiSearch"),
    "SerperSearch": (".serper.serper", "SerperSearch"),
    "TavilySearch": (".tavily.tavily_search", "TavilySearch"),
}


def __getattr__(name: str) -> Any:
    target = _RETRIEVER_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    return getattr(module, attr_name)


__all__ = list(_RETRIEVER_EXPORTS)
