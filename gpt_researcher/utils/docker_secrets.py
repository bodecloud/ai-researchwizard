"""Docker secrets _FILE env var resolution.

Implements the standard Docker secrets pattern: for every environment variable
ending in ``_FILE`` whose value is a readable file path, the file content is
read (stripped of trailing whitespace) and exported as the corresponding env var
**without** the ``_FILE`` suffix — unless that var is already set.

Example:
    OPENAI_API_KEY_FILE=/run/secrets/openai-api-key
    →  OPENAI_API_KEY=<contents of /run/secrets/openai-api-key>

Call :func:`resolve_docker_secrets` *once* at the very top of every entrypoint
(main.py, mcp server.py, …) before ``load_dotenv()`` and any config reading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

_RESOLVED: bool = False


def resolve_docker_secrets(*, verbose: bool = False) -> int:
    """Read ``*_FILE`` env vars and populate their non-suffixed counterparts.

    Returns the number of variables resolved.

    This function is idempotent — calling it more than once is harmless.
    """
    global _RESOLVED  # noqa: PLW0603
    if _RESOLVED:
        return 0
    _RESOLVED = True

    resolved = 0
    for key, filepath in list(os.environ.items()):
        if not key.endswith("_FILE"):
            continue
        real_key = key[: -len("_FILE")]  # e.g. OPENAI_API_KEY
        if not filepath or not filepath.strip():
            continue
        p = Path(filepath.strip())
        if not p.is_file():
            if verbose:
                logger.debug("Docker secret file for %s not found at %s", real_key, p)
            continue
        try:
            value = p.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to read Docker secret %s from %s: %s", real_key, p, exc)
            continue
        if not value:
            if verbose:
                logger.debug("Docker secret file %s is empty, skipping %s", p, real_key)
            continue
        # Only set if the base var is not already defined (explicit env wins)
        if os.environ.get(real_key):
            if verbose:
                logger.debug("%s already set, skipping _FILE resolution", real_key)
            continue
        os.environ[real_key] = value
        resolved += 1
        if verbose:
            logger.debug("Resolved %s from Docker secret %s", real_key, p)

    if resolved:
        logger.info("Resolved %d environment variable(s) from Docker secret files.", resolved)
    return resolved
