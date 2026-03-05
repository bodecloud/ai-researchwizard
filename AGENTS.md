# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

GPT Researcher is an autonomous AI research agent with a Python/FastAPI backend (port 8000) and a Next.js frontend (port 3000). No database is required — reports are stored in a local JSON file. See the `README.md` and `.claude/SKILL.md` for full architecture details.

### Python virtual environment

A Python venv is located at `/workspace/.venv` and is auto-activated via `~/.bashrc`. All Python commands (`python`, `pytest`, `uvicorn`, `ruff`) should use this venv. If it is missing, recreate with `uv venv /workspace/.venv --python 3.12` and install deps with `uv pip install -r requirements.txt` (excluding `langchain-netmind`, which must be installed separately with `--no-deps` due to a dependency conflict).

### Starting services

- **Backend**: `source /workspace/.venv/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload` (from repo root)
- **Frontend**: `cd frontend/nextjs && npx next dev --webpack` (the `--webpack` flag is required because Next.js 16 defaults to Turbopack, which conflicts with the existing webpack config from `@ducanh2912/next-pwa`)

### API keys

The backend and research agent require `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variables (set in `.env` at the repo root) to conduct actual research. Without these keys the UI loads and the API responds, but research queries will fail.

### Linting

- Python: `ruff check .` from repo root (install with `uv pip install ruff` if missing)
- Frontend ESLint has a pre-existing compatibility issue with Next.js 16 and eslint 8; `npx next lint` does not currently work cleanly.

### Testing

- `python -m pytest tests/test_comprehensive_pytest.py -v` — the primary test suite. Some tests have pre-existing failures (`_log_config_section` NameError, `:free` suffix assertions). Tests requiring API access are skipped by default.
- Many test files under `tests/` have import errors due to missing modules (`src.logs_handler`, stale imports). These are pre-existing issues.

### Dependency conflict note

`langchain-netmind==0.1.0` requires `langchain-openai<0.4.0`, which conflicts with the project's `langchain-openai>=1.0.0`. Install it with `--no-deps` to work around this.
