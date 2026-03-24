# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRACK is an AI-powered code review tool that works vendor-agnostically with any LLM provider. It detects issues in GitHub/GitLab pull requests or local codebase changes. Python 3.11+, distributed via pip.

## Common Commands

```bash
# Install for local development
make install          # uv sync --all-groups

# Run tests (sets LLM_API_TYPE=NONE via conftest.py for offline testing)
make test             # pytest --log-cli-level=INFO

# Code formatting and linting
make black            # black .
make cs               # flake8 .

# Build
make build            # uv build
```

Single test: `pytest tests/test_foo.py -k "test_name"`

## Architecture

### Entry Flow
`CRACK.entrypoint:main` → `cli.py` (Typer CLI) → `core.py` (business logic) → `pipeline.py` (step execution)

### Key Modules (under `CRACK/`)
- **cli.py** — CLI commands: `review`, `ask`/`answer`, `setup`, `report`/`render`, `files`, `fix`, `deploy`, `repl`
- **core.py** — Core logic: diff extraction, merge base detection, binary filtering, review orchestration
- **bootstrap.py** — App initialization: logging, env config from `.CRACK/.env`, LLM API validation, CI detection
- **pipeline.py** — Pipeline framework with configurable `PipelineStep`s, environment-aware (local vs CI)
- **project_config.py** — Layered config: bundled `config.toml` → project `.CRACK/config.toml` → env vars
- **report_struct.py** — Data structures: `ReviewTarget`, `RawIssue`, `Issue`, `Report`
- **config.toml** — Bundled defaults (~19KB): prompts, templates, pipeline steps, retry/concurrency settings

### Subpackages
- **commands/** — Subcommand implementations (deploy, fix, GitHub/GitLab posting, Jira/Linear integration, REPL)
- **pipeline_steps/** — Post-processing steps (Jira, Linear integrations)
- **utils/** — Helpers for git, markdown, HTML, string ops
- **utils/git_platform/** — Adapter pattern for GitHub/GitLab APIs (`adapters/base.py` defines the interface)
- **tpl/** — Jinja2 prompt templates

### Key Patterns
- **Vendor agnosticism** via `ai-microcore` (`microcore` / `mc`) abstraction layer for all LLM calls
- **Async-first**: parallelized LLM requests via asyncio, configurable concurrency (`MAX_CONCURRENT_TASKS`, default 40)
- **Pydantic** for data validation on report structures
- **Platform detection**: auto-detects GitHub/GitLab from CI env vars, remote URLs, or filesystem markers

## Code Style
- Black formatter, line length 100
- Flake8 with `E203` ignored, max line length 100
