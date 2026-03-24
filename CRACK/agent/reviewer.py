"""
Agent-based code reviewer using PydanticAI.

Orchestrates the review loop: sends the PR diff as initial context, lets the LLM
call tools to explore the codebase, and collects a structured ReviewResult.
"""

import functools
import logging
import os
from typing import Any, Callable

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.settings import ModelSettings

from .config import AgentConfig
from .models import ReviewEvent, ReviewResult
from .code_checks import AVAILABLE_CODE_CHECKS, CodeCheckSpec
from .tools import ToolRegistry
from .tools.base import ToolContext
from .tools.filesystem import FilesystemToolProvider
from .tools.diff import DiffToolProvider
from .tools.github import GitHubToolProvider
from .tools.embeddings import EmbeddingToolProvider


CODE_CHECK_ENV_VAR = "CRACK_AGENT_CODE_CHECKS"


def _split_checks(raw_checks: str) -> list[str]:
    """Split comma-separated check IDs and normalize them."""
    return [part.strip().lower() for part in raw_checks.split(",") if part.strip()]


def _resolve_enabled_checks(config: AgentConfig) -> list[str]:
    """Resolve requested code-check IDs from config (optional) and environment."""
    checks: list[str] = []

    # Optional config support: allows AgentConfig to add this field later
    config_checks = getattr(config, "code_checks", None)
    if isinstance(config_checks, str):
        checks.extend(_split_checks(config_checks))
    elif isinstance(config_checks, list):
        checks.extend(str(x).strip().lower() for x in config_checks if str(x).strip())

    # Primary workflow toggle
    env_checks = os.getenv(CODE_CHECK_ENV_VAR, "")
    if env_checks:
        checks.extend(_split_checks(env_checks))

    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for check in checks:
        if check not in seen:
            seen.add(check)
            deduped.append(check)

    return deduped


def _resolve_enabled_check_specs(config: AgentConfig) -> list[CodeCheckSpec]:
    """Resolve known code checks from requested IDs and warn on unknown IDs."""
    requested_checks = _resolve_enabled_checks(config)
    if not requested_checks:
        return []

    unknown_checks = [check for check in requested_checks if check not in AVAILABLE_CODE_CHECKS]
    if unknown_checks:
        logging.warning(
            "Unknown code checks requested (ignored): %s",
            ", ".join(unknown_checks),
        )

    return [AVAILABLE_CODE_CHECKS[check] for check in requested_checks if check in AVAILABLE_CODE_CHECKS]


def _merge_review_event(events: list[ReviewEvent]) -> ReviewEvent:
    """Merge per-check events using conservative precedence."""
    if any(event == ReviewEvent.REQUEST_CHANGES for event in events):
        return ReviewEvent.REQUEST_CHANGES
    if any(event == ReviewEvent.COMMENT for event in events):
        return ReviewEvent.COMMENT
    if any(event == ReviewEvent.APPROVE for event in events):
        return ReviewEvent.APPROVE
    return ReviewEvent.COMMENT


def _merge_check_results(results: list[tuple[CodeCheckSpec, ReviewResult]]) -> ReviewResult:
    """Merge multiple per-check ReviewResult payloads into one output object."""
    if not results:
        return ReviewResult(
            summary=(
                "No opt-in code checks were enabled. "
                f"Set {CODE_CHECK_ENV_VAR} to one or more check IDs."
            ),
            event=ReviewEvent.COMMENT,
            comments=[],
        )

    merged_summaries: list[str] = []
    merged_comments = []
    seen_comment_keys: set[tuple[Any, ...]] = set()
    events: list[ReviewEvent] = []

    for check_spec, review in results:
        events.append(review.event)
        merged_summaries.append(f"### {check_spec.check_id}\n{review.summary}")
        for comment in review.comments:
            key = (
                comment.path,
                comment.line,
                comment.side,
                comment.start_line,
                comment.start_side,
                comment.body,
            )
            if key in seen_comment_keys:
                continue
            seen_comment_keys.add(key)
            merged_comments.append(comment)

    return ReviewResult(
        summary="\n\n".join(merged_summaries),
        event=_merge_review_event(events),
        comments=merged_comments,
    )


def _wrap_tool_with_logging(fn: Callable) -> Callable:
    """Wrap a tool function to log its calls and results."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args_str = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logging.info(f"Tool call: {fn.__name__}({args_str})")
        result = fn(*args, **kwargs)
        result_preview = str(result)[:200]
        logging.info(f"Tool result: {fn.__name__} -> {result_preview}...")
        return result

    return wrapper


def _resolve_model(config: AgentConfig) -> Any:
    """
    Resolve the PydanticAI model from config.

    Returns either a model string (e.g., 'openai:gpt-4o') or an OpenAIChatModel
    instance for custom base_url endpoints.
    """
    model_name = config.model
    if not model_name:
        raise ValueError(
            "No model configured. Set CRACK_AGENT_MODEL or MODEL environment variable."
        )

    # If a custom base URL is set, use OpenAIProvider with it
    if config.api_base:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=config.api_base,
                api_key=config.api_key,
            ),
        )

    # Otherwise, use the model string directly (PydanticAI resolves the provider).
    # Set API key via environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    # which PydanticAI reads automatically.
    return model_name


def build_tool_context(
    repo_path: str,
    diff_text: str,
    changed_files: list[dict[str, str]],
    github_token: str | None = None,
    github_repo: str | None = None,
    pr_number: int | None = None,
    config: AgentConfig | None = None,
) -> ToolContext:
    """Build the shared ToolContext from review parameters."""
    config = config or AgentConfig.from_env()
    return ToolContext(
        repo_path=repo_path,
        github_token=github_token,
        github_repo=github_repo,
        pr_number=pr_number,
        diff_text=diff_text,
        changed_files=changed_files,
        max_output_chars=config.max_tool_output_chars,
    )


def build_tool_registry(tool_ctx: ToolContext) -> ToolRegistry:
    """
    Build and initialize the tool registry with all providers.

    This is the extension point: to add new tool providers (e.g., vector search),
    register them here.
    """
    registry = ToolRegistry(tool_ctx)
    registry.register(FilesystemToolProvider)
    registry.register(DiffToolProvider)
    if tool_ctx.github_token and tool_ctx.github_repo:
        registry.register(GitHubToolProvider)
    registry.register(EmbeddingToolProvider)
    registry.initialize_all()
    return registry


async def run_review(
    repo_path: str,
    diff_text: str,
    changed_files: list[dict[str, str]],
    github_token: str | None = None,
    github_repo: str | None = None,
    pr_number: int | None = None,
    config: AgentConfig | None = None,
) -> ReviewResult:
    """
    Run the agent-based code review.

    Args:
        repo_path: Absolute path to the checked-out repository.
        diff_text: The full unified diff of the PR.
        changed_files: List of changed files with status, e.g.
                        [{"path": "src/foo.py", "status": "modified"}, ...]
        github_token: GitHub token for API access (optional).
        github_repo: GitHub repository in "owner/repo" format (optional).
        pr_number: PR number (optional, for context).
        config: Agent configuration. If None, loaded from environment.

    Returns:
        ReviewResult with merged summary, event, and inline comments.
    """
    config = config or AgentConfig.from_env()
    enabled_checks = _resolve_enabled_check_specs(config)
    if not enabled_checks:
        logging.warning(
            "No opt-in checks enabled. Skipping LLM calls. "
            "Set %s to run scoped reviews.",
            CODE_CHECK_ENV_VAR,
        )
        return _merge_check_results([])

    # Build tools
    tool_ctx = build_tool_context(
        repo_path=repo_path,
        diff_text=diff_text,
        changed_files=changed_files,
        github_token=github_token,
        github_repo=github_repo,
        pr_number=pr_number,
        config=config,
    )
    registry = build_tool_registry(tool_ctx)
    raw_tools = registry.get_all_tools()
    tools = [_wrap_tool_with_logging(t) for t in raw_tools]

    logging.info(
        f"Agent review: {len(changed_files)} changed files, "
        f"{len(tools)} tools available, model={config.model}, "
        f"checks={[check.check_id for check in enabled_checks]}"
    )

    # Shared model and prompt payload used by each check run.
    model = _resolve_model(config)
    file_list = "\n".join(f"  {f['status']:>10}  {f['path']}" for f in changed_files)
    user_prompt = (
        f"Please review this pull request.\n\n"
        f"## Changed files\n{file_list}\n\n"
        f"## Diff\n```diff\n{diff_text}\n```"
    )

    # Allocate per-check budgets so total usage stays bounded by config.
    checks_count = len(enabled_checks)
    per_check_requests = max(1, config.max_request_limit // checks_count)
    per_check_tool_calls = max(1, config.max_tool_calls // checks_count)

    check_results: list[tuple[CodeCheckSpec, ReviewResult]] = []
    for check_spec in enabled_checks:
        logging.info("Running code check: %s", check_spec.check_id)
        agent = Agent(
            model,
            output_type=ReviewResult,
            system_prompt=check_spec.system_prompt,
            tools=tools,
            model_settings=ModelSettings(temperature=config.model_temperature),
        )

        result = await agent.run(
            user_prompt,
            usage_limits=UsageLimits(
                request_limit=per_check_requests,
                tool_calls_limit=per_check_tool_calls,
            ),
        )
        review = result.output
        check_results.append((check_spec, review))
        logging.info(
            "Code check complete: %s, event=%s, comments=%s, usage: %s requests, %s tool calls",
            check_spec.check_id,
            review.event.value,
            len(review.comments),
            result.usage().requests,
            result.usage().tool_calls,
        )

    merged_review = _merge_check_results(check_results)
    logging.info(
        "Merged review complete: checks=%s, event=%s, comments=%s",
        len(check_results),
        merged_review.event.value,
        len(merged_review.comments),
    )
    return merged_review
