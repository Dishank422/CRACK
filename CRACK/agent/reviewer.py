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


GENERIC_SYSTEM_PROMPT = """\
You are an expert code reviewer. You are reviewing a pull request (PR) on GitHub.

Your goal is to produce a thorough, actionable code review focused ONLY on the enabled code-check scopes provided by the user message.

You are operating in a multi-turn workflow:
- In each check turn, focus only on the single check scope provided in that turn.
- In the final synthesis turn, combine prior per-check ReviewResult objects into one final ReviewResult.

Workflow requirements:
1. Read the changed files and diff from the user message.
2. You MUST use tools to investigate before producing the final review.
3. Prioritize high-value checks: caller/callee contracts, boundary behavior, and missing tests.
4. Keep findings tightly scoped to enabled checks only.

Your review should contain:
- A concise summary of what the PR does and your overall assessment
- Inline comments on specific lines where you found issues
- Each inline comment should be actionable and explain WHY something is a problem
- Set the event to REQUEST_CHANGES only for genuine bugs or security issues;
  use COMMENT for suggestions and observations
- Use the line numbers from the NEW version of the file (side=RIGHT) unless
  you are specifically commenting on deleted code (side=LEFT)
- IMPORTANT: Inline comments can ONLY be placed on lines that appear in the diff
  (within the @@ hunk ranges). Comments on lines outside the diff will be moved
  to the review body instead of appearing inline. Prefer commenting on changed
  lines for maximum impact.

Do NOT comment on style/formatting or unrelated concerns outside enabled scopes.
"""


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

    return [
        AVAILABLE_CODE_CHECKS[check] for check in requested_checks if check in AVAILABLE_CODE_CHECKS
    ]


def _no_enabled_checks_result() -> ReviewResult:
    """Return a neutral result when no opt-in checks are enabled."""
    return ReviewResult(
        summary=(
            "No opt-in code checks were enabled. "
            f"Set {CODE_CHECK_ENV_VAR} to one or more check IDs."
        ),
        event=ReviewEvent.COMMENT,
        comments=[],
    )


def _enabled_check_ids(enabled_checks: list[CodeCheckSpec]) -> list[str]:
    """Return enabled check IDs for logging."""
    return [check.check_id for check in enabled_checks]


def _build_base_review_prompt(diff_text: str, changed_files: list[dict[str, str]]) -> str:
    """Build the base PR context prompt shared by all turns."""
    file_list = "\n".join(f"  {f['status']:>10}  {f['path']}" for f in changed_files)
    return (
        f"Please review this pull request.\n\n"
        f"## Changed files\n{file_list}\n\n"
        f"## Diff\n```diff\n{diff_text}\n```"
    )


def _format_prior_check_results(results: list[tuple[CodeCheckSpec, ReviewResult]]) -> str:
    """Format intermediate check outputs for subsequent turns and synthesis."""
    if not results:
        return "(none yet)"

    blocks: list[str] = []
    for check, review in results:
        blocks.append(
            "\n".join(
                [
                    f"### {check.check_id}: {check.title}",
                    f"event: {review.event.value}",
                    "summary:",
                    review.summary,
                    "inline_comments_json:",
                    review.model_dump_json(indent=2),
                ]
            )
        )
    return "\n\n".join(blocks)


def _log_intermediate_review(check: CodeCheckSpec, review: ReviewResult) -> None:
    """Log intermediate review summary and inline comments for a check turn."""
    logging.info(
        "Intermediate review result [%s]: event=%s, comments=%s",
        check.check_id,
        review.event.value,
        len(review.comments),
    )
    logging.info("Intermediate summary [%s]:\n%s", check.check_id, review.summary)

    if not review.comments:
        logging.info("Intermediate comments [%s]: (none)", check.check_id)
        return

    for idx, comment in enumerate(review.comments, start=1):
        line_ref = f"L{comment.line}"
        if comment.start_line is not None:
            line_ref = f"L{comment.start_line}-L{comment.line}"
        logging.info(
            "Intermediate comment [%s #%s] %s:%s (%s)\n%s",
            check.check_id,
            idx,
            comment.path,
            line_ref,
            comment.side.value,
            comment.body,
        )


def _build_check_turn_prompt(
    base_prompt: str,
    check: CodeCheckSpec,
    prior_results: list[tuple[CodeCheckSpec, ReviewResult]],
) -> str:
    """Build a single check-turn prompt in the multi-turn review flow."""
    return (
        f"{base_prompt}\n\n"
        "## Multi-turn review mode\n"
        f"Current turn check ID: {check.check_id}\n"
        f"Current turn scope: {check.title}\n"
        f"Scope guidance:\n{check.guidance}\n\n"
        "## Prior turn outputs (for context; avoid duplicates)\n"
        f"{_format_prior_check_results(prior_results)}\n\n"
        "Now produce a ReviewResult for THIS CHECK ONLY."
    )


def _build_final_synthesis_prompt(
    base_prompt: str,
    check_results: list[tuple[CodeCheckSpec, ReviewResult]],
) -> str:
    """Build the final turn prompt to synthesize all intermediate check outputs."""
    return (
        f"{base_prompt}\n\n"
        "## Intermediate per-check ReviewResult outputs\n"
        f"{_format_prior_check_results(check_results)}\n\n"
        "Produce the FINAL consolidated ReviewResult to post on GitHub."
    )


def _merge_review_event(events: list[ReviewEvent]) -> ReviewEvent:
    """Merge events conservatively for compatibility paths."""
    if any(event == ReviewEvent.REQUEST_CHANGES for event in events):
        return ReviewEvent.REQUEST_CHANGES
    if any(event == ReviewEvent.COMMENT for event in events):
        return ReviewEvent.COMMENT
    if any(event == ReviewEvent.APPROVE for event in events):
        return ReviewEvent.APPROVE
    return ReviewEvent.COMMENT


def _merge_check_results(results: list[tuple[CodeCheckSpec, ReviewResult]]) -> ReviewResult:
    """Backwards-compatible helper retained for older call sites."""
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
        args_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
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
        ReviewResult with summary, event, and inline comments.
    """
    config = config or AgentConfig.from_env()
    enabled_checks = _resolve_enabled_check_specs(config)
    if not enabled_checks:
        logging.warning(
            "No opt-in checks enabled. Skipping LLM calls. " "Set %s to run scoped reviews.",
            CODE_CHECK_ENV_VAR,
        )
        return _no_enabled_checks_result()

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
        f"checks={_enabled_check_ids(enabled_checks)}"
    )

    # Shared model and agent.
    model = _resolve_model(config)
    turn_agent = Agent(
        model,
        output_type=ReviewResult,
        system_prompt=GENERIC_SYSTEM_PROMPT,
        tools=tools,
        model_settings=ModelSettings(temperature=config.model_temperature),
    )

    base_prompt = _build_base_review_prompt(diff_text=diff_text, changed_files=changed_files)

    # Budget split: one turn per check + one synthesis turn.
    turn_count = len(enabled_checks) + 1
    per_turn_requests = max(1, config.max_request_limit // turn_count)
    per_check_tool_calls = max(1, config.max_tool_calls // len(enabled_checks))

    check_results: list[tuple[CodeCheckSpec, ReviewResult]] = []
    conversation_messages: list[Any] = []
    for check in enabled_checks:
        turn_prompt = _build_check_turn_prompt(
            base_prompt=base_prompt,
            check=check,
            prior_results=check_results,
        )
        turn_result = await turn_agent.run(
            turn_prompt,
            message_history=conversation_messages,
            usage_limits=UsageLimits(
                request_limit=per_turn_requests,
                tool_calls_limit=per_check_tool_calls,
            ),
        )
        conversation_messages.extend(turn_result.new_messages())
        check_results.append((check, turn_result.output))
        _log_intermediate_review(check, turn_result.output)
        logging.info(
            "Check turn complete: %s, event=%s, comments=%s, usage: %s requests, %s tool calls",
            check.check_id,
            turn_result.output.event.value,
            len(turn_result.output.comments),
            turn_result.usage().requests,
            turn_result.usage().tool_calls,
        )

    synthesis_prompt = _build_final_synthesis_prompt(
        base_prompt=base_prompt,
        check_results=check_results,
    )
    result = await turn_agent.run(
        synthesis_prompt,
        message_history=conversation_messages,
        usage_limits=UsageLimits(
            request_limit=per_turn_requests,
            tool_calls_limit=1,
        ),
    )
    merged_review = result.output
    logging.info(
        "Final synthesis complete: checks=%s, intermediate_results=%s, event=%s, comments=%s, usage: %s requests, %s tool calls",
        len(enabled_checks),
        len(check_results),
        merged_review.event.value,
        len(merged_review.comments),
        result.usage().requests,
        result.usage().tool_calls,
    )
    return merged_review
