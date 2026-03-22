"""
Agent-based code reviewer using PydanticAI.

Orchestrates the review loop: sends the PR diff as initial context, lets the LLM
call tools to explore the codebase, and collects a structured ReviewResult.
"""

import functools
import logging
from typing import Any, Callable

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.settings import ModelSettings

from .config import AgentConfig
from .models import ReviewResult
from .tools import ToolRegistry
from .tools.base import ToolContext
from .tools.filesystem import FilesystemToolProvider
from .tools.diff import DiffToolProvider
from .tools.github import GitHubToolProvider


SYSTEM_PROMPT = """\
You are an expert code reviewer. You are reviewing a pull request (PR) on GitHub.

Your goal is to produce a thorough, actionable code review. Focus on:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- API misuse or incorrect assumptions
- Missing error handling
- Breaking changes or backward compatibility issues

Do NOT comment on:
- Minor style or formatting issues (these are handled by linters)
- Obvious or trivial things that add no value
- Things that are clearly intentional design decisions without real downsides

## Your workflow

YOU MUST USE TOOLS TO INVESTIGATE BEFORE PRODUCING YOUR REVIEW. Do not skip this step.

1. First, read the diff and changed file list provided below to understand what the PR does.
2. Then, BEFORE writing any review, use your tools to gather context. You should make
   at least a few tool calls. Good investigations include:
   - read_file to see the full file around changed code (the diff alone lacks context)
   - search_repo to find callers/usages of modified functions or classes
   - search_repo to check if tests exist for the changed code
   - read_file to follow imports and understand dependencies
   - get_issue_or_pr if you see issue/PR references like #42 or "fixes #123"
   - list_directory to understand project structure if needed
3. Only AFTER investigating with tools, produce your final review.

## Tool usage guidelines

- Be targeted: investigate things that could reveal bugs or missing changes.
- Prioritize: check callers of changed functions, look for missing test coverage,
  verify that API contracts match between caller and callee.
- You have a tool call budget, so focus on the highest-value investigations.
- DO NOT produce your review output without having made at least one tool call first.

## Review output

Your review should contain:
- A concise summary of what the PR does and your overall assessment
- Inline comments on specific lines where you found issues
- Each inline comment should be actionable and explain WHY something is a problem
- Set the event to REQUEST_CHANGES only for genuine bugs or security issues;
  use COMMENT for suggestions and observations
- Use the line numbers from the NEW version of the file (side=RIGHT) unless
  you are specifically commenting on deleted code (side=LEFT)
"""


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
        f"{len(tools)} tools available, model={config.model}"
    )

    # Build the agent
    model = _resolve_model(config)
    agent = Agent(
        model,
        output_type=ReviewResult,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        model_settings=ModelSettings(temperature=config.model_temperature),
    )

    # Build the initial user prompt with the diff included
    file_list = "\n".join(f"  {f['status']:>10}  {f['path']}" for f in changed_files)
    user_prompt = (
        f"Please review this pull request.\n\n"
        f"## Changed files\n{file_list}\n\n"
        f"## Diff\n```diff\n{diff_text}\n```"
    )

    # Run the agent loop
    result = await agent.run(
        user_prompt,
        usage_limits=UsageLimits(
            request_limit=config.max_request_limit,
            tool_calls_limit=config.max_tool_calls,
        ),
    )

    review = result.output
    logging.info(
        f"Agent review complete: {len(review.comments)} inline comments, "
        f"event={review.event.value}, "
        f"usage: {result.usage().requests} requests, "
        f"{result.usage().tool_calls} tool calls"
    )

    return review
