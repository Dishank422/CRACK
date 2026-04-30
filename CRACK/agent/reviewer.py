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
from .models import ReviewResult, ReviewEvent
from .pr_context import PRContext
from .tools import ToolRegistry
from .tools.base import ToolContext
from .tools.filesystem import FilesystemToolProvider
from .tools.diff import DiffToolProvider
from .tools.github import GitHubToolProvider
from .tools.embeddings import EmbeddingToolProvider
from .code_checks.prompts import CODE_CHECK_PROMPTS


SYSTEM_PROMPT = """\
You are an expert code reviewer. You are reviewing a pull request (PR) on GitHub.

Your goal is to produce a thorough, actionable code review. Focus on the requirements for 
the code review as requested by the user. Don't raise issues that are not relevant to the user request.

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
   - semantic_search to find conceptually related code when you don't know exact names
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
- Only comment on code changes in the diff. Do not comment on unchanged code.
- IMPORTANT: Inline comments can ONLY be placed on lines that appear in the diff
  (within the @@ hunk ranges). Comments on lines outside the diff will be moved
  to the review body instead of appearing inline. Prefer commenting on changed
  lines for maximum impact.
"""

INCREMENTAL_REVIEW_ADDENDUM = """\

## Incremental review

You have reviewed this PR before. A timeline of the PR activity (including your
previous review comments) is provided below. New commits have been pushed since
your last review.

Guidelines for incremental reviews:
- Focus your review on the NEW changes (shown in the "Changes since last review"
  section). You do not need to re-review code you already commented on.
- You may raise new issues on OLD changes if the new changes reveal a problem you
  didn't notice before.
- However, do NOT repeat comments on OLD changes that you already raised in a
  previous review, even if the issue was not addressed.
- If a previous comment of yours was addressed by the new changes you can acknowledge
  that briefly in the review summary. Do not attempt to respond to the comment or
  hold a conversation.
- It a reply was posted to your comment, do not attempt to respond to it or hold
  a conversation.
"""


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
    if not os.environ.get("CRACK_SKIP_EMBEDDINGS"):
        registry.register(EmbeddingToolProvider)
    registry.initialize_all()
    return registry


def build_system_prompt(pr_context, changed_files, diff_text, is_incremental) -> str:
    # Build the system prompt
    system_prompt = SYSTEM_PROMPT
    if is_incremental:
        system_prompt += INCREMENTAL_REVIEW_ADDENDUM

    file_list = "\n".join(f"  {f['status']:>10}  {f['path']}" for f in changed_files)
    prompt_parts = []

    # PR metadata (always include if available)
    if pr_context:
        if pr_context.pr_title:
            prompt_parts.append(f"## PR: {pr_context.pr_title}")
            if pr_context.pr_author:
                prompt_parts.append(f"Author: @{pr_context.pr_author}\n")
        if pr_context.pr_body:
            prompt_parts.append(f"## PR Description\n{pr_context.pr_body}\n")
        if pr_context.timeline:
            prompt_parts.append(f"## PR Timeline\n{pr_context.timeline}\n")

    prompt_parts.append(f"## Changed files\n{file_list}\n")
    prompt_parts.append(f"## Full PR Diff\n```diff\n{diff_text}\n```")

    # Incremental diff (only for follow-up reviews)
    if is_incremental and pr_context.incremental_diff:
        prompt_parts.append(
            f"\n## Changes since last review\n"
            f"These are the new changes since your last review. "
            f"Focus your review on these.\n"
            f"```diff\n{pr_context.incremental_diff}\n```"
        )

    system_prompt += "\n\n" + "\n".join(prompt_parts)  # Add PR context to system prompt
    return system_prompt


async def run_review(
    repo_path: str,
    diff_text: str,
    changed_files: list[dict[str, str]],
    github_token: str | None = None,
    github_repo: str | None = None,
    pr_number: int | None = None,
    config: AgentConfig | None = None,
    pr_context: PRContext | None = None,
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
        pr_context: PR context including timeline and previous review info.

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
    
    is_incremental = pr_context and pr_context.last_reviewed_commit is not None
    logging.info(
        f"Agent review: {len(changed_files)} changed files, "
        f"{len(tools)} tools available, model={config.model}"
        f"{', incremental' if is_incremental else ''}"
    )

    system_prompt = build_system_prompt(pr_context, changed_files, diff_text, is_incremental)

    # Build the agent
    model = _resolve_model(config)
    agent = Agent(
        model,
        output_type=ReviewResult,
        system_prompt=system_prompt,
        tools=tools,
        model_settings=ModelSettings(temperature=config.model_temperature),
    )
    message_history = [] # Save conversation history here
    
    review = ReviewResult(
        summary="",
        event=ReviewEvent.APPROVE,
        comments=[],
    )
    review_requests = 0
    review_tool_calls = 0

    for check_num, check in enumerate(config.checks):
        if check in CODE_CHECK_PROMPTS:
            user_prompt = CODE_CHECK_PROMPTS[check]
            if check_num > 0:
                user_prompt += "Please do not repeat previous comments."
            logging.info(f"Adding code check to prompt: {check}")
            current_review_result = await agent.run(
                user_prompt,
                usage_limits=UsageLimits(
                    request_limit=config.max_request_limit,
                    tool_calls_limit=config.max_tool_calls,
                ),
                message_history=message_history,
            )
            message_history += current_review_result.new_messages()
            current_output = current_review_result.output
            review.summary += f"\n\n## {check.capitalize()} Check\n{current_output.summary}"

            if current_output.event in [ReviewEvent.REQUEST_CHANGES, ReviewEvent.COMMENT]:
                review.event = ReviewEvent.REQUEST_CHANGES

            review.comments.extend(current_output.comments)
            review_requests += current_review_result.usage().requests
            review_tool_calls += current_review_result.usage().tool_calls
        else:
            logging.warning(f"Unknown code check specified: {check}")

    logging.info(
        f"Agent review complete: {len(review.comments)} inline comments, "
        f"event={review.event.value}, "
        f"usage: {review_requests} requests, "
        f"{review_tool_calls} tool calls"
    )

    return review
