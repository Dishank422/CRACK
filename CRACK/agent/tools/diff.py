"""
Diff tools: get_pr_diff, get_pr_files.

These provide the LLM with information about what changed in the PR.
The diff text is pre-computed and passed via ToolContext; these tools
just expose it in a structured way.
"""

from typing import Callable

from .base import ToolProvider, ToolContext


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, showing {max_chars}/{len(text)} chars]"


class DiffToolProvider(ToolProvider):
    """Provides tools for accessing the PR diff and changed file list."""

    def get_tools(self) -> list[Callable]:
        ctx = self.ctx

        def get_pr_diff() -> str:
            """Get the full unified diff of the pull request.

            Returns the complete diff showing all changes in the PR.
            """
            if not ctx.diff_text:
                return "No diff available."
            return _truncate(ctx.diff_text, ctx.max_output_chars)

        def get_pr_files() -> str:
            """Get the list of files changed in the pull request, with their change types.

            Returns a list of changed files with their status (added, modified, deleted, renamed).
            """
            if not ctx.changed_files:
                return "No changed files information available."
            lines = []
            for f in ctx.changed_files:
                lines.append(f"{f['status']:>10}  {f['path']}")
            return "\n".join(lines)

        return [get_pr_diff, get_pr_files]
