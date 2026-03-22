"""
GitHub API tools: get_issue_or_pr, search_issues_and_prs.

These allow the LLM to follow issue/PR references (e.g., 'fixes #42')
found in code or commit messages, and to search for related discussions.
"""

import logging
from typing import Callable

import requests

from .base import ToolProvider, ToolContext


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, showing {max_chars}/{len(text)} chars]"


def _github_get(url: str, token: str | None, params: dict | None = None) -> dict | list | None:
    """Make an authenticated GET request to the GitHub API."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    logging.warning(f"GitHub API request failed: {resp.status_code} {resp.reason} for {url}")
    return None


class GitHubToolProvider(ToolProvider):
    """Provides tools for fetching issues, PRs, and searching the GitHub repo."""

    def initialize(self) -> None:
        if not self.ctx.github_token:
            logging.warning(
                "GitHubToolProvider: no github_token provided. "
                "GitHub API tools will have reduced rate limits."
            )

    def get_tools(self) -> list[Callable]:
        ctx = self.ctx

        def get_issue_or_pr(number: int) -> str:
            """Fetch a GitHub issue or pull request by number, including its comments.

            Use this to understand context when you see references like '#42' or 'fixes #123'
            in the code, commit messages, or PR description.

            Args:
                number: The issue or PR number.
            """
            if not ctx.github_repo:
                return "Error: GitHub repository not configured."

            base = f"https://api.github.com/repos/{ctx.github_repo}"

            # Fetch the issue/PR itself
            item = _github_get(f"{base}/issues/{number}", ctx.github_token)
            if item is None:
                return f"Error: could not fetch #{number}."

            is_pr = "pull_request" in item
            kind = "Pull Request" if is_pr else "Issue"

            lines = [
                f"# {kind} #{number}: {item.get('title', '')}",
                f"State: {item.get('state', 'unknown')}",
                f"Author: {item.get('user', {}).get('login', 'unknown')}",
                "",
                item.get("body", "") or "(no description)",
            ]

            # Fetch comments (up to 30)
            comments = _github_get(
                f"{base}/issues/{number}/comments",
                ctx.github_token,
                params={"per_page": 30},
            )
            if comments:
                lines.append(f"\n## Comments ({len(comments)})")
                for c in comments:
                    author = c.get("user", {}).get("login", "unknown")
                    body = c.get("body", "")
                    lines.append(f"\n**{author}:**\n{body}")

            # For PRs, also fetch review comments (inline code comments)
            if is_pr:
                review_comments = _github_get(
                    f"{base}/pulls/{number}/comments",
                    ctx.github_token,
                    params={"per_page": 30},
                )
                if review_comments:
                    lines.append(f"\n## Review Comments ({len(review_comments)})")
                    for rc in review_comments:
                        author = rc.get("user", {}).get("login", "unknown")
                        path = rc.get("path", "")
                        body = rc.get("body", "")
                        lines.append(f"\n**{author}** on `{path}`:\n{body}")

            return _truncate("\n".join(lines), ctx.max_output_chars)

        def search_issues_and_prs(query: str, state: str = "all", max_results: int = 10) -> str:
            """Search the repository's issues and pull requests by keyword.

            Use this to find related discussions, bug reports, or feature requests.

            Args:
                query: Search keywords.
                state: Filter by state: 'open', 'closed', or 'all'. Defaults to 'all'.
                max_results: Maximum results to return. Defaults to 10.
            """
            if not ctx.github_repo:
                return "Error: GitHub repository not configured."

            search_query = f"{query} repo:{ctx.github_repo}"
            if state in ("open", "closed"):
                search_query += f" state:{state}"

            data = _github_get(
                "https://api.github.com/search/issues",
                ctx.github_token,
                params={"q": search_query, "per_page": min(max_results, 30)},
            )
            if data is None:
                return "Error: search request failed."

            items = data.get("items", [])
            if not items:
                return "No results found."

            lines = [f"Found {data.get('total_count', len(items))} results (showing {len(items)}):\n"]
            for item in items:
                number = item.get("number", "?")
                title = item.get("title", "")
                item_state = item.get("state", "")
                is_pr = "pull_request" in item
                kind = "PR" if is_pr else "Issue"
                labels = ", ".join(l.get("name", "") for l in item.get("labels", []))
                label_str = f" [{labels}]" if labels else ""
                lines.append(f"  #{number} ({kind}, {item_state}){label_str}: {title}")

            return "\n".join(lines)

        return [get_issue_or_pr, search_issues_and_prs]
