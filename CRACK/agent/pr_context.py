"""
Fetch PR context from GitHub for incremental reviews.

Builds a chronological timeline of the PR (commits, comments, review comments)
and detects previous agent reviews to enable incremental reviewing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests


def _github_get(url: str, token: str, params: dict | None = None) -> dict | list | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logging.warning(f"GitHub API {resp.status_code} for {url}")
    except Exception as e:
        logging.warning(f"GitHub API error: {e}")
    return None


@dataclass
class PRContext:
    """Context about a PR for the reviewer agent."""

    pr_title: str = ""
    pr_body: str = ""
    pr_author: str = ""
    timeline: str = ""  # Formatted chronological log
    last_reviewed_commit: Optional[str] = None  # SHA of last agent-reviewed commit
    incremental_diff: Optional[str] = None  # Diff since last review (filled in by caller)


def fetch_pr_context(
    github_repo: str,
    pr_number: int,
    github_token: str,
    agent_user: Optional[str] = None,
) -> PRContext:
    """
    Fetch PR metadata and build a timeline for the reviewer.

    Args:
        github_repo: "owner/repo" format.
        pr_number: PR number.
        github_token: GitHub token.
        agent_user: GitHub username of the agent/bot. If None, auto-detected
                     from the token.

    Returns:
        PRContext with timeline and previous review info.
    """
    base = f"https://api.github.com/repos/{github_repo}"
    ctx = PRContext()

    # Auto-detect agent user from token if not provided
    if not agent_user:
        user_data = _github_get("https://api.github.com/user", github_token)
        if user_data:
            agent_user = user_data.get("login", "")

    # Fetch PR metadata
    pr_data = _github_get(f"{base}/pulls/{pr_number}", github_token)
    if not pr_data:
        logging.warning("Could not fetch PR data.")
        return ctx

    ctx.pr_title = pr_data.get("title", "")
    ctx.pr_body = pr_data.get("body", "") or ""
    ctx.pr_author = pr_data.get("user", {}).get("login", "")

    # Fetch commits
    commits = _github_get(
        f"{base}/pulls/{pr_number}/commits",
        github_token,
        params={"per_page": 100},
    ) or []

    # Fetch issue comments (general PR comments)
    issue_comments = _github_get(
        f"{base}/issues/{pr_number}/comments",
        github_token,
        params={"per_page": 100},
    ) or []

    # Fetch reviews
    reviews = _github_get(
        f"{base}/pulls/{pr_number}/reviews",
        github_token,
        params={"per_page": 100},
    ) or []

    # Fetch review comments (inline code comments)
    review_comments = _github_get(
        f"{base}/pulls/{pr_number}/comments",
        github_token,
        params={"per_page": 100},
    ) or []

    # Find last agent review
    agent_reviews = [
        r for r in reviews
        if agent_user and r.get("user", {}).get("login") == agent_user
        and r.get("state") != "PENDING"
    ]
    if agent_reviews:
        last_review = agent_reviews[-1]
        ctx.last_reviewed_commit = last_review.get("commit_id")
        logging.info(
            f"Found {len(agent_reviews)} previous agent review(s), "
            f"last at commit {ctx.last_reviewed_commit}"
        )

    # Build chronological timeline
    events = []

    for c in commits:
        sha = c.get("sha", "")[:8]
        author = c.get("author", {}).get("login", "") or c.get("commit", {}).get("author", {}).get("name", "unknown")
        message = c.get("commit", {}).get("message", "").split("\n")[0]  # First line only
        date = c.get("commit", {}).get("committer", {}).get("date", "")
        events.append((date, f"[commit {sha}] \"{message}\" by @{author}"))

    for r in reviews:
        author = r.get("user", {}).get("login", "unknown")
        state = r.get("state", "")
        body = (r.get("body", "") or "").strip()
        date = r.get("submitted_at", "")
        is_agent = agent_user and author == agent_user
        label = " (this agent)" if is_agent else ""
        body_preview = body[:200] + "..." if len(body) > 200 else body
        if body_preview:
            events.append((date, f"[review by @{author}{label}] {state}: {body_preview}"))
        else:
            events.append((date, f"[review by @{author}{label}] {state}"))

    for c in issue_comments:
        author = c.get("user", {}).get("login", "unknown")
        body = (c.get("body", "") or "").strip()
        date = c.get("created_at", "")
        body_preview = body[:200] + "..." if len(body) > 200 else body
        events.append((date, f"[comment by @{author}] {body_preview}"))

    for rc in review_comments:
        author = rc.get("user", {}).get("login", "unknown")
        path = rc.get("path", "")
        body = (rc.get("body", "") or "").strip()
        date = rc.get("created_at", "")
        is_agent = agent_user and author == agent_user
        label = " (this agent)" if is_agent else ""
        body_preview = body[:150] + "..." if len(body) > 150 else body
        events.append((date, f"[inline comment by @{author}{label} on {path}] {body_preview}"))

    # Sort chronologically and format
    events.sort(key=lambda x: x[0])
    ctx.timeline = "\n".join(e[1] for e in events)

    return ctx
