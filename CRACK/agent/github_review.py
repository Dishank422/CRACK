"""
Post a code review to GitHub using the Pull Request Reviews API.

Uses POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews to submit
a proper review with inline comments attached to specific lines.
"""

import logging
from typing import Optional

import requests

from .models import ReviewResult, InlineComment


def post_github_review(
    review: ReviewResult,
    github_repo: str,
    pr_number: int,
    github_token: str,
    commit_sha: Optional[str] = None,
) -> bool:
    """
    Post a ReviewResult as a GitHub pull request review with inline comments.

    Args:
        review: The structured review output from the agent.
        github_repo: Repository in "owner/repo" format.
        pr_number: Pull request number.
        github_token: GitHub token with pull-requests:write permission.
        commit_sha: Optional commit SHA to anchor the review to.

    Returns:
        True if the review was posted successfully.
    """
    url = f"https://api.github.com/repos/{github_repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }

    # Build the comments array for the API
    api_comments = []
    for comment in review.comments:
        c = {
            "path": comment.path,
            "body": comment.body,
            "line": comment.line,
            "side": comment.side.value,
        }
        if comment.start_line is not None:
            c["start_line"] = comment.start_line
            c["start_side"] = (comment.start_side or comment.side).value
        api_comments.append(c)

    body = {
        "body": review.summary,
        "event": review.event.value,
        "comments": api_comments,
    }
    if commit_sha:
        body["commit_id"] = commit_sha

    resp = requests.post(url, headers=headers, json=body)

    if 200 <= resp.status_code < 300:
        logging.info(
            f"Posted review to {github_repo}#{pr_number}: "
            f"{len(api_comments)} inline comments, event={review.event.value}"
        )
        return True

    logging.error(
        f"Failed to post review: {resp.status_code} {resp.reason}\n{resp.text}"
    )

    # If the batch fails (e.g., a comment references an invalid line),
    # try posting just the summary without inline comments as a fallback.
    if api_comments:
        logging.info("Retrying without inline comments...")
        fallback_body = _build_fallback_body(review)
        body_no_comments = {
            "body": fallback_body,
            "event": review.event.value,
            "comments": [],
        }
        if commit_sha:
            body_no_comments["commit_id"] = commit_sha

        resp2 = requests.post(url, headers=headers, json=body_no_comments)
        if 200 <= resp2.status_code < 300:
            logging.info("Posted review summary (without inline comments) as fallback.")
            return True
        logging.error(
            f"Fallback also failed: {resp2.status_code} {resp2.reason}\n{resp2.text}"
        )

    return False


def _build_fallback_body(review: ReviewResult) -> str:
    """
    Build a fallback review body that includes inline comments as text,
    in case the API rejects the inline comment format.
    """
    parts = [review.summary]
    if review.comments:
        parts.append("\n---\n## Inline Comments\n")
        for c in review.comments:
            line_ref = f"L{c.line}"
            if c.start_line is not None:
                line_ref = f"L{c.start_line}-L{c.line}"
            parts.append(f"**`{c.path}:{line_ref}`**\n{c.body}\n")
    return "\n".join(parts)
