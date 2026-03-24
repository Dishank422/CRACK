"""
Post a code review to GitHub using the Pull Request Reviews API.

Uses POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews to submit
a proper review with inline comments attached to specific lines.

GitHub's API only allows inline comments on lines within the diff hunks.
Comments targeting lines outside the diff are moved into the review body.
"""

import logging
import re
from typing import Optional

import requests

from .models import ReviewResult, InlineComment


def _parse_valid_lines(diff_text: str) -> dict[str, set[int]]:
    """
    Parse a unified diff to extract valid (file, line) pairs for RIGHT-side comments.

    Returns a dict mapping file paths to sets of valid line numbers (new file side).
    These are the only lines GitHub's API will accept for inline comments.
    """
    valid = {}
    current_file = None

    for line in diff_text.splitlines():
        # Detect file header: +++ b/path/to/file
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file not in valid:
                valid[current_file] = set()
            continue

        # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
        if line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                # All lines in the hunk range are valid for RIGHT-side comments
                # (additions + context lines)
                valid[current_file].update(range(start, start + count))

    return valid


def post_github_review(
    review: ReviewResult,
    github_repo: str,
    pr_number: int,
    github_token: str,
    commit_sha: Optional[str] = None,
    diff_text: Optional[str] = None,
) -> bool:
    """
    Post a ReviewResult as a GitHub pull request review with inline comments.

    Args:
        review: The structured review output from the agent.
        github_repo: Repository in "owner/repo" format.
        pr_number: Pull request number.
        github_token: GitHub token with pull-requests:write permission.
        commit_sha: Optional commit SHA to anchor the review to.
        diff_text: The PR diff text, used to validate inline comment line numbers.

    Returns:
        True if the review was posted successfully.
    """
    url = f"https://api.github.com/repos/{github_repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }

    # Validate comments against the diff — only lines in diff hunks are allowed
    valid_lines = _parse_valid_lines(diff_text) if diff_text else None
    valid_comments = []
    invalid_comments = []

    for comment in review.comments:
        if valid_lines is not None:
            file_lines = valid_lines.get(comment.path, set())
            if comment.line not in file_lines:
                invalid_comments.append(comment)
                continue
        valid_comments.append(comment)

    if invalid_comments:
        logging.info(
            f"{len(invalid_comments)} comment(s) on lines outside the diff, "
            f"moving to review body"
        )

    # Build the comments array for the API (valid comments only)
    api_comments = []
    for comment in valid_comments:
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

    # Build review body — include invalid comments as text
    review_body = review.summary
    if invalid_comments:
        review_body += _format_non_inline_comments(invalid_comments)

    body = {
        "body": review_body,
        "event": review.event.value,
        "comments": api_comments,
    }
    if commit_sha:
        body["commit_id"] = commit_sha

    resp = requests.post(url, headers=headers, json=body, timeout=30)

    if 200 <= resp.status_code < 300:
        logging.info(
            f"Posted review to {github_repo}#{pr_number}: "
            f"{len(api_comments)} inline comments, event={review.event.value}"
        )
        return True

    logging.error(
        f"Failed to post review: {resp.status_code} {resp.reason}\n{resp.text}"
    )

    # If inline comments still had issues (e.g., multi-line range errors),
    # retry with everything in the body. Only worth retrying on 422.
    if resp.status_code == 422 and api_comments:
        logging.info("Retrying without inline comments...")
        all_comments = valid_comments + invalid_comments
        fallback_body = review.summary + _format_non_inline_comments(all_comments)
        body_no_comments = {
            "body": fallback_body,
            "event": review.event.value,
            "comments": [],
        }
        if commit_sha:
            body_no_comments["commit_id"] = commit_sha

        resp2 = requests.post(url, headers=headers, json=body_no_comments, timeout=30)
        if 200 <= resp2.status_code < 300:
            logging.info("Posted review summary (without inline comments) as fallback.")
            return True
        logging.error(
            f"Fallback also failed: {resp2.status_code} {resp2.reason}\n{resp2.text}"
        )

    return False


def _format_non_inline_comments(comments: list[InlineComment]) -> str:
    """Format comments as markdown text for inclusion in the review body."""
    parts = ["\n\n---\n### Additional Comments\n"]
    for c in comments:
        line_ref = f"L{c.line}"
        if c.start_line is not None:
            line_ref = f"L{c.start_line}-L{c.line}"
        parts.append(f"**`{c.path}:{line_ref}`**\n{c.body}\n")
    return "\n".join(parts)
