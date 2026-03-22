"""
Pydantic models for the agent-based code reviewer.

These models are designed to map directly to the GitHub Pull Request Reviews API
(POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews) so that the LLM output
can be posted with minimal transformation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CommentSide(str, Enum):
    """Which side of the diff a comment refers to."""

    LEFT = "LEFT"  # The "before" version (deletions)
    RIGHT = "RIGHT"  # The "after" version (additions / context)


class ReviewEvent(str, Enum):
    """The action to take when submitting a review."""

    COMMENT = "COMMENT"
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class InlineComment(BaseModel):
    """
    A review comment attached to a specific line (or line range) in a file.

    Maps directly to the `comments` array items in the GitHub reviews API.
    The LLM produces these; they are posted as-is.
    """

    path: str = Field(description="Relative file path in the repository.")
    body: str = Field(description="The review comment text (Markdown).")
    line: int = Field(
        description=(
            "The line number in the file to attach the comment to. "
            "For single-line comments, the target line. "
            "For multi-line comments, the LAST line of the range."
        )
    )
    side: CommentSide = Field(
        default=CommentSide.RIGHT,
        description="Which side of the diff: RIGHT (new code) or LEFT (old code).",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="For multi-line comments: the FIRST line of the range.",
    )
    start_side: Optional[CommentSide] = Field(
        default=None,
        description="Side for the starting line of a multi-line comment.",
    )


class ReviewResult(BaseModel):
    """
    The complete output of the agent-based code reviewer.

    Contains an overall summary and a list of inline comments, mapping
    directly to the GitHub reviews API shape.
    """

    summary: str = Field(
        description=(
            "A concise overall summary of the review. Covers the main findings, "
            "overall code quality assessment, and any high-level suggestions. "
            "This becomes the review body on GitHub."
        )
    )
    event: ReviewEvent = Field(
        default=ReviewEvent.COMMENT,
        description=(
            "The review verdict: COMMENT (neutral), APPROVE, or REQUEST_CHANGES."
        ),
    )
    comments: list[InlineComment] = Field(
        default_factory=list,
        description="Inline comments attached to specific lines in changed files.",
    )
