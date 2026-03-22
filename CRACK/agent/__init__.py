"""
Agent-based code reviewer for CRACK.

This package provides an agentic code review pipeline using PydanticAI,
with tools for exploring the codebase and GitHub context.
"""

from .models import ReviewResult, InlineComment, ReviewEvent, CommentSide
from .config import AgentConfig
from .reviewer import run_review

__all__ = [
    "ReviewResult",
    "InlineComment",
    "ReviewEvent",
    "CommentSide",
    "AgentConfig",
    "run_review",
]
