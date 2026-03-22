"""
Base class for tool providers in the agent-based code reviewer.

Each tool provider groups related tools (e.g., filesystem tools, GitHub API tools)
and can optionally perform initialization before the agent loop starts.
This supports tools that need setup (e.g., loading an embedding model for vector search).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    """
    Shared context available to all tool providers.

    Passed during initialization so tools have access to repo information,
    tokens, and configuration without global state.
    """

    repo_path: str
    github_token: str | None = None
    github_repo: str | None = None  # "owner/repo"
    pr_number: int | None = None
    diff_text: str | None = None  # The full PR diff as text
    changed_files: list[dict[str, str]] | None = None  # [{path, status}, ...]
    max_output_chars: int = 10000  # Truncation limit for tool output
    extra: dict[str, Any] = field(default_factory=dict)


class ToolProvider(ABC):
    """
    Base class for tool providers.

    Subclasses group related tools and optionally perform initialization.
    Tools are plain functions that get registered with the PydanticAI agent.
    """

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def initialize(self) -> None:
        """
        Optional initialization step, called before the agent loop starts.

        Override this for tools that need setup (e.g., building an index,
        loading a model, connecting to a service).
        """
        pass

    @abstractmethod
    def get_tools(self) -> list[Callable]:
        """
        Return the list of tool functions to register with the agent.

        Each function should have type annotations and a docstring,
        which PydanticAI uses to generate the tool schema automatically.
        """
        ...
