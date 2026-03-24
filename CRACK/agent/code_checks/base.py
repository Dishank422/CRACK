"""Base types for opt-in agent code checks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeCheckSpec:
    """A single scoped code-review check configuration."""

    check_id: str
    title: str
    system_prompt: str
