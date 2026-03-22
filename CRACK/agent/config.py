"""
Configuration for the agent-based code reviewer.

Reads from environment variables, with sensible defaults for a GitHub Actions context.
Separate from the existing CRACK ProjectConfig to avoid coupling.
"""

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the agent reviewer."""

    # LLM settings
    model: str = ""
    api_key: str = ""
    api_base: str | None = None

    # Agent loop limits
    max_tool_calls: int = 15
    max_request_limit: int = 30

    # Tool output limits
    max_tool_output_chars: int = 10000

    # Review behavior
    model_temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        return cls(
            model=os.getenv("CRACK_AGENT_MODEL", os.getenv("MODEL", "")),
            api_key=os.getenv("CRACK_AGENT_API_KEY", os.getenv("LLM_API_KEY", "")),
            api_base=os.getenv("CRACK_AGENT_API_BASE", os.getenv("LLM_API_BASE", None)),
            max_tool_calls=int(os.getenv("CRACK_AGENT_MAX_TOOL_CALLS", "15")),
            max_request_limit=int(os.getenv("CRACK_AGENT_MAX_REQUESTS", "30")),
            max_tool_output_chars=int(os.getenv("CRACK_AGENT_MAX_TOOL_OUTPUT", "10000")),
            model_temperature=float(os.getenv("CRACK_AGENT_TEMPERATURE", "0.2")),
        )
