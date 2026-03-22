"""
Tool registry for the agent-based code reviewer.

Collects tool providers, initializes them, and returns the flat list of
tool functions for PydanticAI agent registration.
"""

import logging
from typing import Callable

from .base import ToolContext, ToolProvider


class ToolRegistry:
    """
    Manages tool providers: initializes them and collects their tools.

    Usage:
        registry = ToolRegistry(ctx)
        registry.register(FilesystemToolProvider)
        registry.register(DiffToolProvider)
        registry.initialize_all()
        tools = registry.get_all_tools()
    """

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._providers: list[ToolProvider] = []

    def register(self, provider_cls: type[ToolProvider]) -> None:
        """Instantiate and add a tool provider."""
        provider = provider_cls(self.ctx)
        self._providers.append(provider)

    def initialize_all(self) -> None:
        """Initialize all registered providers. Call before get_all_tools()."""
        for provider in self._providers:
            name = provider.__class__.__name__
            logging.debug(f"Initializing tool provider: {name}")
            provider.initialize()
            logging.debug(f"Initialized tool provider: {name}")

    def get_all_tools(self) -> list[Callable]:
        """Return the flat list of all tool functions from all providers."""
        tools = []
        for provider in self._providers:
            tools.extend(provider.get_tools())
        return tools
