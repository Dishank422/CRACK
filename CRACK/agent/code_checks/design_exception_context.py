"""Opt-in code check: exception context and diagnosability."""

from .base import CodeCheckSpec


CHECK_ID = "design-exception-context"


GUIDANCE = """\
- Each exception should provide enough context to identify source and location.
- Error messages should mention the failed operation and failure type.
- Flag generic messages (e.g., "failed", "error occurred") when key context is missing.
- Prefer preserving causal chains when wrapping/re-raising exceptions.
"""


CHECK = CodeCheckSpec(
    check_id=CHECK_ID,
    title="Design review: exception context and error diagnosability",
    guidance=GUIDANCE,
)
