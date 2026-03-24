"""Opt-in code check: architecture modularity and domain boundaries."""

from .base import CodeCheckSpec


CHECK_ID = "architecture-modularity"


GUIDANCE = """\
- Verify domains/concerns are modularized (domain logic, integration, transport, persistence).
- Flag tightly coupled cross-domain logic and invasive controller/orchestration code.
- Prefer thin controller/orchestration layers and domain logic in domain modules.
- Flag changes that increase coupling or blur boundaries without clear justification.
"""


CHECK = CodeCheckSpec(
    check_id=CHECK_ID,
    title="Architecture review: modular domains and minimally invasive controllers",
    guidance=GUIDANCE,
)
