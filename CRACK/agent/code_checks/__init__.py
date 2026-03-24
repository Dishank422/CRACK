"""Registry for opt-in scoped code checks."""

from .architecture_modularity import CHECK as ARCHITECTURE_MODULARITY_CHECK
from .base import CodeCheckSpec
from .design_exception_context import CHECK as DESIGN_EXCEPTION_CONTEXT_CHECK


AVAILABLE_CODE_CHECKS: dict[str, CodeCheckSpec] = {
    DESIGN_EXCEPTION_CONTEXT_CHECK.check_id: DESIGN_EXCEPTION_CONTEXT_CHECK,
    ARCHITECTURE_MODULARITY_CHECK.check_id: ARCHITECTURE_MODULARITY_CHECK,
}


__all__ = [
    "CodeCheckSpec",
    "AVAILABLE_CODE_CHECKS",
]
