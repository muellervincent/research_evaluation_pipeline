"""
Generic protocols for atomic research models and orchestration.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel
from ..clients.provider_protocol import ModelProvider

T_Output = TypeVar("T_Output", bound=BaseModel)


@runtime_checkable
class Model(Protocol[T_Output]):
    """
    Protocol for an atomic research processing model.
    Focuses on the build -> generate cycle for a specific transformation.
    """

    def build_prompt(self, *args, **kwargs) -> Any:
        """Construct the prompt for this model."""
        ...

    async def generate(self, provider: ModelProvider, prompt: Any) -> T_Output:
        """Generate the structured output for this model."""
        ...


class Orchestrator(Protocol):
    """
    Protocol for an orchestrator that coordinates multiple models.
    """

    async def run(self, *args, **kwargs) -> Any:
        """Run the orchestrated workflow."""
        ...
