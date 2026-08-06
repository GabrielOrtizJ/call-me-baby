from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class Parameter(BaseModel):
    """Represents a function parameter."""

    type: str


class ReturnType(BaseModel):
    """Represents a function return type."""

    type: str


class FunctionDefinition(BaseModel):
    """Represents a function definition."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class Prompt(BaseModel):
    """Represents an input prompt."""

    prompt: str


class FunctionCall(BaseModel):
    """Represents the output for a processed prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]
