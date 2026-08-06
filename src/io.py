"""Input and output helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast, List, Dict, Any
from .models import FunctionCall
from .models import FunctionDefinition
from .models import Prompt


DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions."""

    data = _load_json(path)

    return [
        FunctionDefinition.model_validate(item)
        for item in data
    ]


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate prompts."""

    data = _load_json(path)

    return [
        Prompt.model_validate(item)
        for item in data
    ]


def write_results(
    path: str,
    results: list[FunctionCall],
) -> None:
    """
    Write generated function calls to JSON.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                result.model_dump()
                for result in results
            ],
            file,
            indent=4,
            ensure_ascii=False,
        )


def _load_json(path: str) -> list[dict]:
    """Load a JSON file."""

    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
            return cast(List[Dict[str, Any]], data)

    except FileNotFoundError as exc:

        raise RuntimeError(
            f"File not found: {path}"
        ) from exc

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"Invalid JSON: {path}"
        ) from exc
