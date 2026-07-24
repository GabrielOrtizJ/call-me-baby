"""Constrained decoder."""

from .llm import LLM
from .models import FunctionCall
from .models import FunctionDefinition
from .models import Prompt


class Decoder:
    """Generate function calls from prompts."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialize the decoder."""
        self._functions = functions
        self._llm = LLM()

    def process(self, prompts: list[Prompt]) -> list[FunctionCall]:
        """Process every prompt."""
        results = []

        for prompt in prompts:
            results.append(self._process_prompt(prompt))

        return results

    def _process_prompt(self, prompt: Prompt) -> FunctionCall:
        """Process one prompt."""

        print("\n==============================")
        print("PROMPT")
        print(prompt.prompt)
        print("==============================")

        self._show_function_tokens()

        return FunctionCall(
            prompt=prompt.prompt,
            name="",
            parameters={},
        )

    def _show_function_tokens(self) -> None:
        """Print the tokenization of every function."""

        for function in self._functions:
            token_ids = self._llm.encode(function.name)

            print(f"\nFunction: {function.name}")
            print(f"Token ids: {token_ids}")
            print(f"Decoded : {self._llm.decode(token_ids)}")