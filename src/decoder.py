# ABOUTME: Constrained decoder for function calling.
# ABOUTME: Selects valid functions and produces structured outputs.


import re
from typing import List

from .models import (
    FunctionCall,
    FunctionDefinition,
    Prompt,
)


class Decoder:
    """
    Main constrained decoder.
    """

    def __init__(
        self,
        functions: List[FunctionDefinition],
        llm,
    ) -> None:

        self.functions = functions
        self.llm = llm


    def process(
        self,
        prompts: List[Prompt],
    ) -> List[FunctionCall]:
        """
        Process all prompts.
        """

        results = []

        for prompt in prompts:
            result = self.decode(prompt)
            results.append(result)

        return results


    def decode(
        self,
        prompt: Prompt,
    ) -> FunctionCall:
        """
        Decode one prompt.
        """

        selected_function = self.select_function(
            prompt.prompt
        )

        arguments = self.extract_arguments(
            prompt.prompt,
            selected_function,
        )

        return FunctionCall(
            prompt=prompt.prompt,
            name=selected_function.name,
            parameters=arguments,
        )


    def select_function(
        self,
        text: str,
    ) -> FunctionDefinition:
        """
        Temporary function selection.

        This will later be replaced by
        constrained token decoding.
        """

        text = text.lower()


        if "sum" in text or "add" in text:
            return self._find(
                "fn_add_numbers"
            )


        if "greet" in text:
            return self._find(
                "fn_greet"
            )


        if "reverse" in text:
            return self._find(
                "fn_reverse_string"
            )


        if "square root" in text:
            return self._find(
                "fn_get_square_root"
            )


        if "replace" in text or "substitute" in text:
            return self._find(
                "fn_substitute_string_with_regex"
            )


        return self.functions[0]


    def extract_arguments(
        self,
        text: str,
        function: FunctionDefinition,
    ) -> dict:
        """
        Temporary argument extraction.

        Later replaced by constrained JSON generation.
        """


        if function.name == "fn_add_numbers":

            numbers = [
                int(number)
                for number in re.findall(
                    r"\d+",
                    text,
                )
            ]

            if len(numbers) >= 2:

                return {
                    "a": numbers[0],
                    "b": numbers[1],
                }

            return {
                "a": None,
                "b": None,
            }


        if function.name == "fn_get_square_root":

            numbers = [
                int(number)
                for number in re.findall(
                    r"\d+",
                    text,
                )
            ]

            if numbers:

                return {
                    "a": numbers[0],
                }

            return {
                "a": None,
            }


        if function.name == "fn_greet":

            words = text.split()

            if len(words) > 1:

                return {
                    "name": words[-1],
                }

            return {
                "name": "",
            }


        if function.name == "fn_reverse_string":

            match = re.search(
                r"'([^']+)'",
                text,
            )

            if match:

                return {
                    "s": match.group(1),
                }

            return {
                "s": "",
            }


        if function.name == "fn_substitute_string_with_regex":

            return {
                "source_string": text,
                "regex": "",
                "replacement": "",
            }


        return {}

    def _find(
        self,
        name: str,
    ) -> FunctionDefinition:
        """
        Find a function by name.
        """

        for function in self.functions:

            if function.name == name:
                return function

        return self.functions[0]