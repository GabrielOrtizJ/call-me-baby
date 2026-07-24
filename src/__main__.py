"""Entry point of the project."""

from .decoder import Decoder
from .io import (
    load_functions,
    load_prompts,
    parse_args,
    write_results,
)
from .llm import LLM


def main() -> None:
    """Run the function calling pipeline."""

    args = parse_args()

    functions = load_functions(
        args.functions_definition
    )

    prompts = load_prompts(
        args.input
    )

    llm = LLM()

    decoder = Decoder(
        functions,
        llm,
    )

    results = decoder.process(
        prompts
    )

    write_results(
        args.output,
        results
    )


if __name__ == "__main__":
    main()
