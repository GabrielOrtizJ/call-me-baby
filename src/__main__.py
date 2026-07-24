"""Entry point of the project."""

from .decoder import Decoder
from .io import (
    load_functions,
    load_prompts,
    parse_args,
    write_results,
)


def main() -> None:
    """Run the function calling pipeline."""
    args = parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    decoder = Decoder(functions)
    results = decoder.process(prompts)

    write_results(args.output, results)


if __name__ == "__main__":
    main()