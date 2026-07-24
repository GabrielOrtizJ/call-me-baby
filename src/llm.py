"""Wrapper around the provided LLM SDK."""

from llm_sdk import Small_LLM_Model


class LLM:
    """Simple wrapper for the language model."""

    def __init__(self) -> None:
        """Initialize the language model."""
        self._model = Small_LLM_Model()

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._model.encode(text)[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids into text."""
        return self._model.decode(token_ids)

    def get_logits(self, token_ids: list[int]) -> list[float]:
        """Return the logits for the next token."""
        return self._model.get_logits_from_input_ids(token_ids)

    def get_vocab_path(self) -> str:
        """Return the vocabulary file path."""
        return self._model.get_path_to_vocab_file()
