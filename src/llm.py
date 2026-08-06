"""LLM wrapper around Small_LLM_Model."""
from __future__ import annotations

from typing import List, cast
import json
from llm_sdk import Small_LLM_Model


class LLM:
    """Wrapper for the small LLM model."""

    def __init__(self) -> None:
        self._model = Small_LLM_Model()
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> dict[int, str]:
        """
        Load the vocabulary file provided by the SDK.
        Maps token_id -> string.
        """
        path = self._model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # SDK usually gives {"token": id}, invert it
        vocab = {}
        for token, idx in data.items():
            vocab[int(idx)] = token

        return vocab

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a flat list[int].
        The SDK returns a tensor shaped like [[ids]].
        """
        tensor = self._model.encode(text)

        # Convert to Python list
        ids = tensor.tolist()

        if isinstance(ids[0], list):
            ids = ids[0]

        return [int(x) for x in ids]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        return cast(str, self._model.decode(token_ids))

    def get_logits(self, input_ids: List[int]) -> List[float]:
        """
        Get logits for the next token.
        Must receive List[int], not nested lists.
        """
        return cast(
            List[float], self._model.get_logits_from_input_ids(input_ids)
            )
