"""Constrained decoder for function calling."""

from typing import List
import re

from .models import (
    FunctionCall,
    FunctionDefinition,
    Prompt,
)


class TokenTree:
    """Prefix tree used for constrained decoding."""

    END = "__end__"

    def __init__(self) -> None:
        self.root: dict = {}

    def insert(self, token_ids: list[int], value: str) -> None:
        node = self.root
        for token in token_ids:
            node = node.setdefault(token, {})
        node[self.END] = value

    def next_tokens(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            if token not in node:
                return []
            node = node[token]
        return [token for token in node if token != self.END]

    def get_value(self, prefix: list[int]) -> str | None:
        node = self.root
        for token in prefix:
            if token not in node:
                return None
            node = node[token]
        return node.get(self.END)


class Decoder:
    """Main constrained decoder."""

    def __init__(self, functions: List[FunctionDefinition], llm) -> None:
        self.functions = functions
        self.llm = llm
        self.function_tree = self._build_function_tree()

    def process(self, prompts: List[Prompt]) -> List[FunctionCall]:
        return [self.decode(prompt) for prompt in prompts]

    def decode(self, prompt: Prompt) -> FunctionCall:
        selected = self._select_function(prompt.prompt)
        params = self._extract_parameters(prompt.prompt, selected)
        return FunctionCall(
            prompt=prompt.prompt,
            name=selected.name,
            parameters=params,
        )

    def _select_function(self, text: str) -> FunctionDefinition:
        prompt = self._build_function_prompt(text)
        input_ids = self.llm.encode(prompt)
        prefix: list[int] = []

        while True:
            allowed = self.function_tree.next_tokens(prefix)
            if not allowed:
                break

            token = self._next_token(input_ids, allowed)
            input_ids.append(token)
            prefix.append(token)

            name = self.function_tree.get_value(prefix)
            if name is not None:
                return self._find(name)

        return self.functions[0]

    def _find(self, name: str) -> FunctionDefinition:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return self.functions[0]

    def _build_function_tree(self) -> TokenTree:
        tree = TokenTree()
        for fn in self.functions:
            token_ids = self.llm.encode(fn.name)
            tree.insert(token_ids, fn.name)
        return tree

    def _next_token(self, input_ids: list[int],
                    allowed_tokens: list[int]) -> int:
        logits = self.llm.get_logits(input_ids)
        allowed = set(allowed_tokens)
        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = float("-inf")
        return max(range(len(logits)), key=lambda t: logits[t])

    def _build_function_prompt(self, text: str) -> str:
        lines = ["Available functions:", ""]
        for fn in self.functions:
            lines.append(fn.name)
            lines.append(fn.description)
            lines.append("")
        lines.extend([
            "User request:",
            text,
            "",
            "Answer using only one function name.",
        ])
        return "\n".join(lines)

    def _extract_parameters(self, text: str, fn: FunctionDefinition) -> dict:
        params: dict[str, object] = {}

        # Extract numbers (int or float)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        num_index = 0

        # Extract quoted strings (single or double quotes)
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        strings = [s1 or s2 for s1, s2 in quoted]
        str_index = 0

        # Extract file paths (Linux, Windows)
        paths = re.findall(
            r"(\/[\w\/\.\-]+|[A-Za-z]:\\\\[\w\\\\\.\-]+)", text,)
        path_index = 0

        # Extract SQL-like fragments (even without semicolon)
        sql = re.findall(
            r"(SELECT .*?FROM .*"
            r"|INSERT .*?VALUES .*"
            r"|UPDATE .*?SET .*"
            r"|DELETE .*?FROM .*)",
            text,
            re.IGNORECASE,
        )

        sql_index = 0

        # Fallback: last word
        fallback = text.split()[-1]

        # fn_execute_sql_query(query: string, database: string)
        if fn.name == "fn_execute_sql_query":
            # query: first quoted string or first SQL fragment
            if str_index < len(strings):
                query = strings[str_index]
            elif sql_index < len(sql):
                query = sql[sql_index]
            else:
                query = text

            # database: word before "database"
            m = re.search(r"(\w+)\s+database", text)
            database = m.group(1) if m else fallback

            return {
                "query": query,
                "database": database,
            }

        # fn_read_file(path: string, encoding: string)
        if fn.name == "fn_read_file":
            # path: explicit path or word after "Read" / "Read the file at"
            if path_index < len(paths):
                path = paths[path_index]
            else:
                m = re.search(r"Read(?: the file at)?\s+(\S+)", text)
                path = m.group(1) if m else fallback

            # encoding: word before "encoding"
            m_enc = re.search(r"(\S+)\s+encoding", text)
            encoding = m_enc.group(1) if m_enc else fallback

            return {
                "path": path,
                "encoding": encoding,
            }

        # fn_format_template(template: string)
        if fn.name == "fn_format_template":
            m = re.search(r"Format template:\s*(.*)", text)
            template = m.group(1) if m else text
            return {
                "template": template,
            }

        for name, param in fn.parameters.items():
            ptype = param.type.lower()

            if ptype == "number":
                if num_index < len(numbers):
                    params[name] = float(numbers[num_index])
                    num_index += 1
                else:
                    params[name] = 0.0

            elif ptype == "integer":
                if num_index < len(numbers):
                    params[name] = int(float(numbers[num_index]))
                    num_index += 1
                else:
                    params[name] = 0

            elif ptype == "string":
                # 1. SQL fragment
                if sql_index < len(sql):
                    params[name] = sql[sql_index]
                    sql_index += 1
                    continue

                # 2. File path
                if path_index < len(paths):
                    params[name] = paths[path_index]
                    path_index += 1
                    continue

                # 3. Quoted strings
                if str_index < len(strings):
                    params[name] = strings[str_index]
                    str_index += 1
                    continue

                # 4. Fallback: last word
                params[name] = fallback

            elif ptype == "boolean":
                params[name] = "true" in text.lower()

            else:
                params[name] = None

        return params
