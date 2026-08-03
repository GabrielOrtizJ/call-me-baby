> **This project has been created as part of the 42 curriculum by gortiz-j.**

# 📞 Call Me Maybe

> **Function Calling with Constrained Decoding**

A deterministic function-calling pipeline powered by **Qwen3-0.6B**, designed to convert natural language prompts into **structured JSON function calls** with guaranteed schema compliance.

---

# ✨ Features

- 🤖 LLM-based function selection
- 🔒 Constrained decoding (no hallucinated function names)
- 🧩 Deterministic parameter extraction
- ✅ 100% valid JSON output
- 📋 JSON Schema compliance
- ⚡ Fast and reproducible execution

---

# 📖 Description

The goal of this project is to transform natural language into structured function calls.

Unlike traditional LLM function-calling systems, the language model is **only responsible for selecting the target function**. All parameter extraction is performed deterministically, ensuring predictable and reproducible results.

The system receives:

- A list of available functions (`functions_definition.json`)
- A collection of natural language prompts

and produces:

```json
{
  "prompt": "...",
  "name": "...",
  "parameters": {}
}
```

The project **does not execute functions**. Its only responsibility is generating the function call.

---

# 🏗 Architecture

```
          Prompt
             │
             ▼
      Prompt Builder
             │
             ▼
   Qwen3-0.6B Language Model
             │
             ▼
   Constrained Decoding
    (TokenTree Masking)
             │
             ▼
     Selected Function
             │
             ▼
 Deterministic Argument Parser
             │
             ▼
      JSON Construction
             │
             ▼
      Valid Function Call
```

---

# 🚀 Installation

## Requirements

- Python 3.11+
- uv

Install dependencies:

```bash
uv sync
```

---

# ▶️ Usage

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

---

# 📂 Project Structure

```
.
├── data/
│   ├── input/
│   └── output/
├── src/
├── tests/
├── README.md
└── pyproject.toml
```

---

# 🧠 Algorithm

## 1. Function Selection

The language model is used **only** to identify the correct function.

### Process

1. Build a prompt listing all available functions.
2. Encode the prompt.
3. Create a `TokenTree` containing every function name as token sequences.
4. During decoding:
   - Retrieve model logits.
   - Mask every token that cannot continue a valid function name.
   - Select the highest-scoring valid token.
5. Stop when a complete function name is produced.

### Guarantees

- No hallucinated function names
- No invalid outputs
- Deterministic behavior
- Exact function matching

---

## 2. Parameter Extraction

Arguments are extracted without using the language model.

Extraction relies on:

- JSON Schema types
- Regular expressions
- Function-specific patterns

Supported value types:

- Numbers
- Integers
- Strings
- Booleans
- File paths
- SQL queries
- Templates
- Encodings

This guarantees predictable and reproducible extraction.

---

## 3. JSON Construction

The final output always follows the required schema.

Example:

```json
{
  "prompt": "...",
  "name": "...",
  "parameters": {}
}
```

Guaranteed properties:

- Valid JSON
- Schema compliant
- No unexpected fields
- Deterministic output

---

# 🎯 Design Decisions

### LLM only for function selection

The model never generates JSON or parameters, avoiding common hallucination issues.

### Constrained decoding

A `TokenTree` masks invalid tokens during generation, ensuring only existing function names can be produced.

### Deterministic parameter extraction

Arguments are parsed using schema-aware logic instead of relying on model interpretation.

### Function-specific rules

Certain functions (SQL, templates, file paths, etc.) require additional parsing rules for improved accuracy.

---

# 📊 Performance

| Metric | Result |
|---------|--------|
| Function selection | 100% |
| JSON validity | 100% |
| Deterministic execution | ✅ |
| Argument extraction | ~90%+ |

Processes dozens of prompts in well under a second on standard hardware.

---

# 🧪 Testing

### Unit Tests

- Number extraction
- String extraction
- Boolean detection
- Path detection
- SQL detection

### Integration Tests

- Complete pipeline
- JSON validation
- Schema compliance

### Edge Cases

- Empty prompts
- Missing arguments
- Multiple quoted strings
- Complex file paths
- Unstructured prompts

---

# ⚠ Challenges

- Understanding constrained decoding
- Mapping function names into token sequences
- Preventing invalid token generation
- Building a deterministic parser
- Handling SQL queries and file paths
- Ensuring schema compliance
- Producing valid JSON under every scenario

---

# 📚 References

- JSON Schema
- Pydantic Documentation
- Qwen Documentation
- Tokenization and Vocabulary Mapping
- Constrained Decoding Literature
- 42 Subject — *Call Me Maybe*

---

# 🤖 Use of AI

AI assistance was limited to:

- Brainstorming ideas
- Clarifying technical concepts
- Reviewing explanations
- Improving documentation

No AI-generated code was used without review and adaptation.

---

# 💻 Example

### Input

```json
{
    "prompt": "What is the product of 3 and 5?"
}
```

### Output

```json
{
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {
        "a": 3.0,
        "b": 5.0
    }
}
```

---

# 📄 License

This project was developed as part of the **42 School** curriculum and is intended for educational purposes.