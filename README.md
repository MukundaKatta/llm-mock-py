# llm-mock-py

Deterministic mock LLM for testing agents without real API calls.

```bash
pip install llm-mock-py
```

## Quick start

```python
from llm_mock import MockLLM

mock = MockLLM(responses=["Hello!", "How can I help?"])

r = mock.chat([{"role": "user", "content": "hi"}])
assert r["content"] == "Hello!"

mock.assert_called_once()
```

## Keyword matching

```python
mock = MockLLM(
    responses=["Default reply."],
    keyword_map={"error": "Something went wrong. Please try again."},
)

r = mock.chat([{"role": "user", "content": "there was an error"}])
# r["content"] == "Something went wrong. Please try again."
```

## API

```python
MockLLM(responses=None, keyword_map=None)
mock.chat(messages, **kwargs) -> dict          # {"role": "assistant", "content": "..."}
mock(messages)                                 # callable alias
mock.calls -> list[MockCall]                   # all recorded calls
mock.call_count -> int
mock.last_messages() -> list[dict] | None
mock.reset()                                   # clear calls, restart cycle
mock.assert_called_n_times(n)
mock.assert_called_once()
mock.assert_not_called()
```

## How responses work

- **Cycling**: responses rotate in order. After the last, wraps back to the first.
- **Keyword priority**: if any message content matches a keyword_map key (case-insensitive), that response is returned and the cycle index is not advanced.

## Zero dependencies
