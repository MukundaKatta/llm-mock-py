"""llm-mock-py — deterministic mock LLM for testing agents without real API calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockCall:
    messages: list[dict]
    response: str
    index: int


class MockLLM:
    """
    Deterministic mock LLM for unit-testing agents.

    Supports two response modes:
    - Index-based cycling: responses are served in order, cycling when exhausted.
    - Keyword matching: if any message content contains a registered keyword,
      the mapped response is returned (checked before index cycling).

    Example::

        mock = MockLLM(responses=["Hello!", "How can I help?"])
        r = mock.chat([{"role": "user", "content": "hi"}])
        assert r["content"] == "Hello!"
        mock.assert_called_n_times(1)
    """

    def __init__(
        self,
        responses: list[str | dict] | None = None,
        keyword_map: dict[str, str | dict] | None = None,
    ) -> None:
        self._responses: list[str | dict] = responses or ["Mock response."]
        self._keyword_map: dict[str, str | dict] = keyword_map or {}
        self._calls: list[MockCall] = []
        self._index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def calls(self) -> list[MockCall]:
        """All recorded calls in order."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def chat(self, messages: list[dict], **_kwargs: Any) -> dict:
        """Return the next mock response. Records the call."""
        response = self._pick_response(messages)
        if isinstance(response, str):
            content = response
        else:
            content = response.get("content", "Mock response.")

        call = MockCall(messages=messages, response=content, index=self._index)
        self._calls.append(call)

        # Advance cycle index (keyword matches don't advance it)
        if not self._matched_keyword(messages):
            self._index = (self._index + 1) % len(self._responses)

        return {"role": "assistant", "content": content}

    # Alias for OpenAI-style callers
    def __call__(self, messages: list[dict], **kwargs: Any) -> dict:
        return self.chat(messages, **kwargs)

    def reset(self) -> None:
        """Clear call history and reset the cycle index."""
        self._calls.clear()
        self._index = 0

    def assert_called_n_times(self, n: int) -> None:
        if len(self._calls) != n:
            raise AssertionError(
                f"Expected {n} call(s), got {len(self._calls)}"
            )

    def assert_called_once(self) -> None:
        self.assert_called_n_times(1)

    def assert_not_called(self) -> None:
        self.assert_called_n_times(0)

    def last_messages(self) -> list[dict] | None:
        """Return the messages from the most recent call, or None."""
        return self._calls[-1].messages if self._calls else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matched_keyword(self, messages: list[dict]) -> bool:
        combined = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        ).lower()
        for kw in self._keyword_map:
            if kw.lower() in combined:
                return True
        return False

    def _pick_response(self, messages: list[dict]) -> str | dict:
        combined = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        ).lower()
        for kw, resp in self._keyword_map.items():
            if kw.lower() in combined:
                return resp
        return self._responses[self._index % len(self._responses)]


__all__ = ["MockLLM", "MockCall"]
