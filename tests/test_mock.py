import pytest
from llm_mock import MockLLM, MockCall


# ---------------------------------------------------------------------------
# Basic cycling
# ---------------------------------------------------------------------------

def test_single_response_cycles():
    m = MockLLM(responses=["Hello!"])
    assert m.chat([{"role": "user", "content": "hi"}])["content"] == "Hello!"
    assert m.chat([{"role": "user", "content": "hi"}])["content"] == "Hello!"


def test_multiple_responses_cycle():
    m = MockLLM(responses=["A", "B", "C"])
    assert m.chat([])["content"] == "A"
    assert m.chat([])["content"] == "B"
    assert m.chat([])["content"] == "C"
    assert m.chat([])["content"] == "A"  # wraps around


def test_default_response_when_none_given():
    m = MockLLM()
    r = m.chat([])
    assert isinstance(r["content"], str)
    assert len(r["content"]) > 0


# ---------------------------------------------------------------------------
# Role in response
# ---------------------------------------------------------------------------

def test_response_has_assistant_role():
    m = MockLLM(responses=["Hi"])
    r = m.chat([{"role": "user", "content": "hey"}])
    assert r["role"] == "assistant"


# ---------------------------------------------------------------------------
# Call recording
# ---------------------------------------------------------------------------

def test_calls_recorded():
    m = MockLLM(responses=["ok"])
    m.chat([{"role": "user", "content": "test"}])
    assert len(m.calls) == 1
    assert m.calls[0].response == "ok"


def test_call_count():
    m = MockLLM(responses=["x"])
    for _ in range(5):
        m.chat([])
    assert m.call_count == 5


def test_last_messages():
    m = MockLLM(responses=["y"])
    msgs = [{"role": "user", "content": "final"}]
    m.chat([{"role": "user", "content": "first"}])
    m.chat(msgs)
    assert m.last_messages() == msgs


def test_last_messages_none_before_any_call():
    m = MockLLM()
    assert m.last_messages() is None


def test_calls_are_snapshot_not_live():
    m = MockLLM(responses=["a"])
    m.chat([])
    snap = m.calls
    m.chat([])
    assert len(snap) == 1  # snapshot not mutated


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def test_assert_called_n_times_passes():
    m = MockLLM(responses=["a"])
    m.chat([])
    m.chat([])
    m.assert_called_n_times(2)  # no exception


def test_assert_called_n_times_fails():
    m = MockLLM(responses=["a"])
    m.chat([])
    with pytest.raises(AssertionError):
        m.assert_called_n_times(2)


def test_assert_called_once():
    m = MockLLM(responses=["a"])
    m.chat([])
    m.assert_called_once()


def test_assert_not_called():
    m = MockLLM()
    m.assert_not_called()


def test_assert_not_called_fails_after_call():
    m = MockLLM()
    m.chat([])
    with pytest.raises(AssertionError):
        m.assert_not_called()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_calls():
    m = MockLLM(responses=["A", "B"])
    m.chat([])
    m.reset()
    assert m.call_count == 0


def test_reset_restarts_cycle():
    m = MockLLM(responses=["A", "B"])
    m.chat([])
    m.chat([])
    m.reset()
    assert m.chat([])["content"] == "A"


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def test_keyword_match_takes_priority():
    m = MockLLM(responses=["default"], keyword_map={"error": "Error response"})
    r = m.chat([{"role": "user", "content": "there was an error"}])
    assert r["content"] == "Error response"


def test_keyword_match_case_insensitive():
    m = MockLLM(keyword_map={"hello": "Hi there"})
    r = m.chat([{"role": "user", "content": "HELLO world"}])
    assert r["content"] == "Hi there"


def test_keyword_no_match_falls_through_to_cycle():
    m = MockLLM(responses=["fallback"], keyword_map={"x": "X"})
    r = m.chat([{"role": "user", "content": "hello"}])
    assert r["content"] == "fallback"


def test_keyword_match_does_not_advance_index():
    m = MockLLM(responses=["A", "B"], keyword_map={"err": "E"})
    m.chat([{"role": "user", "content": "err"}])  # keyword match
    r = m.chat([{"role": "user", "content": "ok"}])  # should still be A
    assert r["content"] == "A"


def test_keyword_dict_response():
    m = MockLLM(keyword_map={"bye": {"content": "Goodbye!", "role": "assistant"}})
    r = m.chat([{"role": "user", "content": "bye"}])
    assert r["content"] == "Goodbye!"


# ---------------------------------------------------------------------------
# Callable interface
# ---------------------------------------------------------------------------

def test_callable_interface():
    m = MockLLM(responses=["callable ok"])
    r = m([{"role": "user", "content": "test"}])
    assert r["content"] == "callable ok"


# ---------------------------------------------------------------------------
# MockCall fields
# ---------------------------------------------------------------------------

def test_mock_call_index():
    m = MockLLM(responses=["a", "b"])
    m.chat([])
    m.chat([])
    assert m.calls[0].index == 0
    assert m.calls[1].index == 1


def test_mock_call_messages_stored():
    m = MockLLM(responses=["r"])
    msgs = [{"role": "user", "content": "stored?"}]
    m.chat(msgs)
    assert m.calls[0].messages == msgs


# ---------------------------------------------------------------------------
# Empty messages
# ---------------------------------------------------------------------------

def test_empty_messages_list():
    m = MockLLM(responses=["ok"])
    r = m.chat([])
    assert r["content"] == "ok"
