"""Unit tests for structured error formatting sent back to the LLM."""

import logging

from pydantic import BaseModel, ValidationError

from mcp_use.errors.error_formatting import format_error


class _SampleArgs(BaseModel):
    """Small model used to produce a real pydantic ValidationError."""

    limit: int
    name: str


def _make_validation_error() -> ValidationError:
    try:
        _SampleArgs(limit="ten", name=123)
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected a ValidationError to be raised")


class TestFormatErrorGeneric:
    """Behavior for plain, non-validation exceptions."""

    def test_no_stack_key_in_result(self):
        """The raw traceback must never reach the LLM-facing dict."""
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            result = format_error(e)

        assert "stack" not in result

    def test_generic_fallback_hint(self):
        """Non-validation errors get the generic, non-fabricated hint."""
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            result = format_error(e)

        assert result["error"] == "RuntimeError"
        assert result["details"] == "boom"
        assert result["hint"] == "Check that all required arguments were provided with the correct types and retry."

    def test_context_kwargs_are_merged(self):
        """Extra **context kwargs still merge into the returned dict."""
        try:
            raise ValueError("bad value")
        except ValueError as e:
            result = format_error(e, tool="my_tool", extra_field=42)

        assert result["tool"] == "my_tool"
        assert result["extra_field"] == 42
        assert result["error"] == "ValueError"

    def test_logs_error(self, caplog):
        """format_error logs the structured error (and the traceback at debug level)."""
        with caplog.at_level(logging.DEBUG, logger="mcp_use"):
            try:
                raise RuntimeError("boom")
            except RuntimeError as e:
                format_error(e)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Structured error" in m for m in messages)
        assert any("traceback" in m.lower() for m in messages)


class TestFormatErrorValidation:
    """Behavior for pydantic ValidationErrors."""

    def test_hint_names_offending_fields(self):
        error = _make_validation_error()
        result = format_error(error)

        assert result["error"] == "ValidationError"
        assert "stack" not in result
        # Both invalid fields should be mentioned in the hint.
        assert "limit" in result["hint"]
        assert "name" in result["hint"]

    def test_hint_mentions_expected_and_actual_type(self):
        error = _make_validation_error()
        result = format_error(error)

        # 'limit' expected an int but got a str; the hint should say so without
        # dumping the raw pydantic error dict.
        assert "limit" in result["hint"]
        assert "str" in result["hint"] or "int" in result["hint"]
        assert "{'type':" not in result["hint"]
