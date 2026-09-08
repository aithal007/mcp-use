import traceback

from mcp_use.logging import logger

_GENERIC_HINT = "Check that all required arguments were provided with the correct types and retry."


def _build_hint(error: Exception) -> str:
    """Best-effort, structured hint for the LLM about how to fix the error.

    If ``error`` looks like a Pydantic ``ValidationError`` (i.e. it exposes an
    ``.errors()`` method), the hint lists each invalid field along with what was
    expected vs. what was actually received, e.g.:
        "Field 'limit' expected an integer, got str: 'ten'."
    Otherwise a generic, non-fabricated fallback hint is returned.
    """
    errors_method = getattr(error, "errors", None)
    if not callable(errors_method):
        return _GENERIC_HINT

    try:
        validation_errors = errors_method()
    except Exception:
        return _GENERIC_HINT

    if not validation_errors:
        return _GENERIC_HINT

    field_hints = []
    for err in validation_errors:
        loc = err.get("loc") or ()
        field = ".".join(str(part) for part in loc) or "<root>"
        expected = err.get("type", "a valid value")
        received = err.get("input", None)
        received_type = type(received).__name__
        field_hints.append(f"Field '{field}' expected {expected}, got {received_type}: {received!r}.")

    if not field_hints:
        return _GENERIC_HINT

    return " ".join(field_hints)


def format_error(error: Exception, **context) -> dict:
    """
    Formats an exception into a structured, LLM-facing format.

    The returned dict is intended to become tool-call feedback that an LLM reads
    back (e.g. as `ToolMessage` content), so it intentionally does NOT include the
    full Python traceback - that would leak host file paths into model context and
    rarely helps the model self-correct. The full traceback is still emitted
    server-side via the logger (at debug level) so developers debugging mcp-use
    retain it.

    Instead, a best-effort `hint` field is included: for Pydantic `ValidationError`s
    it names the offending field(s) with expected vs. actual types; otherwise it
    falls back to a generic, non-fabricated suggestion to check argument
    correctness and retry (concrete, structured feedback like this helps an LLM
    self-correct better than a raw stack trace - see arXiv:2304.05128).

    Args:
        error: The exception to format.
        **context: Additional context to include in the formatted error.

    Returns:
        A dictionary containing the formatted error: `error` (exception type name),
        `details` (str(error)), `hint` (best-effort remediation guidance), `code`,
        and any extra `**context` merged in.
    """
    formatted_context = {
        "error": type(error).__name__,
        "details": str(error),
        "hint": _build_hint(error),
        "code": getattr(error, "code", "UNKNOWN"),
    }
    formatted_context.update(context)

    logger.error(f"Structured error: {formatted_context}")  # For observability (maybe remove later)
    logger.debug(f"Full traceback for structured error above:\n{traceback.format_exc()}")
    return formatted_context
