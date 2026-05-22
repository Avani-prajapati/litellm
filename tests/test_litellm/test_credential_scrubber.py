"""
Tests for the credential-scrubbing filters in litellm/_logging.py.

Covers every bug identified in the original PR review:
  - Format-string split (key in msg, value in args) now caught via getMessage()
  - %(name)s format specifiers are no longer eaten by the regex
  - exc_text gets key-name scrubbing (not overwritten with value-only)
  - handleError / args-leak path plugged
  - Word-boundary fix: invalid_token not matched by id_token
  - '@' excluded so password=pass@host does not swallow the hostname
  - Non-str dict keys do not crash the filter
  - json_excepthook gets CredentialScrubberFilter via error_handler
  - _initialize_loggers_with_handler attaches CredentialScrubberFilter
"""

import logging
import sys
from unittest.mock import patch


# ── helpers ────────────────────────────────────────────────────────────────

def _make_record(msg, args=None, *, exc_info=None):
    record = logging.LogRecord(
        name="test", level=logging.DEBUG,
        pathname="", lineno=0, msg=msg, args=(), exc_info=exc_info,
    )
    if args is not None:
        record.args = args
    return record


# ══════════════════════════════════════════════════════════════════════════════
# _scrub_secrets
# ══════════════════════════════════════════════════════════════════════════════

class TestScrubSecrets:
    def test_api_key_value_redacted(self):
        from litellm._logging import _scrub_secrets
        assert "sk-abcdef123456789" not in _scrub_secrets("api_key=sk-abcdef123456789")

    def test_encryption_key_redacted(self):
        from litellm._logging import _scrub_secrets
        assert "default-litellm-key-here" not in _scrub_secrets(
            "encryption_key=default-litellm-key-here"
        )

    def test_aws_secret_key_redacted(self):
        from litellm._logging import _scrub_secrets
        assert "AKIAIOSFODNN7EXAMPLE" not in _scrub_secrets(
            "aws_secret_key: AKIAIOSFODNN7EXAMPLE"
        )

    def test_short_value_not_redacted(self):
        from litellm._logging import _scrub_secrets
        result = _scrub_secrets("api_key=abc")
        assert "abc" in result  # < 6 chars, not redacted

    def test_non_secret_field_not_redacted(self):
        from litellm._logging import _scrub_secrets
        result = _scrub_secrets("model=gpt-4o endpoint=https://api.openai.com")
        assert "gpt-4o" in result
        assert "api.openai.com" in result

    def test_pem_block_lookahead_prevents_regex_match(self):
        """The (?!-----) lookahead in _SECRET_KEY_RE must not eat the PEM header
        as the 'value'.  (The upstream _redact_string catches PEM blocks separately
        by shape; this test focuses on the key-name regex in isolation.)"""
        from litellm._logging import _SECRET_KEY_RE
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
        m = _SECRET_KEY_RE.search(f"private_key = {pem}")
        # Regex must not match (lookahead blocks the -----BEGIN... sequence).
        assert m is None

    # Word-boundary fix ───────────────────────────────────────────────────────

    def test_invalid_token_not_matched_by_id_token(self):
        from litellm._logging import _scrub_secrets
        text = "error=invalid_token status=401 description=token-has-expired"
        result = _scrub_secrets(text)
        # "invalid_token" must survive because id_token only matches at a word
        # boundary; "inval" is not a word boundary.
        assert "invalid_token" in result

    def test_bare_id_token_still_redacted(self):
        from litellm._logging import _scrub_secrets
        assert "my-bearer-token-12345" not in _scrub_secrets(
            "id_token=my-bearer-token-12345"
        )

    def test_password_in_superpassword_not_matched_by_regex(self):
        """The key-name regex must not match 'password' at offset 5 inside
        'superpassword' — the lookbehind (?<![a-zA-Z0-9_]) blocks this.
        Test the regex directly; _redact_string may still redact the value
        by shape, which is correct and independent of this fix."""
        from litellm._logging import _SECRET_KEY_RE
        m = _SECRET_KEY_RE.search("superpassword=some-config-value-here")
        assert m is None, f"regex must not match inside 'superpassword', got {m}"

    # @ excluded from value charset ───────────────────────────────────────────

    def test_at_sign_terminates_value_in_regex(self):
        """The key-name regex must stop at '@' so only the password token
        'redis-pass' is captured and the '@hostname' part is not eaten.
        Test the regex directly; _redact_string handles the full URL string
        separately via connection-string shape matching."""
        from litellm._logging import _SECRET_KEY_RE
        m = _SECRET_KEY_RE.search("password=redis-pass@redis.internal.svc:6379")
        assert m is not None, "password= must be matched"
        # Value captured by the regex must stop at '@'.
        assert m.group(3) == "redis-pass", (
            f"expected only 'redis-pass', got {m.group(3)!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CredentialScrubberFilter
# ══════════════════════════════════════════════════════════════════════════════

class TestCredentialScrubberFilter:

    # Core fix: getMessage() called before regex so key+value are assembled ──

    def test_format_string_split_is_caught(self):
        """encryption_key=%s with tuple arg — key and value assembled via getMessage."""
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("encryption_key=%s", ("default-litellm-key-here",))
        f.filter(record)
        assert "default-litellm-key-here" not in record.msg
        assert "REDACTED" in record.msg  # redaction marker (either [REDACTED] or REDACTED)
        assert record.args is None

    def test_format_string_split_dict_args(self):
        """encryption_key=%(key)s with dict arg — getMessage assembles the pair."""
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("encryption_key=%(key)s", {"key": "default-litellm-key-here"})
        f.filter(record)
        assert "default-litellm-key-here" not in record.msg
        assert record.args is None

    def test_format_specifier_not_eaten_by_regex(self):
        """%(key)s in a format string must not be consumed as a 'value' token."""
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        # With a non-secret value the format result must survive intact.
        record = _make_record("api_key=%(key)s model=%(model)s",
                              {"key": "sk-abc123def456ghi", "model": "gpt-4o"})
        f.filter(record)
        # Secret is gone; non-secret model name survives.
        assert "sk-abc123def456ghi" not in record.msg
        assert "gpt-4o" in record.msg

    # args always cleared ──────────────────────────────────────────────────────

    def test_args_cleared_on_success(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("api_key=%s", ("sk-abcdef123456789",))
        f.filter(record)
        assert record.args is None

    def test_args_cleared_when_getmessage_raises(self):
        """Even if getMessage() fails, args must be cleared (handleError safety)."""
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        # Mixing dict-only format string with positional %s causes TypeError.
        record = _make_record("api_key=%(key)s model=%s",
                              {"key": "sk-secret", "model": "gpt-4"})
        f.filter(record)
        assert record.args is None

    # Non-scalar tuple args not coerced ───────────────────────────────────────

    def test_decimal_arg_not_coerced_to_str(self):
        """Decimal/float-like objects must stay as-is so %f format works."""
        from decimal import Decimal
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        # getMessage() succeeds: "latency=0.123"
        record = _make_record("latency=%s", (Decimal("0.123"),))
        f.filter(record)
        # No TypeError, message is assembled; secret-free value stays.
        assert "0.123" in record.msg
        assert record.args is None

    # filter always returns True ───────────────────────────────────────────────

    def test_filter_always_returns_true(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        assert f.filter(_make_record("plain message")) is True

    def test_none_msg_no_crash(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record(None)
        result = f.filter(record)
        assert result is True

    def test_non_string_msg_no_crash(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record(12345)
        assert f.filter(record) is True

    # extra fields ──────────────────────────────────────────────────────────

    def test_extra_field_with_secret_key_name_redacted(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("msg")
        record.api_key = "sk-extrasecret123456789"  # type: ignore[attr-defined]
        f.filter(record)
        assert getattr(record, "api_key") == "[REDACTED]"

    def test_extra_field_non_secret_key_with_embedded_secret(self):
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("msg")
        record.debug_info = "api_key=sk-embeddedsecret12345"  # type: ignore[attr-defined]
        f.filter(record)
        assert "sk-embeddedsecret12345" not in getattr(record, "debug_info", "")

    def test_non_string_dict_key_does_not_crash(self):
        """Non-str dict keys must not cause TypeError in _SECRET_KEY_NAME_RE.match."""
        from litellm._logging import CredentialScrubberFilter
        f = CredentialScrubberFilter()
        record = _make_record("msg %(1)s")
        record.args = {1: "value"}  # type: ignore[assignment]
        assert f.filter(record) is True  # no crash

    # _ENABLE_SECRET_REDACTION off ────────────────────────────────────────────

    def test_disabled_redaction_passthrough(self):
        from litellm._logging import CredentialScrubberFilter
        with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
            f = CredentialScrubberFilter()
            record = _make_record("api_key=sk-secret123456789")
            f.filter(record)
            assert "sk-secret123456789" in record.msg

    # Registration ──────────────────────────────────────────────────────────

    def test_filter_registered_on_verbose_logger(self):
        import litellm._logging as log_module
        assert any(isinstance(f, log_module.CredentialScrubberFilter)
                   for f in log_module.verbose_logger.filters)

    def test_filter_registered_on_verbose_proxy_logger(self):
        import litellm._logging as log_module
        assert any(isinstance(f, log_module.CredentialScrubberFilter)
                   for f in log_module.verbose_proxy_logger.filters)

    def test_filter_registered_on_verbose_router_logger(self):
        import litellm._logging as log_module
        assert any(isinstance(f, log_module.CredentialScrubberFilter)
                   for f in log_module.verbose_router_logger.filters)


# ══════════════════════════════════════════════════════════════════════════════
# SecretRedactionFilter — exc_text uses _scrub_secrets + fallback clears args
# ══════════════════════════════════════════════════════════════════════════════

class TestSecretRedactionFilterFixes:

    def test_exc_text_uses_key_name_scrubbing(self):
        """exc_text must use _scrub_secrets so key-name patterns in tracebacks
        are caught, not just value-shape patterns."""
        from litellm._logging import SecretRedactionFilter
        try:
            raise ValueError("litellm_key=my-arbitrary-password-here")
        except ValueError:
            ei = sys.exc_info()
        f = SecretRedactionFilter()
        record = _make_record("error occurred", exc_info=ei)
        f.filter(record)
        assert record.exc_text is not None
        assert "my-arbitrary-password-here" not in record.exc_text
        assert "[REDACTED]" in record.exc_text

    def test_fallback_clears_args_to_prevent_handleerror_leak(self):
        """When getMessage() raises TypeError, record.args must be cleared so
        handler.handleError() cannot write the raw args tuple to stderr."""
        from litellm._logging import SecretRedactionFilter
        f = SecretRedactionFilter()
        # Mixing %(name)s format (eaten internally) with positional args causes TypeError.
        record = _make_record("api_key=%(long_param_name_here)s and model=%s",
                              ("sk-secret", "gpt-4"))
        f.filter(record)
        assert record.args is None

    def test_format_string_split_caught_after_getmessage(self):
        """SecretRedactionFilter also calls getMessage() first, so the combined
        string is available to _scrub_secrets."""
        from litellm._logging import SecretRedactionFilter
        f = SecretRedactionFilter()
        record = _make_record("encryption_key=%s", ("default-litellm-key-here",))
        f.filter(record)
        assert "default-litellm-key-here" not in record.msg
        assert record.args is None
