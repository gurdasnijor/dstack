"""
Helpers for redacting sensitive values in API-serialized specs.

Submitted environment values and registry credentials may contain secrets in
cleartext (for example a Hugging Face token passed as `env: HF_TOKEN=hf_...`).
API responses must never return those values. Redaction keeps the key names so
clients can still show which variables a run defines, and preserves pure
`${{ secrets.<name> }}` references, which carry no secret value themselves.
"""

import re

SENSITIVE_VALUE_PLACEHOLDER = "[redacted]"

_PURE_SECRET_REFERENCE_REGEX = re.compile(r"^\s*\$\{\{\s*secrets\.[a-zA-Z0-9_]+\s*\}\}\s*$")


def is_pure_secret_reference(value: str) -> bool:
    """Returns True for values like `${{ secrets.NAME }}` that carry no secret data."""
    return _PURE_SECRET_REFERENCE_REGEX.match(value) is not None


def redact_value(value: str) -> str:
    """Returns the value unchanged if it is a pure secret reference, else the placeholder."""
    if is_pure_secret_reference(value):
        return value
    return SENSITIVE_VALUE_PLACEHOLDER
