# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for bounding text sent to embedding providers."""

from __future__ import annotations

import re
from unicodedata import east_asian_width

EMBEDDING_TRUNCATION_SUFFIX = "\n...(truncated for embedding)"
_ASCII_DIGIT = re.compile("[0-9]")


def _scan_embedding_input(
    text: str, max_tokens: int | None = None, suffix_tokens: int = 0
) -> tuple[int, int]:
    """Count estimated tokens and locate a prefix that leaves room for the suffix.

    Stop once the full budget is exceeded. Keep the earlier prefix boundary so
    text that fits the full budget is returned intact, without a second scan.
    """
    scan_length = len(text) if max_tokens is None else min(len(text), max_tokens * 4 + 1)
    if text.isascii() and not _ASCII_DIGIT.search(text, 0, scan_length):
        tokens = (scan_length + 3) // 4
        end = (
            (max_tokens - suffix_tokens) * 4
            if max_tokens is not None and tokens > max_tokens
            else len(text)
        )
        return tokens, end

    max_units = None if max_tokens is None else max_tokens * 4
    prefix_units = 0 if max_tokens is None else (max_tokens - suffix_tokens) * 4
    units = prefix_end = digit_offset = 0
    for end, char in enumerate(text, 1):
        code_point = ord(char)
        if 48 <= code_point <= 57:
            # Each consecutive group of up to three ASCII digits costs a token.
            if digit_offset == 0:
                units += 4
            digit_offset = (digit_offset + 1) % 3
        else:
            digit_offset = 0
            # Quarter-token units keep classification and accumulation inline.
            if code_point <= 0x7F:
                units += 1
            elif 0x4E00 <= code_point <= 0x9FFF:
                units += 6
            elif code_point > 0xFFFF:
                units += 8
            elif east_asian_width(char) in ("W", "F"):
                units += 6
            else:
                units += 4

        if max_units is not None:
            if units > max_units:
                return (units + 3) // 4, prefix_end
            if units <= prefix_units:
                prefix_end = end
    return (units + 3) // 4, len(text)


def estimate_embedding_input_tokens(text: str) -> int:
    """Estimate embedding input without a provider-specific tokenizer.

    ASCII prose costs 1/4 token per character; consecutive digits cost one
    token per group of up to three. Wide/fullwidth BMP characters cost 1.5,
    supplementary-plane characters cost 2, and other non-ASCII characters
    cost 1. These are heuristics, not an upper bound on provider token counts.
    """
    return _scan_embedding_input(text)[0]


def truncate_embedding_input(
    text: str,
    max_tokens: int,
    suffix: str = EMBEDDING_TRUNCATION_SUFFIX,
) -> str:
    """Fit text and its marker into a positive estimated token budget.

    Preserve text that already fits. Omit the marker if it cannot fit by itself.
    """
    if not text:
        return text
    if max_tokens <= 0:
        return suffix.lstrip()
    # No character costs more than two estimated tokens under this heuristic.
    if len(text) * 2 <= max_tokens:
        return text

    suffix_tokens = estimate_embedding_input_tokens(suffix)
    if suffix_tokens > max_tokens:
        suffix = ""
        suffix_tokens = 0
    tokens, end = _scan_embedding_input(text, max_tokens, suffix_tokens)
    if tokens <= max_tokens:
        return text
    return text[:end].rstrip() + suffix


def resolve_embedding_max_input_tokens(
    config: dict[str, object] | None,
    default: int | None = None,
) -> int | None:
    """Read and normalize max_input_tokens from an embedder config dict."""
    raw_value = (config or {}).get("max_input_tokens", default)
    if raw_value is None:
        return default

    try:
        max_tokens = int(raw_value)
    except (TypeError, ValueError):
        return default

    if max_tokens <= 0:
        return default
    return max_tokens
