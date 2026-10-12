# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for bounding text sent to embedding providers."""

from __future__ import annotations

import re

EMBEDDING_TRUNCATION_SUFFIX = "\n...(truncated for embedding)"
_CJK_RUN = re.compile("[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_WEIGHTED_RUN = re.compile(f"({_CJK_RUN.pattern})|([0-9]+)|([\U00010000-\U0010ffff]+)")
# Both counting and truncation use quarter-token units.
_CJK_UNITS = 4
_DIGIT_UNITS = 2
_SUPPLEMENTARY_UNITS = 8
_RUN_UNITS = (_CJK_UNITS, _DIGIT_UNITS, _SUPPLEMENTARY_UNITS)


def estimate_embedding_input_tokens(text: str) -> int:
    """Estimate CJK=1, supplementary=2, ASCII digit=1/2, other=1/4 tokens."""
    other = text if text.isascii() else _CJK_RUN.sub("", text)
    data = other.encode("utf-8", "surrogatepass")
    digits = len(data) - len(data.translate(None, b"0123456789"))
    # Each supplementary code point has one UTF-8 leading byte in F0..F4.
    supplementary = (
        0 if other.isascii() else len(data) - len(data.translate(None, b"\xf0\xf1\xf2\xf3\xf4"))
    )
    units = (len(text) - len(other)) * _CJK_UNITS + len(other)
    units += digits * (_DIGIT_UNITS - 1) + supplementary * (_SUPPLEMENTARY_UNITS - 1)
    return (units + 3) // 4


def truncate_embedding_input(
    text: str,
    max_tokens: int,
    suffix: str = EMBEDDING_TRUNCATION_SUFFIX,
) -> str:
    """Fit text and its marker into a positive estimated token budget.

    Preserve text that already fits. Omit the marker if it cannot fit by itself.
    The character heuristic is not an upper bound on provider token counts.
    """
    if not text:
        return text
    if max_tokens <= 0:
        return suffix.lstrip()
    if len(text) * max(_RUN_UNITS) <= max_tokens * 4:
        return text

    suffix_tokens = estimate_embedding_input_tokens(suffix)
    if suffix_tokens > max_tokens:
        suffix = ""
        suffix_tokens = 0
    # Quarter-token units avoid rounding each run separately. Scan at most the
    # prefix that could fit at the cheapest character cost, plus one character.
    max_units = max_tokens * 4
    prefix_units = (max_tokens - suffix_tokens) * 4
    scan_end = min(len(text), max_units + 1)
    units = position = prefix_end = 0
    for match in _WEIGHTED_RUN.finditer(text, 0, scan_end):
        start, end = match.span()
        # A weighted run is preceded by a possibly empty ordinary-character run.
        weighted_cost = _RUN_UNITS[match.lastindex - 1]
        for run_end, cost in ((start, 1), (end, weighted_cost)):
            length = run_end - position
            if units <= prefix_units:
                prefix_end = position + min(length, (prefix_units - units) // cost)
            units += length * cost
            if units > max_units:
                return text[:prefix_end].rstrip() + suffix
            position = run_end

    # Only ordinary characters remain after the final match. Keep the input
    # intact if it fits the full budget without needing a marker.
    if units + scan_end - position <= max_units:
        return text
    if units <= prefix_units:
        prefix_end = position + prefix_units - units
    return text[:prefix_end].rstrip() + suffix


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
