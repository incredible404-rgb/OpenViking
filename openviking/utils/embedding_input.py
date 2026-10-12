# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for bounding text sent to embedding providers."""

from __future__ import annotations

import re

EMBEDDING_TRUNCATION_SUFFIX = "\n...(truncated for embedding)"
_CJK_RUN = re.compile("[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def estimate_embedding_input_tokens(text: str) -> int:
    """Keep the existing CJK=1, other=1/4 heuristic without a Python character loop."""
    if text.isascii():
        return (len(text) + 3) // 4
    other_chars = len(_CJK_RUN.sub("", text))
    return len(text) - other_chars + (other_chars + 3) // 4


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
    # Every character costs at most one token under the existing heuristic.
    if len(text) <= max_tokens:
        return text

    suffix_tokens = estimate_embedding_input_tokens(suffix)
    if suffix_tokens > max_tokens:
        suffix = ""
        suffix_tokens = 0
    if text.isascii():
        if len(text) <= max_tokens * 4:
            return text
        return text[: (max_tokens - suffix_tokens) * 4].rstrip() + suffix

    # Quarter-token units avoid rounding each run separately. Scan at most the
    # prefix that could fit at the cheapest character cost, plus one character.
    max_units = max_tokens * 4
    prefix_units = (max_tokens - suffix_tokens) * 4
    scan_end = min(len(text), max_units + 1)
    units = position = prefix_end = 0
    for match in _CJK_RUN.finditer(text, 0, scan_end):
        start, end = match.span()
        # Each CJK run is preceded by a possibly empty run of other characters.
        for run_end, cost in ((start, 1), (end, 4)):
            length = run_end - position
            if units <= prefix_units:
                prefix_end = position + min(length, (prefix_units - units) // cost)
            units += length * cost
            if units > max_units:
                return text[:prefix_end].rstrip() + suffix
            position = run_end

    # Only non-CJK characters remain after the final match. Keep the input
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
