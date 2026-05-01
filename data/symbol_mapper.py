from __future__ import annotations

import re

from config import DEFAULT_EGX_SYMBOLS, KNOWN_SYMBOL_NAMES


EGX_SUFFIX = ".CA"


def normalize_symbol(symbol: str) -> str:
    """Normalize user input to a bare EGX ticker such as COMI."""
    cleaned = re.sub(r"\s+", "", str(symbol or "").upper())
    if cleaned.endswith(EGX_SUFFIX):
        cleaned = cleaned[: -len(EGX_SUFFIX)]
    return cleaned


def to_yahoo_symbol(symbol: str) -> str:
    """Convert an EGX ticker to Yahoo Finance format."""
    bare = normalize_symbol(symbol)
    if not bare:
        return ""
    return f"{bare}{EGX_SUFFIX}"


def display_symbol(symbol: str) -> str:
    bare = normalize_symbol(symbol)
    return f"{bare}{EGX_SUFFIX}" if bare else ""


def company_name(symbol: str) -> str:
    return KNOWN_SYMBOL_NAMES.get(normalize_symbol(symbol), normalize_symbol(symbol))


def default_symbols(include_suffix: bool = False) -> list[str]:
    if include_suffix:
        return [to_yahoo_symbol(symbol) for symbol in DEFAULT_EGX_SYMBOLS]
    return list(DEFAULT_EGX_SYMBOLS)


def parse_symbol_list(raw: str) -> list[str]:
    parts = re.split(r"[\n,; ]+", raw or "")
    symbols = []
    seen = set()
    for part in parts:
        bare = normalize_symbol(part)
        if bare and bare not in seen:
            seen.add(bare)
            symbols.append(bare)
    return symbols
