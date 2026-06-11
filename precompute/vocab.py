"""Token-id -> display string via the official tokenizer, with byte-level cleanup."""
from __future__ import annotations

from typing import Iterable

from typedefs import Tokenizer, TokenId


def decode_token(tokenizer: Tokenizer, token_id: TokenId) -> str:
    """Raw decode of a single id (plumbing)."""
    return tokenizer.decode([token_id])


def display_token(tokenizer: Tokenizer, token_id: TokenId) -> str:
    """Human-readable UI label for a token id: decode, then make whitespace visible
    (space → ␣, newline → ↵)."""


    text = decode_token(tokenizer, token_id)

    return (
        text
        .replace(" ", "␣")
        .replace("\n", "↵")
    )


def build_vocab_map(tokenizer: Tokenizer, referenced_ids: Iterable[TokenId]) -> dict[TokenId, str]:
    """{id: display_token} for only the ids referenced in a file (keeps the JSON light)."""
    disp_map: dict[TokenId, str] = {}

    for ref_id in referenced_ids:
        disp_map[ref_id] = display_token(tokenizer, ref_id)

    return disp_map
