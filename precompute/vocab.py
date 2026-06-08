"""Token-id -> display string via the official tokenizer, with byte-level cleanup."""
from __future__ import annotations

from typing import Iterable

from typedefs import Tokenizer, TokenId


def decode_token(tokenizer: Tokenizer, token_id: TokenId) -> str:
    """Raw decode of a single id (plumbing)."""
    return tokenizer.decode([token_id])


def display_token(tokenizer: Tokenizer, token_id: TokenId) -> str:
    """Human-readable UI label — the cleanup CONVENTION is yours.
    Qwen is byte-level BPE: leading-space marker, newline marker, partial-byte tokens, specials.
    Design qs: space as ' ' or visible '␣'? newline as '↵'? mark un-renderable partial-byte tokens?
    label specials (<|im_start|> ...)?"""
    # TODO(unnat): implement the display cleanup convention.
    raise NotImplementedError


def build_vocab_map(tokenizer: Tokenizer, referenced_ids: Iterable[TokenId]) -> dict[TokenId, str]:
    """{id: display_token} for ONLY the ids referenced in a file (keeps it light)."""
    # TODO(unnat): implement.
    raise NotImplementedError
