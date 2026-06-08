"""Token-id -> display string via the official tokenizer, with byte-level cleanup."""


def decode_token(tokenizer, token_id):
    """Raw decode of a single id (plumbing)."""
    return tokenizer.decode([token_id])


def display_token(tokenizer, token_id):
    """Human-readable UI label — the cleanup CONVENTION is yours.
    Qwen is byte-level BPE: leading-space marker, newline marker, partial-byte tokens, specials.
    Design qs: space as ' ' or visible '␣'? newline as '↵'? mark un-renderable partial-byte tokens?
    label specials (<|im_start|> ...)?"""
    # TODO(unnat): implement the display cleanup convention.
    raise NotImplementedError


def build_vocab_map(tokenizer, referenced_ids):
    """{id: display_token} for ONLY the ids referenced in a file (keeps it light)."""
    # TODO(unnat): implement.
    raise NotImplementedError
