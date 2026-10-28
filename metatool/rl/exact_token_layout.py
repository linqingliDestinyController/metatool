"""Exact byte spans for sampled Qwen byte-BPE response IDs.

This is an alignment safety fix. It does not define reward or token weights.
"""
from __future__ import annotations


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(map(chr, cs), bs))


_BYTE_DECODER = _byte_decoder()


def response_byte_layout(ids, tokenizer):
    ids = [int(x) for x in ids]
    tokenizer_length = len(tokenizer)
    if any(tid < 0 or tid >= tokenizer_length for tid in ids):
        raise RuntimeError("Sampled response contains an ID outside tokenizer range")
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    special = set(tokenizer.all_special_ids)
    pieces, offsets, end = [], [], 0
    for tid, tok in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        try:
            piece = tok.encode("utf-8") if tid in special else bytes(_BYTE_DECODER[c] for c in tok)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot reconstruct generated token ID {tid}") from exc
        pieces.append(piece)
        offsets.append((end, end + len(piece)))
        end += len(piece)
    if b"".join(pieces).decode("utf-8", errors="replace") != text:
        raise RuntimeError("Sampled ID bytes do not reconstruct decoded response exactly")
    return text, offsets


def span_to_token_span(text, char_lo, char_hi, offsets):
    if char_hi <= char_lo:
        return 0, 0
    lo = len(text[:char_lo].encode("utf-8"))
    hi = len(text[:char_hi].encode("utf-8"))
    selected = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    if not selected:
        raise RuntimeError(f"No sampled token overlaps character span {char_lo}:{char_hi}")
    return selected[0], selected[-1] + 1
