"""metatool additive credit spans. Pure Python; never rewrites or scores a response.

Structural coverage is lexical: every exact control-tag spelling gets credit,
even if unpaired/repeated (or present literally in a JSON string). Call-field
parsing is JSON/string aware: quoted tags do not open/close blocks, and nested
`name` keys are not top-level tool names. Existing metatool assignments are retained.
"""
from __future__ import annotations
import json
import re
from numbers import Integral

STRUCTURAL_TAG = re.compile(r"</?(?:think|response|tool_call|toolcall)>")
CALL_TAG = re.compile(r"<(\/?)(?:tool_call|toolcall)>")
DECODER = json.JSONDecoder()


def structural_spans(text):
    for match in STRUCTURAL_TAG.finditer(text):
        yield match.start(), match.end()


def _ws(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _fields(text, start, end):
    """Walk immediate members of an already decoded complete JSON object."""
    pos = _ws(text, start + 1)
    while pos < end and text[pos] != '}':
        key, pos = DECODER.raw_decode(text, pos)
        pos = _ws(text, pos)
        if text[pos] != ':':
            raise ValueError('Decoded object has no member separator')
        lo = _ws(text, pos + 1)
        value, hi = DECODER.raw_decode(text, lo)
        if key == 'name' and isinstance(value, str) and hi > lo + 2:
            yield 'name', lo + 1, hi - 1
        elif key in ('arguments', 'parameters') and isinstance(value, dict):
            yield 'args', lo, hi
        pos = _ws(text, hi)
        if text[pos] == ',':
            pos = _ws(text, pos + 1)


def _malformed_end(text, start):
    """Skip damaged JSON without promoting a nested object into a new call.

    An unquoted call tag is a recovery boundary. No braces/values are invented;
    the first-object fallback assignments remain available for malformed text.
    """
    depth, quoted, escaped = 0, False, False
    for pos in range(start, len(text)):
        char = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '<' and CALL_TAG.match(text, pos):
            return pos
        elif char in '{[':
            depth += 1
        elif char in '}]':
            depth -= 1
            if depth <= 0:
                return pos + 1
    return len(text)


def call_field_spans(text):
    """All complete top-level call objects in call blocks, including JSONL.

    Repeated openers reset the block boundary. Complete objects in an unclosed
    block are usable spans; this does not make the response score as valid.
    Flat JSON arrays of calls are supported without recursively promoting
    parameter objects. Returned offsets refer to the original escaped text.
    """
    pos, active = 0, False
    while pos < len(text):
        tag = CALL_TAG.match(text, pos)
        if tag:
            active = not bool(tag.group(1))
            pos = tag.end()
            continue
        if text[pos] in '{["':
            try:
                value, end = DECODER.raw_decode(text, pos)
            except json.JSONDecodeError:
                pos = _malformed_end(text, pos) if text[pos] in '{[' else pos + 1
                continue
            if active and isinstance(value, dict) and isinstance(value.get('name'), str):
                yield from _fields(text, pos, end)
            elif active and isinstance(value, list):
                cursor = _ws(text, pos + 1)
                while cursor < end and text[cursor] != ']':
                    item, item_end = DECODER.raw_decode(text, cursor)
                    if isinstance(item, dict) and isinstance(item.get('name'), str):
                        yield from _fields(text, cursor, item_end)
                    cursor = _ws(text, item_end)
                    if text[cursor] == ',':
                        cursor = _ws(text, cursor + 1)
            pos = end
        else:
            pos += 1


def eos_ids(explicit, fallback=None):
    """Only declared EOS IDs; never infer EOS from pad_token_id."""
    value = fallback if explicit is None else explicit
    values = [value] if isinstance(value, Integral) else value
    if not isinstance(values, (list, tuple, set)) or not values:
        raise ValueError('metatool requires eos_token_ids or tokenizer.eos_token_id')
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in values):
        raise ValueError(f'Invalid EOS token IDs: {values!r}')
    return frozenset(int(i) for i in values)


def generated_eos_indices(sampled_ids, valid_length, terminals):
    if len(sampled_ids) < valid_length:
        raise ValueError('Fewer response IDs than valid mask positions')
    return [i for i in range(valid_length) if int(sampled_ids[i]) in terminals]
