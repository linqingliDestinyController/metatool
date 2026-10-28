# === Token Weighting Module ===
"""
Token weighting module for reinforcement learning training.
This module provides functionality to assign different weights to tokens
based on their semantic importance in the training process.
"""

import json
import re
from typing import Dict, List, Tuple, Union

import torch
from metatool.rl.exact_token_layout import response_byte_layout, span_to_token_span
from metatool.rl.token_coverage import (
    structural_spans, call_field_spans, eos_ids, generated_eos_indices,
)


# Constants for weight adjustment
DEFAULT_GAMMA_FORMAT = 1.0
DEFAULT_STEP_WEIGHT_INCREASE = 0.002
DEFAULT_BASE_THINK_WEIGHT = 0.6
DEFAULT_MAX_GAMMA_ARGS = 1.2
DEFAULT_NORMALIZATION_EPS = 1e-8

# Regex patterns for tag matching
THINK_TAG_PATTERN = r"<think>(.*?)</think>"
RESPONSE_TAG_PATTERN = r"<response>(.*?)</response>"
TOOL_CALL_TAG_PATTERN = r"<tool_call>(.*?)</tool_call>"
TOOL_CALL_CONTENT_PATTERN = r"(?:<tool_call>|<toolcall>)(.*?)(?:</tool_call>|</toolcall>)"
TOOL_NAME_PATTERN = r'"name"\s*:\s*"([^"]+)"'
TOOL_ARGS_PATTERN = r'"(?:arguments|parameters)"\s*:\s*'


def char_span_to_token_span(text, char_lo, char_hi, tokenizer, offsets):
    """Map character spans onto the exact sampled IDs; include the rightmost overlap."""
    return span_to_token_span(text, char_lo, char_hi, offsets)

def _find_braced_span(s: str, open_idx: int) -> Tuple[int, int]:
    """Find the matching closing brace for a JSON object.
    
    Args:
        s: Input string
        open_idx: Index of the opening brace
        
    Returns:
        Tuple of (start_idx, end_idx) for the complete braced span
    """
    depth = 0
    i = open_idx
    in_str = False
    esc = False
    
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return open_idx, i + 1
        i += 1
    return open_idx, open_idx  # No matching brace found


def _assign_tag_weights(
    text: str,
    tag_pattern: str,
    token_weight: torch.Tensor,
    row_idx: int,
    weight_value: float,
    tokenizer,
    offsets
) -> None:
    """Assign weights to opening and closing tags.
    
    Args:
        text: Input text
        tag_pattern: Regex pattern for tag matching
        token_weight: Token weight tensor to update
        row_idx: Row index in the batch
        weight_value: Weight value to assign
        tokenizer: Tokenizer instance
    """
    for m in re.finditer(tag_pattern, text, flags=re.S):
        # Assign weight to opening tag
        start_tag_lo, start_tag_hi = m.span(0)
        start_tag_end = start_tag_lo + text[start_tag_lo:start_tag_hi].find('>') + 1
        t_lo, t_hi = char_span_to_token_span(text, start_tag_lo, start_tag_end, tokenizer, offsets)
        if t_hi > t_lo:
            token_weight[row_idx, t_lo:t_hi] = token_weight[row_idx, t_lo:t_hi].maximum(
                torch.full((t_hi - t_lo,), weight_value, device=token_weight.device)
            )
        
        # Assign weight to closing tag
        end_tag_start = start_tag_lo + text[start_tag_lo:start_tag_hi].rfind('<')
        end_tag_hi = start_tag_hi
        t_lo, t_hi = char_span_to_token_span(text, end_tag_start, end_tag_hi, tokenizer, offsets)
        if t_hi > t_lo:
            token_weight[row_idx, t_lo:t_hi] = token_weight[row_idx, t_lo:t_hi].maximum(
                torch.full((t_hi - t_lo,), weight_value, device=token_weight.device)
            )


def build_token_weight(
    batch: Dict[str, Union[torch.Tensor, List[str], int]], 
    tokenizer,
    gamma_name: float = 1.2, 
    gamma_args: float = 0.6,
    alpha: float = 0.5, 
    beta: float = 5.0,
    w_min: float = 0.001, 
    w_max: float = 8.0,
) -> torch.Tensor:
    """Build token weights for reinforcement learning training.
    
    Assigns weights to tokens in <tool_call>/<toolcall> blocks:
    - "name" field gets moderate weight
    - "arguments"/"parameters" use the unchanged metatool step schedule
    - metatool adds every structural tag, all complete call objects, and actual EOS
    - Respond body remains unweighted except tokens overlapping covered spans
    - metatool assignments remain; normalization includes newly covered tokens
    
    Args:
        batch: Dictionary containing:
            - "loss_mask": Tensor of shape (B, T) indicating valid positions
            - "response_text": List of response text strings
            - "response_ids": Actual sampled IDs, never re-tokenized output
            - "eos_token_ids": Same int/list used for the response mask;
              tokenizer.eos_token_id is only a direct-call fallback
            - "step": Current training step number
        tokenizer: Tokenizer instance for text processing
        gamma_name: Weight multiplier for tool name fields
        gamma_args: Weight multiplier for tool arguments
        alpha: Alpha parameter for length-dependent weighting  
        beta: Beta parameter for length-dependent weighting
        w_min: Minimum weight value
        w_max: Maximum weight value
        
    Returns:
        Token weight tensor of shape (B, T)
        
    Raises:
        ValueError: If batch format is invalid
        KeyError: If required keys are missing from batch
        TypeError: If input types are incorrect
    """

    # Extract current training step
    step = batch["step"]
    print(f"Current training step: {step}")
    if step is None:
        raise RuntimeError("No global_steps attribute found in _global_trainer, unable to get training step.")

    # Dynamic adjustment of format and think section weights
    # gamma_format = 0.8 - DEFAULT_STEP_WEIGHT_INCREASE * step  # High initially, decreases with step
    gamma_format = DEFAULT_GAMMA_FORMAT

    gamma_args = DEFAULT_BASE_THINK_WEIGHT + DEFAULT_STEP_WEIGHT_INCREASE * step  # Argument weight increases with step
    gamma_args = min(gamma_args, DEFAULT_MAX_GAMMA_ARGS)

    # Input validation
    if batch is None or not isinstance(batch, dict):
        raise ValueError(f"build_token_weight: input batch must be dict, got: {type(batch)}")
    if "loss_mask" not in batch:
        raise KeyError("build_token_weight: missing 'loss_mask'")
    if "response_text" not in batch:
        raise KeyError("build_token_weight: missing 'response_text'")
    if tokenizer is None:
        raise ValueError("build_token_weight: tokenizer cannot be None")

    loss_mask = batch["loss_mask"]  # (B, T)
    if not torch.is_tensor(loss_mask):
        raise TypeError(f"'loss_mask' must be torch.Tensor, got {type(loss_mask)}")
    B, T = loss_mask.shape

    response_texts = batch["response_text"]
    response_ids = batch["response_ids"]
    if not isinstance(response_texts, (list, tuple)) or len(response_texts) != B:
        raise ValueError("'response_text' must be List[str] with length equal to batch size")

    token_weight = torch.zeros_like(loss_mask, dtype=torch.float32)



    # Preserve metatool coefficients: at step100, args=.8 and name=1.2.
    A = float(gamma_args)
    Bv = float(gamma_name)

    if len(response_ids) != B:
        raise ValueError("response_ids length differs from batch size")
    terminal_ids = eos_ids(batch.get("eos_token_ids"), tokenizer.eos_token_id)
    for i in range(B):
        text = response_texts[i]
        n = int(loss_mask[i].sum().item())
        if not bool(((loss_mask[i, :n] == 1).all() & (loss_mask[i, n:] == 0).all()).item()):
            raise ValueError("metatool requires the unchanged contiguous binary response mask")
        if len(response_ids[i]) < n:
            raise ValueError("Fewer response IDs than valid mask positions")
        sampled_text, offsets = response_byte_layout(response_ids[i][:n], tokenizer)
        if sampled_text != text:
            raise RuntimeError("response_text differs from actual generated response IDs")
        if not isinstance(text, str):
            raise TypeError(f"response_text[{i}] must be str, got {type(text)}")

        # 1) Match and assign weights to format sections
        # 1a) <think>...</think> section - assign weight to tags and content
        _assign_tag_weights(text, THINK_TAG_PATTERN, token_weight, i, gamma_format, tokenizer, offsets)
        
        # Assign adaptive weight to <think> content
        for m in re.finditer(THINK_TAG_PATTERN, text, flags=re.S):
            content_lo, content_hi = m.span(1)
            c_t_lo, c_t_hi = char_span_to_token_span(text, content_lo, content_hi, tokenizer, offsets)
            if c_t_hi > c_t_lo:
                # Adaptive weight for think content based on training step
                gamma_think = DEFAULT_BASE_THINK_WEIGHT + DEFAULT_STEP_WEIGHT_INCREASE * step  # Increases with step
                gamma_think = min(gamma_think, DEFAULT_GAMMA_FORMAT)
                token_weight[i, c_t_lo:c_t_hi] = token_weight[i, c_t_lo:c_t_hi].maximum(
                    torch.full((c_t_hi - c_t_lo,), gamma_think, device=token_weight.device)
                )

        # 1b) <response>...</response> section - only assign weight to tags themselves
        _assign_tag_weights(text, RESPONSE_TAG_PATTERN, token_weight, i, gamma_format, tokenizer, offsets)

        # 1c) <tool_call>...</tool_call> section - only assign weight to tags themselves
        _assign_tag_weights(text, TOOL_CALL_TAG_PATTERN, token_weight, i, gamma_format, tokenizer, offsets)

        # 2) Match specific content in <tool_call> or <toolcall>
        for m in re.finditer(TOOL_CALL_CONTENT_PATTERN, text, flags=re.S):
            block = m.group(1)
            base_lo, base_hi = m.span(1)

            # 2a) "name": "..." - assign moderate weight to tool names
            nm = re.search(TOOL_NAME_PATTERN, block)
            if nm:
                c_lo = base_lo + nm.start(1)
                c_hi = base_lo + nm.end(1)
                t_lo, t_hi = char_span_to_token_span(text, c_lo, c_hi, tokenizer, offsets)
                if t_hi > t_lo:
                    token_weight[i, t_lo:t_hi] = token_weight[i, t_lo:t_hi].maximum(
                        torch.full((t_hi - t_lo,), Bv, device=token_weight.device)
                    )

            # 2b) First-object argument span; keep its existing schedule.
            km = re.search(TOOL_ARGS_PATTERN, block)
            if km:
                brace_pos = block.find('{', km.end())
                if brace_pos != -1:
                    loc_lo, loc_hi = _find_braced_span(block, brace_pos)
                    if loc_hi > loc_lo:
                        c_lo = base_lo + loc_lo
                        c_hi = base_lo + loc_hi
                        t_lo, t_hi = char_span_to_token_span(text, c_lo, c_hi, tokenizer, offsets)
                        if t_hi > t_lo:
                            token_weight[i, t_lo:t_hi] = token_weight[i, t_lo:t_hi].maximum(
                                torch.full((t_hi - t_lo,), A, device=token_weight.device)
                            )

        # metatool additive coverage. Keep metatool assignments above, including its
        # malformed-JSON fallback; do not remove old positive support.
        spans = [(lo, hi, gamma_format) for lo, hi in structural_spans(text)]
        spans += [(lo, hi, Bv if kind == "name" else A)
                  for kind, lo, hi in call_field_spans(text)]
        for c_lo, c_hi, value in spans:
            t_lo, t_hi = char_span_to_token_span(text, c_lo, c_hi, tokenizer, offsets)
            if t_hi > t_lo:
                token_weight[i, t_lo:t_hi] = token_weight[i, t_lo:t_hi].clamp_min(value)
        # A generated pad-ID is covered only if metadata declares it an EOS
        # and the original response mask marks that position as valid.
        for token_idx in generated_eos_indices(response_ids[i], n, terminal_ids):
            token_weight[i, token_idx] = token_weight[i, token_idx].clamp_min(gamma_format)

        # 3) Clip only positive weights, then normalize on "positive weights ∩ valid positions" to mean=1
        pos_mask = (token_weight[i] > 0) & (loss_mask[i] != 0)
        if pos_mask.any():
            tw = token_weight[i][pos_mask].clamp(w_min, w_max)
            den = tw.mean().clamp_min(DEFAULT_NORMALIZATION_EPS)
            token_weight[i][pos_mask] = tw / den
        # 4) Keep only valid positions
        token_weight[i] *= (loss_mask[i] != 0).float()

    return token_weight
