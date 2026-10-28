"""Fixed, reward-schedule-independent validation for the isolated metatool method."""
import hashlib
import json
import statistics
from pathlib import Path


def _tool_list(text):
    try:
        inside = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        tools = [json.loads(line) for line in inside.splitlines() if line.strip()]
        if not tools or any(not isinstance(x, dict) or "name" not in x or "parameters" not in x for x in tools):
            return None
        return sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in tools)
    except (IndexError, ValueError, TypeError):
        return None


def score_one(response, ground_truth, truncated):
    response = response.strip()
    needs_tool = "<tool_call>" in ground_truth
    common = (response.count("<think>") == response.count("</think>") == 1 and
              response.find("<think>") < response.find("</think>"))
    if needs_tool:
        valid = (common and response.count("<tool_call>") == response.count("</tool_call>") == 1 and
                 "<response>" not in response and "</response>" not in response and
                 response.find("</think>") < response.find("<tool_call>") <
                 response.find("</tool_call>") and _tool_list(response) is not None)
        exact = bool(valid and not truncated and _tool_list(response) == _tool_list(ground_truth))
        return bool(valid and not truncated), exact, None
    valid = (common and response.count("<response>") == response.count("</response>") == 1 and
             "<tool_call>" not in response and "</tool_call>" not in response and
             response.find("</think>") < response.find("<response>") < response.find("</response>"))
    decision = bool(valid and not truncated)
    return decision, None, decision


def score_batch(batch, generated, tokenizer, max_response_length):
    rows = []
    prompts = generated.batch["prompts"]
    responses = generated.batch["responses"]
    attention = generated.batch["attention_mask"]
    prompt_width = prompts.shape[1]
    tokenizer_length = len(tokenizer)
    for i in range(len(batch)):
        prompt_ids = prompts[i][attention[i, :prompt_width].bool()].tolist()
        length = int(attention[i, prompt_width:].sum().item())
        response_ids = responses[i, :length].tolist()
        assert all(0 <= token_id < tokenizer_length for token_id in response_ids)
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        ground_truth = batch.non_tensor_batch["reward_model"][i]["ground_truth"]
        truncated = length >= max_response_length and (not response_ids or response_ids[-1] != tokenizer.eos_token_id)
        fmt, tool_exact, no_tool_exact = score_one(response, ground_truth, truncated)
        rows.append({
            "index": len(rows), "prompt_sha256": hashlib.sha256(bytes(str(prompt_ids), "utf-8")).hexdigest(),
            "ground_truth": ground_truth, "output": response, "response_token_ids": response_ids,
            "output_tokens": length, "finish_reason": "length" if truncated else "eos_or_stop",
            "truncated": bool(truncated), "format_valid": fmt,
            "tool_exact": tool_exact, "no_tool_decision_exact": no_tool_exact,
        })
    return rows


def persist(run_dir, step, rows):
    assert len(rows) == 80, "Expected all 80 fixed validation items"
    folder = Path(run_dir) / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step_{step:03d}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            row["index"] = index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tool = [x for x in rows if x["tool_exact"] is not None]
    no_tool = [x for x in rows if x["no_tool_decision_exact"] is not None]
    assert len(tool) == 71 and len(no_tool) == 9
    lengths = [x["output_tokens"] for x in rows]
    result = {
        "step": step, "tool_exact_count": sum(x["tool_exact"] for x in tool),
        "tool_exact_n": 71, "no_tool_decision_exact_count": sum(x["no_tool_decision_exact"] for x in no_tool),
        "no_tool_n": 9, "format_valid_count": sum(x["format_valid"] for x in rows),
        "format_valid_n": 80, "truncation_count": sum(x["truncated"] for x in rows),
        "truncation_n": 80, "mean_output_tokens": statistics.mean(lengths),
        "median_output_tokens": statistics.median(lengths),
        "predictions": str(path),
    }
    with (Path(run_dir) / "validation_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result
