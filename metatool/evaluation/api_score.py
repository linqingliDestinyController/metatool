"""CPU-only API-Bank scoring using the actual bundled metatool source blocks.

The paper/source metric tolerates empty thoughts, multiple calls and a parameter
only fallback. It is intentionally not a strict format/one-call diagnostic.
"""
import ast
import contextlib
import io
import json
import re
import statistics
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_cases(data_dir):
    cases = {}
    for level, expected in ((1, 399), (2, 67), (3, 131)):
        rows = json.loads((Path(data_dir) / f"level-{level}-api_processed.json").read_text())
        if len(rows) != expected:
            raise ValueError(f"Unexpected API-Bank level {level} size")
        cases.update({f"Level{level}_{i}": row for i, row in enumerate(rows)})
    return cases


def load_predictions(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    predictions = {row["id"]: row for row in rows}
    if len(predictions) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return predictions


def scoring_blocks():
    from metatool.evaluation.api_blocks import PARSER_SOURCE, SCORER_SOURCE
    return compile(PARSER_SOURCE,"api_parser","exec"),compile(SCORER_SOURCE,"api_match","exec")


def score(cases, predictions):
    if set(cases) != set(predictions):
        raise ValueError("Require all 597 predictions; no missing-item denominator changes")
    parser, scorer = scoring_blocks()
    results = {}
    for key, prediction in predictions.items():
        env = {"assistant_output": prediction["output"].strip(), "json": json}
        exec(parser, env)
        results[key] = {"data": cases[key], "tool_calls": env["all_tool_calls"]}
    env = {"results": results, "scores": {}, "json": json}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(scorer, env)
    scores = {key: int(row["score"]) for key, row in results.items()}
    by_level = {}
    for level in (1, 2, 3):
        values = [v for key, v in scores.items() if key.startswith(f"Level{level}_")]
        by_level[str(level)] = {"correct": sum(values), "n": len(values), "accuracy": sum(values) / len(values)}
    tokens = [p["output_tokens"] for p in predictions.values() if "output_tokens" in p]
    metrics = {"metric": "uploaded metatool API-Bank parser + SOFT MATCH loop",
               "correct": sum(scores.values()), "n": len(scores), "accuracy": sum(scores.values()) / len(scores),
               "levels": by_level,
               "diagnostics": {"empty_think": sum(bool(re.search(r"<think>\s*</think>", p["output"])) for p in predictions.values()),
                               "truncated": sum(bool(p.get("truncated", False)) for p in predictions.values()),
                               "mean_output_tokens": statistics.mean(tokens) if tokens else None}}
    return metrics, scores


