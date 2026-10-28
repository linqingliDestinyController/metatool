"""Minimal scalar adapter; the training manager calls the four-component reward directly."""
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None,
                          sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    from metatool.rl.reward import compute_score
    if "rlla" not in data_source:
        raise ValueError("Unsupported training data source")
    return float(compute_score(solution_str, ground_truth, step=0)[0])
_default_compute_score = default_compute_score
