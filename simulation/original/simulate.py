"""Streaming simulation on 10,000 tokens with a genuine K-token history.

The score-relevant regime is the parity of all K token identity bits. Token
pair identities depend on the preceding token, so scores share a tractable
two-regime representation while generation operates on the full vocabulary.
"""

import numpy as np


def simulate(p, q, score_arrays, floors, K, lengths, n_paths, seed,
             source="human", epsilon=0, attack="minimum_zero"):
    """Return (score snapshots, coverage metadata) without token trajectories.

    p is the target kernel and q the human kernel over local groups, shape
    (2,G). Every group has two equiprobable token identity bits in clean
    generation. ``source='human'`` uses q and ignores contamination; either
    ``source='machine'`` or ``source='target'`` uses the contaminated p kernel.
    The initial K-token history consists of zeros. Scores have shape (4,2,G).
    Raw/clipped snapshots have shape (n_paths,4) and contain path averages.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 2 or p.shape[0] != 2 or p.shape != q.shape or not 1 <= p.shape[1] <= 5000:
        raise ValueError("p and q must both have shape (2,G), 1 <= G <= 5000")
    for kernel in (p, q):
        if not np.all(np.isfinite(kernel)) or np.any(kernel < 0) or not np.allclose(kernel.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("kernel rows must be finite nonnegative probabilities summing to one")
    score_arrays = np.asarray(score_arrays, dtype=float)
    floors = np.asarray(floors, dtype=float)
    if score_arrays.shape != (4, *p.shape) or floors.shape != (4,):
        raise ValueError("score_arrays must be (4,2,G) and floors must be (4,)")
    if not np.all(np.isfinite(score_arrays)) or not np.all(np.isfinite(floors)):
        raise ValueError("scores and floors must be finite")
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    checkpoints = list(lengths)
    if not checkpoints or any(not isinstance(n, (int, np.integer)) or n < 1 for n in checkpoints):
        raise ValueError("lengths must be positive integers")
    checkpoints = set(int(n) for n in checkpoints)
    if source not in {"human", "machine", "target", "contaminated"}:
        raise ValueError("unknown source")
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if attack not in {"minimum_zero", "minimum_force_zero", "human_mixture"}:
        raise ValueError("unknown contamination attack")

    rng = np.random.default_rng(seed)
    cdf_p, cdf_q = np.cumsum(p, axis=1), np.cumsum(q, axis=1)
    cdf_p[:, -1] = cdf_q[:, -1] = 1.0
    minimum = np.argmin(score_arrays[0], axis=1)
    history = np.zeros((n_paths, K), dtype=np.int32)
    regime = np.zeros(n_paths, dtype=np.intp)
    last_pair = np.zeros(n_paths, dtype=np.intp)
    totals = np.zeros((n_paths, 4), dtype=float)
    clipped_totals = np.zeros_like(totals)
    visited = np.zeros(10000, dtype=bool)
    regime_visits = np.zeros(2, dtype=np.int64)
    contamination_count = 0
    snapshots = {}
    path_index = np.arange(n_paths)
    for step in range(1, max(checkpoints) + 1):
        position = (step - 1) % K
        outgoing_bit = history[:, position] & 1
        groups = np.empty(n_paths, dtype=np.intp)
        uniforms = rng.random(n_paths)
        base_cdf = cdf_q if source == "human" else cdf_p
        for z in (0, 1):
            indices = np.flatnonzero(regime == z)
            groups[indices] = np.searchsorted(base_cdf[z], uniforms[indices], side="right")
            regime_visits[z] += indices.size
        bits = rng.integers(0, 2, n_paths, dtype=np.intp)
        if source != "human" and epsilon > 0:
            contaminated = rng.random(n_paths) < epsilon
            contamination_count += int(contaminated.sum())
            if attack == "human_mixture":
                human_uniforms = rng.random(n_paths)
                for z in (0, 1):
                    indices = np.flatnonzero(contaminated & (regime == z))
                    groups[indices] = np.searchsorted(cdf_q[z], human_uniforms[indices], side="right")
            else:
                groups[contaminated] = minimum[regime[contaminated]]
                if attack == "minimum_zero":
                    bits[contaminated] = 0
                else:
                    bits[contaminated] = regime[contaminated] ^ outgoing_bit[contaminated]
        local_scores = score_arrays[:, regime, groups].T
        totals += local_scores
        clipped_totals += np.maximum(local_scores, floors[None, :])
        pair = np.where(groups == 0, 0, 1 + ((last_pair + groups - 1) % 4999))
        tokens = 2 * pair + bits
        visited[tokens] = True
        history[path_index, position] = tokens
        regime = regime ^ outgoing_bit ^ bits
        last_pair = pair
        if step in checkpoints:
            snapshots[step] = {"raw": totals.copy() / step,
                               "clipped": clipped_totals.copy() / step}
    metadata = {
        "vocabulary_size": 10000,
        "visited_token_count": int(visited.sum()),
        "visited_regime_count": int(np.count_nonzero(regime_visits)),
        "regime_visits": regime_visits.tolist(),
        "transitions": int(max(checkpoints) * n_paths),
        "contaminated_transitions": contamination_count,
        "history_length": int(K),
        "initial_history": "all zero tokens",
        "source": source,
        "attack": attack if source != "human" else None,
        "seed": int(seed),
        "final_history_parity_check": bool(np.array_equal(np.bitwise_xor.reduce(history & 1, axis=1), regime)),
    }
    return snapshots, metadata


if __name__ == "__main__":
    probabilities = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
    scores = np.stack([np.log(probabilities)] * 4)
    floors = np.full(4, -1.1)
    first, metadata = simulate(probabilities, probabilities, scores, floors, 3, [1, 10], 100, 7)
    again, _ = simulate(probabilities, probabilities, scores, floors, 3, [10], 100, 7)
    assert np.array_equal(first[10]["raw"], again[10]["raw"])
    assert np.all(first[10]["clipped"] >= first[10]["raw"])
    assert metadata["final_history_parity_check"]
    forced, info = simulate(probabilities, probabilities, scores, floors, 3, [10], 100, 8,
                            source="machine", epsilon=1, attack="minimum_force_zero")
    assert info["regime_visits"] == [1000, 0]
    assert np.allclose(forced[10]["raw"], np.log(0.2))
    assert info["final_history_parity_check"]
    print("Large-vocabulary simulator validation passed.")
