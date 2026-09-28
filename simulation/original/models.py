"""Analytically chosen conditional profiles for paired-token supports.

Each group splits its probability equally across two actual vocabulary tokens.
All supported groups have positive mass. Group weights have strict ordering,
giving token competition ranks 1, 3, ..., 2G-1. These are synthetic profiles,
not estimates of a trained language model.
"""
import numpy as np


def score_arrays(p):
    """Return four oriented token-score arrays, one value per fair-bit pair."""
    p = np.asarray(p, dtype=float)
    group_rank = (p[:, None, :] > p[:, :, None]).sum(axis=2)
    rank = 1 + 2 * group_rank
    logp = np.log(p / 2.0)
    entropy = -(p * logp).sum(axis=1)
    return {
        "loglik": logp,
        "negative_rank": -rank.astype(float),
        "negative_logrank": -np.log(rank.astype(float)),
        "negative_entropy_gap": logp + entropy[:, None],
    }


def _profile(groups, masses):
    counts = [max(1, int(round(0.04 * groups))),
              max(1, int(round(0.46 * groups)))]
    counts.append(groups - sum(counts))
    blocks = []
    for count, mass in zip(counts, masses):
        slope = np.linspace(1.025, 0.975, count)
        blocks.append(mass * slope / slope.sum())
    return np.concatenate(blocks)


def make_profiles(support_size):
    """Return [regime, group] probabilities for the requested token support.

    The supported token count is twice the number of groups. Mapping these
    positive groups into the full vocabulary is done by the history model.
    """
    if support_size % 2 or support_size < 50:
        raise ValueError("support_size must be even and at least 50")
    groups = support_size // 2
    base = np.array([0.70, 0.20, 0.10])
    human = np.array([0.15, 0.70, 0.15])
    cases = []
    for name, shift, description in [
        ("mild", 0.005, "Three probability tiers with small regime variation and positive probabilities throughout the support."),
        ("larger_heterogeneity", 0.08, "Same score-generating construction with appreciable regime variation."),
    ]:
        p = np.vstack([_profile(groups, base),
                       _profile(groups, base + shift * np.array([-1., .6, .4]))])
        q = np.vstack([_profile(groups, human), _profile(groups, human)])
        cases.append(dict(name=name, p=p, q=q, description=description))
    cases.append(dict(
        name="tailheavy_control",
        p=cases[0]["p"].copy(),
        q=np.vstack([_profile(groups, [.45, .10, .45])] * 2),
        description="Human mass concentrated on head and rare tail; intended to expose conservatism rather than to assume clipping cannot help.",
    ))
    return cases
