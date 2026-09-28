# Technical details for Appendix C.3

This file describes the simulation implementation and evaluation for Appendix C.3.
See the [simulation guide](../README.md) for reproduction commands and
[models.py](models.py) for the token construction. Exact probability profiles are
recorded in [results/profiles.json](results/profiles.json), and numerical power
results are in [results/monte_carlo.csv](results/monte_carlo.csv).

## Source model and the theorem

The simulation uses the common finite history class reachable from the
all-zero history. It does not verify irreducibility on the full Cartesian
space of all vocabulary tuples required by the original source assumption.
On the reachable class, both clean kernels are irreducible and aperiodic,
with strictly positive supported transition probabilities. The human
convergence proof applies there, as does the supported minimum-score attack.

For arbitrary contamination, an off-support observation causes the operational
test to reject immediately. In the auxiliary bounded-score process used for
concentration, use the actual clipped scores before the first off-support
observation, then assign the upper score endpoint at that observation and
all later steps. Before exit the conditional-mean lower bound is valid; after
exit it remains valid because the upper endpoint is at least that lower
bound. The operational acceptance event is contained in the auxiliary
acceptance event. Thus the martingale proof extends without assumptions on
histories outside the reachable class or on contaminated stationary weights.

This extension is a qualification of the model, not an empirical inference
from finite sampled coverage. All four simulated scores are bounded additive
scores. Fixing a threshold can sometimes express a ratio detector as an
additive contrast, but clipping that contrast is not generally componentwise
ratio clipping or recalibration of the ratio test.

## Score definitions and code notation

For regime \(z\) and local group \(g\), set
$$
\ell_z(g)=\log(p_z(g)/2),\qquad
H_z=-\sum_g p_z(g)\ell_z(g),\qquad
R_z(g)=1+2\#\{g':p_z(g')>p_z(g)\}.
$$
The scores are \(\ell_z(g)\), \(-R_z(g)\), \(-\log R_z(g)\), and
\(\ell_z(g)+H_z\). Competition ranking counts both equally likely tokens
in each group; this explains the factor two. The last score has clean
conditional mean zero because the same target probabilities define both
generation and the entropy centering.

| Code/output field | Paper quantity |
|---|---|
| `b` | \(\underline\phi\) |
| `mean_min`, `mean_max` | \(\mu_1-\delta,\mu_1+\delta\) |
| `rho` | \(\rho_0=F_0(\underline\phi+r)\) |
| `epsilon_raw` | \(\epsilon_-\) |
| `epsilon_clip` | Uncapped upper fraction; the paper uses \(\epsilon_+=\min\{1,\texttt{epsilon\_clip}\}\) |

## Band, contamination, and floor selection

`analysis.audit` first checks the mean ordering and sets
$$
r_{\max}=\min\{\mu_1-\delta-\underline\phi,
\overline\phi-\underline\phi\}.
$$
It takes `cap = np.nextafter(r_max, 0.)`, the largest representable
floating-point value below this positive limit. The candidate set is the
sorted union of 1,500 equally spaced values from `cap/1500` through `cap`
and all distinct score gaps strictly between zero and \(r_{\max}\).
The band mass uses a strict comparison, `score - b < r`.

For every candidate, the audit computes the endpoints and the margin
$$
\frac{\mu_1-\mu_0}{\mu_1-\underline\phi}
-\rho_0-\frac{2\delta+\eta}{r}.
$$
It selects the candidate maximizing
\(\min\{1,\texttt{epsilon\_clip}\}-\epsilon_-\).
An exact tie selects the first, hence smallest, candidate in the sorted set.
Certification requires a width greater than \(10^{-10}\).
`strong_condition` requires the displayed margin at the selected candidate
to exceed \(10^{-10}\). `best_strong_margin` instead maximizes that margin
over the entire grid. These are different selection objectives; the paper's
condition counts use `strong_condition`, not `best_strong_margin`.

For a certified score the selected contamination is
$$
\epsilon=\frac{\epsilon_-+\min\{1,\texttt{epsilon\_clip}\}}{2}.
$$
Writing
$$
t_{\min}=
\frac{\mu_0-(1-\epsilon)(\mu_1-\delta)-\epsilon\underline\phi}
{\epsilon-\rho_0},
$$
`floor_for` checks \(0<t_{\min}<r\) and chooses
\(a=\underline\phi+(t_{\min}+r)/2\).
All choices precede Monte Carlo evaluation. There is no global optimization
over all band widths and no assumption of a common interval across scores.
`common_interval` records whether the four intervals intersect, but the
simulation still uses detector-specific midpoints. The runner evaluates the
union of these midpoints and marks the matching rows as `theorem_selected`.
Uncertified scores use the fallback floor \(\underline\phi+r/2\); such
rows are not certified examples. If no score is certified, the runner uses
contamination levels 0.2, 0.4, and 0.6 for exploratory evaluation.

## Contamination strategies

- `minimum_zero`: choose a minimum-score group and set the new identity bit to zero.
- `minimum_force_zero`: choose a minimum-score group and set the new bit to
  the current regime XOR the outgoing oldest bit, forcing the next regime to zero.
- `human_mixture`: draw the contaminating group from the human conditional
  profile and draw a fair identity bit.

Since pair members have identical scores, either bit is available to a
minimum-score attacker. `analytic_attack_mean` uses the exact stationary
probability of regime one: \((1-\epsilon^K)/2\) for `minimum_zero`,
\((1-\epsilon)/2\) for `minimum_force_zero`, and \(1/2\) for
`human_mixture`. Independent validation enumerated small bit-window chains
to check these expressions. These means describe the specified attacks,
not an optimization over all adaptive strategies.

## Calibration, evaluation, and uncertainty

The frozen run uses seed 20260906, 2,000 evaluation paths, 4,000 independent
human calibration paths, and lengths 128, 512, 2048, and 4096. Initial
histories contain \(K\) zero tokens. Raw and clipped scores share paths;
different lengths use prefixes of the same paths. Calibration and evaluation
paths are independent, but reported rows are not independent repetitions.

For `calibrated_5pct`, each score and aggregation has its own human 95th
percentile, computed by `np.quantile(..., .95, method='higher')`; rejection
uses a strict `score > threshold` comparison. Empirical calibration does not
guarantee a population false-positive rate of exactly 5%.

For `fixed_theorem`, the clipped threshold is the midpoint of the human
upper bound \(\mu_0+\rho_0(a-\underline\phi)\) and contaminated lower
bound \((1-\epsilon)(\mu_1-\delta)+\epsilon a\). The raw threshold is
\(\mu_0+0.01(\overline\phi-\underline\phi)\).
These fixed thresholds are distinct from empirical 5% calibration.

The CSV's 95% Wilson intervals are conditional on the fitted thresholds.
They exclude calibration-sample uncertainty and simultaneous coverage across
configurations. The main figure reports point estimates.

## Saved results summarized in the paper

There are 48 audited configurations, 480 exact-mean rows, and 4,800 Monte
Carlo rows. At the selected bands, 40 configurations have positive certified
intervals and 24 satisfy the stronger proposition condition. For every
support size and history length, the counts out of four scores are:

| Profile | Proposition condition | Positive certified interval |
|---|---:|---:|
| Mild | 4 | 4 |
| Larger heterogeneity | 0 | 4 |
| Tail-heavy human control | 2 | 2 |

The following estimates use the mild profile, \(K=100\), \(M=1000\),
\(n=4096\), and `minimum_force_zero` at each score's selected contamination.

| Score | Contamination | Raw power | Clipped power | Raw FPR | Clipped FPR |
|---|---:|---:|---:|---:|---:|
| Log-likelihood | 0.6972 | 0.0000 | 0.9990 | 0.0465 | 0.0470 |
| Negative rank | 0.2153 | 0.0000 | 1.0000 | 0.0405 | 0.0460 |
| Negative log-rank | 0.5190 | 0.0000 | 0.9995 | 0.0450 | 0.0430 |
| Negative entropy gap | 0.6966 | 0.0000 | 0.9985 | 0.0460 | 0.0470 |

For the same configuration under `fixed_theorem`, estimated clipped powers
in that order are 0.9975, 0.9955, 0.9390, and 0.9950; estimated clipped FPRs
are 0.0035, 0, 0, and 0.0035. The fixed-level table does not demonstrate
false-positive probability tending to zero.

All 144 simulated batches visited all 10,000 vocabulary identities by the
largest checkpoint, and all rolling-history parity checks passed. These
diagnostics supplement the analytical reachability argument. Neither the
coverage counts nor the selected-attack simulations prove uniform robustness
or establish the assumptions for trained language models.
