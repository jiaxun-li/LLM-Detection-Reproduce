# Technical reference

Implementation details for the three studies in the paper.


## Experimental methods

The repository contains three related studies with separate fitting and
uncertainty procedures. Saved empirical summaries report seven detectors:
log likelihood, rank, log rank, LRR, entropy, entropy gap, and `binocular_origin`.
Detector components and scoring windows are described in [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md).

## Finite-state simulation

The simulation examines four bounded additive scores on a reachable finite-state
class, using vocabulary size 10,000, history lengths 10/100, and support sizes
500/1,000. Saved outputs cover 48 audited configurations, 480 exact-mean rows,
and 4,800 Monte Carlo rows. Full-run parameters use 2,000 evaluated paths,
4,000 calibration paths, and lengths 128, 512, 2,048, and 4,096.

Its guarantees concern the constructed finite-state class. They do not establish
the assumptions for trained language models, all seven empirical detectors, or
clipped component ratios. Simulated attacks are selected supported constructions;
the extension outside support is analytical. Wilson intervals condition on fitted
thresholds and do not include calibration uncertainty.

## Controlled contamination

Each of the nine cells selects 3,000 source groups: 500 for clipping tuning,
500 for calibration, and 2,000 for testing. Sources are selected deterministically
with seed 1731. SQuAD contexts are deduplicated and short contexts may be packed;
a selected source is not necessarily one original question/context pair.

Generation uses a nominal 30-token prefix, target length 220, minimum length
210, temperature 0.8, top-p 0.95, and seeds 101/202/303 assigned across sources.
Human and generated continuations are length-matched before contamination.
Scoring uses prompt context but aggregates response predictions, with a
512-token combined limit. No chat template is introduced.

At each positive replacement ratio (5%, 10%, 20%, 30%, 40%, 50%), three random
token-splice draws and one high-NLL-donor suffix construction are generated.
Together with clean human and machine rows, this gives 26 rows per source and
78,000 per cell. These are controlled token splices, not certified human edits;
they can disrupt syntax and conditioning beyond the replaced positions.

The clean tuning split determines score direction except for LRR (fixed +1)
and origin (fixed -1). LRR divides mean negative log likelihood by mean log
rank plus a stabilizer; it is not a two-model likelihood ratio. Entropy gap is
NLL minus entropy and is not variance-normalized Fast-DetectGPT.

Generic clipping lower-bounds oriented token evidence. LRR caps numerator and
denominator components; origin caps numerator NLL only. Candidate quantiles are
0.80, 0.85, 0.90, 0.95, 0.975, 0.99, and 0.995, plus no clipping. Primary selection
maximizes 0.8 pooled attacked tuning AUROC plus 0.2 clean tuning AUROC. It uses
10–50% attacks with 3:1 random-to-tail row weighting; 5% is excluded from fitting.
One frozen specification applies across attack modes, ratios, and FPR targets.

The primary origin ratio is a prompt-conditioned adaptation using the Falcon-7B
observer and Falcon-7B-Instruct performer on saved response features. It differs
from RAID's output-only implementation. Exactly constant/structurally saturated
candidates are rejected. Current code also rejects nonpositive LRR denominator
caps. Saved results preserve their accepted analysis versions; new analyses use
the current safeguards.

Raw and clipped thresholds are separately calibrated using clean human data at
1% and 5% target FPR. Held-out FPR can differ and should accompany TPR comparisons.
Intervals use 2,000 source-cluster percentile bootstraps, keeping variants and
raw/clipped contrasts paired. Directions, bounds, and thresholds remain fixed.
Each cell has 392 publication metric rows; all nine have 3,528.

## RAID: shared data and scoring

The reference uses the labeled English release with eight domains. One complete
generation family per source contributes one human document, one clean machine
document, and eleven attacked variants. The 500 sources used in development
were excluded before the domain-stratified 40%/20%/40% tuning/calibration/test
split. The reference contains 12,871 sources and 167,323 score rows. Selected source/split/generation identities are distributed in `reference/raid/`.
The original exclusion file is included at `reference/provenance/raid/excluded_sources.json`; see [PROVENANCE.md](PROVENANCE.md).

Single-model features use Falcon-7B. Origin uses Falcon-7B as observer and
Falcon-7B-Instruct as performer, with output-only tokenizer-native input and
right truncation to 512 input tokens. Causal shifted NLL and official cross-entropy
windows differ. Numerator clipping is anchored to the saved official document
numerator, adding its float64 token-clipping delta; the denominator is unchanged.
The upstream component gate must pass before corrected scoring.

### Current four-scope analysis

All seven detectors maximize 0.8 cross-fitted attacked TPR plus 0.2 cross-fitted
clean TPR. Attacked documents have equal weight within the selected fitting
support; attack categories are not reweighted. Five source-grouped folds are
balanced within domain. Each fold's thresholds are estimated from human tuning
texts in the other folds at 5% target FPR, separately by domain. Bounds and
directions use the entire clean tuning set: only threshold estimation is
cross-fitted. AUROC is a diagnostic, not the objective.

Rate-specific and attack-specific fits are the main analyses. They assume a
rate range or attack mechanism is supplied. Full-universal and eligible-universal
fits are secondary comparisons. All four use the same TPR objective and the
same candidate/constant-score/positive-LRR-cap safeguards. There is no hard
clean-loss constraint or extra FPR penalty. Each rate bin requires 250 attacked
tuning rows and has no fallback; no attack-by-rate grid is fitted.

Final domain-specific thresholds use independent calibration humans. The 2,000
source-cluster bootstrap replicates pair clipped with raw while holding the
fitted rules fixed. Neither specific fits nor cross-fitting guarantee improvement
on test data. See [current RAID analysis](TECHNICAL_REFERENCE.md) for all fit,
output, and interpretation contracts.

## Reading the evidence

Bootstrap intervals quantify test-source variation conditional on fitting and
calibration; they exclude those stages' uncertainty and multiplicity correction.
The simulation uses Wilson intervals instead. Neither successful validation nor
an improved curve proves uniform benefit at identical held-out FPR. The RAID analysis has the exploratory status stated in Appendix G of the paper.

Detailed definitions are in the [controlled-study workflow](TECHNICAL_REFERENCE.md),
[current RAID analysis](TECHNICAL_REFERENCE.md), and
[detector implementation](TECHNICAL_REFERENCE.md).

## Detector definitions

The paper reports seven detectors: log likelihood, rank, log rank, LRR, entropy,
entropy gap and Binoculars (`binocular_origin`). Higher oriented scores indicate
machine-generated text. LRR has direction +1 and Binoculars -1; the remaining
directions come from clean tuning means.

Additive scores are lower-clipped after orientation, then averaged. LRR averages
capped NLL and capped log-rank components before taking their ratio. Its
denominator cap must be positive. Binoculars caps numerator NLL components and
leaves its cross-entropy denominator unchanged. Exactly constant clean scores
and structural saturation are rejected. The quantile grid is
0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, with no clipping also available.

## Scoring windows

Controlled-study scores condition on the prompt and aggregate response-token
predictions. The Falcon pair uses the same prompt-conditioned continuation
policy. RAID instead uses output-only tokenizer-native input, right-truncated
to 512 input tokens. Its causal NLL and official observer-to-performer
cross-entropy have distinct windows, as implemented in `RAID/binocular_origin.py`.

For RAID, the clipped numerator is anchored to the saved official numerator:

$$
A_U = \min\left\{A_{\mathrm{off}},\,
\max\left(0,\, A_{\mathrm{off}}
+ \operatorname{mean}_{64}\!\left[\min(a_i,U)\right]
- \operatorname{mean}_{64}\!\left[a_i\right]\right)\right\}.
$$

Here \(a_i\) is token NLL, \(U\) is its cap, and \(A_{\mathrm{off}}\) is the saved official numerator. The means use float64 arithmetic. This retains the official numerator
when no token is capped, while applying the deterministic float64 clipping
change. The scoring gate checks the upstream numerator/denominator components
on identical logits before full scoring. Exact component parity does not imply
identical source selection or an independently reproduced experiment.

## Entry points

The controlled preparation/scoring stage saves the component arrays used by
`tools/reanalysis/reevaluate_detector_revision.py`, which produces the seven
paper detectors. Run that analysis after `run_experiment.py`; do not substitute
a pre-existing document score for the required component ratio.

The current RAID entry is `RAID/run_raid.py`, or `RAID/analyze.py` for existing
token packs. No change to detector definitions is made by table regeneration.

## Controlled-study workflow

This document describes the reference analysis and its interpretation. See [the repository guide](../README.md),
[detector definitions](TECHNICAL_REFERENCE.md) and
[RAID's separate protocol](TECHNICAL_REFERENCE.md).

## 1. Scope and evidence hierarchy

The released primary study crosses XSum, SQuAD and WritingPrompts with
`ibm-granite/granite-3.3-8b-base`,
`mistralai/Mistral-Small-24B-Base-2501` and `Qwen/Qwen2.5-32B`:
**nine primary cells**.
RAID is a separate realistic-attack benchmark.

The accompanying paper defines the reported experiment. Saved numerical results
and per-cell configurations provide its implementation evidence. Use
`configs/paper_cells/` for recorded settings rather than generic defaults. Run
`tools/reanalysis/reevaluate_detector_revision.py` after the base preparation and
scoring stage to produce the seven paper detectors.

## 2. Research question and interpretation

The study tests document-detector robustness to controlled token-splice
contamination, comparing raw and clipped aggregation on identical texts.
Contaminated machine continuations remain machine-positive; clean human
continuations are negative. Replacement material comes from human text, but
the resulting token splices are **not certified grammatical human edits**.
They can disrupt syntax, token boundaries and conditional context. Gains may
reflect robustness to these disruptions, not specifically human authorship.

A realized replacement budget is not the theoretical conditional mixture
probability in a contamination theorem: conditional distributions downstream of
an edit can change beyond directly replaced positions. Decreasing detection
with contamination and improvement under clipping are hypotheses, not enforced
properties. Null/adverse results remain results.

## 3. Sources and splits

| Dataset | Dataset ID / split | Source field |
|---|---|---|
| XSum | `EdinburghNLP/xsum`, validation | `document` |
| SQuAD | `rajpurkar/squad`, train | Deduplicated `context`; short unique contexts packed without reuse |
| WritingPrompts | `euclaise/writingprompts`, validation | First present of `story`, `text`, `completion` |

`select_source_manifest` in
[`pipeline.py`](../experiment_core/preparation/pipeline.py) normalizes,
deduplicates, applies a 260-word filter (packing where enabled), and deterministically
hash-selects sources. Its manifest signature is independent of target model.
Before a new paired cross-model analysis, verify exact source overlap from saved
source manifests rather than assuming it from current code alone.

Each cell uses 3,000 selected sources: 500 `clipping_tuning`, 500 `calibration`,
2,000 `test`, with selection seed 1731. Generation seeds 101/202/303 divide the
sources round-robin, not multiply their count. A packed SQuAD source can contain
multiple original contexts; 3,000 sources does not mean 3,000 original QA pairs.
All variants and random draws for a source are dependent observations.

## 4. Generation, scoring context and saved overrides

All nine saved revision configurations specify Transformers generation, nominal
30-token source prefix, target/max 220 new tokens, minimum 210, temperature 0.8,
top-p 0.95, batch 8, checkpoint 96, length bucket 32 and BF16. Actual continuation
length can differ from 220. Human and machine continuations are length-matched
before corruption. No chat template is introduced.

`prepare_scoring_tokens` in
[`scoring.py`](../experiment_core/detectors/scoring.py) separately encodes
prompt and response with no added special tokens and concatenates their IDs.
Only response predictions contribute; prompt tokens provide context. If combined
length exceeds 512 tokens, the response is right-truncated to the space left
after the prompt. The saved Falcon pair uses this same policy. 220 target-model
tokens need not be 220 Falcon tokens; generated length and scored length differ.

The saved manifests establish these differences from current defaults:

| Cells | Generation/scoring device map | Recorded generation TP | Pooled hidden state saved | Base round-trip guard | Explicit constructed guard |
|---|---|---:|---|---:|---:|
| Granite, all 3 | `auto` | 2 | true | 8 | absent |
| Mistral, all 3 | `auto` | 4 | false | 12 | absent |
| Qwen 32 XSum/SQuAD | `auto` | 4 | false | 12 | absent |
| Qwen 32 WritingPrompts | `auto` | 4 | false | 12 | 20 |

Recorded `tensor_parallel_size` is not evidence of actual tensor-parallel
execution under the Transformers/device-map backend. Scoring batch/microbatch
are 4, saved top-k 10, vocabulary chunk 8192. Hidden states are auxiliary outputs,
not inputs to the seven detectors.

Current preparation records decode/re-tokenize drift, can canonically normalize
and length-match base continuations outside their guard, and falls back to the
base guard when no separate constructed guard exists. Do not retroactively
describe all historical cells with current 12/20 defaults. Historical per-row
normalization and empirical length distributions require original prepared/base
rows, which are not included in the public distribution.

## 5. Contamination construction and counts

Ratios are 0/5/10/20/30/40/50%. For original length \(n\) and positive ratio
\(r\), token budget is \(\min(n,\max(1,\operatorname{round}(nr)))\), using
Python rounding. Replacement preserves the token-array length before decoding.
Recorded realized ratio uses donor count/final re-tokenized length, not edit
distance and not necessarily the fraction of distinct token IDs that changed.
See [`data.py`](../experiment_core/preparation/data.py) and `pipeline.py`.

- **Random:** sentence-like human spans are shuffled, truncated to fill the
  budget, and placed in non-overlapping random recipient token windows. Three
  draws per positive ratio/source; windows can cut recipient sentences.
- **Tail:** candidate human spans are scored once for mean target-model NLL
  conditional on the prompt, ordered highest first with candidate-ID tie breaking,
  cached across ratios, then used to replace a suffix. This is a white-box
  high-NLL donor/suffix attack, not optimization against all seven detectors
  or proof of a worst-case attack.

Spans use a regex punctuation/whitespace/newline splitter, not a linguistic
sentence parser. Corruption base seed 99173 is combined with dataset, target,
sample ID and draw ID; ratio is not part of that seed. Ratios/draws are not
independent experimental replications.

Each source has two clean rows (human/machine) plus six positive ratios times
four constructions (three random, one tail): 26 rows/source, 78,000 rows/cell.
Target and Falcon score packs must cover those same provenance keys. A test
random condition has 6,000 positive rows from 2,000 sources; a tail condition has
2,000. Clean machine TPR appears in both mode panels for display, without
duplicating the original clean sample.

## 6. Final detector definitions

Let \(\ell_i=\log p(x_i\mid\mathrm{prefix})\), \(a_i=-\ell_i\),
\(r_i=1+\#\{v:\mathrm{logit}(v)>\mathrm{logit}(x_i)\}\), and
\(h_i=-\sum_v p_i(v)\log p_i(v)\). Logs are natural, rank ties are competition
ties, and bars denote response-token arithmetic means.

| Final detector ID | Raw score | Final direction |
|---|---|---|
| `log_likelihood` | \(\bar\ell\) | Learned from clean tuning means |
| `rank` | \(\bar r\) | Learned |
| `log_rank` | \(\overline{\log r}\) | Learned |
| `lrr` | \(\bar a/(\overline{\log r}+10^{-12})\) | Fixed+1: larger = machine-like |
| `entropy` | \(\bar h\) | Learned |
| `entropy_gap` | \(\overline{a-h}\) | Learned |
| `binocular_origin` | Performer mean NLL / observer-to-performer mean cross-entropy | Fixed-1: smaller = machine-like |

For learned direction, choose+1 when the machine clean tuning mean is at least
the human mean, otherwise-1. It is not selected using test AUROC. LRR means
DetectLLM likelihood/log-rank ratio, **not a likelihood ratio between two
probability models**. Entropy gap is unnormalized NLL minus entropy; do not
label it variance-normalized Fast-DetectGPT. Literature names must refer to the
exact implemented statistic.

Falcon performer is `tiiuae/falcon-7b-instruct`, observer `tiiuae/falcon-7b`.
Observer probabilities weight performer log probabilities in the cross-entropy.
Primary origin components share the saved prompt-conditioned response window:
label it **conditional Binoculars-ratio adaptation**, not exact output-only
upstream reproduction. RAID's official-window/BF16-anchored path is separate.
The publication detector identifier is `binocular_origin`.

## 7. Clipping, candidate selection and degeneracy guards

Implementation: `_candidate_specs`, `tune_clipping_spec`, `orientation`,
`evaluate` in [`evaluation.py`](../experiment_core/analysis/evaluation.py),
and guards in [`detector_revision.py`](../experiment_core/detectors/detector_revision.py).

Generic oriented token evidence \(z_i=d s_i\) is replaced by
\(\max(z_i,L)\) and averaged. LRR caps both components:

$$
S_{\mathrm{LRR,clip}}=
\frac{\overline{\min(a_i,U_a)}}{\overline{\min(\log r_i,U_r)}+10^{-12}}.
$$

Primary origin caps only numerator NLL; its cross-entropy denominator is
unchanged. This is an experimental extension, not the published raw detector.

Candidate quantiles pool **clean human and clean machine tuning tokens**;
documents with more tokens contribute more to these quantiles. Grid:
0.80/0.85/0.90/0.95/0.975/0.99/0.995 plus no clipping. Generic bounds are lower
\(1-q\) quantiles of oriented evidence; origin bounds are upper NLL quantiles.
LRR uses the seven-by-seven Cartesian grid: 49 cap pairs plus raw before
rejection/duplicate values.

The attacked tuning mixture pools random and tail rows at 10/20/30/40/50%,
excluding 5%. With three random draws versus one tail draw it has **3: 1
random:tail row weighting**, not equal-weight attack-average AUROC. Objective:

$$
J=0.8\,\mathrm{AUROC}(H_{\rm tune},M_{\rm attacked,tune})
 +0.2\,\mathrm{AUROC}(H_{\rm tune},M_{\rm clean,tune}).
$$

The clean term is a soft reward, not a hard zero-clean-loss constraint. RAID
pilot selectors and clean-loss budgets must not be imported into this primary
description. No-clipping is considered first; improvement must exceed
\(10^{-15}\), leaving the earlier candidate on ties. One frozen specification
serves every mode, ratio and FPR for a dataset/model/detector. Optional
mode-specific oracle is disabled.

Revised evaluation rejects nonempty candidates with constant clean tuning
document scores or exact structural full saturation; it adds no near-constant
tolerance. A constant origin numerator alone is not sufficient for rejection
if its denominator retains information. LRR requires a finite strictly positive
log-rank cap: zero would erase rank evidence even when the score still varies.
Rejections appear in `clipping_rejections.json`. Qwen 32-SQuAD V4.2 specifically
corrects this denominator-collapse candidate using saved features, not new
inference. This fix must not be described merely as flipping a plotted curve.

## 8. Calibration, metrics and uncertainty

After fixing direction/specification, calibrate raw and clipped thresholds
separately using 500 clean human calibration documents at 1% and 5% target FPR.
For sorted scores and \(m=\lfloor\alpha n\rfloor\), use the next representable
number above zero-based order statistic \(n-m-1\); classify scores greater than
or equal to threshold. At most \(m\) calibration observations exceed it.
Held-out test FPR need not equal target FPR and must accompany TPR claims.

Report TPR, actual FPR, AUROC, normalized partial AUROC over 0–5% FPR, paired
clipped-minus-raw differences, and normalized TPR-versus-contamination robustness
AUC. Partial AUROC is trapezoidal ROC area divided by 0.05, **not** the
McClish/sklearn standardized partial-AUC convention. Robustness AUC is the
trapezoidal contamination-curve area divided by its ratio range (0.5).

Final intervals use 2,000 percentile bootstrap repetitions, 2.5th/97.5th
percentiles, base seed 481516 and deterministic condition-specific seeds.
`_metric_bootstrap` resamples `sample_id` clusters, taking all relevant
human/machine rows and random draws with each sampled source. Raw/clipped
differences share the resample. `_robustness_bootstrap` keeps each resampled
source across ratios. Bounds and calibration thresholds remain fixed: these
are test-source sampling intervals conditional on fitting/calibration, not
uncertainty of the complete tuning procedure. Intervals are pointwise, not
simultaneous or multiplicity-adjusted. Repeated clean/robustness values in tidy
rows do not create additional independent observations.

## 9. Reference artifacts

`reference/empirical/primary/` supplies the final numerical results used to regenerate the figures.
`reference/provenance/primary/` supplies resolved settings, versions and source
manifest filenames. `configs/paper_cells/` supplies nine runnable configurations.
Source identities for all three datasets and instructions for restoring the original
source manifests are provided; token-score packs are not included. See
[PROVENANCE.md](PROVENANCE.md). Keep the source groups paired within each cell.

## RAID analysis

## Scope and evidence

The current analysis fits seven detectors in all four scopes using cross-fitted
TPR. Rate-specific and attack-specific fits are the main analyses; universal
fits are secondary comparisons. The controlled-contamination study retains its
own AUROC objective, and the simulation is unchanged.

The distributed reference contains **119 fits, 462 result rows, and 2,000
bootstrap repetitions** in `reference/raid/`. Use `python reproduce.py raid-tables
--output outputs/raid-tables` to regenerate the corresponding manuscript tables.
The reference includes calibration, full candidate diagnostics, final thresholds,
source identities, measured edit rates and tuning folds. See [PROVENANCE.md](PROVENANCE.md).

## Fitting objective and folds

For every detector and fitting scope, maximize

$$
0.8\,\widehat{\mathrm{TPR}}_{\mathrm{attacked,CV}}
+0.2\,\widehat{\mathrm{TPR}}_{\mathrm{clean,CV}}.
$$

Attacked documents receive equal weight within the fitting support. Represented
attack types are not reweighted. Five source folds are balanced within domains,
with seed 20260917. All variants of one source stay together. For each candidate,
a held-out fold uses thresholds estimated from human tuning texts in the other
four folds, separately by domain at 5% target FPR. Hits and counts are pooled
across folds, rather than taking an unweighted average of fold TPRs.

Bounds and directions use all clean tuning texts. **Only threshold estimation
is cross-fitted**, not candidate generation or the entire model-selection
procedure. Calibration and test rows never select candidates. Final thresholds
are subsequently estimated from independent calibration humans.

The candidate quantile grid remains 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995
plus no clipping. LRR considers the component-cap grid and rejects nonpositive
log-rank caps. Exactly constant clean scores and structural saturation remain
rejection criteria. No-clipping is first; an objective improvement must exceed
\(10^{-12}\), so ties retain the earlier candidate. There is **no hard clean-loss
constraint, extra FPR penalty, or AUROC fitting objective**. AUROC and cross-fit
human FPR are recorded as diagnostics. Clean machine texts enter only the clean
objective term, not attacked support.

| Scope | Attacked tuning support | Role |
|---|---|---|
| Rate-specific | Each of four positive-rate bins up to 0.5 | Main; assumes a rate range is supplied |
| Attack-specific | One attack across all its measured rates | Main; assumes a mechanism is supplied |
| Full-universal | All eleven attacks, including measured-zero and dense rows | Secondary; one common bound |
| Eligible-universal | Attacked rows with \(0<\rho\leq0.5\) | Secondary; one bound for restricted support |

Rate bins are \((0,0.05]\), \((0.05,0.10]\), \((0.10,0.20]\), and
\((0.20,0.50]\). Every rate fit requires at least **250 attacked tuning rows**;
insufficient support is an error, with no fallback. There is no attack-by-rate
grid. 

## Shared population, scoring, and calibration

The completed reference uses 12,871 sources, 167,323 score rows, and
5,148/2,574/5,149 tuning/calibration/test sources. The original 500 development
sources were excluded. The selected-source identities are distributed. Exact sample reconstruction
also uses the original exclusion file described in [PROVENANCE.md](PROVENANCE.md).

The seven methods are log likelihood, rank, log rank, LRR, entropy, entropy gap,
and `binocular_origin`. LRR direction is fixed +1 and origin -1; other directions
come from clean tuning means. Falcon single-model features and official
output-only Falcon observer/performer components retain their scoring semantics.
Origin caps only numerator losses, anchoring the float64 clipping change to the
saved official numerator; the denominator is unchanged. This is separate from
the primary study's prompt-conditioned ratio. 

Realized rate is minimum token edit distance divided by the maximum clean/attacked
scored-target-window length. It uses the shared Falcon target window, not a
separately recomputed official-denominator rate. Native attack-rate defaults are
not measured token rates. See [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) for component and window definitions.

Each fitted rule is finally calibrated on independent human calibration texts,
separately by domain at 5% target FPR. Actual held-out FPR need not equal 5% and
must accompany TPR. The domain-stratified source bootstrap uses 2,000 paired
replicates, holding directions, selected bounds, and thresholds fixed. It does
not include fitting/calibration uncertainty or multiplicity correction.

## Run from score packs

Use reader-provided storage and CPU resources; this analysis performs no model
inference. The public entry accepts explicit inputs and writes a fresh output:

```sh
python -B RAID/analyze.py --falcon-scores FALCON_SCORES_JSONL --origin-scores ORIGIN_SCORES_JSONL --run-id raid-tpr-01 --output outputs/raid-tpr-01 --bootstrap-repetitions 2000 --bootstrap-seed 481516
```

Repeat `--origin-scores` for additional shards. Optional
`--contamination-records` supplies an aligned rate CSV; otherwise rates are
measured from scored and full token IDs. Token components are needed for new
bound selection; summary results cannot replace them. Follow
[EXPERIMENTS.md](EXPERIMENTS.md) for preparation and scoring. Choose a fresh
output directory and never write into the reference directories.

## Output interpretation

- `selected_specs.json` contains 77 attack, 28 rate, seven full, and seven eligible
  fits, including candidate diagnostics and final thresholds: 119 fits total.
- `results.csv` contains 210 attacked/clean rows for specific fits and 252 rows
  for universal fits: 462 total. Each universal fit is tested on 18 populations:
  clean, all attacks, eligible attacks, eleven individual attacks, and four bins.
- Universal `group` names fitting support (`all_attacks` or `eligible_attacks`);
  `condition` names test support (`clean`, `all_attacks`, `eligible_attacks`,
  `attack:<name>`, or `rate:<bin>`).
- `calibration.csv` records calibration thresholds, calibration FPR, and test FPR
  by fit and domain.
- The manifest and completion marker record protocol, folds, input/code
  fingerprints, and output integrity. Read them alongside fit and row counts.

Intervals pair **clipped minus raw**, not one fitting scope minus another.
Repeated human FPRs across conditions reuse the same observations. These outputs
do not establish the FPR of a single automatically routed system. All-attack
pooled TPR equals an equal-attack mean when each source has one row per attack;
eligible pooling generally does not. Report clean TPR and achieved FPR,
including adverse outcomes. Specific fits are not guaranteed upper bounds on
universal performance. The analysis is exploratory, as stated in Appendix G.