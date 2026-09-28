# Software reuse and external resources

The software and accompanying software documentation are available under the
[MIT license](LICENSE). Preserve its notice when redistributing copies or
substantial portions of the software.

This license does not relicense external datasets, model weights, dependencies,
upstream source files or manuscript materials. Obtain those resources under
their own terms and retain their attribution. Reference summaries and decision
arrays contain research outputs; their inclusion does not grant rights in
underlying source text. No model weights or raw dataset copies are bundled.

## Dataset and model attribution

The controlled study uses [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum),
[SQuAD](https://huggingface.co/datasets/rajpurkar/squad), and
[WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts).
The attack benchmark is [RAID](https://raid-bench.xyz/), described in the
[RAID paper](https://aclanthology.org/2024.acl-long.674/). Follow the source
projects' citation and access instructions when obtaining or using their data.

The target models are
[Granite 3.3 8B Base](https://huggingface.co/ibm-granite/granite-3.3-8b-base),
[Mistral Small 24B Base](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501),
and [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B).
The detector models are [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) and
[Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct).
Model cards provide the respective model terms and attribution. A repository
URL names a resource; record a resolved revision for a new experiment.

## Binoculars reference

The origin ratio follows the component conventions of the
[Binoculars project](https://github.com/ahans30/Binoculars). The separate upstream
`binoculars/metrics.py` used by the numerical parity gate is an external input;
it is not included here. Obtain a reviewed, pinned version with its license and
notices. The repository implements its own scoring and clipping workflow;
referencing upstream formulas does not transfer this repository's license to
the upstream implementation. Primary prompt-conditioned scoring and RAID
output-only scoring are distinct adaptations documented in [TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md).

## Distribution scope

The maintained reference directory contains explicitly derived copies with
identifying execution metadata removed. Its inventories authenticate those
distributed bytes, not omitted private execution records. Numerical results are
preserved. Local manuscript and historical download folders are outside this
maintained distribution and are not needed to run its reproduction commands.
No publication, acceptance, repository DOI, or compliance with a particular
venue's submission rules is implied. Use the accompanying paper for study
attribution when available.
