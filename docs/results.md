# Sanitized results and limitations

Historical end-to-end observations on RTX3060 12 GB, i7-13700F, approximately
62 GB RAM. These are measured results, not promises for other systems or inputs.
Data is aggregated to exclude calibration contents, sample IDs, prompts,
generated transcripts, personal paths and machine/account identifiers.

| Workload | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| 24 measured warm 4K requests, aggregate | 1189.87 | 23.63 |
| Minimum among those requests | 1181.42 | 21.94 |
| Maximum among those requests | 1192.20 | 26.37 |
| Separate difficult code workload, mean | 1189.66 | 20.13 |
| Fresh-process CLI, 3540-token prompt | 952.49 | 23.33 |
| 26-token short CLI request | 15.25 | 21.68 |

The short decode measurement has only two cached steps and is insufficient for
a robust sustained decode claim. Fresh process is not a cold boot: existing
compiler and filesystem caches were retained. Initial model loading and graph
preparation are not included in request prefill/decode rates; the CLI reports
startup separately. Prefill creates the first generated token; decode counts only
subsequent cached steps. Inputs, EOS and output lengths affect timing.

The 24-request matrix covers eight held-out 4K prompts, three measured repeats
each, following one warmup pair per prompt; each measured request has 32 cached
steps. Detailed tokenization/domain-selection records and source examples are not
published. This repository does not claim independent reproducibility of private
dataset metrics from the provided aggregate alone. The public neutral-prompt
smoke harness is independently runnable but is a different workload.

## Quality versus original BF16

On 1024 held-out token positions across eight 4096-token prefixes:

- Top-1 agreement: 98.046875%; top-5 overlap: 95.566407%.
- Mean KL divergence: 0.0052861.
- NLL: 1.060325 versus 1.054530; difference +0.0057944.
- Perplexity: 2.887309 versus 2.870627.
- No high-margin flips among 771 high-margin positions.

Frozen source/incremental quality gates passed in local experiments. W8A8 is
not equivalent to W8A16: eight of 32 free-generation pairs in the held-out
benchmark differed. No broad BF16 free-generation equivalence is claimed.
A separate development comparison approached its per-window NLL limit;
held-out success does not establish uniform behavior on all domains.

## Release validation versus research evidence

Kernel arithmetic is retained; package imports and deployment entry points are
reorganized, unused experiments removed, and explicit source relocation added.
The public entry point is tested separately with a neutral prompt smoke run.
That check must not be presented as a rerun of the private full quality suite.

The packaged public CLI was also benchmarked on one reproducible synthetic
4084-token prompt, with one warmup followed by three measured requests. Warm
aggregate throughput was **1095.82 prefill / 22.63 decode tokens/s**; all three
measured requests met 1000/20. The first request was only **856.22 / 22.49** and
is retained in `release_benchmark.json`. This different workload does not replace
the historical eight-domain measurements or establish a cold-start guarantee.

No guarantee is made for short/cold requests, >4K prompts, >33 generated tokens,
two streams, RTX5060, 128K/192K/256K contexts, vision, MTP, or production service
security. The historical 1000-prefill/20-decode target is not universally met.
