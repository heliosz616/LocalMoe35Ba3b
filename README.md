# Qwen 35B-A3B INT8 deployment

Consumer-GPU text inference for **Qwen3.6-35B-A3B**, whose Transformers
implementation uses the `Qwen3_5Moe` architecture. This is a small,
FreeToken-inspired runtime, not an official Qwen or FreeToken distribution.

**No model, tokenizer, dataset, user prompts, or private experiment transcripts
are included.** Bring your own appropriately licensed local checkpoint.

## Tested configuration and scope

- RTX 3060 12 GB, i7-13700F, approximately 62 GB system RAM, Linux x86-64.
- Python 3.12, PyTorch 2.12.0+cu132, Triton 3.7.0, Transformers 5.15.0.
- INT8 symmetric group-128 weights with FP32 scales; original-precision CPU
  embeddings and small sensitive parameters. This is absmax round-to-nearest,
  not a claim that the quantizer is globally optimal.
- CPU-pinned expert bank, 7 GiB GPU expert arena, GPU-resident non-expert weights,
  double-buffered prefill, GPU routing metadata, prepared GDN decode graphs.
- 2048-token prefill chunks; causal frequency retention of 45 experts per layer.
- Default decode-only W8A8 experts; `--activation w8a16` retains BF16 activations.
- One serial stream. **4096 prompt tokens and at most 33 output tokens** in this
  validated release. Not a general long-context or concurrent serving system.

On 24 measured warm 4K requests, W8A8 averaged approximately **1190 prefill /
23.63 decode tokens/s**; the minima were **1181 / 21.94**. A separate difficult
code workload reached approximately **1190 / 20.13**. A fresh-process 3540-token
CLI request reached only **952 / 23.33**. Short requests are much slower.
**There is no universal 1000/20 throughput guarantee.** See [results](docs/results.md).

## Install

Use a separate Python 3.12 environment and a CUDA-capable NVIDIA driver. The
original driver was 595.84; other environments are not validated here. CUDA 13
wheels are large; inspect `requirements.lock.txt` and available disk space first.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Run commands from the repository root. Triton compiles kernels on first use;
there is no separate native build. Do not run multiple model processes on this
12 GB GPU. Approximately 64 GB RAM is recommended; pinned expert weights alone
occupy about 33.22 GB. The artifact payload is about 36.26 GB (decimal units).

## Convert your local checkpoint

```bash
.venv/bin/python -m qwen35_deploy.convert \
  --source /path/to/original-checkpoint \
  --output /path/to/int8-model
```

The converter audits shard hashes, writes per-tensor safetensors, and resumes
validated completed tensors after interruption. It does not download anything.
Keep the original checkpoint until conversion and validation finish. Conversion
requires additional disk space, CPU RAM, and CUDA. Vision and MTP are excluded.
Generated manifests contain your local source path: keep them private and outside
this repository. They are **not** files to publish with this source release.

## Run

```bash
.venv/bin/python -m qwen35_deploy \
  --model /path/to/int8-model \
  --source /path/to/original-config-and-tokenizer \
  --prompt "Explain why the sky appears blue." --max-new-tokens 32
```

`--source` is explicit so a converted model can be relocated without editing its
manifest. Inference needs original configuration/tokenizer files, not original
weight shards. Source configuration and every converted tensor are hash-checked.
Startup reports graph-preparation time separately; it does not hide it in model
throughput. The first request may still pay additional initialization costs.

For a persistent process, replace `--prompt ...` with `--jsonl` and send:

```json
{"prompt":"What is 7 plus 5? Reply with only the number.","max_new_tokens":8}
```

Each response is one JSON object on stdout; readiness is JSON on stderr. This is
not HTTP, OpenAI-compatible, token-streaming, or a multi-user server. Prompts and
generated output are not saved by the runtime, but are visible on its input/output;
protect terminal logs and any redirects yourself. No telemetry or network calls
are required for inference.

`release.json` binds shipped module hashes, critical dependency versions, and a
path-independent tested payload fingerprint. A different payload requires
`--allow-unvalidated-model`; this explicitly invalidates the published quality and
performance scope. A matching fingerprint is an integrity check, not a signature
or a general quality guarantee. Code changes require revalidation, not merely
regenerating hashes.

## Tests and organization

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python tools/audit_release.py
.venv/bin/python tools/smoke.py --model /path/to/int8-model \
  --source /path/to/original-config-and-tokenizer --output outputs/smoke.json
.venv/bin/python tools/benchmark.py --model /path/to/int8-model \
  --source /path/to/original-config-and-tokenizer --output outputs/benchmark.json
```

- `qwen35_deploy/operators/`: W8A16/W8A8 Triton operations and GDN graph wrapper.
- `qwen35_deploy/runtime/`: pinned-bank pipeline, cache, routing and preparation.
- `qwen35_deploy/model.py`: local text-model loading and tensor verification.
- `qwen35_deploy/convert.py`: resumable INT8 conversion.
- `qwen35_deploy/generation.py`, `cli.py`: serial generation and deployment CLI.
- `docs/`: architecture, limitations, sanitized aggregate results.
- `tools/`: local smoke testing and publication hygiene checks.

Failed medium-tail/cross-chunk/gather experiments are intentionally not deployed
or included. See [NOTICE](NOTICE.md) for provenance and licensing scope.
