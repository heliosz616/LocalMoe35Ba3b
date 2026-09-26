# Runtime and numerical contract

This release targets 40 text blocks, 30 GDN and 10 full-attention blocks, hidden
size 2048, 256 routed experts per layer, top-8 routing, and expert intermediate
size 512. The Transformers architecture name is Qwen3_5Moe; the tested checkpoint
is Qwen3.6-35B-A3B. Other geometries are unsupported.

The expert bank is INT8 g128 in CPU pinned RAM. A fixed 7 GiB CUDA slab holds
the two prefill banks, retained experts and subsequent decode LRU. Non-expert
weights remain on GPU. Original-precision token embeddings stay on CPU.
Whole expert layers are prefetched two layers ahead. CUDA release/ready events
protect both bank reuse and consumers. After the final prefill chunk, selected
experts are copied to slots outside the two banks; completed banks become free
decode slots. Misses fetch the exact routed experts. There is no speculative
expert omission, cross-request KV reuse, or prediction-based substitution.

Frequency selection uses only the final prefill chunk. Last-token experts are
always retained; remaining slots favor frequent experts with deterministic ID
tie-breaking. The system must execute on one CUDA-dispatch thread. Shared GDN
graph memory pools are not safe for concurrent calls.

Non-expert and prefill computation use W8A16 arithmetic, including the original
BF16 rounding/reduction order. The optional/default W8A8 decode expert path
quantizes each input row in 128-element groups, performs INT32 tensor-core dot
products, and accumulates scaled outputs in FP32. This changes numerical results
and may change routes and generated text. It is not lossless conversion.

GDN decode graphs capture original Transformers operations with private static
state. Startup prepares these graphs using zeros, without processing a prompt
or priming the expert cache. Prefill uses the original forward. BF16 reduced-
precision matrix accumulation is disabled as in the validated experiments.

Approximate weight-side accounting: 36.26 GB artifact payload, 33.22 GB complete
expert-bank prefill traffic per chunk, 2.03 GB resident non-expert weights, and
1.02 GB CPU embeddings. The 7 GiB arena is additional GPU allocation; KV,
activations and runtime structures require remaining headroom. Do not interpret
7 GiB as total VRAM consumption or 36.26 GB as peak process RAM.
