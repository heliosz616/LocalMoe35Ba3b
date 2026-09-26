# Source extraction and publication checks

The release was assembled with a new Git history, not by publishing research
history. Only runtime/conversion source, tests, pinned dependencies, documentation
and sanitized aggregate results were selected. Model weights, configuration and
tokenizer assets, calibration/test datasets, prompts from datasets, generated
experiment transcripts, raw traces, binaries, local paths and credentials were
excluded. Git commits use a generic maintainer identity.

The source extraction was checked structurally against the locally tested code:
all nine Triton kernel function ASTs match, as do the converter quantizer and
generation function. Package imports, loading/source relocation, CLI setup and
release validation were reorganized. Unused experimental branches were removed;
this check is not a proof that arbitrary surrounding changes preserve behavior.

Nine CPU tests pass: frequency-selection and installation guards, release source
and dependency integrity, portable model fingerprinting, and quantizer arithmetic.
The curated file list and `tools/audit_release.py` scan check content, secrets,
personal-path/email patterns, file types, symlinks and file sizes. This heuristic
scan supplements manual allowlist review; it cannot prove absence of every
possible form of sensitive information in arbitrary future contributions.

`release_smoke.json` records the actual new entry point against a local checkpoint,
for both W8A16 and W8A8. It stores counts and timings only. Checks include invalid
input/output limits, subsequent valid-request recovery, nonempty generation and
simple arithmetic. It does not rerun the private held-out evaluation, establish
all-length throughput, or justify hiding cold/short-request failures.

Future contributors should run the scan and inspect `git diff --cached` before
pushing. Never add generated manifests, prompts, model shards or raw traces merely
because a filename is not yet covered by `.gitignore`.
