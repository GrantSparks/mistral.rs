---
title: Isolated BF16 reference
---

This reference runs the pinned original Qwen/Qwen3.8-27B BF16 base through the
multimodal Qwen3.5 loader with text-only requests. It measures full API usage and
HTTP wall time separately. It does not derive decode throughput from either.
The fixed CLI arguments request 64 layers split `0:32;1:32`, BF16, one sequence,
and loopback serving. NCCL is disabled at runtime. Effective loader placement is
reported unavailable until structured runtime evidence proves it.

`make build-bf16-reference-cli` builds the release CLI with only `cuda,nccl`, no
default features, using CUDA 13.3 by default and the existing Cargo configuration
and cache. It does not invent a compute capability flag. The target captures
source/build environment identity before compilation and rejects source changes
before publishing the binary SHA256 receipt. Coordinate the existing Cargo slot with the owner before running it.

`make test-bf16-reference-launcher` runs portable subprocess, loopback HTTP,
lease, deadline, identity, and payload-redaction fixtures without GPU queries.
`make run-bf16-reference BF16_REFERENCE_PLAN=/absolute/plan.json` is a hardware
operation and must only run after the concrete plan is reviewed and both devices
are available. The launcher never stops or restarts production.

The plan is JSON with exactly these fields (placeholder hashes must be replaced
with identities of the reviewed candidate):

```json
{
  "schemaVersion": 1,
  "binary": {"path": "/absolute/target/release/mistralrs", "sha256": "<64 lowercase hex>"},
  "buildIdentityPath": "/absolute/target/bf16-cli-build-identity.json",
  "source": {"root": "/absolute/fork", "baseRevision": "<40 lowercase hex>", "sourcePatchSha256": "<64 lowercase hex>"},
  "model": {"root": "/absolute/pinned-bf16-base", "manifestSha256": "<64 lowercase hex>", "repository": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
  "request": {"path": "/absolute/request.json", "sha256": "<64 lowercase hex>"},
  "gpuUuids": ["GPU-735d11ff-3eb7-f0b3-0e5b-4cabe70cdc6a", "GPU-2e0136cd-44e3-52c1-fd62-81a0b225286a"],
  "port": 19081,
  "outputDirectory": "/tmp/fresh-bf16-reference",
  "readinessSeconds": 600,
  "requestSeconds": 300,
  "totalSeconds": 900,
  "outputTokens": [24]
}
```

The source patch SHA256 is deterministic: concatenate `git diff --binary
--no-ext-diff --no-textconv`, then for each bytewise-sorted NUL-delimited path from
`git ls-files --others --exclude-standard -z`, concatenate `git diff --no-index
--binary --no-ext-diff --no-textconv -- /dev/null <path>`. Hash those bytes.
`source_identity` implements this recipe and includes Cargo.lock/config hashes.
Generated output belongs in ignored `target/` or outside the checkout so it does
not mutate the candidate identity. The manifest and model metadata are hashed;
large weight files are checked by pinned manifest identity and sizes, explicitly
reported as not rehashed. The revision authority is the owner's pinned input.

The request uses the worker ServingRequest JSON shape. Text parts are joined
with newline without altering their content or roles. Thinking and JSON mode
are preserved; output length may be overridden by the one or two reviewed
lengths. The report records original/effective/message digests without payload
text. JSON parsing success is independent of schema validation; no quality
schema is invented for an ordinary chat response.

The launcher takes both UUID file leases in the shared qualification namespace,
then refuses active or unknown compute processes. It creates a fresh directory
under `/tmp` or source `target`, rejects traversal and ancestor symlinks, and
retains an open directory descriptor plus device/inode for exclusive report
publication. Requests have absolute deadlines even if response bytes trickle.
Readiness is bounded, total time is at most 900 seconds, and eight seconds are
reserved for process-group TERM/KILL, reaping, and log draining before lease
release. Diagnostics retain bounded warning categories, source locations, line
hashes and byte counts; raw prompt/output/log text is discarded.

Reports include UTC/monotonic anchors with bracket uncertainty, child process
PID/group/binary identity, request start/end boundaries (including failed
requests), and bounded GPU samples before load and after requests. GPU samples
record driver, total/used/free MiB and power when available; unavailable queries
are explicit. These boundary samples do not establish peak allocation or
minimum free headroom during prefill. Build toolkit environment is retained;
CUDA runtime version remains unavailable without a runtime hook.

Readiness requires the real server model list: the `default` alias plus the exact
`--model-id` path with status `loaded` (MultimodalPipeline returns the original
model ID). Before sending any HTTP payload, the launcher connects and checks
Linux `/proc` listener socket inodes against the spawned process group. Foreign
model IDs, foreign listeners, and unavailable ownership evidence are refused.

For a reviewed launcher-only diagnostic correction, the optional absolute plan
field `compiledSourcePatchPath` may reference the archived patch whose SHA256
exactly matches the existing binary build receipt. The launcher compares both
patches after excluding only the launcher script, its portable tests, and this
BF16 reference document. Every other patch byte, including Makefile, and the
base/root/Cargo lock/config identities must remain equal. Reports preserve
`compiledSource` separately from current `source` and mark
`launcherOnlyDeltaVerified`; the original binary receipt is never rewritten.

HTTP failure reports retain the response hash/size and extract fixed allowlisted
CUDA/cuBLAS symbols and technical messages only from the top-level `message`.
The server's `partial_response` is never inspected because it may hold generated
text. Log diagnostics use the same allowlist. Unrecognized free text is discarded
and technical detail is explicitly unavailable. These diagnostics can identify a
CUDA status without exposing arbitrary request/output text; they do not establish
which kernel or layer caused it.

Error-only Qwen layer/stage and dense GEMM contexts may supply fixed
`BF16_OP_CONTEXT` records. Reports retain at most eight records from the same
bounded top-level error message or diagnostic log line. The parser accepts only
the exact known fields, stage/backend/device/dtype enums, layer indices through
4095, device ordinals through 255, and matching shape/stride vectors of rank one
through four with integer values through 1,000,000,000. Unknown fields or invalid
records are discarded. Tensor values and the original error's free text or
backtrace are never retained. These contexts preserve the original error chain
and identify the operation that returned an error; an earlier asynchronous CUDA
failure may still be its cause. They add no synchronization or dispatch changes.
GEMM records include actual input and weight device locations and ordinals;
the cuBLAS handle's current context remains unavailable.
