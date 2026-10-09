---
title: Serving observations
description: Bounded engine diagnostics and measurement boundaries.
---

`Usage.engine_observation` is optional and SDK-accessible as `EngineObservation`. Ordinary single-choice causal generation supports it. Originally multiple-choice groups remain unsupported after a choice fails. Noncausal and agentic aggregate usage omit it. Entering an actual speculative driver or verifier permanently suppresses this observation and clears any prior group snapshot; the sampler probability-mode flag does not identify speculative execution. Legacy usage deserializes omitted observations as unknown.

Set `MISTRALRS_INFERENCE_OBSERVATIONS=0` before startup to disable new observation clocks, counters, metadata construction and publication. The setting defaults to enabled and is cached once. Existing timers and behavior remain unchanged.

Token markers use integer nanoseconds from sequence construction. Their boundary is the accepted sampler result before token decoding, rather than GPU completion or streamed text. The count includes EOS, stop and nonvisible output tokens, excluding grammar redraw candidates and speculative drafts. Bind sequence id and response index to the worker lifecycle. A UTC nanosecond anchor is sampled between monotonic origin creation and a second monotonic reading, before observation metadata allocations. The bracket duration is retained as conservative alignment uncertainty; this does not make sequence and worker origins equal. Out-of-range system clocks explicitly omit the anchor. Finish reason `length` indicates truncation. Failed, canceled, speculative, aggregate or truncated turns must not receive exact decode rates. A complete ordinary causal turn with N > 1 and a positive marker interval permits `(N - 1) / (last - first)`.

Prefix accounting separates the KV candidate, restored recurrent boundary and final reused prefix after multimodal prompt preparation. The suffix uses that final boundary. A missing snapshot is never inferred to be an eviction. Snapshot bytes, entries and capacity describe retained host cache state, not request-owned or peak device memory. Restore and lookup timings measure host elapsed calls without adding synchronization. Raw prompt and completion seconds retain floating point precision; legacy durations retain millisecond accounting. The stop reason reflects the engine stopping condition before response tool-call normalization; a length stop remains truncated even if buffered output parses as a tool call.

Prefill count and maximum size describe planned sequence chunks. `prefill_host_forward_sec` sums host forward-call elapsed intervals, potentially shared across sequences. Asynchronous CUDA work may remain incomplete at that boundary.

All `worker_*` graph/kernel counters have `worker_process_cumulative` scope. They count host dispatch attempts including graph warmup/capture. They do not count GPU launches during graph replay. Actual branches observe FP8 native/dequantization, unquantized GEMV/matmul and fused single-token CUDA GDN recurrence. `fp8Native` counts every native blockwise-FP8 provider dispatch: the W8A8 tensor-core GEMV (BF16 activations of at most 32 rows), the cuTile W8A16 and FP8 GEMM providers, the DeepGEMM and CUTLASS SM90 providers, and the legacy blockwise kernel. `fp8Dequant` counts the dense BF16 GEMM over the dequantized weight, which is the route for larger prefill row counts when no cuTile GEMM is registered, and the non-CUDA fallback. A dequantized FP8 dispatch also counts once under the unquantized GEMV/matmul route it lands on. A thread-local scope around each synchronous engine forward call labels actual prefill/decode dispatch phases; unscoped calls remain unknown. Nested scopes restore the previous phase on return or panic. The scope never crosses an await. Cumulative phase counters still cannot attribute a request or elapsed kernel cost. Request-local attribution remains explicitly unavailable and requires correlated profiling. Graph configuration, eligible shapes, capture attempts/successes/failures, replays and latest eager reason are separate; they follow the target-model decode-graph dispatch decisions and capture/replay outcomes behind `mistralrs_cuda_graph_dispatch_total` and `mistralrs_cuda_graph_events_total`, and exclude draft-model graphs. Relaxed reads do not form an atomic cross-counter snapshot.

Paged causal execution exposes activation/resolved KV dtype, block size and selected build features. Load provenance, full attention route details, GPU architecture, weight/block layout, non-recurrent cache writes/evictions and memory peaks remain unavailable. These hooks alone do not satisfy real-run backend or overhead acceptance.

Focused portable checks are `make observation-check` and `make observation-test`. Format only the changed Rust files with `make observation-format OBSERVATION_FILES='<files>'`. These checks require no model download, GPU execution or installation.

Request-local recurrent snapshot fields are `snapshot_write_count`,
`snapshot_write_bytes`, `snapshot_evicted_entries`, `snapshot_evicted_bytes`,
`snapshot_host_copy_ns` and `snapshot_write_measurement_complete`. They follow
actual chunk/final-boundary writes by the same supported original single-choice
Sequence. A successful host copy and store counts one write, including a set
immediately removed because it exceeds capacity. Replacing an existing key is a
write and does not count as an eviction. Only capacity-driven removal contributes
eviction entries/bytes. Bytes use the sum of the underlying conv/recurrent tensor
payloads, the same convention as the authoritative retained snapshot store;
they do not represent allocator usage or transfer bandwidth. Counters saturate.

Fields stay unknown until a relevant supported gather/write scope is tracked;
lookup alone never establishes zero writes. Within that scope zero is observed,
including a disabled or zero-capacity insertion. A gather or host-copy failure
makes measurement completeness false permanently for the Sequence while retaining
known counts and attempted-copy host time. Successful later chunks cannot erase
that failure. Host copy nanoseconds bracket attempted `to_device(Cpu)` calls,
including failed attempts, with no additional GPU synchronization. Already-host
snapshots still count stored writes and their host call time.

Recurrent miss reasons follow actual branches: `no_matching_kv_prefix`,
`unaligned_candidate`, `no_recurrent_slot`, `not_retained`, `restore_failed` and
`disabled`. They are recorded when the staged prefix validation commits after KV
admission: a zero-token KV candidate is `no_matching_kv_prefix`; a candidate the
replay policy or block alignment rejects is `unaligned_candidate`; a sequence
without a recurrent slot is `no_recurrent_slot`; an absent checkpoint is
`not_retained`. `not_retained` says only that the matching recurrent snapshot is
absent now; it never claims it was evicted rather than never inserted. The CPU
A/warm-A/B/C/A fixture gives the store room for two snapshot sets, demonstrates
A's KV candidate surviving after A's snapshot removal, and observes final reuse
zero with `not_retained`. Aggregate snapshot state is distinct from these
request-local events. The SDK forwards the existing EngineConfig host-capacity
value through both primary builder paths; the default remains 64 GiB.

After construction, SDK `Model::prefix_cache_host_capacity_bytes()` returns the
accepted capacity of the default engine; the `_with_model(Option<&str>)` variant
uses the existing model/default/alias resolution rules. It reads the accepted
engine reboot configuration and applies the exact same usize normalization as
snapshot manager construction. It works with observations disabled and does not
measure occupancy, retention effectiveness, allocated host memory or device
headroom. Missing model IDs return the existing inference error style.
