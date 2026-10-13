---
title: Serving observations
description: Bounded engine diagnostics and measurement boundaries.
---

`Usage.engine_observation` is optional and SDK-accessible as `EngineObservation`. Ordinary single-choice causal generation supports it. Originally multiple-choice groups remain unsupported after a choice fails. Noncausal and agentic aggregate usage omit it. A speculative driver or verifier can retain it with an optional `SpeculativeObservation` block; the sampler probability-mode flag does not identify speculative execution. The block separates inspected acceptance decisions from actual output appends and marks available canceled/error snapshots as partial before publication. Genuine unsupported paths still permanently clear prior group observations. Legacy usage deserializes omitted observations as unknown, and ordinary observation JSON omits the speculative block. See [speculative request observations](/guides/perf/speculative-decoding/#request-observations) for counters and validity flags.

Set `MISTRALRS_INFERENCE_OBSERVATIONS=0` before startup to disable new observation clocks, counters, metadata construction and publication. The setting defaults to enabled and is cached once. Existing timers and behavior remain unchanged.

Token markers use integer nanoseconds from sequence construction. Ordinary markers use the accepted sampler result before token decoding, rather than GPU completion or streamed text. The count includes EOS, stop and nonvisible output tokens, excluding grammar redraw candidates and uncommitted speculative drafts. Speculative markers use actual post-append timestamps and require `speculative.token_markers_valid`; mixed earlier sampler-boundary markers remain unknown while committed counts stay known. Bind sequence id and response index to the worker lifecycle. A UTC nanosecond anchor is sampled between monotonic origin creation and a second monotonic reading, before observation metadata allocations. The bracket duration is retained as conservative alignment uncertainty; this does not make sequence and worker origins equal. Out-of-range system clocks explicitly omit the anchor. Finish reason `length` indicates truncation. Failed, canceled, aggregate or truncated turns must not receive exact decode rates. Speculative legacy phase totals require their validity flags and currently do not provide exact decode rates. A complete ordinary causal turn with N > 1 and a positive marker interval permits `(N - 1) / (last - first)`.

Prefix accounting separates the KV candidate, restored recurrent boundary and final reused prefix after multimodal prompt preparation. The suffix uses that final boundary. A missing snapshot is never inferred to be an eviction. Snapshot bytes, entries and capacity describe retained host cache state, not request-owned or peak device memory. Restore and lookup timings measure host elapsed calls without adding synchronization. Raw prompt and completion seconds retain floating point precision; legacy durations retain millisecond accounting. The stop reason reflects the engine stopping condition before response tool-call normalization; a length stop remains truncated even if buffered output parses as a tool call.

For speculative output, the new block's `terminal_reason` follows the final response refinement, including `tool_calls`, while the legacy engine finish reason retains that original stopping condition. A model-forward error marks the block incomplete with reason `error` before constructing partial ModelError or CompletionModelError Usage; the existing sequence error transition still follows response publication. `measurement_complete` describes observed counters at the output boundary, not subsequent cache/state commit or successful delivery to a disconnected client. Missing validity flags are false; only existing direct cache facts and successful exact prefill geometry retain authority. Shared host dispatch durations remain unknown and must not be mapped to zero or exact request timing.

Prefill count and maximum size aggregate successful prompt host-forward dispatches for each Sequence across scheduler steps. Count is the number of dispatches that process that Sequence; maximum size is the largest retained query range (`end - start`) attributed through the processed sequence indices. Decode and failed forwards do not contribute. Fields remain unknown before an observed dispatch, and ambiguous, unavailable or invalid geometry (including count overflow) permanently clears both fields for that Sequence. Suffix-only prefill token views and deferred multimodal prompt preparation lack proven retained query ranges; a successful prompt forward in a batch containing either permanently clears geometry for its participating Sequences. Default-backend prompt forwards also lack retained query ranges and clear geometry. Prefix lookup and final-prefix clamps do not alter dispatch geometry. These fields do not measure padded batch tensor size, physical workspace, peak memory or GPU completion. `prefill_host_forward_sec` sums host forward-call elapsed intervals, potentially shared across sequences. Asynchronous CUDA work may remain incomplete at that boundary.

All `worker_*` graph/kernel counters have `worker_process_cumulative` scope. They count host dispatch attempts including graph warmup/capture. They do not count GPU launches during graph replay. Actual branches observe FP8 native/dequantization, unquantized GEMV/matmul and fused single-token CUDA GDN recurrence. `fp8Native` counts every native blockwise-FP8 provider dispatch: the W8A8 tensor-core GEMV (BF16 activations of at most 32 rows), the cuTile W8A16 and FP8 GEMM providers, the DeepGEMM and CUTLASS SM90 providers, and the legacy blockwise kernel. `fp8Dequant` counts the dense BF16 GEMM over the dequantized weight, which is the route for larger prefill row counts when no cuTile GEMM is registered, and the non-CUDA fallback. A dequantized FP8 dispatch also counts once under the unquantized GEMV/matmul route it lands on. A thread-local scope around each synchronous engine forward call labels actual prefill/decode dispatch phases; unscoped calls remain unknown. Nested scopes restore the previous phase on return or panic. The scope never crosses an await. Cumulative phase counters still cannot attribute a request or elapsed kernel cost. Request-local attribution remains explicitly unavailable and requires correlated profiling. Graph configuration, eligible shapes, capture attempts/successes/failures, replays and latest eager reason are separate; they follow the target-model decode-graph dispatch decisions and capture/replay outcomes behind `mistralrs_cuda_graph_dispatch_total` and `mistralrs_cuda_graph_events_total`, and exclude draft-model graphs. Relaxed reads do not form an atomic cross-counter snapshot.

Paged causal execution exposes activation/resolved KV dtype, block size and selected build features. Load provenance, full attention route details, GPU architecture, weight/block layout, non-recurrent cache writes/evictions and memory peaks remain unavailable. These hooks alone do not satisfy real-run backend or overhead acceptance.

Focused portable checks are `make observation-check` and `make observation-test`. Format only the changed Rust files with `make observation-format OBSERVATION_FILES='<files>'`. These checks require no model download, GPU execution or installation.

Request-local recurrent snapshot fields are `snapshot_write_count`,
`snapshot_write_bytes`, `snapshot_evicted_entries`, `snapshot_evicted_bytes`,
`snapshot_host_copy_ns` and `snapshot_write_measurement_complete`. They follow
actual interval-checkpoint writes by the same supported original single-choice
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

`MISTRALRS_CUDA_ALLOCATOR_OBSERVATIONS=1` opts each newly constructed Sequence
into optional `EngineObservation.cuda_allocator` queries, only while inference
observations are enabled. Other values leave the block absent and preserve
existing observation JSON bytes. Four named optional slots bound retained queries:
`pipeline_begin`, `prefill_before`, `prefill_after`, and `terminal`. Producer phase
labels are fixed as `PipelineBegin`, `PrefillBefore`, `PrefillAfter`, and `Terminal`;
each is queried at most once. Missing boundaries remain absent, never zero.

`PipelineBegin` uses the actual primary pipeline device after scheduler/admission,
excluding load, startup and scheduler waiting. `PrefillBefore` precedes the first
actual prompt forward outside its existing measured interval. `PrefillAfter`
follows a successful single-pass default prompt, or a paged prompt chunk with
proven final metadata, after the existing execution timer boundary. This includes
existing CPU conversion and, for paged execution, speculative/computed-update
post-forward processing. Failed forwards and missing final-chunk metadata leave
that slot absent. `Terminal` precedes final choice/Usage snapshots, including the
existing model-error publication boundary; no new query occurs after a known
response disconnect. Cancellation without a usable device/publication boundary
can leave it missing. Retention does not attest successful response delivery.

Samples have `primary_device` scope and the actual `cuda_device_ordinal`. Integer
`query_start_sequence_elapsed_ns` and `query_end_sequence_elapsed_ns` bracket the
host query on the existing sequence origin. They do not synchronize CUDA or prove
GPU completion or cross-clock alignment. `total_bytes` and `free_bytes` are raw
device observations; optional `async_pool` and `graph_pool` retain their separate
current used/reserved and observed used/reserved highwater values. The async pool
also retains its release threshold. These sequential reads are not an atomic
snapshot. Missing pools are unknown while known zero remains zero. Highwaters
are allocator/device observations, not physical request or phase peaks, and may
reflect external resets; this observer never resets them, trims pools or changes
synchronization. Do not sum pools, subtract highwaters or infer request-owned
memory. A failed query retains `unavailable=query_failed` with unknown metrics;
non-CUDA devices retain `non_cuda_device`, and unavailable allocator/feature
branches use fixed reasons. No inference error is introduced by these queries.

`terminal` retains the first observed terminal publication boundary. A later
publication error does not requery or replace it; that earlier sample does not
describe the later error state. Present terminal samples never establish
measurement completeness or successful delivery. Supplied pool/graph metrics and
sample phase/scope/query brackets are required when deserializing a present
object; incomplete objects reject rather than silently create known zeros.
