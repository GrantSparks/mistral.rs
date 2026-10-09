use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) static GRAPH_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_SUCCESSES: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_FAILURES: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_REPLAYS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_ELIGIBLE: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GDN_FUSED_DECODE: mistralrs_quant::serving_observation::HostDispatchCounter =
    mistralrs_quant::serving_observation::HostDispatchCounter::new();
static GRAPH_REASON: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum GraphEagerReason {
    Unknown = 0,
    ConfigurationDisabled,
    ModelUnsupported,
    SpeculativeProposer,
    NoPagedAttention,
    PromptShape,
    UnsupportedDecodeShape,
    NoCacheConfig,
    DisabledAfterFailure,
    CaptureOrReplayFailed,
    GraphUsed,
    MemoryPressure,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphEvent {
    Eligible,
    CaptureAttempt,
    CaptureSuccess,
    Replay,
    Failure,
    Eager(GraphEagerReason),
}

#[cfg(any(feature = "cuda", test))]
#[derive(Default)]
struct GraphSnapshot {
    attempts: usize,
    successes: usize,
    failures: usize,
    replays: usize,
    eligible: usize,
    reason: Option<GraphEagerReason>,
}

#[cfg(any(feature = "cuda", test))]
impl GraphSnapshot {
    fn transition(mut self, event: GraphEvent) -> Self {
        match event {
            GraphEvent::Eligible => self.eligible = self.eligible.saturating_add(1),
            GraphEvent::CaptureAttempt => self.attempts = self.attempts.saturating_add(1),
            GraphEvent::CaptureSuccess => {
                self.successes = self.successes.saturating_add(1);
                self.reason = Some(GraphEagerReason::GraphUsed);
            }
            GraphEvent::Replay => {
                self.replays = self.replays.saturating_add(1);
                self.reason = Some(GraphEagerReason::GraphUsed);
            }
            GraphEvent::Failure => {
                self.failures = self.failures.saturating_add(1);
                self.reason = Some(GraphEagerReason::CaptureOrReplayFailed);
            }
            GraphEvent::Eager(reason) => self.reason = Some(reason),
        }
        self
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn record_graph(event: GraphEvent) {
    if !mistralrs_quant::serving_observation::enabled() {
        return;
    }
    let delta = GraphSnapshot::default().transition(event);
    for (counter, increment) in [
        (&GRAPH_ATTEMPTS, delta.attempts),
        (&GRAPH_SUCCESSES, delta.successes),
        (&GRAPH_REPLAYS, delta.replays),
        (&GRAPH_FAILURES, delta.failures),
        (&GRAPH_ELIGIBLE, delta.eligible),
    ] {
        if increment != 0 {
            counter.fetch_add(increment, Ordering::Relaxed);
        }
    }
    if let Some(reason) = delta.reason {
        GRAPH_REASON.store(reason as usize, Ordering::Relaxed);
    }
}

/// Map the target decode-graph dispatch decision, as labelled on
/// `mistralrs_cuda_graph_dispatch_total`, onto the observation vocabulary.
/// A replayed or captured-then-replayed step is an eligible step; every
/// skipped or eager decision becomes the latest eager reason. Draft-model
/// (DFlash) graphs are not part of the target-model observation.
#[cfg(feature = "cuda")]
pub(crate) fn graph_event_for_dispatch(
    component: crate::pipeline::cuda_graph::CudaGraphComponent,
    mode: crate::pipeline::cuda_graph::CudaGraphDispatchMode,
    reason: crate::pipeline::cuda_graph::CudaGraphDispatchReason,
) -> Option<GraphEvent> {
    use crate::pipeline::cuda_graph::{
        CudaGraphComponent, CudaGraphDispatchMode as Mode, CudaGraphDispatchReason as Reason,
    };
    if !matches!(component, CudaGraphComponent::Target) {
        return None;
    }
    let reason = match (mode, reason) {
        (_, Reason::CacheHit) | (Mode::Eager, Reason::CachePopulation) => {
            return Some(GraphEvent::Eligible)
        }
        (_, Reason::CachePopulation) => return None,
        (_, Reason::Fallback) => GraphEagerReason::CaptureOrReplayFailed,
        (_, Reason::Disabled) => GraphEagerReason::ConfigurationDisabled,
        (_, Reason::ModelUnsupported) => GraphEagerReason::ModelUnsupported,
        (_, Reason::SpeculativeConflict) => GraphEagerReason::SpeculativeProposer,
        (_, Reason::PagedAttentionUnavailable) => GraphEagerReason::NoPagedAttention,
        (_, Reason::Prefill) => GraphEagerReason::PromptShape,
        (_, Reason::IncompatibleShape | Reason::BatchUnsupported | Reason::PaddingUnavailable) => {
            GraphEagerReason::UnsupportedDecodeShape
        }
        (_, Reason::CacheConfigUnavailable) => GraphEagerReason::NoCacheConfig,
        (_, Reason::RuntimeDisabled) => GraphEagerReason::DisabledAfterFailure,
        (_, Reason::MemoryPressure) => GraphEagerReason::MemoryPressure,
    };
    Some(GraphEvent::Eager(reason))
}

/// Map a completed target-graph capture or replay, as labelled on
/// `mistralrs_cuda_graph_events_total`, onto observation transitions. Eager
/// fallbacks are observed through their dispatch decision instead.
#[cfg(feature = "cuda")]
pub(crate) fn graph_events_for_event(
    component: crate::pipeline::cuda_graph::CudaGraphComponent,
    event: crate::pipeline::cuda_graph::CudaGraphEvent,
    success: bool,
) -> &'static [GraphEvent] {
    use crate::pipeline::cuda_graph::{CudaGraphComponent, CudaGraphEvent as Event};
    if !matches!(component, CudaGraphComponent::Target) {
        return &[];
    }
    match (event, success) {
        (Event::Capture, true) => &[GraphEvent::CaptureAttempt, GraphEvent::CaptureSuccess],
        (Event::Capture, false) => &[GraphEvent::CaptureAttempt, GraphEvent::Failure],
        (Event::Replay, true) => &[GraphEvent::Replay],
        (Event::Replay, false) => &[GraphEvent::Failure],
        (Event::EagerFallback, _) => &[],
    }
}

pub(crate) fn populate(observation: &mut crate::EngineObservation) {
    if !mistralrs_quant::serving_observation::enabled() {
        return;
    }
    observation.worker_graph_capture_attempts = GRAPH_ATTEMPTS.load(Ordering::Relaxed);
    observation.worker_graph_capture_successes = GRAPH_SUCCESSES.load(Ordering::Relaxed);
    observation.worker_graph_failures = GRAPH_FAILURES.load(Ordering::Relaxed);
    observation.worker_graph_replays = GRAPH_REPLAYS.load(Ordering::Relaxed);
    observation.worker_graph_eligible = GRAPH_ELIGIBLE.load(Ordering::Relaxed);
    observation.worker_graph_eager_reason =
        graph_reason(GRAPH_REASON.load(Ordering::Relaxed)).into();
    observation.graph_requested = crate::perf_flags::cuda_graphs_enabled();
    let [native, dequant, gemv, matmul] = mistralrs_quant::serving_observation::counters();
    observation.worker_gdn_fused_decode = GDN_FUSED_DECODE.total();
    observation.build_cuda = cfg!(feature = "cuda");
    observation.build_flash_attn = cfg!(feature = "flash-attn");
    observation.build_metal = cfg!(feature = "metal");
    let phases = mistralrs_quant::serving_observation::phase_counters();
    observation.worker_fp8_native_unknown = phases[0][0];
    observation.worker_fp8_native_prefill = phases[0][1];
    observation.worker_fp8_native_decode = phases[0][2];
    observation.worker_fp8_dequant_unknown = phases[1][0];
    observation.worker_fp8_dequant_prefill = phases[1][1];
    observation.worker_fp8_dequant_decode = phases[1][2];
    observation.worker_unquant_gemv_unknown = phases[2][0];
    observation.worker_unquant_gemv_prefill = phases[2][1];
    observation.worker_unquant_gemv_decode = phases[2][2];
    observation.worker_unquant_matmul_unknown = phases[3][0];
    observation.worker_unquant_matmul_prefill = phases[3][1];
    observation.worker_unquant_matmul_decode = phases[3][2];
    let gdn_phases = GDN_FUSED_DECODE.phase_counts();
    observation.worker_gdn_fused_decode_unknown = gdn_phases[0];
    observation.worker_gdn_fused_decode_prefill = gdn_phases[1];
    observation.worker_gdn_fused_decode_decode = gdn_phases[2];
    observation.worker_fp8_native = native;
    observation.worker_fp8_dequant = dequant;
    observation.worker_unquant_gemv = gemv;
    observation.worker_unquant_matmul = matmul;
}

fn graph_reason(reason: usize) -> &'static str {
    match reason {
        value if value == GraphEagerReason::ConfigurationDisabled as usize => {
            "configuration_disabled"
        }
        value if value == GraphEagerReason::ModelUnsupported as usize => "model_unsupported",
        value if value == GraphEagerReason::SpeculativeProposer as usize => "speculative_proposer",
        value if value == GraphEagerReason::NoPagedAttention as usize => "no_paged_attention",
        value if value == GraphEagerReason::PromptShape as usize => "prompt_shape",
        value if value == GraphEagerReason::UnsupportedDecodeShape as usize => {
            "unsupported_decode_shape"
        }
        value if value == GraphEagerReason::NoCacheConfig as usize => "no_cache_config",
        value if value == GraphEagerReason::DisabledAfterFailure as usize => {
            "disabled_after_failure"
        }
        value if value == GraphEagerReason::CaptureOrReplayFailed as usize => {
            "capture_or_replay_failed"
        }
        value if value == GraphEagerReason::GraphUsed as usize => "graph_used",
        value if value == GraphEagerReason::MemoryPressure as usize => "memory_pressure",
        value if value == GraphEagerReason::Unknown as usize && !cfg!(feature = "cuda") => {
            "cuda_not_built"
        }
        _ => "unavailable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_contract_graph_capture_replay_failure_and_disable_transitions() {
        let mut snapshot = GraphSnapshot::default()
            .transition(GraphEvent::Eager(GraphEagerReason::ConfigurationDisabled));
        assert_eq!(
            snapshot.reason,
            Some(GraphEagerReason::ConfigurationDisabled)
        );
        assert_eq!(snapshot.attempts, 0);
        snapshot = snapshot
            .transition(GraphEvent::Eligible)
            .transition(GraphEvent::CaptureAttempt)
            .transition(GraphEvent::CaptureSuccess);
        assert_eq!(snapshot.reason, Some(GraphEagerReason::GraphUsed));
        snapshot = snapshot
            .transition(GraphEvent::Replay)
            .transition(GraphEvent::Failure);
        assert_eq!(
            snapshot.reason,
            Some(GraphEagerReason::CaptureOrReplayFailed)
        );
        snapshot = snapshot.transition(GraphEvent::Eager(GraphEagerReason::DisabledAfterFailure));
        assert_eq!(
            (
                snapshot.eligible,
                snapshot.attempts,
                snapshot.successes,
                snapshot.replays,
                snapshot.failures
            ),
            (1, 1, 1, 1, 1)
        );
        assert_eq!(
            snapshot.reason,
            Some(GraphEagerReason::DisabledAfterFailure)
        );
    }

    #[test]
    fn observation_contract_unsupported_graph_cannot_count_capture_or_replay() {
        let snapshot = GraphSnapshot::default()
            .transition(GraphEvent::Eager(GraphEagerReason::ModelUnsupported));
        assert_eq!(
            (snapshot.attempts, snapshot.successes, snapshot.replays),
            (0, 0, 0)
        );
        assert_eq!(
            graph_reason(snapshot.reason.expect("known eager reason") as usize),
            "model_unsupported"
        );
        assert_eq!(
            graph_reason(GraphEagerReason::MemoryPressure as usize),
            "memory_pressure"
        );
    }

    #[test]
    fn observation_contract_failed_capture_has_no_success_or_replay() {
        let snapshot = GraphSnapshot::default()
            .transition(GraphEvent::Eligible)
            .transition(GraphEvent::CaptureAttempt)
            .transition(GraphEvent::Failure);
        assert_eq!(
            (
                snapshot.attempts,
                snapshot.failures,
                snapshot.successes,
                snapshot.replays
            ),
            (1, 1, 0, 0)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn observation_contract_target_graph_labels_map_to_observation_vocabulary() {
        use crate::pipeline::cuda_graph::{
            CudaGraphComponent, CudaGraphDispatchMode, CudaGraphDispatchReason, CudaGraphEvent,
        };
        assert_eq!(
            graph_event_for_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Replay,
                CudaGraphDispatchReason::CacheHit
            ),
            Some(GraphEvent::Eligible)
        );
        assert_eq!(
            graph_event_for_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::CachePopulation
            ),
            Some(GraphEvent::Eligible)
        );
        for (reason, expected) in [
            (CudaGraphDispatchReason::Disabled, "configuration_disabled"),
            (
                CudaGraphDispatchReason::ModelUnsupported,
                "model_unsupported",
            ),
            (
                CudaGraphDispatchReason::SpeculativeConflict,
                "speculative_proposer",
            ),
            (
                CudaGraphDispatchReason::PagedAttentionUnavailable,
                "no_paged_attention",
            ),
            (CudaGraphDispatchReason::Prefill, "prompt_shape"),
            (
                CudaGraphDispatchReason::IncompatibleShape,
                "unsupported_decode_shape",
            ),
            (
                CudaGraphDispatchReason::BatchUnsupported,
                "unsupported_decode_shape",
            ),
            (
                CudaGraphDispatchReason::PaddingUnavailable,
                "unsupported_decode_shape",
            ),
            (
                CudaGraphDispatchReason::CacheConfigUnavailable,
                "no_cache_config",
            ),
            (
                CudaGraphDispatchReason::RuntimeDisabled,
                "disabled_after_failure",
            ),
            (
                CudaGraphDispatchReason::Fallback,
                "capture_or_replay_failed",
            ),
            (CudaGraphDispatchReason::MemoryPressure, "memory_pressure"),
        ] {
            let Some(GraphEvent::Eager(eager)) = graph_event_for_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                reason,
            ) else {
                panic!("{reason:?} must map to an eager reason");
            };
            assert_eq!(graph_reason(eager as usize), expected, "{reason:?}");
        }
        assert!(graph_event_for_dispatch(
            CudaGraphComponent::DFlash,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit
        )
        .is_none());

        let snapshot =
            graph_events_for_event(CudaGraphComponent::Target, CudaGraphEvent::Capture, true)
                .iter()
                .chain(graph_events_for_event(
                    CudaGraphComponent::Target,
                    CudaGraphEvent::Replay,
                    true,
                ))
                .chain(graph_events_for_event(
                    CudaGraphComponent::Target,
                    CudaGraphEvent::Replay,
                    false,
                ))
                .fold(GraphSnapshot::default(), |snapshot, &event| {
                    snapshot.transition(event)
                });
        assert_eq!(
            (
                snapshot.attempts,
                snapshot.successes,
                snapshot.replays,
                snapshot.failures
            ),
            (1, 1, 1, 1)
        );
        let failed =
            graph_events_for_event(CudaGraphComponent::Target, CudaGraphEvent::Capture, false)
                .iter()
                .fold(GraphSnapshot::default(), |snapshot, &event| {
                    snapshot.transition(event)
                });
        assert_eq!(
            (failed.attempts, failed.successes, failed.failures),
            (1, 0, 1)
        );
        assert!(
            graph_events_for_event(CudaGraphComponent::DFlash, CudaGraphEvent::Capture, true)
                .is_empty()
        );
        assert!(graph_events_for_event(
            CudaGraphComponent::Target,
            CudaGraphEvent::EagerFallback,
            true
        )
        .is_empty());
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SequenceClockAnchor {
    pub(crate) origin: std::time::Instant,
    pub(crate) utc_unix_ns: Option<u64>,
    pub(crate) uncertainty_ns: u64,
}

impl SequenceClockAnchor {
    pub(crate) fn capture() -> Option<Self> {
        if !mistralrs_quant::serving_observation::enabled() {
            return None;
        }
        let origin = std::time::Instant::now();
        let utc = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok();
        let uncertainty = origin.elapsed();
        Some(Self {
            origin,
            utc_unix_ns: utc.and_then(|duration| u64::try_from(duration.as_nanos()).ok()),
            uncertainty_ns: u64::try_from(uncertainty.as_nanos()).unwrap_or(u64::MAX),
        })
    }
}

/// Return the cached engine observation mode used by the serving hooks.
pub fn inference_observations_enabled() -> bool {
    mistralrs_quant::serving_observation::enabled()
}

/// Translate a recurrent prefix validation miss, as reported on
/// `mistralrs_paged_recurrent_prefix_validation_total`, into the observation's
/// recurrent miss vocabulary. A zero-token KV candidate is reported as the KV
/// miss it is, whichever validation branch rejected it.
pub(crate) fn recurrent_miss_reason(
    validation_reason: &'static str,
    cached_tokens: usize,
) -> &'static str {
    if cached_tokens == 0 {
        return "no_matching_kv_prefix";
    }
    match validation_reason {
        "missing_slot" => "no_recurrent_slot",
        "unaligned_boundary" | "no_replay_boundary" => "unaligned_candidate",
        "checkpoint_unavailable" => "not_retained",
        other => other,
    }
}

#[cfg(test)]
mod snapshot_candidate_tests {
    use super::recurrent_miss_reason;

    #[test]
    fn observation_contract_recurrent_candidate_real_branch_reasons() {
        assert_eq!(
            recurrent_miss_reason("no_replay_boundary", 0),
            "no_matching_kv_prefix"
        );
        assert_eq!(
            recurrent_miss_reason("missing_slot", 0),
            "no_matching_kv_prefix"
        );
        assert_eq!(
            recurrent_miss_reason("missing_slot", 8),
            "no_recurrent_slot"
        );
        assert_eq!(
            recurrent_miss_reason("unaligned_boundary", 7),
            "unaligned_candidate"
        );
        assert_eq!(
            recurrent_miss_reason("no_replay_boundary", 7),
            "unaligned_candidate"
        );
        assert_eq!(
            recurrent_miss_reason("checkpoint_unavailable", 8),
            "not_retained"
        );
    }
}
