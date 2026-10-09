use std::{
    error::Error,
    fmt::{Debug, Display},
    sync::Arc,
};

use candle_core::Tensor;
#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::Serialize;

use crate::{sampler::TopLogprob, tools::ToolCallResponse};

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
#[serde(default)]
/// Sequence-local causal output markers and explicitly worker-wide dispatch counters.
/// Accepted output includes EOS, stop and nonvisible tokens; rejected candidates and drafts are excluded.
pub struct EngineObservation {
    pub sequence_id: usize,
    pub clock_origin: String,
    pub clock_anchor_unavailable: String,
    pub sequence_origin_utc_unix_ns: Option<u64>,
    pub sequence_origin_uncertainty_ns: Option<u64>,
    pub activation_dtype: Option<String>,
    pub kv_dtype: Option<String>,
    pub paged_block_size: Option<usize>,
    pub worker_graph_capture_attempts: usize,
    pub worker_graph_capture_successes: usize,
    pub worker_graph_failures: usize,
    pub worker_graph_replays: usize,
    pub worker_graph_eligible: usize,
    pub worker_graph_eager_reason: String,
    pub graph_requested: bool,
    pub graph_counter_scope: String,
    pub build_cuda: bool,
    pub build_flash_attn: bool,
    pub build_metal: bool,
    pub worker_gdn_fused_decode: usize,
    pub token_count_convention: String,
    pub worker_fp8_native_prefill: usize,
    pub worker_fp8_native_decode: usize,
    pub worker_fp8_native_unknown: usize,
    pub worker_fp8_dequant_prefill: usize,
    pub worker_fp8_dequant_decode: usize,
    pub worker_fp8_dequant_unknown: usize,
    pub worker_unquant_gemv_prefill: usize,
    pub worker_unquant_gemv_decode: usize,
    pub worker_unquant_gemv_unknown: usize,
    pub worker_unquant_matmul_prefill: usize,
    pub worker_unquant_matmul_decode: usize,
    pub worker_unquant_matmul_unknown: usize,
    pub worker_gdn_fused_decode_prefill: usize,
    pub worker_gdn_fused_decode_decode: usize,
    pub worker_gdn_fused_decode_unknown: usize,
    pub worker_fp8_native: usize,
    pub worker_fp8_dequant: usize,
    pub worker_unquant_gemv: usize,
    pub worker_unquant_matmul: usize,
    pub kernel_counter_scope: String,
    pub response_index: usize,
    pub prompt_tokens: usize,
    pub candidate_prefix_tokens: Option<usize>,
    pub recurrent_restored_tokens: Option<usize>,
    pub reused_prefix_tokens: Option<usize>,
    pub uncached_suffix_tokens: Option<usize>,
    pub cache_outcome: Option<String>,
    pub recurrent_outcome: Option<String>,
    pub prefix_lookup_sec: Option<f64>,
    pub recurrent_lookup_sec: Option<f64>,
    pub recurrent_restore_sec: Option<f64>,
    pub snapshot_host_bytes: Option<usize>,
    pub snapshot_entries: Option<usize>,
    pub snapshot_capacity_bytes: Option<usize>,
    pub snapshot_placement: Option<String>,
    pub snapshot_write_count: Option<u64>,
    pub snapshot_write_bytes: Option<u64>,
    pub snapshot_evicted_entries: Option<u64>,
    pub snapshot_evicted_bytes: Option<u64>,
    pub snapshot_host_copy_ns: Option<u64>,
    pub snapshot_write_measurement_complete: Option<bool>,
    pub prefill_chunk_count: Option<usize>,
    pub prefill_chunk_max_tokens: Option<usize>,
    pub prefill_host_forward_sec: Option<f64>,
    pub first_token_sequence_elapsed_ns: Option<u64>,
    pub last_token_sequence_elapsed_ns: Option<u64>,
    pub sampled_tokens: usize,
    pub final_token_marker: bool,
    pub finish_reason: Option<String>,
    pub token_marker_boundary: String,
    pub kernel_phase_unavailable: String,
    pub raw_prompt_sec: f64,
    pub raw_completion_sec: f64,
}

impl EngineObservation {
    pub(crate) fn begin_snapshot_write(&mut self) {
        self.snapshot_write_count.get_or_insert(0);
        self.snapshot_write_bytes.get_or_insert(0);
        self.snapshot_evicted_entries.get_or_insert(0);
        self.snapshot_evicted_bytes.get_or_insert(0);
        self.snapshot_host_copy_ns.get_or_insert(0);
        self.snapshot_write_measurement_complete.get_or_insert(true);
    }

    pub(crate) fn snapshot_write_failed(&mut self) {
        self.begin_snapshot_write();
        self.snapshot_write_measurement_complete = Some(false);
    }

    pub(crate) fn record_snapshot_write(
        &mut self,
        outcome: &crate::prefix_cacher::SnapshotWriteOutcome,
    ) {
        for (counter, delta) in [
            (&mut self.snapshot_write_count, outcome.written_count),
            (&mut self.snapshot_write_bytes, outcome.written_bytes),
            (&mut self.snapshot_evicted_entries, outcome.evicted_entries),
            (&mut self.snapshot_evicted_bytes, outcome.evicted_bytes),
        ] {
            *counter = Some(counter.unwrap_or(0).saturating_add(delta));
        }
    }

    pub(crate) fn accepted_token(&mut self, elapsed_ns: u64) {
        self.first_token_sequence_elapsed_ns
            .get_or_insert(elapsed_ns);
        self.last_token_sequence_elapsed_ns = Some(elapsed_ns);
        self.sampled_tokens = self.sampled_tokens.saturating_add(1);
    }

    pub(crate) fn recurrent_restore(&mut self, restored: bool, boundary: usize) {
        self.recurrent_restored_tokens = Some(if restored { boundary } else { 0 });
        self.recurrent_outcome = Some(
            if restored {
                "restored"
            } else {
                "restore_failed"
            }
            .into(),
        );
    }

    pub(crate) fn recurrent_miss(&mut self, reason: &str) {
        self.recurrent_outcome = Some(reason.into());
        self.recurrent_restored_tokens = Some(0);
    }

    pub(crate) fn prefix_candidate(&mut self, enabled: bool, candidate: usize) {
        self.candidate_prefix_tokens = Some(candidate);
        self.cache_outcome = Some(
            if !enabled {
                "disabled"
            } else if candidate == 0 {
                "no_matching_kv_prefix"
            } else {
                "candidate_hit"
            }
            .into(),
        );
    }

    pub(crate) fn final_prefix(&mut self, prompt_tokens: usize, reused: usize) {
        self.reused_prefix_tokens = Some(reused);
        self.uncached_suffix_tokens = Some(prompt_tokens.saturating_sub(reused));
    }
}

pub const SYSTEM_FINGERPRINT: &str = "local";

macro_rules! generate_repr {
    ($t:ident) => {
        #[cfg(feature = "pyo3_macros")]
        #[pymethods]
        impl $t {
            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    };
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion response message.
pub struct ResponseMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content from Harmony format (separate from final content).
    /// This contains chain-of-thought reasoning that is not intended for end users.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

generate_repr!(ResponseMessage);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Delta in content for streaming response.
pub struct Delta {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content delta from Harmony format.
    /// This contains incremental chain-of-thought reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

generate_repr!(Delta);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// A logprob with the top logprobs for this token.
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

generate_repr!(ResponseLogprob);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Logprobs per token.
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

generate_repr!(Logprobs);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion choice.
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

generate_repr!(Choice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming chunk choice.
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

generate_repr!(ChunkChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming chunk choice.
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<ResponseLogprob>,
    pub finish_reason: Option<String>,
}

generate_repr!(CompletionChunkChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
/// OpenAI compatible (superset) usage during a request.
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_observation: Option<EngineObservation>,
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
    pub total_time_sec: f32,
    pub total_prompt_time_sec: f32,
    pub total_completion_time_sec: f32,
}

generate_repr!(Usage);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct AgenticToolCallRecord {
    pub round: usize,
    pub name: String,
    pub arguments: String,
    pub result_content: String,
    /// Base64-encoded PNG images.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub result_images_base64: Vec<String>,
    /// Resolve via the response's top-level `files` or `/v1/files/{id}`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
}

generate_repr!(AgenticToolCallRecord);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible chat completion response.
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
    /// Ordered record of all tool calls made during the agentic loop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic_tool_calls: Option<Vec<AgenticToolCallRecord>>,
    /// Files surfaced by tool calls. Bodies inline up to the wire-embed cap; larger files are fetch-by-id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<crate::files::File>>,
    /// Reuse in later requests to keep agentic state across messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

generate_repr!(ChatCompletionResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming request chunk.
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Option<Usage>,
    /// Set on the final chunk so streaming clients can read it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

generate_repr!(ChatCompletionChunkResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<Logprobs>,
}

generate_repr!(CompletionChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible completion response.
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}

generate_repr!(CompletionResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChunkResponse {
    pub id: String,
    pub choices: Vec<CompletionChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

generate_repr!(CompletionChunkResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageChoice {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

generate_repr!(ImageChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u128,
    pub data: Vec<ImageChoice>,
}

generate_repr!(ImageGenerationResponse);

/// Tool-specific structured progress data for agentic tool calls.
#[derive(Debug, Clone)]
pub enum AgenticToolCallData {
    /// Python code execution.
    CodeExecution {
        code: Option<String>,
        stdout: Option<String>,
        stderr: Option<String>,
        exception: Option<String>,
        images: Vec<image::DynamicImage>,
        video_frames: Vec<image::DynamicImage>,
        video_frame_count: Option<usize>,
        working_directory: Option<String>,
        execution_time_ms: Option<u64>,
    },
    /// Web search or content extraction.
    WebSearch {
        query: Option<String>,
        results_count: Option<usize>,
        sources: Vec<String>,
    },
    /// Shell command execution.
    Shell {
        commands: Vec<String>,
        stdout: Option<String>,
        stderr: Option<String>,
        exit_code: Option<i64>,
        status: Option<String>,
        working_directory: Option<String>,
        timed_out: Option<bool>,
    },
    /// User callback, MCP, or HTTP dispatch. Opaque to the engine.
    Custom { arguments: String, content: String },
}

/// Phase of an agentic tool call.
#[derive(Debug, Clone)]
pub enum AgenticToolCallPhase {
    /// Tool call parsed, about to execute.
    Calling(AgenticToolCallData),
    /// Execution complete.
    Complete(AgenticToolCallData),
}

#[derive(Debug, Clone)]
pub struct BlockDenoisingProgress {
    pub index: usize,
    pub step: usize,
    pub total_steps: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub finished: bool,
    pub final_block: bool,
}

/// The response enum contains 3 types of variants:
/// - Error (-Error suffix)
/// - Chat (no prefix)
/// - Completion (Completion- prefix)
pub enum Response {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    // Chat
    ModelError(String, ChatCompletionResponse),
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionModelError(String, CompletionResponse),
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(ImageGenerationResponse),
    // Speech generation
    Speech {
        pcm: Arc<Vec<f32>>,
        rate: usize,
        channels: usize,
    },
    // Raw
    Raw {
        logits_chunks: Vec<Tensor>,
        tokens: Vec<u32>,
    },
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
    /// Progress event emitted by the agentic loop during tool execution.
    AgenticToolCallProgress {
        round: usize,
        tool_name: String,
        phase: AgenticToolCallPhase,
    },
    BlockDenoisingProgress(BlockDenoisingProgress),
    AgenticToolApprovalRequired {
        approval_id: String,
        session_id: String,
        round: usize,
        tool: crate::AgentToolMetadata,
        arguments: serde_json::Value,
    },
    /// Emitted as soon as the runtime reads a file out of the working directory.
    File(crate::files::File),
}

#[derive(Debug, Clone)]
pub enum ResponseOk {
    // Chat
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(ImageGenerationResponse),
    // Speech generation
    Speech {
        pcm: Arc<Vec<f32>>,
        rate: usize,
        channels: usize,
    },
    // Raw
    Raw {
        logits_chunks: Vec<Tensor>,
        tokens: Vec<u32>,
    },
    // Embeddings
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
    // Agentic tool progress
    AgenticToolCallProgress {
        round: usize,
        tool_name: String,
        phase: AgenticToolCallPhase,
    },
    BlockDenoisingProgress(BlockDenoisingProgress),
    AgenticToolApprovalRequired {
        approval_id: String,
        session_id: String,
        round: usize,
        tool: crate::AgentToolMetadata,
        arguments: serde_json::Value,
    },
    File(crate::files::File),
}

pub enum ResponseErr {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    ModelError(String, ChatCompletionResponse),
    CompletionModelError(String, CompletionResponse),
}

impl Display for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Display::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl Debug for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Debug::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl std::error::Error for ResponseErr {}

impl Response {
    /// Convert the response into a result form.
    pub fn as_result(self) -> Result<ResponseOk, Box<ResponseErr>> {
        match self {
            Self::Done(x) => Ok(ResponseOk::Done(x)),
            Self::Chunk(x) => Ok(ResponseOk::Chunk(x)),
            Self::CompletionDone(x) => Ok(ResponseOk::CompletionDone(x)),
            Self::CompletionChunk(x) => Ok(ResponseOk::CompletionChunk(x)),
            Self::InternalError(e) => Err(Box::new(ResponseErr::InternalError(e))),
            Self::ValidationError(e) => Err(Box::new(ResponseErr::ValidationError(e))),
            Self::ModelError(e, x) => Err(Box::new(ResponseErr::ModelError(e, x))),
            Self::CompletionModelError(e, x) => {
                Err(Box::new(ResponseErr::CompletionModelError(e, x)))
            }
            Self::ImageGeneration(x) => Ok(ResponseOk::ImageGeneration(x)),
            Self::Speech {
                pcm,
                rate,
                channels,
            } => Ok(ResponseOk::Speech {
                pcm,
                rate,
                channels,
            }),
            Self::Raw {
                logits_chunks,
                tokens,
            } => Ok(ResponseOk::Raw {
                logits_chunks,
                tokens,
            }),
            Self::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => Ok(ResponseOk::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            }),
            Self::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            } => Ok(ResponseOk::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            }),
            Self::BlockDenoisingProgress(progress) => {
                Ok(ResponseOk::BlockDenoisingProgress(progress))
            }
            Self::AgenticToolApprovalRequired {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            } => Ok(ResponseOk::AgenticToolApprovalRequired {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            }),
            Self::File(f) => Ok(ResponseOk::File(f)),
        }
    }
}

#[cfg(test)]
mod observation_contract_tests {
    use super::*;

    #[test]
    fn observation_contract_token_markers_include_terminal_sample() {
        let mut observation = EngineObservation::default();
        observation.accepted_token(100);
        observation.accepted_token(350);
        observation.final_token_marker = true;
        assert_eq!(observation.sampled_tokens, 2);
        assert_eq!(observation.first_token_sequence_elapsed_ns, Some(100));
        assert_eq!(observation.last_token_sequence_elapsed_ns, Some(350));
    }

    #[test]
    fn observation_contract_final_clamp_keeps_candidate_and_restore_boundary() {
        let mut observation = EngineObservation {
            candidate_prefix_tokens: Some(96),
            recurrent_restored_tokens: Some(64),
            ..Default::default()
        };
        observation.final_prefix(101, 32);
        assert_eq!(observation.candidate_prefix_tokens, Some(96));
        assert_eq!(observation.recurrent_restored_tokens, Some(64));
        assert_eq!(observation.reused_prefix_tokens, Some(32));
        assert_eq!(observation.uncached_suffix_tokens, Some(69));
    }

    #[test]
    fn observation_contract_restore_failed_is_distinct_from_hit() {
        let mut observation = EngineObservation::default();
        observation.recurrent_restore(false, 64);
        assert_eq!(observation.recurrent_restored_tokens, Some(0));
        assert_eq!(
            observation.recurrent_outcome.as_deref(),
            Some("restore_failed")
        );
        observation.recurrent_restore(true, 64);
        assert_eq!(observation.recurrent_restored_tokens, Some(64));
        assert_eq!(observation.recurrent_outcome.as_deref(), Some("restored"));
    }

    #[test]
    fn observation_contract_snapshot_fields_are_optional_and_roundtrip() {
        let legacy: EngineObservation =
            serde_json::from_str("{}").expect("legacy engine observation");
        assert_eq!(legacy.snapshot_write_count, None);
        assert_eq!(legacy.snapshot_host_copy_ns, None);
        assert_eq!(legacy.snapshot_write_measurement_complete, None);
        let observed = EngineObservation {
            snapshot_write_count: Some(2),
            snapshot_write_bytes: Some(4096),
            snapshot_evicted_entries: Some(1),
            snapshot_evicted_bytes: Some(2048),
            snapshot_host_copy_ns: Some(123),
            snapshot_write_measurement_complete: Some(false),
            ..Default::default()
        };
        let encoded = serde_json::to_string(&observed).expect("snapshot observations serialize");
        let decoded: EngineObservation =
            serde_json::from_str(&encoded).expect("snapshot observations deserialize");
        assert_eq!(decoded.snapshot_write_count, observed.snapshot_write_count);
        assert_eq!(decoded.snapshot_write_bytes, observed.snapshot_write_bytes);
        assert_eq!(
            decoded.snapshot_evicted_entries,
            observed.snapshot_evicted_entries
        );
        assert_eq!(
            decoded.snapshot_evicted_bytes,
            observed.snapshot_evicted_bytes
        );
        assert_eq!(
            decoded.snapshot_host_copy_ns,
            observed.snapshot_host_copy_ns
        );
        assert_eq!(decoded.snapshot_write_measurement_complete, Some(false));
    }

    #[test]
    fn observation_contract_legacy_usage_has_unknown_observation() {
        let usage: Usage = serde_json::from_str(r#"{"completion_tokens":1,"prompt_tokens":2,"total_tokens":3,"avg_tok_per_sec":0.0,"avg_prompt_tok_per_sec":0.0,"avg_compl_tok_per_sec":0.0,"total_time_sec":0.0,"total_prompt_time_sec":0.0,"total_completion_time_sec":0.0}"#).expect("legacy fixture");
        assert!(usage.engine_observation.is_none());
        let observation: EngineObservation = serde_json::from_str("{}").expect("optional fields");
        assert!(observation.candidate_prefix_tokens.is_none());
        assert!(observation.first_token_sequence_elapsed_ns.is_none());
    }
}
