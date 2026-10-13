pub(crate) mod debug;
pub(crate) mod gguf_metadata;
pub(crate) mod memory_usage;
pub(crate) mod model_config;
pub(crate) mod normal;
pub(crate) mod progress;
pub(crate) mod tiktoken;
pub(crate) mod tokenizer;
pub(crate) mod tokens;
pub(crate) mod unvarbuilder;
pub(crate) mod varbuilder_utils;

pub(crate) fn prepare_pipeline_error_observation(seq: &mut crate::sequence::Sequence) {
    seq.mark_speculative_incomplete("error");
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            // This prevents deadlock when a spawned async task busy-loops while
            // another task holds the lock across an await point.
            std::thread::yield_now();
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
                return;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_request_error {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                if $response
                    .send(Response::ValidationError(e.into()))
                    .await
                    .is_err()
                {
                    tracing::warn!("Receiver disconnected");
                }
                return;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_ok {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_stateaware_ok {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                if let Err(_) = $seq
                    .responder()
                    .send(Response::InternalError(e.into()))
                    .await
                {
                    tracing::warn!("Receiver disconnected");
                }
                $seq.set_state(SequenceState::Error);
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_pipeline_forward_error {
    ($stage: tt, $fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt, $prefix_cacher:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                // Auto-retry on iOS Metal background GPU error: when the iOS app
                // goes to background, Metal rejects command buffers. We detect this,
                // reset cache, sleep, and let the engine loop retry. Sequences stay
                // in the scheduler (still in Running state) and are re-scheduled.
                #[cfg(feature = "metal")]
                {
                    let err_str = e.to_string();
                    if err_str.contains("Insufficient Permission")
                        || err_str.contains("BackgroundExecutionNotPermitted")
                    {
                        tracing::warn!(
                            "Metal GPU background error detected (iOS app likely in background). \
                             Pausing 1s before retry..."
                        );
                        {
                            let p = get_mut_arcmutex!($pipeline);
                            if let Err(reset_err) =
                                p.set_none_cache($seq_slice, true, true, false)
                            {
                                tracing::error!("Failed to reset model cache: {reset_err}");
                            }
                        }
                        get_mut_arcmutex!($prefix_cacher).evict_all_caches().unwrap();
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        continue $label;
                    }
                }

                let (tokenizer, pipeline_name, allocator_device) = {
                    let pipeline = get_mut_arcmutex!($pipeline);
                    let pipeline_name = pipeline.name();
                    let tokenizer = pipeline.tokenizer();
                    let allocator_device = if $seq_slice.iter_mut().any(|seq| seq.allocator_observation_pending($crate::sequence::AllocatorObservationPhase::Terminal)) {
                        Some(pipeline.device())
                    } else {
                        None
                    };
                    (tokenizer, pipeline_name, allocator_device)
                };
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::response::SYSTEM_FINGERPRINT;
                use tracing::error;
                error!("{} - Model failed with error: {:?}", $stage, &e);
                for seq in $seq_slice.iter_mut() {
                    // Step 1: Add all choices to groups
                    let start = seq.prompt_tokens().min(seq.get_toks().len());
                    let res = match &tokenizer {
                        Some(tok) => match tok.decode(&seq.get_toks()[start..], false) {
                            Ok(t) => t,
                            Err(_) => "".to_string(),
                        },
                        None => "".to_string(),
                    };

                    if let Some(device) = allocator_device.as_ref() {
                        seq.observe_cuda_allocator($crate::sequence::AllocatorObservationPhase::Terminal, device);
                    }
                    $crate::utils::prepare_pipeline_error_observation(seq);
                    if seq.get_mut_group().is_chat {
                        let choice = Choice {
                            finish_reason: "error".to_string(),
                            stop_sequence: None,
                            index: seq.get_response_index(),
                            message: ResponseMessage {
                                content: Some(res),
                                role: "assistant".to_string(),
                                tool_calls: None,
                                reasoning_content: None,
                            },
                            logprobs: None,
                        };
                        seq.add_choice_to_group(choice);
                    } else {
                        let choice = CompletionChoice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            text: res,
                            logprobs: None,
                        };
                        seq.add_completion_choice_to_group(choice);
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 2: Respond with all groups
                    let group = seq.get_mut_group();

                    if group.is_chat {
                        let partial_completion_response = ChatCompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline_name.clone(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "chat.completion".to_string(),
                            usage: group.get_usage(),
                            adapter_generation: seq
                                .adapter_generation()
                                .map(|generation| generation.to_string()),
                            agentic_tool_calls: None,
                            files: None,
                            session_id: None,
                        };

                        if let Err(send_err) = seq.responder()
                            .send(Response::ModelError(
                                e.to_string(),
                                partial_completion_response,
                            ))
                            .await
                        {
                            tracing::warn!(
                                "Failed to send chat model error to client for seq {}: {} (client likely disconnected)",
                                seq.id(),
                                send_err
                            );
                        }
                    } else {
                        let partial_completion_response = CompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_completion_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline_name.clone(),
                            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                            object: "text_completion".to_string(),
                            usage: group.get_usage(),
                            adapter_generation: seq
                                .adapter_generation()
                                .map(|generation| generation.to_string()),
                        };

                        if let Err(send_err) = seq.responder()
                            .send(Response::CompletionModelError(
                                e.to_string(),
                                partial_completion_response,
                            ))
                            .await
                        {
                            tracing::warn!(
                                "Failed to send completion model error to client for seq {}: {} (client likely disconnected)",
                                seq.id(),
                                send_err
                            );
                        }
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
                    seq.set_state(SequenceState::Error);
                }

                let p = get_mut_arcmutex!($pipeline);
                // Also reset non granular state because:
                // - The sequence is gone
                // - We should reset the state then, including draft.
                if let Err(reset_err) = p.set_none_cache($seq_slice, true, true, false) {
                    tracing::error!("Failed to reset model cache: {reset_err}");
                }
                get_mut_arcmutex!($prefix_cacher).evict_all_caches().unwrap();

                continue $label;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_group {
    ($this:expr) => {
        loop {
            if let Ok(inner) = $this.group.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            std::thread::yield_now();
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const fn paged_attn_supported() -> bool {
    true
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub const fn paged_attn_supported() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub const fn using_flash_attn() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
pub const fn using_flash_attn() -> bool {
    true
}

#[cfg(test)]
mod observation_contract_tests {
    use crate::{
        sampler::Logprobs,
        sequence::{SequenceState, StopReason},
    };

    #[test]
    fn observation_contract_pipeline_error_usage_is_partial_before_error_state() {
        for is_chat in [true, false] {
            for reason in [StopReason::Eos, StopReason::Length(1)] {
                let mut seq = crate::sequence::tests::make_observed_test_sequence();
                seq.get_mut_group().is_chat = is_chat;
                seq.begin_speculative_verification(7);
                seq.observe_speculative_token(true);
                seq.add_token(
                    Logprobs {
                        token: 9,
                        logprob: 0.0,
                        bytes: None,
                        top_logprobs: None,
                    },
                    b"partial".to_vec(),
                    Some(reason),
                );
                seq.mark_final_token(&reason);
                seq.set_state(SequenceState::Done(reason));
                assert!(
                    seq.engine_observation
                        .speculative
                        .as_ref()
                        .expect("prior terminal")
                        .measurement_complete
                );
                super::prepare_pipeline_error_observation(&mut seq);
                if is_chat {
                    seq.add_choice_to_group(crate::Choice {
                        finish_reason: "error".into(),
                        stop_sequence: None,
                        index: 0,
                        message: crate::ResponseMessage {
                            content: Some("partial".into()),
                            role: "assistant".into(),
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        logprobs: None,
                    });
                } else {
                    seq.add_completion_choice_to_group(crate::CompletionChoice {
                        finish_reason: "error".into(),
                        index: 0,
                        text: "partial".into(),
                        logprobs: None,
                    });
                }
                assert_eq!(seq.getstate(), SequenceState::Done(reason));
                let group = seq.get_mut_group();
                let usage = group.get_usage();
                if is_chat {
                    assert_eq!(group.get_choices()[0].finish_reason, "error");
                } else {
                    assert_eq!(group.get_completion_choices()[0].finish_reason, "error");
                }
                let observation = usage
                    .engine_observation
                    .expect("published error observation");
                let speculative = observation
                    .speculative
                    .expect("published speculative error");
                assert!(!speculative.measurement_complete);
                assert_eq!(speculative.terminal_reason.as_deref(), Some("error"));
                assert_eq!(speculative.verification_steps, 1);
                assert_eq!(speculative.proposed_draft_tokens, 7);
                assert_eq!(speculative.accepted_draft_tokens, 1);
                assert_eq!(speculative.committed_draft_tokens, 1);
                assert_eq!(speculative.committed_output_tokens, 1);
                assert_eq!(
                    observation.finish_reason.as_deref(),
                    Some(reason.to_string().as_str())
                );
                drop(group);
                seq.set_state(SequenceState::Error);
                assert_eq!(seq.getstate(), SequenceState::Error);
            }
        }
    }

    #[test]
    fn observation_contract_pipeline_error_observer_preserves_ordinary_and_unsupported_usage() {
        for unsupported in [false, true] {
            let mut seq = crate::sequence::tests::make_observed_test_sequence();
            seq.observation_causal = true;
            seq.observe_sampled_token();
            seq.add_token(
                Logprobs {
                    token: 9,
                    logprob: 0.0,
                    bytes: None,
                    top_logprobs: None,
                },
                b"ordinary".to_vec(),
                Some(StopReason::Length(1)),
            );
            seq.mark_final_token(&StopReason::Length(1));
            if unsupported {
                seq.disable_exact_observation();
            }
            seq.update_time_info();
            let before =
                serde_json::to_string(&seq.get_mut_group().get_usage()).expect("ordinary usage");
            super::prepare_pipeline_error_observation(&mut seq);
            let after = serde_json::to_string(&seq.get_mut_group().get_usage())
                .expect("prepared ordinary usage");
            assert_eq!(before, after);
            assert!(seq.engine_observation.speculative.is_none());
        }
    }
}
