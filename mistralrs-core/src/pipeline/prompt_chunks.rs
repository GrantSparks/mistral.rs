use crate::{
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy},
    speculative::{target::clamp_speculative_prefix_cache_hit, SpeculativePrefixReplay},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptChunkPlan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) attention_policy: MultimodalAttentionPolicy,
}

/// Hybrid prefill also checkpoints recurrent state at every multiple of this many tokens. A
/// prompt that shares only a leading span with an earlier prompt (a long system prompt, a
/// warmup) then resumes from the last multiple inside that span; the earlier prompt's own
/// final-block checkpoint lies past the shared span and can never serve it.
///
/// It must be a multiple of the paged block size so every multiple is a block boundary a prefix
/// key can name; 4096 is a multiple of every supported size (8, 16, 32). It also equals the
/// default prefill token budget, so prompts prefilled in 512- or 4096-token chunks from position
/// 0 already end chunks there. Each multiple costs one recurrent snapshot copy to host memory.
pub(crate) const RECURRENT_PREFIX_CHECKPOINT_INTERVAL: usize = 4096;

/// Whether `end` is an interval multiple inside a `total_len`-token prompt. The end of the prompt
/// itself is excluded: a prefix hit must leave at least one token to compute.
pub(crate) fn is_recurrent_interval_boundary(end: usize, total_len: usize) -> bool {
    end > 0 && end < total_len && end.is_multiple_of(RECURRENT_PREFIX_CHECKPOINT_INTERVAL)
}

pub(crate) fn next_prompt_chunk_group(
    plan_indices: &[usize],
    chunk_plans: &[Vec<PromptChunkPlan>],
    require_uniform_query_len: bool,
) -> Option<(Vec<usize>, MultimodalAttentionPolicy, bool)> {
    let (attention_policy, is_final, query_len) =
        plan_indices
            .iter()
            .zip(chunk_plans)
            .find_map(|(&plan_idx, plan)| {
                plan.get(plan_idx).map(|chunk| {
                    (
                        chunk.attention_policy,
                        plan_idx + 1 == plan.len(),
                        chunk.end - chunk.start,
                    )
                })
            })?;
    let active_indices = plan_indices
        .iter()
        .zip(chunk_plans)
        .enumerate()
        .filter_map(|(idx, (&plan_idx, plan))| {
            plan.get(plan_idx)
                .filter(|chunk| {
                    chunk.attention_policy == attention_policy
                        && (plan_idx + 1 == plan.len()) == is_final
                        && (!require_uniform_query_len || chunk.end - chunk.start == query_len)
                })
                .map(|_| idx)
        })
        .collect();
    Some((active_indices, attention_policy, is_final))
}

fn noncausal_component_end(
    pos: usize,
    total_len: usize,
    features: &[&MultiModalFeature],
) -> Option<usize> {
    let mut end = features
        .iter()
        .filter(|feature| {
            feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                && feature.offset <= pos
                && feature.end() > pos
        })
        .map(|feature| feature.end())
        .max()?
        .min(total_len);

    loop {
        let extended_end = features
            .iter()
            .filter(|feature| {
                feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                    && feature.offset < end
                    && feature.end() > pos
            })
            .map(|feature| feature.end())
            .max()
            .unwrap_or(end)
            .min(total_len);
        if extended_end == end {
            return Some(end);
        }
        end = extended_end;
    }
}

fn causal_unsplittable_component_end(
    pos: usize,
    total_len: usize,
    features: &[&MultiModalFeature],
) -> Option<usize> {
    let mut end = features
        .iter()
        .filter(|feature| {
            feature.attention_policy == MultimodalAttentionPolicy::Causal
                && !feature.splittable
                && feature.offset <= pos
                && feature.end() > pos
        })
        .map(|feature| feature.end())
        .max()?
        .min(total_len);

    loop {
        let extended_end = features
            .iter()
            .filter(|feature| {
                feature.attention_policy == MultimodalAttentionPolicy::Causal
                    && !feature.splittable
                    && feature.offset < end
                    && feature.end() > pos
            })
            .map(|feature| feature.end())
            .max()
            .unwrap_or(end)
            .min(total_len);
        if extended_end == end {
            return Some(end);
        }
        end = extended_end;
    }
}

fn normalize_prefix_boundary(
    mut boundary: usize,
    block_size: usize,
    features: &[&MultiModalFeature],
) -> usize {
    for feature in features.iter().rev() {
        if feature.offset < boundary && boundary < feature.end() {
            boundary = feature.offset / block_size * block_size;
        }
    }
    boundary
}

pub(crate) fn effective_recurrent_prefix_boundary(
    max_cached_tokens: usize,
    minimum_exclusive: usize,
    block_size: usize,
    replay: SpeculativePrefixReplay,
    features: &[MultiModalFeature],
) -> Option<usize> {
    if block_size == 0 || max_cached_tokens == 0 {
        return None;
    }
    let mut features = features.iter().collect::<Vec<_>>();
    features.sort_by_key(|feature| feature.offset);
    let max_cached_tokens = normalize_prefix_boundary(max_cached_tokens, block_size, &features);
    let replay_boundary = clamp_speculative_prefix_cache_hit(max_cached_tokens, block_size, replay);
    let replay_boundary = normalize_prefix_boundary(replay_boundary, block_size, &features);
    (minimum_exclusive < replay_boundary).then_some(replay_boundary)
}

pub(crate) fn recurrent_checkpoint_boundary(
    total_len: usize,
    prefix_len: usize,
    block_size: Option<usize>,
    replay: SpeculativePrefixReplay,
    features: &[MultiModalFeature],
) -> Option<usize> {
    let block_size = block_size.filter(|size| *size > 0)?;
    let max_cached_tokens = total_len.saturating_sub(1) / block_size * block_size;
    effective_recurrent_prefix_boundary(max_cached_tokens, prefix_len, block_size, replay, features)
        .filter(|boundary| *boundary < total_len)
}

fn cap_at_reusable_boundary(pos: usize, end: usize, boundary: Option<usize>) -> usize {
    boundary
        .filter(|boundary| pos < *boundary && *boundary < end)
        .unwrap_or(end)
}

/// `block_align` ends text chunks on paged-attention block boundaries. Hybrid models can only
/// checkpoint recurrent state between forwards, so without this the boundary a prefix lookup asks
/// for is never observed. Callers pass it only for hybrid models; with it, text chunks also end
/// at each [`RECURRENT_PREFIX_CHECKPOINT_INTERVAL`] multiple the chunk loop can checkpoint.
pub(crate) fn build_prompt_chunk_plan(
    total_len: usize,
    prefix_len: usize,
    chunk_size: usize,
    block_align: Option<usize>,
    replay: SpeculativePrefixReplay,
    features: &[MultiModalFeature],
) -> Vec<PromptChunkPlan> {
    let mut pos = prefix_len.min(total_len);
    let mut chunks = Vec::new();
    let reusable_boundary =
        recurrent_checkpoint_boundary(total_len, pos, block_align, replay, features);
    let mut features = features
        .iter()
        .filter(|feature| feature.offset < total_len && feature.end() > pos)
        .collect::<Vec<_>>();
    features.sort_by_key(|feature| feature.offset);

    while pos < total_len {
        if let Some(end) = noncausal_component_end(pos, total_len, &features) {
            let end = cap_at_reusable_boundary(pos, end, reusable_boundary);
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
            });
            pos = end;
            continue;
        }

        if let Some(end) = causal_unsplittable_component_end(pos, total_len, &features) {
            let end = features
                .iter()
                .filter(|feature| {
                    feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                        && feature.offset > pos
                        && feature.offset < end
                })
                .map(|feature| feature.offset)
                .min()
                .unwrap_or(end);
            let end = cap_at_reusable_boundary(pos, end, reusable_boundary);
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy: MultimodalAttentionPolicy::Causal,
            });
            pos = end;
            continue;
        }

        let active_features = features
            .iter()
            .filter(|feature| feature.offset <= pos && feature.end() > pos)
            .collect::<Vec<_>>();
        if !active_features.is_empty() {
            let attention_policy = if active_features
                .iter()
                .any(|feature| feature.attention_policy == MultimodalAttentionPolicy::NonCausal)
            {
                MultimodalAttentionPolicy::NonCausal
            } else {
                MultimodalAttentionPolicy::Causal
            };
            let next_feature_start = features
                .iter()
                .filter(|feature| feature.offset > pos)
                .map(|feature| feature.offset)
                .min()
                .unwrap_or(total_len);
            let next_feature_end = active_features
                .iter()
                .map(|feature| feature.end())
                .min()
                .unwrap_or(total_len);
            let end = next_feature_start
                .min(next_feature_end)
                .min(pos.saturating_add(chunk_size))
                .min(total_len);
            let end = cap_at_reusable_boundary(pos, end, reusable_boundary);
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy,
            });
            pos = end;
            continue;
        }

        let next_feature_start = features
            .iter()
            .filter(|feature| feature.offset > pos)
            .map(|feature| feature.offset)
            .min()
            .unwrap_or(total_len);
        let mut end = (pos + chunk_size).min(next_feature_start).min(total_len);
        if let Some(block_size) = block_align.filter(|size| *size > 0) {
            let aligned = end / block_size * block_size;
            if aligned > pos && aligned < end {
                end = aligned;
            }
            // No media overlaps this chunk, so only replay can move a checkpoint off the
            // interval; the chunk loop snapshots exactly where this check holds.
            let interval = (pos / RECURRENT_PREFIX_CHECKPOINT_INTERVAL + 1)
                * RECURRENT_PREFIX_CHECKPOINT_INTERVAL;
            if interval < end
                && clamp_speculative_prefix_cache_hit(interval, block_size, replay) == interval
            {
                end = interval;
            }
        }
        end = cap_at_reusable_boundary(pos, end, reusable_boundary);
        chunks.push(PromptChunkPlan {
            start: pos,
            end,
            attention_policy: MultimodalAttentionPolicy::Causal,
        });
        pos = end;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::MultimodalKind;

    fn feature(
        offset: usize,
        length: usize,
        attention_policy: MultimodalAttentionPolicy,
    ) -> MultiModalFeature {
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset,
            length,
            attention_policy,
            splittable: false,
        }
    }

    fn policies(chunks: Vec<PromptChunkPlan>) -> Vec<(usize, usize, MultimodalAttentionPolicy)> {
        chunks
            .into_iter()
            .map(|chunk| (chunk.start, chunk.end, chunk.attention_policy))
            .collect()
    }

    #[test]
    fn chunk_groups_do_not_mix_final_and_nonfinal_sequences() {
        let plans = vec![
            vec![PromptChunkPlan {
                start: 0,
                end: 4,
                attention_policy: MultimodalAttentionPolicy::Causal,
            }],
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 2,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 2,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
        ];

        assert_eq!(
            next_prompt_chunk_group(&[0, 0], &plans, false),
            Some((vec![0], MultimodalAttentionPolicy::Causal, true))
        );
        assert_eq!(
            next_prompt_chunk_group(&[1, 0], &plans, false),
            Some((vec![1], MultimodalAttentionPolicy::Causal, false))
        );
    }

    #[test]
    fn uniform_chunk_group_separates_unequal_final_tails() {
        let plans = vec![
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 4,
                    end: 5,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 4,
                    end: 7,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
        ];

        assert_eq!(
            next_prompt_chunk_group(&[1, 1], &plans, false),
            Some((vec![0, 1], MultimodalAttentionPolicy::Causal, true))
        );
        assert_eq!(
            next_prompt_chunk_group(&[1, 1], &plans, true),
            Some((vec![0], MultimodalAttentionPolicy::Causal, true))
        );
    }

    #[test]
    fn keeps_media_spans_policy_homogeneous() {
        let chunks = build_prompt_chunk_plan(
            25,
            0,
            8,
            None,
            SpeculativePrefixReplay::NotRequired,
            &[feature(10, 6, MultimodalAttentionPolicy::NonCausal)],
        );

        assert_eq!(
            chunks,
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 8,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 8,
                    end: 10,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 10,
                    end: 16,
                    attention_policy: MultimodalAttentionPolicy::NonCausal,
                },
                PromptChunkPlan {
                    start: 16,
                    end: 24,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 24,
                    end: 25,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ]
        );
    }

    #[test]
    fn only_splittable_causal_media_respects_chunk_size() {
        let unsplittable = feature(2, 10, MultimodalAttentionPolicy::Causal);
        let mut splittable = unsplittable.clone();
        splittable.splittable = true;

        assert_eq!(
            policies(build_prompt_chunk_plan(
                16,
                0,
                4,
                None,
                SpeculativePrefixReplay::NotRequired,
                &[unsplittable],
            )),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
        assert_eq!(
            policies(build_prompt_chunk_plan(
                16,
                0,
                4,
                None,
                SpeculativePrefixReplay::NotRequired,
                &[splittable],
            )),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 6, MultimodalAttentionPolicy::Causal),
                (6, 10, MultimodalAttentionPolicy::Causal),
                (10, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn overlapping_unsplittable_causal_media_keeps_its_atomic_boundary() {
        let unsplittable = feature(2, 10, MultimodalAttentionPolicy::Causal);
        let mut splittable = feature(2, 4, MultimodalAttentionPolicy::Causal);
        splittable.splittable = true;

        assert_eq!(
            policies(build_prompt_chunk_plan(
                16,
                0,
                4,
                None,
                SpeculativePrefixReplay::NotRequired,
                &[unsplittable, splittable],
            )),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn overlapping_noncausal_media_is_one_atomic_chunk() {
        let chunks = build_prompt_chunk_plan(
            20,
            0,
            8,
            None,
            SpeculativePrefixReplay::NotRequired,
            &[
                feature(4, 8, MultimodalAttentionPolicy::NonCausal),
                feature(8, 4, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 4, MultimodalAttentionPolicy::Causal),
                (4, 12, MultimodalAttentionPolicy::NonCausal),
                (12, 20, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn transitively_overlapping_noncausal_media_is_one_atomic_chunk() {
        let chunks = build_prompt_chunk_plan(
            24,
            0,
            8,
            None,
            SpeculativePrefixReplay::NotRequired,
            &[
                feature(4, 6, MultimodalAttentionPolicy::NonCausal),
                feature(8, 6, MultimodalAttentionPolicy::NonCausal),
                feature(13, 5, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 4, MultimodalAttentionPolicy::Causal),
                (4, 18, MultimodalAttentionPolicy::NonCausal),
                (18, 24, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn mixed_overlapping_policies_use_non_causal() {
        let chunks = build_prompt_chunk_plan(
            14,
            0,
            8,
            None,
            SpeculativePrefixReplay::NotRequired,
            &[
                feature(2, 8, MultimodalAttentionPolicy::Causal),
                feature(4, 4, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 4, MultimodalAttentionPolicy::Causal),
                (4, 8, MultimodalAttentionPolicy::NonCausal),
                (8, 10, MultimodalAttentionPolicy::Causal),
                (10, 14, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn block_align_splits_the_prompt_tail() {
        let spans = |plan: Vec<PromptChunkPlan>| {
            plan.into_iter()
                .map(|chunk| (chunk.start, chunk.end))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            spans(build_prompt_chunk_plan(
                65,
                0,
                4096,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            )),
            vec![(0, 64), (64, 65)]
        );
        assert_eq!(
            spans(build_prompt_chunk_plan(
                64,
                0,
                4096,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            )),
            vec![(0, 32), (32, 64)]
        );
        // Shorter than one block: nothing to snapshot, so do not split.
        assert_eq!(
            spans(build_prompt_chunk_plan(
                20,
                0,
                4096,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            )),
            vec![(0, 20)]
        );
        assert_eq!(
            spans(build_prompt_chunk_plan(
                6712,
                0,
                4096,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            )),
            vec![(0, 4096), (4096, 6688), (6688, 6712)]
        );
        assert_eq!(
            spans(build_prompt_chunk_plan(
                65,
                0,
                4096,
                None,
                SpeculativePrefixReplay::NotRequired,
                &[],
            )),
            vec![(0, 65)]
        );
    }

    #[test]
    fn block_aligned_long_prompt_exposes_the_maximum_reusable_prefix() {
        let spans = build_prompt_chunk_plan(
            65_536,
            0,
            4096,
            Some(32),
            SpeculativePrefixReplay::NotRequired,
            &[],
        )
        .into_iter()
        .map(|chunk| (chunk.start, chunk.end))
        .collect::<Vec<_>>();

        assert_eq!(spans.len(), 17);
        assert_eq!(
            spans[spans.len() - 2..],
            [(61_440, 65_504), (65_504, 65_536)]
        );
    }

    #[test]
    fn target_checkpoint_boundary_handles_aligned_and_unaligned_prompts() {
        assert_eq!(
            recurrent_checkpoint_boundary(
                65_536,
                0,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            ),
            Some(65_504)
        );
        assert_eq!(
            recurrent_checkpoint_boundary(
                65_537,
                0,
                Some(32),
                SpeculativePrefixReplay::NotRequired,
                &[],
            ),
            Some(65_536)
        );
    }

    #[test]
    fn suffix_replay_checkpoint_boundary_handles_aligned_and_unaligned_prompts() {
        let replay = SpeculativePrefixReplay::Suffix(2048);
        assert_eq!(
            recurrent_checkpoint_boundary(65_536, 0, Some(32), replay, &[]),
            Some(63_456)
        );
        assert_eq!(
            recurrent_checkpoint_boundary(65_537, 0, Some(32), replay, &[]),
            Some(63_488)
        );

        let aligned = build_prompt_chunk_plan(65_536, 0, 4096, Some(32), replay, &[]);
        let unaligned = build_prompt_chunk_plan(65_537, 0, 4096, Some(32), replay, &[]);
        assert!(aligned.iter().any(|chunk| chunk.end == 63_456));
        assert!(unaligned.iter().any(|chunk| chunk.end == 63_488));
    }

    #[test]
    fn full_replay_has_no_recurrent_checkpoint_boundary() {
        assert_eq!(
            recurrent_checkpoint_boundary(65_536, 0, Some(32), SpeculativePrefixReplay::Full, &[],),
            None
        );
    }

    #[test]
    fn reusable_prefix_boundary_stays_outside_atomic_media() {
        let chunks = build_prompt_chunk_plan(
            64,
            0,
            32,
            Some(8),
            SpeculativePrefixReplay::NotRequired,
            &[feature(50, 10, MultimodalAttentionPolicy::Causal)],
        );
        let spans = chunks
            .into_iter()
            .map(|chunk| (chunk.start, chunk.end))
            .collect::<Vec<_>>();

        assert!(spans.iter().any(|(_, end)| *end == 48));
        assert!(!spans.iter().any(|(_, end)| 50 < *end && *end < 60));
    }

    #[test]
    fn replay_boundary_normalizes_transitive_media_in_one_sorted_pass() {
        let features = [
            feature(50, 10, MultimodalAttentionPolicy::Causal),
            feature(45, 4, MultimodalAttentionPolicy::Causal),
        ];
        assert_eq!(
            effective_recurrent_prefix_boundary(
                56,
                0,
                8,
                SpeculativePrefixReplay::NotRequired,
                &features,
            ),
            Some(40)
        );
        assert_eq!(
            effective_recurrent_prefix_boundary(
                56,
                0,
                8,
                SpeculativePrefixReplay::Suffix(8),
                &features,
            ),
            Some(32)
        );
    }

    fn chunk_spans(chunks: Vec<PromptChunkPlan>) -> Vec<(usize, usize)> {
        chunks
            .into_iter()
            .map(|chunk| (chunk.start, chunk.end))
            .collect()
    }

    /// The chunk ends the pipeline chunk loop snapshots for a hybrid plan at block size 32: the
    /// final reusable boundary, and each interval boundary its eligibility leaves at the chunk end.
    fn checkpoint_ends(
        total_len: usize,
        prefix_len: usize,
        chunk_size: usize,
        replay: SpeculativePrefixReplay,
        features: &[MultiModalFeature],
    ) -> Vec<usize> {
        let final_boundary =
            recurrent_checkpoint_boundary(total_len, prefix_len, Some(32), replay, features);
        build_prompt_chunk_plan(
            total_len,
            prefix_len,
            chunk_size,
            Some(32),
            replay,
            features,
        )
        .into_iter()
        .filter(|chunk| {
            final_boundary == Some(chunk.end)
                || (is_recurrent_interval_boundary(chunk.end, total_len)
                    && effective_recurrent_prefix_boundary(
                        chunk.end,
                        chunk.start,
                        32,
                        replay,
                        features,
                    ) == Some(chunk.end))
        })
        .map(|chunk| chunk.end)
        .collect()
    }

    #[test]
    fn recurrent_interval_boundary_excludes_the_prompt_end() {
        assert!(is_recurrent_interval_boundary(4096, 4097));
        assert!(is_recurrent_interval_boundary(8192, 9586));
        assert!(!is_recurrent_interval_boundary(4096, 4096));
        assert!(!is_recurrent_interval_boundary(0, 9586));
        assert!(!is_recurrent_interval_boundary(4064, 9586));
        // The paged attention block sizes it accepts (`SUPPORTED_BLOCK_SIZE`).
        assert!([8, 16, 32]
            .iter()
            .all(|block_size| RECURRENT_PREFIX_CHECKPOINT_INTERVAL.is_multiple_of(*block_size)));
    }

    #[test]
    fn hybrid_plans_end_chunks_at_recurrent_checkpoint_intervals() {
        let replay = SpeculativePrefixReplay::NotRequired;
        // 9,586 tokens at block size 32: the final reusable boundary is 9,568.
        for chunk_size in [512, 4096, 8192, 16_384] {
            let plan = build_prompt_chunk_plan(9586, 0, chunk_size, Some(32), replay, &[]);
            assert_eq!(plan.first().map(|chunk| chunk.start), Some(0));
            assert_eq!(plan.last().map(|chunk| chunk.end), Some(9586));
            assert!(plan.windows(2).all(|pair| pair[0].end == pair[1].start));
            assert!(plan
                .iter()
                .all(|chunk| chunk.start < chunk.end && chunk.end - chunk.start <= chunk_size));
            let ends = plan.iter().map(|chunk| chunk.end).collect::<Vec<_>>();
            for boundary in [4096, 8192, 9568, 9586] {
                assert!(
                    ends.contains(&boundary),
                    "chunk_size={chunk_size}: no chunk ends at {boundary}: {ends:?}"
                );
            }
            assert_eq!(
                checkpoint_ends(9586, 0, chunk_size, replay, &[]),
                [4096, 8192, 9568]
            );
        }
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                9586,
                0,
                16_384,
                Some(32),
                replay,
                &[]
            )),
            [(0, 4096), (4096, 8192), (8192, 9568), (9568, 9586)]
        );

        // A restored prefix off the 512-token grid still stops at the next interval.
        let resumed = build_prompt_chunk_plan(9586, 1056, 512, Some(32), replay, &[]);
        assert!(resumed
            .iter()
            .any(|chunk| (chunk.start, chunk.end) == (3616, 4096)));
        assert_eq!(
            checkpoint_ends(9586, 1056, 512, replay, &[]),
            [4096, 8192, 9568]
        );
        assert_eq!(checkpoint_ends(9586, 4096, 512, replay, &[]), [8192, 9568]);
    }

    #[test]
    fn interval_split_leaves_other_plans_unchanged() {
        let replay = SpeculativePrefixReplay::NotRequired;
        // No interval inside the prompt, or the interval already is the final boundary.
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                4000,
                0,
                16_384,
                Some(32),
                replay,
                &[]
            )),
            [(0, 3968), (3968, 4000)]
        );
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                4096,
                0,
                16_384,
                Some(32),
                replay,
                &[]
            )),
            [(0, 4064), (4064, 4096)]
        );
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                4097,
                0,
                16_384,
                Some(32),
                replay,
                &[]
            )),
            [(0, 4096), (4096, 4097)]
        );
        // Callers align only hybrid plans; without alignment nothing is split.
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(9586, 0, 16_384, None, replay, &[])),
            [(0, 9586)]
        );
        // Replay moves every checkpoint off the interval, so the plan is not split there.
        let suffix = SpeculativePrefixReplay::Suffix(2048);
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                9586,
                0,
                16_384,
                Some(32),
                suffix,
                &[]
            )),
            [(0, 7520), (7520, 9568), (9568, 9586)]
        );
        assert_eq!(checkpoint_ends(9586, 0, 512, suffix, &[]), [7520]);
        let full = SpeculativePrefixReplay::Full;
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                9586,
                0,
                16_384,
                Some(32),
                full,
                &[]
            )),
            [(0, 9568), (9568, 9586)]
        );
        assert!(checkpoint_ends(9586, 0, 512, full, &[]).is_empty());
    }

    #[test]
    fn interval_inside_atomic_media_is_not_a_chunk_end() {
        let replay = SpeculativePrefixReplay::NotRequired;
        let features = [feature(4000, 200, MultimodalAttentionPolicy::Causal)];
        assert_eq!(
            chunk_spans(build_prompt_chunk_plan(
                9586,
                0,
                16_384,
                Some(32),
                replay,
                &features
            )),
            [
                (0, 4000),
                (4000, 4200),
                (4200, 8192),
                (8192, 9568),
                (9568, 9586)
            ]
        );
        assert_eq!(
            checkpoint_ends(9586, 0, 16_384, replay, &features),
            [8192, 9568]
        );
    }
}
