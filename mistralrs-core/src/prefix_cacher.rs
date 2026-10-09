use candle_core::{Device, Result, Tensor};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::{debug, info};

use crate::{
    kv_cache::RecurrentStateSnapshot, paged_attention::block_hash::BlockHash, pipeline::KvCache,
    sequence::Sequence,
};

/// Host memory the recurrent-state snapshot store may hold, in bytes.
///
/// A hybrid model's snapshot set is one gather of every recurrent layer's
/// state: for a 48-layer gated-delta-net stack of 48 heads x 128 x 128 f32
/// that is ~146 MiB per retained prefix. Keeping those sets resident on the
/// device — which is what this store used to do — spends VRAM that the paged
/// KV pool and the prefill workspace both need, and it spends it in a way that
/// byte arithmetic does not predict: a resident store fragments the device
/// allocator, so long prompts fail to find a contiguous workspace while the
/// totals still say they fit. Host memory has neither problem, and a set
/// crosses PCIe 4.0 x16 in tens of milliseconds against the tens of seconds of
/// prefill that a prefix hit saves. So residency lives on the host and the
/// bound is stated in bytes of RAM rather than in a sequence count, which never
/// described what the store actually held.
///
/// 64 GiB is ~440 sets of the shape above — far more retention than any served
/// traffic pattern has been measured to need, and a modest share of the RAM a
/// machine large enough to serve such a model has. It is a ceiling that keeps
/// the store from being unbounded, not a target: the store only ever holds what
/// traffic inserts. Set it to 0 to keep no recurrent snapshots at all.
pub const DEFAULT_PREFIX_CACHE_HOST_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[derive(PartialEq, Eq, Debug, Hash)]
struct Tokens(Vec<u32>);

impl Tokens {
    /// Returns the length of the common prefix shared with `other`.
    fn shared_prefix_len(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    /// Recurrent state snapshots for hybrid models (one per recurrent layer)
    recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
    video_hashes: Option<Vec<u64>>,
}

impl CacheElement {
    fn can_rewind_to(&self, len: usize) -> bool {
        self.cache
            .iter()
            .flatten()
            .all(|layer| layer.try_set_len(len).is_ok())
    }
}

/// A retained snapshot set, held in host memory with its footprint precomputed
/// so the store can stay inside its byte bound without re-walking tensors.
struct HostSnapshotSet {
    snapshots: Vec<RecurrentStateSnapshot>,
    bytes: usize,
}

fn tensor_bytes(tensor: &Tensor) -> usize {
    tensor.elem_count() * tensor.dtype().size_in_bytes()
}

fn snapshot_set_bytes(snapshots: &[RecurrentStateSnapshot]) -> usize {
    snapshots
        .iter()
        .map(|snapshot| {
            tensor_bytes(&snapshot.conv_state) + tensor_bytes(&snapshot.recurrent_state)
        })
        .sum()
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct SnapshotWriteOutcome {
    pub written_count: u64,
    pub written_bytes: u64,
    pub evicted_entries: u64,
    pub evicted_bytes: u64,
}

impl SnapshotWriteOutcome {
    fn record_eviction(&mut self, bytes: usize) {
        self.evicted_entries = self.evicted_entries.saturating_add(1);
        self.evicted_bytes = self
            .evicted_bytes
            .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
    }
}

fn observe_host_copy<T>(
    observation: Option<&mut crate::EngineObservation>,
    start: &std::time::Instant,
    copy: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let copied = copy();
    if let Some(observation) = observation {
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        observation.snapshot_host_copy_ns = Some(
            observation
                .snapshot_host_copy_ns
                .unwrap_or(0)
                .saturating_add(elapsed),
        );
        if copied.is_err() {
            observation.snapshot_write_failed();
        }
    }
    copied
}

pub(crate) fn snapshot_host_capacity_bytes(configured: u64) -> usize {
    usize::try_from(configured).unwrap_or(usize::MAX)
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<Tokens, CacheElement>,
    /// Recurrent-state snapshots keyed by paged block-hash prefix. Every value
    /// here is host-resident; see [`DEFAULT_PREFIX_CACHE_HOST_BYTES`].
    paged_recurrent_caches: IndexMap<Vec<BlockHash>, HostSnapshotSet>,
    /// Live host footprint of `paged_recurrent_caches`, maintained on every
    /// insert and eviction.
    paged_recurrent_host_bytes: usize,
    /// Ceiling for `paged_recurrent_host_bytes`; the store is trimmed
    /// oldest-first until it fits.
    paged_recurrent_host_capacity_bytes: usize,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        videos_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
}

impl PrefixCacheManagerV2 {
    pub fn new(
        n_on_device: usize,
        no_prefix_cache: bool,
        has_paged_attention: bool,
        host_capacity_bytes: u64,
    ) -> Self {
        if !no_prefix_cache && !has_paged_attention {
            info!("Prefix caching enabled (sequence-level, non-paged attention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        let paged_recurrent_host_capacity_bytes = snapshot_host_capacity_bytes(host_capacity_bytes);
        if !no_prefix_cache && has_paged_attention {
            info!(
                "Recurrent-state prefix snapshots are host-resident, bounded at {} MiB of RAM.",
                paged_recurrent_host_capacity_bytes / (1024 * 1024)
            );
        }
        PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            paged_recurrent_caches: IndexMap::new(),
            paged_recurrent_host_bytes: 0,
            paged_recurrent_host_capacity_bytes,
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return;
        }

        // For paged attention, prefix caching is handled by the KVCacheManager.
        // PrefixCacheManagerV2 only handles non-paged attention caching.
        if !self.has_paged_attention {
            let cache = seq.normal_cache().to_vec();

            self.caches.insert(
                seq.get_toks().to_vec().into(),
                CacheElement {
                    cache,
                    recurrent_snapshots,
                    image_hashes: seq.image_hashes().map(|x| x.to_vec()),
                    audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
                    video_hashes: seq.video_hashes().map(|x| x.to_vec()),
                },
            );
        }
    }

    /// Evict the caches. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        let mut n_on_device = 0;
        for cache in self.caches.values() {
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Shared { .. } => continue,
            };

            if !matches!(cache_device, Device::Cpu) {
                n_on_device += 1;
            }
        }
        let mut n_evicted = 0;
        // Intentionally evict the first ones first, as they are the oldest
        for cache in self.caches.values_mut() {
            if n_on_device - n_evicted <= self.n_on_device {
                break;
            }
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Shared { .. } => continue,
            };

            if !matches!(cache_device, Device::Cpu) {
                cache.cache.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();
        self.caches.clear();
        self.paged_recurrent_caches.clear();
        self.paged_recurrent_host_bytes = 0;
        Ok(len)
    }

    /// Add a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// This is used by hybrid models to restore recurrent states alongside paged KV prefix hits.
    ///
    /// The set is copied to host memory before it is retained, so the device
    /// keeps only the set being gathered here and the set being scattered back
    /// on a hit. `to_device` is a no-op for a snapshot that is already
    /// host-resident, so a CPU pipeline pays nothing for this.
    pub fn add_paged_recurrent_prefix(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
    ) -> Result<()> {
        self.add_paged_recurrent_inner(key, snapshots, None).map(|_| ())
    }

    fn add_paged_recurrent_inner(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
        mut observation: Option<&mut crate::EngineObservation>,
    ) -> Result<SnapshotWriteOutcome> {
        if let Some(observation) = observation.as_deref_mut() {
            observation.begin_snapshot_write();
        }
        if self.no_prefix_cache
            || !self.has_paged_attention
            || self.paged_recurrent_host_capacity_bytes == 0
            || key.is_empty()
            || snapshots.is_empty()
        {
            return Ok(SnapshotWriteOutcome::default());
        }

        let offload_start = std::time::Instant::now();
        let host_snapshots = observe_host_copy(observation.as_deref_mut(), &offload_start, || {
            snapshots
                .iter()
                .map(|snapshot| {
                    Ok(RecurrentStateSnapshot {
                        conv_state: snapshot.conv_state.to_device(&Device::Cpu)?,
                        recurrent_state: snapshot.recurrent_state.to_device(&Device::Cpu)?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let bytes = snapshot_set_bytes(&host_snapshots);
        debug!(
            "Offloaded a {} MiB recurrent snapshot set to host memory in {:.1} ms",
            bytes / (1024 * 1024),
            offload_start.elapsed().as_secs_f64() * 1000.0
        );

        let observed = mistralrs_quant::serving_observation::enabled();
        let mut outcome = SnapshotWriteOutcome::default();
        if observed {
            outcome.written_count = 1;
            outcome.written_bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        }

        // Maintain LRU order by reinserting on update.
        if let Some(replaced) = self.paged_recurrent_caches.shift_remove(&key) {
            self.paged_recurrent_host_bytes = self
                .paged_recurrent_host_bytes
                .saturating_sub(replaced.bytes);
        }
        self.paged_recurrent_caches.insert(
            key,
            HostSnapshotSet {
                snapshots: host_snapshots,
                bytes,
            },
        );
        self.paged_recurrent_host_bytes = self.paged_recurrent_host_bytes.saturating_add(bytes);

        // Intentionally evict the oldest first. A set larger than the whole
        // bound evicts itself rather than being retained over the bound.
        while self.paged_recurrent_host_bytes > self.paged_recurrent_host_capacity_bytes {
            let Some((_, evicted)) = self.paged_recurrent_caches.shift_remove_index(0) else {
                break;
            };
            self.paged_recurrent_host_bytes = self
                .paged_recurrent_host_bytes
                .saturating_sub(evicted.bytes);
            if observed {
                outcome.record_eviction(evicted.bytes);
            }
        }
        if let Some(observation) = observation {
            observation.record_snapshot_write(&outcome);
        }
        Ok(outcome)
    }

    pub(crate) fn add_paged_recurrent_observed(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
        observation: Option<&mut crate::EngineObservation>,
    ) -> Result<()> {
        let observation = observation.filter(|_| mistralrs_quant::serving_observation::enabled());
        self.add_paged_recurrent_inner(key, snapshots, observation)
            .map(|_| ())
    }

    /// Lookup a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// Returns a cloned snapshot and updates LRU order.
    ///
    /// The returned snapshots are host-resident; restoring them into a device
    /// pool is what copies them back.
    pub fn get_paged_recurrent_prefix(
        &mut self,
        key: &[BlockHash],
    ) -> Option<Vec<RecurrentStateSnapshot>> {
        if self.no_prefix_cache || !self.has_paged_attention || key.is_empty() {
            return None;
        }

        let key = key.to_vec();
        let entry = self.paged_recurrent_caches.shift_remove(&key)?;
        let out = entry.snapshots.clone();
        self.paged_recurrent_caches.insert(key, entry);
        Some(out)
    }

    pub(crate) fn recurrent_observation(&self, observation: &mut crate::EngineObservation) {
        if !mistralrs_quant::serving_observation::enabled() {
            return;
        }
        observation.snapshot_host_bytes = Some(self.paged_recurrent_host_bytes);
        observation.snapshot_entries = Some(self.paged_recurrent_caches.len());
        observation.snapshot_capacity_bytes = Some(self.paged_recurrent_host_capacity_bytes);
        observation.snapshot_placement = Some("host".into());
        if self.no_prefix_cache
            || !self.has_paged_attention
            || self.paged_recurrent_host_capacity_bytes == 0
        {
            observation.recurrent_outcome = Some("disabled".into());
        }
    }

    /// Host memory currently held by retained recurrent-state snapshots.
    pub fn paged_recurrent_host_bytes(&self) -> usize {
        self.paged_recurrent_host_bytes
    }

    pub fn get_longest_paged_recurrent_prefix(
        &mut self,
        block_hashes: &[BlockHash],
        max_blocks: usize,
    ) -> Option<(usize, Vec<RecurrentStateSnapshot>)> {
        if self.no_prefix_cache || !self.has_paged_attention {
            return None;
        }

        let max_blocks = max_blocks.min(block_hashes.len());
        for n_blocks in (1..=max_blocks).rev() {
            if let Some(snapshots) = self.get_paged_recurrent_prefix(&block_hashes[..n_blocks]) {
                return Some((n_blocks, snapshots));
            }
        }
        None
    }

    pub(crate) fn lookup_paged_recurrent_observed(
        &mut self,
        block_hashes: &[BlockHash],
        max_blocks: usize,
        observation: &mut crate::EngineObservation,
    ) -> Option<(usize, Vec<RecurrentStateSnapshot>)> {
        let start = mistralrs_quant::serving_observation::enabled().then(std::time::Instant::now);
        self.recurrent_observation(observation);
        let matching = self.get_longest_paged_recurrent_prefix(block_hashes, max_blocks);
        if let Some(start) = start {
            observation.recurrent_lookup_sec = Some(start.elapsed().as_secs_f64());
            if matching.is_none() {
                if observation.recurrent_outcome.as_deref() != Some("disabled") {
                    observation.recurrent_outcome = Some("not_retained".into());
                }
                observation.recurrent_restored_tokens = Some(0);
            }
        }
        matching
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
        video_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if self.has_paged_attention {
            // For paged attention, prefix caching is handled by the KVCacheManager.
            // PrefixCacheManagerV2 only handles non-paged attention caching.
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement, usize, usize, usize)> = None;
        for (k, v) in &self.caches {
            let match_len = toks.shared_prefix_len(k);
            if match_len == 0 {
                continue;
            }

            let images_match_until = match image_hashes {
                Some(input_hashes) => match &v.image_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            let audios_match_until = match audio_hashes {
                Some(input_hashes) => match &v.audio_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            // Vision/audio models can use repeated placeholder tokens (e.g. <image_soft_token>)
            // that are identical regardless of the actual image/audio content. This means
            // token-level matching can extend through image/audio regions even when the
            // underlying content differs, leaving stale vision/audio encoder outputs in the
            // KV cache. Skip entries where any overlapping images or audios diverge.
            let cached_image_count = v.image_hashes.as_ref().map_or(0, |h| h.len());
            let input_image_count = image_hashes.map_or(0, |h| h.len());
            if images_match_until < input_image_count.min(cached_image_count) {
                continue;
            }

            let cached_audio_count = v.audio_hashes.as_ref().map_or(0, |h| h.len());
            let input_audio_count = audio_hashes.map_or(0, |h| h.len());
            if audios_match_until < input_audio_count.min(cached_audio_count) {
                continue;
            }

            let videos_match_until = match video_hashes {
                Some(input_hashes) => match &v.video_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            let cached_video_count = v.video_hashes.as_ref().map_or(0, |h| h.len());
            let input_video_count = video_hashes.map_or(0, |h| h.len());
            if videos_match_until < input_video_count.min(cached_video_count) {
                continue;
            }

            // The cache holds kv only for forwarded positions; a finished sequence's
            // final sampled token has no kv row, so clamp the text match to coverage.
            // Shared layers mirror their owner and always report zero, so skip them.
            let cache_len = v
                .cache
                .iter()
                .flatten()
                .filter(|l| !matches!(l, KvCache::Shared { .. }))
                .map(|l| l.current_seq_len())
                .min()
                .unwrap_or(0);
            let match_len = match_len.min(cache_len);
            if match_len == 0 {
                continue;
            }

            // Sliding/rotating caches only retain a fixed tail. If a cache has already
            // truncated older tokens, it can still safely serve an exact extension of the
            // cached prefix, but it cannot be rewound to an earlier logical length. Skip such
            // candidates here so a rolled-over cache does not block a shorter valid prefix hit.
            if !v.can_rewind_to(match_len) {
                continue;
            }

            if best_match
                .as_ref()
                .is_none_or(|(len, _, _, _, _)| match_len > *len)
            {
                best_match = Some((
                    match_len,
                    v,
                    images_match_until,
                    audios_match_until,
                    videos_match_until,
                ));
            }
        }

        if let Some((
            match_len,
            cache_element,
            images_match_until,
            audios_match_until,
            videos_match_until,
        )) = best_match
        {
            let new_toks = toks.0[match_len..].to_vec();
            if new_toks.is_empty() {
                return Ok(None);
            }

            let mut cache = cache_element.clone();
            // Count how many input images are not already cached
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(audios_match_until)
            } else {
                0
            };
            let videos_to_keep = if let Some(input_hashes) = video_hashes {
                input_hashes.len().saturating_sub(videos_match_until)
            } else {
                0
            };
            for layer in cache.cache.iter_mut().flatten() {
                if layer.try_set_len(match_len).is_err() {
                    return Ok(None);
                }
            }
            for layer in cache.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            return Ok(Some(MatchingCache::Normal {
                normal: cache.cache,
                recurrent_snapshots: cache.recurrent_snapshots,
                images_to_keep,
                audios_to_keep,
                videos_to_keep,
                toks: new_toks,
                offset: match_len,
            }));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        observe_host_copy, snapshot_set_bytes, CacheElement, MatchingCache, PrefixCacheManagerV2,
        SnapshotWriteOutcome, DEFAULT_PREFIX_CACHE_HOST_BYTES,
    };
    use crate::kv_cache::{KvCache, RecurrentStateSnapshot, RotatingCache, SingleCache};
    use crate::paged_attention::block_hash::{hash_block_tokens, BlockHash};

    fn make_cache_tensor(len: usize) -> candle_core::Result<Tensor> {
        Tensor::zeros((1, 1, len, 1), DType::F32, &Device::Cpu)
    }

    fn make_rotating_kv_cache(
        logical_len: usize,
        sliding_window: usize,
    ) -> candle_core::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = RotatingCache::new(2, sliding_window, sliding_window);
        let mut v = RotatingCache::new(2, sliding_window, sliding_window);
        let _ = k.append(&src)?;
        let _ = v.append(&src)?;
        Ok(KvCache::Rotating { k, v })
    }

    fn make_normal_kv_cache(logical_len: usize) -> candle_core::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = SingleCache::new(2, logical_len, logical_len);
        let mut v = SingleCache::new(2, logical_len, logical_len);
        k.append(&src)?;
        v.append(&src)?;
        Ok(KvCache::Normal { k, v })
    }

    /// A snapshot set shaped like a hybrid model's, only small enough to make
    /// the byte bound easy to state: two recurrent layers of a fixed size.
    fn make_snapshot_set() -> candle_core::Result<Vec<RecurrentStateSnapshot>> {
        (0..2)
            .map(|_| {
                Ok(RecurrentStateSnapshot {
                    conv_state: Tensor::zeros((1, 4, 64), DType::F32, &Device::Cpu)?,
                    recurrent_state: Tensor::zeros((1, 4, 16, 16), DType::F32, &Device::Cpu)?,
                })
            })
            .collect()
    }

    fn make_key(seed: u32, len: usize) -> Vec<BlockHash> {
        let mut parent = None;
        let mut key = Vec::with_capacity(len);
        for block in 0..len {
            #[allow(clippy::cast_possible_truncation)]
            let hash = hash_block_tokens(parent, &[seed, block as u32], None);
            key.push(hash);
            parent = Some(hash);
        }
        key
    }

    #[test]
    fn observation_contract_real_cache_cold_warm_changed_suffix_and_prefix(
    ) -> candle_core::Result<()> {
        use crate::paged_attention::{
            block_hash::compute_block_hashes, kv_cache_manager::KVCacheManager,
        };
        let tokens: Vec<u32> = (1..=12).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        let mut kv = KVCacheManager::new(16, 4, true, vec![0]);
        let mut recurrent =
            PrefixCacheManagerV2::new(1, false, true, DEFAULT_PREFIX_CACHE_HOST_BYTES);
        let mut observation = crate::EngineObservation::default();
        let cold = kv.get_computed_blocks(&hashes, tokens.len());
        observation.prefix_candidate(true, cold.num_computed_tokens);
        assert_eq!(
            observation.cache_outcome.as_deref(),
            Some("no_matching_kv_prefix")
        );
        assert!(recurrent
            .lookup_paged_recurrent_observed(&hashes, 3, &mut observation)
            .is_none());
        assert_eq!(
            observation.recurrent_outcome.as_deref(),
            Some("not_retained")
        );

        assert!(kv.allocate_slots(1, tokens.len(), &[]).is_some());
        kv.cache_blocks(1, &hashes, tokens.len());
        kv.free(1);
        recurrent.add_paged_recurrent_prefix(hashes[..2].to_vec(), make_snapshot_set()?)?;
        recurrent.add_paged_recurrent_prefix(hashes.to_vec(), make_snapshot_set()?)?;
        let warm = kv.get_computed_blocks(&hashes, tokens.len());
        observation.prefix_candidate(true, warm.num_computed_tokens);
        assert_eq!(warm.num_computed_tokens, 8);
        let (boundary, _) = recurrent
            .lookup_paged_recurrent_observed(
                &hashes,
                warm.num_computed_tokens / 4,
                &mut observation,
            )
            .expect("warm recurrent prefix");
        observation.recurrent_restore(true, boundary * 4);
        observation.final_prefix(tokens.len(), boundary * 4);
        assert_eq!(observation.uncached_suffix_tokens, Some(4));

        let extended: Vec<u32> = (1..=13).collect();
        let extended_hashes = compute_block_hashes(&extended, 4, &[], &[]);
        let extended_hit = kv.get_computed_blocks(&extended_hashes, extended.len());
        assert_eq!(extended_hit.num_computed_tokens, 12);
        let (boundary, _) = recurrent
            .lookup_paged_recurrent_observed(&extended_hashes, 3, &mut observation)
            .expect("extended recurrent prefix");
        observation.final_prefix(extended.len(), boundary * 4);
        assert_eq!(observation.uncached_suffix_tokens, Some(1));

        let changed_suffix = [1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102];
        let changed_hashes = compute_block_hashes(&changed_suffix, 4, &[], &[]);
        assert_eq!(
            kv.get_computed_blocks(&changed_hashes, changed_suffix.len())
                .num_computed_tokens,
            8
        );
        assert_eq!(
            recurrent
                .lookup_paged_recurrent_observed(&changed_hashes, 3, &mut observation)
                .expect("shared prefix survives suffix change")
                .0,
            2
        );
        let changed_prefix = [999, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let changed_hashes = compute_block_hashes(&changed_prefix, 4, &[], &[]);
        assert_eq!(
            kv.get_computed_blocks(&changed_hashes, changed_prefix.len())
                .num_computed_tokens,
            0
        );
        assert!(recurrent
            .lookup_paged_recurrent_observed(&changed_hashes, 3, &mut observation)
            .is_none());
        assert_eq!(
            observation.recurrent_outcome.as_deref(),
            Some("not_retained")
        );
        assert_eq!(observation.snapshot_entries, Some(2));
        assert_eq!(observation.snapshot_placement.as_deref(), Some("host"));
        assert!(observation
            .snapshot_host_bytes
            .is_some_and(|bytes| bytes > 0));
        Ok(())
    }

    #[test]
    fn observation_contract_real_cache_disabled_and_multimodal_clamp() -> candle_core::Result<()> {
        use crate::paged_attention::{
            block_hash::{
                clamp_prefix_cache_hit_len, compute_block_hashes, MultiModalFeature,
                MultimodalAttentionPolicy, MultimodalKind,
            },
            kv_cache_manager::KVCacheManager,
        };
        let hashes = compute_block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &[], &[]);
        let kv = KVCacheManager::new(16, 4, false, vec![0]);
        let mut observation = crate::EngineObservation::default();
        observation.prefix_candidate(
            false,
            kv.get_computed_blocks(&hashes, 8).num_computed_tokens,
        );
        assert_eq!(observation.cache_outcome.as_deref(), Some("disabled"));
        for (disabled, capacity) in [(true, DEFAULT_PREFIX_CACHE_HOST_BYTES), (false, 0)] {
            let mut recurrent = PrefixCacheManagerV2::new(1, disabled, true, capacity);
            recurrent.add_paged_recurrent_prefix(hashes.to_vec(), make_snapshot_set()?)?;
            assert!(recurrent
                .lookup_paged_recurrent_observed(&hashes, 2, &mut observation)
                .is_none());
            assert_eq!(observation.recurrent_outcome.as_deref(), Some("disabled"));
        }
        let features = [MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 5,
            length: 4,
            attention_policy: MultimodalAttentionPolicy::NonCausal,
            splittable: false,
        }];
        observation.prefix_candidate(true, 8);
        observation.recurrent_restore(true, 8);
        let clamped = crate::sequence::clamp_prefix_cache_len_for_mm_features(8, 4, &features);
        let clamped = clamp_prefix_cache_hit_len(clamped, 4, &features);
        observation.final_prefix(13, clamped);
        assert_eq!(observation.candidate_prefix_tokens, Some(8));
        assert_eq!(observation.recurrent_restored_tokens, Some(8));
        assert_eq!(observation.reused_prefix_tokens, Some(4));
        assert_eq!(observation.uncached_suffix_tokens, Some(9));
        Ok(())
    }

    #[test]
    fn observation_contract_real_recurrent_restore_failure() -> candle_core::Result<()> {
        use crate::kv_cache::{
            HybridCache, HybridCacheConfig, HybridLayerType, RecurrentLayerConfig,
        };
        let mut cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent],
                max_seq_len: 32,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 4,
                    conv_width: 64,
                    state_dims: vec![4, 16, 16],
                    recurrent_dtype: None,
                },
            },
            DType::F32,
            &Device::Cpu,
        )?;
        let slot = cache.allocate_seq().expect("fixture slot");
        let mut snapshots = make_snapshot_set()?;
        let mut observation = crate::EngineObservation::default();
        observation.recurrent_restore(cache.restore_recurrent_state(slot, &snapshots).is_ok(), 8);
        assert_eq!(observation.recurrent_outcome.as_deref(), Some("restored"));
        snapshots[0].conv_state = Tensor::zeros((1, 4, 1), DType::F32, &Device::Cpu)?;
        observation.recurrent_restore(cache.restore_recurrent_state(slot, &snapshots).is_ok(), 8);
        observation.final_prefix(
            13,
            observation
                .recurrent_restored_tokens
                .expect("known restore result"),
        );
        assert_eq!(
            observation.recurrent_outcome.as_deref(),
            Some("restore_failed")
        );
        assert_eq!(observation.reused_prefix_tokens, Some(0));
        assert_eq!(observation.uncached_suffix_tokens, Some(13));
        Ok(())
    }

    #[test]
    fn observation_contract_snapshot_capacity_eviction_preserves_kv_candidate(
    ) -> candle_core::Result<()> {
        use crate::paged_attention::{
            block_hash::compute_block_hashes, kv_cache_manager::KVCacheManager,
        };
        let set_bytes = snapshot_set_bytes(&make_snapshot_set()?);
        let mut recurrent = PrefixCacheManagerV2::new(1, false, true, (set_bytes * 2) as u64);
        let mut kv = KVCacheManager::new(64, 4, true, vec![0]);
        let prompts: Vec<Vec<u32>> = (0..3)
            .map(|seed| (1..=12).map(|token| token + seed * 100).collect())
            .collect();
        let hashes: Vec<_> = prompts
            .iter()
            .map(|tokens| compute_block_hashes(tokens, 4, &[], &[]))
            .collect();
        let mut write_a = crate::EngineObservation::default();
        assert_eq!(
            kv.get_computed_blocks(&hashes[0], 12).num_computed_tokens,
            0
        );
        assert!(kv.allocate_slots(1, 12, &[]).is_some());
        kv.cache_blocks(1, &hashes[0], 12);
        kv.free(1);
        recurrent.add_paged_recurrent_observed(
            hashes[0][..2].to_vec(),
            make_snapshot_set()?,
            Some(&mut write_a),
        )?;
        assert_eq!(write_a.snapshot_write_count, Some(1));
        assert_eq!(write_a.snapshot_write_bytes, Some(set_bytes as u64));
        assert_eq!(write_a.snapshot_evicted_entries, Some(0));
        assert_eq!(write_a.snapshot_write_measurement_complete, Some(true));

        let mut warm_a = crate::EngineObservation::default();
        let candidate = kv.get_computed_blocks(&hashes[0], 12).num_computed_tokens;
        warm_a.prefix_candidate(true, candidate);
        let (boundary, _) = recurrent
            .lookup_paged_recurrent_observed(&hashes[0], candidate / 4, &mut warm_a)
            .expect("warm A snapshot");
        warm_a.recurrent_restore(true, boundary * 4);
        warm_a.final_prefix(12, boundary * 4);
        assert_eq!(warm_a.reused_prefix_tokens, Some(8));
        assert_eq!(warm_a.snapshot_write_count, None);
        assert_eq!(warm_a.snapshot_write_measurement_complete, None);

        for index in 1..3 {
            assert!(kv.allocate_slots(index + 1, 12, &[]).is_some());
            kv.cache_blocks(index + 1, &hashes[index], 12);
            kv.free(index + 1);
            let mut written = crate::EngineObservation::default();
            recurrent.add_paged_recurrent_observed(
                hashes[index][..2].to_vec(),
                make_snapshot_set()?,
                Some(&mut written),
            )?;
            assert_eq!(written.snapshot_write_count, Some(1));
            assert_eq!(
                written.snapshot_evicted_entries,
                Some(if index == 2 { 1 } else { 0 })
            );
            assert_eq!(
                written.snapshot_evicted_bytes,
                Some(if index == 2 { set_bytes as u64 } else { 0 })
            );
        }
        let mut final_a = crate::EngineObservation::default();
        let candidate = kv.get_computed_blocks(&hashes[0], 12).num_computed_tokens;
        assert_eq!(candidate, 8);
        final_a.prefix_candidate(true, candidate);
        assert!(recurrent
            .lookup_paged_recurrent_observed(&hashes[0], candidate / 4, &mut final_a)
            .is_none());
        final_a.final_prefix(12, 0);
        assert_eq!(final_a.recurrent_outcome.as_deref(), Some("not_retained"));
        assert_eq!(final_a.candidate_prefix_tokens, Some(8));
        assert_eq!(final_a.reused_prefix_tokens, Some(0));
        assert_eq!(final_a.uncached_suffix_tokens, Some(12));
        assert_eq!(final_a.snapshot_evicted_entries, None);
        assert_eq!(recurrent.paged_recurrent_host_bytes(), set_bytes * 2);
        Ok(())
    }

    #[test]
    fn observation_contract_snapshot_disabled_too_small_replacement_and_default(
    ) -> candle_core::Result<()> {
        assert_eq!(DEFAULT_PREFIX_CACHE_HOST_BYTES, 64 * 1024 * 1024 * 1024);
        let default = PrefixCacheManagerV2::new(1, false, true, DEFAULT_PREFIX_CACHE_HOST_BYTES);
        assert_eq!(
            default.paged_recurrent_host_capacity_bytes,
            usize::try_from(DEFAULT_PREFIX_CACHE_HOST_BYTES).unwrap_or(usize::MAX)
        );
        let maximum = PrefixCacheManagerV2::new(1, false, true, u64::MAX);
        assert_eq!(maximum.paged_recurrent_host_capacity_bytes, usize::MAX);
        let set_bytes = snapshot_set_bytes(&make_snapshot_set()?);
        let key = make_key(1, 2);
        let mut disabled =
            PrefixCacheManagerV2::new(1, true, true, DEFAULT_PREFIX_CACHE_HOST_BYTES);
        let mut disabled_observed = crate::EngineObservation::default();
        disabled.add_paged_recurrent_observed(
            key.clone(),
            make_snapshot_set()?,
            Some(&mut disabled_observed),
        )?;
        assert_eq!(disabled_observed.snapshot_write_count, Some(0));
        assert_eq!(disabled_observed.snapshot_host_copy_ns, Some(0));
        assert_eq!(
            disabled_observed.snapshot_write_measurement_complete,
            Some(true)
        );
        for capacity in [0, set_bytes as u64 - 1] {
            let mut cache = PrefixCacheManagerV2::new(1, false, true, capacity);
            let mut observed = crate::EngineObservation::default();
            cache.add_paged_recurrent_observed(
                key.clone(),
                make_snapshot_set()?,
                Some(&mut observed),
            )?;
            assert_eq!(observed.snapshot_write_count, Some(u64::from(capacity > 0)));
            assert_eq!(
                observed.snapshot_write_bytes,
                Some(if capacity > 0 { set_bytes as u64 } else { 0 })
            );
            assert_eq!(
                observed.snapshot_evicted_entries,
                Some(u64::from(capacity > 0))
            );
            assert_eq!(
                observed.snapshot_evicted_bytes,
                observed.snapshot_write_bytes
            );
            assert_eq!(observed.snapshot_write_measurement_complete, Some(true));
            assert_eq!(cache.paged_recurrent_host_bytes(), 0);
            assert!(cache.get_paged_recurrent_prefix(&key).is_none());
            if capacity == 0 {
                assert_eq!(observed.snapshot_host_copy_ns, Some(0));
            }
        }
        let mut cache = PrefixCacheManagerV2::new(1, false, true, set_bytes as u64);
        let mut observed = crate::EngineObservation::default();
        for _ in 0..2 {
            cache.add_paged_recurrent_observed(
                key.clone(),
                make_snapshot_set()?,
                Some(&mut observed),
            )?;
        }
        assert_eq!(observed.snapshot_write_count, Some(2));
        assert_eq!(observed.snapshot_write_bytes, Some((set_bytes * 2) as u64));
        assert_eq!(observed.snapshot_evicted_entries, Some(0));
        assert_eq!(observed.snapshot_evicted_bytes, Some(0));
        assert_eq!(cache.paged_recurrent_host_bytes(), set_bytes);
        let unobserved = crate::EngineObservation::default();
        cache.add_paged_recurrent_observed(key, make_snapshot_set()?, None)?;
        assert_eq!(unobserved.snapshot_write_count, None);
        assert_eq!(unobserved.snapshot_host_copy_ns, None);
        Ok(())
    }

    #[test]
    fn observation_contract_snapshot_failed_copy_is_sticky_and_counters_saturate() {
        let mut observed = crate::EngineObservation::default();
        observed.begin_snapshot_write();
        let failed: candle_core::Result<()> =
            observe_host_copy(Some(&mut observed), &std::time::Instant::now(), || {
                candle_core::bail!("injected copy failure")
            });
        assert!(failed.is_err());
        assert_eq!(observed.snapshot_write_count, Some(0));
        assert_eq!(observed.snapshot_write_measurement_complete, Some(false));
        assert!(observed.snapshot_host_copy_ns.is_some());
        observed.begin_snapshot_write();
        observed.record_snapshot_write(&SnapshotWriteOutcome {
            written_count: u64::MAX,
            written_bytes: u64::MAX,
            evicted_entries: u64::MAX,
            evicted_bytes: u64::MAX,
        });
        observed.record_snapshot_write(&SnapshotWriteOutcome {
            written_count: 1,
            written_bytes: 1,
            evicted_entries: 1,
            evicted_bytes: 1,
        });
        assert_eq!(observed.snapshot_write_count, Some(u64::MAX));
        assert_eq!(observed.snapshot_write_bytes, Some(u64::MAX));
        assert_eq!(observed.snapshot_evicted_entries, Some(u64::MAX));
        assert_eq!(observed.snapshot_evicted_bytes, Some(u64::MAX));
        assert_eq!(observed.snapshot_write_measurement_complete, Some(false));
        observed.snapshot_host_copy_ns = Some(u64::MAX);
        assert_eq!(
            observe_host_copy(Some(&mut observed), &std::time::Instant::now(), || Ok(42))
                .expect("successful copy result"),
            42
        );
        assert_eq!(observed.snapshot_host_copy_ns, Some(u64::MAX));
        assert_eq!(observed.snapshot_write_measurement_complete, Some(false));
        let mut eviction = SnapshotWriteOutcome {
            evicted_entries: u64::MAX,
            evicted_bytes: u64::MAX,
            ..Default::default()
        };
        eviction.record_eviction(1);
        assert_eq!(eviction.evicted_entries, u64::MAX);
        assert_eq!(eviction.evicted_bytes, u64::MAX);
    }

    #[test]
    fn paged_recurrent_store_is_bounded_in_bytes_and_evicts_oldest_first() -> candle_core::Result<()>
    {
        let set_bytes = snapshot_set_bytes(&make_snapshot_set()?);
        assert!(set_bytes > 0);
        // Room for exactly three sets.
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, true, (set_bytes * 3) as u64);

        let keys: Vec<_> = (0..3).map(|seed| make_key(seed, 2)).collect();
        for key in &keys {
            prefix_cacher.add_paged_recurrent_prefix(key.clone(), make_snapshot_set()?)?;
        }
        assert_eq!(prefix_cacher.paged_recurrent_host_bytes(), set_bytes * 3);
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 3);

        // A lookup is a use: it must move that key to the front of the eviction
        // order, so the next insert takes the genuinely oldest entry instead.
        assert!(prefix_cacher.get_paged_recurrent_prefix(&keys[0]).is_some());
        let fourth = make_key(3, 2);
        prefix_cacher.add_paged_recurrent_prefix(fourth.clone(), make_snapshot_set()?)?;

        assert_eq!(prefix_cacher.paged_recurrent_host_bytes(), set_bytes * 3);
        assert!(prefix_cacher.get_paged_recurrent_prefix(&keys[1]).is_none());
        for surviving in [&keys[0], &keys[2], &fourth] {
            assert!(
                prefix_cacher
                    .get_paged_recurrent_prefix(surviving)
                    .is_some(),
                "a recently used key was evicted ahead of an older one"
            );
        }

        Ok(())
    }

    #[test]
    fn retained_recurrent_snapshots_are_host_resident() -> candle_core::Result<()> {
        let mut prefix_cacher =
            PrefixCacheManagerV2::new(1, false, true, DEFAULT_PREFIX_CACHE_HOST_BYTES);
        let key = make_key(7, 3);
        prefix_cacher.add_paged_recurrent_prefix(key.clone(), make_snapshot_set()?)?;

        let restored = prefix_cacher
            .get_paged_recurrent_prefix(&key)
            .expect("the set just inserted must be retained");
        for snapshot in &restored {
            assert!(snapshot.conv_state.device().is_cpu());
            assert!(snapshot.recurrent_state.device().is_cpu());
        }

        Ok(())
    }

    #[test]
    fn a_zero_byte_host_bound_retains_nothing() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, true, 0);
        let key = make_key(11, 2);
        prefix_cacher.add_paged_recurrent_prefix(key.clone(), make_snapshot_set()?)?;

        assert_eq!(prefix_cacher.paged_recurrent_host_bytes(), 0);
        assert!(prefix_cacher.get_paged_recurrent_prefix(&key).is_none());

        Ok(())
    }

    #[test]
    fn skips_rolled_over_rotating_candidate_that_cannot_rewind() -> candle_core::Result<()> {
        let mut prefix_cacher =
            PrefixCacheManagerV2::new(1, false, false, DEFAULT_PREFIX_CACHE_HOST_BYTES);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 9].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(6)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 99],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 5);
                assert_eq!(toks, vec![6, 7, 99]);
            }
            None => panic!("expected a shorter valid prefix-cache hit"),
        }

        Ok(())
    }

    #[test]
    fn allows_exact_extension_from_rolled_over_rotating_cache() -> candle_core::Result<()> {
        let mut prefix_cacher =
            PrefixCacheManagerV2::new(1, false, false, DEFAULT_PREFIX_CACHE_HOST_BYTES);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 10);
                assert_eq!(toks, vec![11]);
            }
            None => panic!("expected exact-extension prefix-cache hit"),
        }

        Ok(())
    }
}
