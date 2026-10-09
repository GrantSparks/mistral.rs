#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, DeviceLocation, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};

use super::config::{LayerType, TextConfig};
use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    gdn::{GatedDeltaNet, GdnConfig, GdnInputProjectionKind, GdnLayerCache},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{self, GemmaRmsNorm, Qwen3VLRotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalLoadingMetadata,
        RecurrentBatchKind,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

const BF16_CONTEXT_MAX_INTEGER: usize = 1_000_000_000;
const BF16_CONTEXT_MAX_LAYER: usize = 4095;
const BF16_CONTEXT_MAX_ORDINAL: usize = 255;

#[derive(Clone, Copy)]
enum Bf16ForwardStage {
    AttentionQkv,
    AttentionSdpa,
    AttentionOutput,
    Gdn,
    MlpGate,
    MlpUp,
    MlpDown,
    FullAttentionLayer,
    LinearAttentionLayer,
    LmHead,
}

impl Bf16ForwardStage {
    fn name(self) -> &'static str {
        match self {
            Self::AttentionQkv => "attention_qkv",
            Self::AttentionSdpa => "attention_sdpa",
            Self::AttentionOutput => "attention_output",
            Self::Gdn => "gdn",
            Self::MlpGate => "mlp_gate",
            Self::MlpUp => "mlp_up",
            Self::MlpDown => "mlp_down",
            Self::FullAttentionLayer => "full_attention_layer",
            Self::LinearAttentionLayer => "linear_attention_layer",
            Self::LmHead => "lm_head",
        }
    }
}

fn forward_error_context(
    error: candle_core::Error,
    layer: usize,
    stage: Bf16ForwardStage,
    input: &Tensor,
) -> candle_core::Error {
    let (device, ordinal) = match input.device().location() {
        DeviceLocation::Cpu => ("cpu", 0),
        DeviceLocation::Cuda { gpu_id } => ("cuda", gpu_id),
        DeviceLocation::Metal { gpu_id } => ("metal", gpu_id),
    };
    if layer > BF16_CONTEXT_MAX_LAYER
        || ordinal > BF16_CONTEXT_MAX_ORDINAL
        || !(1..=4).contains(&input.rank())
        || input
            .dims()
            .iter()
            .chain(input.layout().stride())
            .any(|&value| value > BF16_CONTEXT_MAX_INTEGER)
    {
        return error;
    }
    let dims = input
        .dims()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let strides = input
        .layout()
        .stride()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    error.context(format!(
        "BF16_OP_CONTEXT{{kind=layer;layer={layer};stage={};device={device};ordinal={ordinal};dtype={:?};dims={dims};strides={strides}}}",
        stage.name(), input.dtype()
    ))
}

impl GdnConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
}

// ====================== Full Attention layer with MRoPE ======================

#[allow(dead_code)]
struct FullAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // q_proj outputs num_heads * head_dim * 2 (doubled for gate)
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;

        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm)?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        layer_idx: usize,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q_gate, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj).map_err(
                |error| forward_error_context(error, layer_idx, Bf16ForwardStage::AttentionQkv, x),
            )?;
        // Split q_gate into q and gate
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Reshape to (batch, heads, seq, head_dim)
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = self.rotary_emb.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        // Standard attention
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn
                    .forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        Some(key_cache),
                        Some(value_cache),
                        input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )
                    .map_err(|error| {
                        forward_error_context(error, layer_idx, Bf16ForwardStage::AttentionSdpa, &q)
                    })?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn
                        .forward(
                            &q,
                            &k,
                            &v,
                            attention_mask,
                            None,
                            None,
                            &input_metadata,
                            &self.sdpa_params,
                            Some(flash_params),
                        )
                        .map_err(|error| {
                            forward_error_context(
                                error,
                                layer_idx,
                                Bf16ForwardStage::AttentionSdpa,
                                &q,
                            )
                        })?
                }
            },
            None => {
                let (cache_k, cache_v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &cache_k,
                    &cache_v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )
                .map_err(|error| {
                    forward_error_context(error, layer_idx, Bf16ForwardStage::AttentionSdpa, &q)
                })?
            }
        };

        // Prefill and SDPA paths return head-major rank-4 `[b, h, s, d]`; the paged-decode
        // kernels return token-major rank-3 `[n, h, d]`. The attention-mask state is not a
        // reliable proxy for which one ran: paged prefill chunks and prefix-cache-hit prefills
        // carry `AttentionMask::None` yet still produce rank-4 output. Discriminate on rank.
        y = if y.rank() == 4 {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let res = self.o_proj.forward(&y).map_err(|error| {
            forward_error_context(error, layer_idx, Bf16ForwardStage::AttentionOutput, &y)
        })?;
        Ok(res)
    }
}

// ====================== Dense MLP ======================

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quant_config: &Option<mistralrs_quant::QuantizedConfig>,
        act_fn: crate::layers::Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_size,
            hidden_size,
            quant_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, layer_idx: usize, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs).map_err(|error| {
            forward_error_context(error, layer_idx, Bf16ForwardStage::MlpGate, xs)
        })?;
        let up = self.up_proj.forward(xs).map_err(|error| {
            forward_error_context(error, layer_idx, Bf16ForwardStage::MlpUp, xs)
        })?;
        let activated = crate::ops::mul_and_act(&gate, &up, self.act_fn)?;
        let res = self.down_proj.forward(&activated).map_err(|error| {
            forward_error_context(error, layer_idx, Bf16ForwardStage::MlpDown, &activated)
        })?;
        Ok(res)
    }
}

// ====================== Decoder Layer ======================

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        layer_idx: usize,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(
            layer_idx,
            &x,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(layer_idx, &normed)?;
        ffn_out + residual
    }

    fn forward_linear(
        &self,
        layer_idx: usize,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn
            .forward(&x, cache, batch_kind)
            .map_err(|error| forward_error_context(error, layer_idx, Bf16ForwardStage::Gdn, &x))?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(layer_idx, &normed)?;
        ffn_out + residual
    }
}

// ====================== Text Model ======================

pub struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    pub(super) norm: GemmaRmsNorm,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
}

fn tensor_for_layer<'a>(
    tensors: &'a mut HashMap<DeviceLocation, Tensor>,
    tensor: &Tensor,
    mapper: &dyn DeviceMapper,
    layer_idx: usize,
    device: &Device,
) -> Result<&'a Tensor> {
    match tensors.entry(device.location()) {
        std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
        std::collections::hash_map::Entry::Vacant(entry) => {
            Ok(entry.insert(mapper.map(tensor.clone(), layer_idx)?))
        }
    }
}

fn rotary_for_layer<'a>(
    tensors: &'a mut HashMap<DeviceLocation, (Tensor, Tensor)>,
    cos_sin: &(Tensor, Tensor),
    mapper: &dyn DeviceMapper,
    layer_idx: usize,
    device: &Device,
) -> Result<&'a (Tensor, Tensor)> {
    match tensors.entry(device.location()) {
        std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
        std::collections::hash_map::Entry::Vacant(entry) => Ok(entry.insert((
            mapper.map(cos_sin.0.clone(), layer_idx)?,
            mapper.map(cos_sin.1.clone(), layer_idx)?,
        ))),
    }
}

impl Qwen3_5TextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let vb_m = if vb.contains_tensor("language_model.model.embed_tokens.weight") {
            vb.pp("language_model").pp("model")
        } else {
            vb.pp("model").pp("language_model")
        };

        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let layer_types = cfg.layer_types();

        // Create MRoPE embeddings (one per device, using rot_dim not head_dim)
        let rot_dim = cfg.rot_dim();
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if *layer_type != LayerType::FullAttention {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    Qwen3VLRotaryEmbedding::new(
                        cfg.rope_theta() as f32,
                        rot_dim,
                        device,
                        cfg.mrope_section().to_vec(),
                    )
                    .expect("Failed to create rotary embedding"),
                )
            });
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;

            let layer_impl = match layer_types[layer_idx] {
                LayerType::FullAttention => {
                    let device = mapper
                        .device_for(layer_idx, false)
                        .unwrap_or(&normal_loading_metadata.real_device);
                    let rotary_emb = ropes
                        .get(&device.location())
                        .expect("No RoPE for device location!")
                        .clone();
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_l.pp(layer_idx),
                        cfg,
                        &*mapper,
                        layer_idx,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => LayerImpl::LinearAttention(GatedDeltaNet::load(
                    vb_l.pp(layer_idx),
                    cfg as &dyn GdnConfig,
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    GdnInputProjectionKind::Split,
                )?),
            };

            let input_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_l.pp(layer_idx).pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("post_attention_layernorm"),
                    false,
                ),
            )?;

            let mlp = Mlp::new(
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("mlp"),
                    normal_loading_metadata.loading_isq,
                ),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                &comm,
            )?;

            Ok(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                mlp,
            })
        })?;

        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !tie {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(
                candle_nn::Linear::new(
                    mapper.cast_nm_device(
                        embed_tokens.embeddings(),
                        normal_loading_metadata.loading_isq,
                    )?,
                    None,
                ),
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        };

        // Create pipeline hybrid cache
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.linear_conv_dim(),
                conv_width: cfg.linear_conv_kernel_dim,
                state_dims: vec![
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                ],
                recurrent_dtype: Some(DType::F32),
            },
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new_with_layer_devices(hybrid_cache_config, vb_m.dtype(), |layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
            })
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            layer_types: layer_types.clone(),
            lm_head,
            cache: EitherCache::Hybrid(pipeline_cache),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            mapper,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn supports_cuda_decode_graphs(&self) -> bool {
        // Capture tracks only the input device's stream; staged layer transfers need eager execution.
        self.mapper
            .get_unique_devices()
            .iter()
            .all(|device| device.same_device(&self.device))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: &AttentionMask,
        position_ids: &Tensor,
        _seqlen_offsets: &[usize],
        ctx: &ModelForwardContext<'_>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        if self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention))
            && recurrent_metadata.is_none()
        {
            candle_core::bail!(
                "Hybrid recurrent metadata is required for linear-attention layers."
            );
        }

        // Compute MRoPE cos/sin using first full-attention layer's rotary embedding
        let cos_sin = {
            let first_attn_idx = self
                .layer_types
                .iter()
                .position(|lt| *lt == LayerType::FullAttention)
                .expect("No full attention layer found");
            match &self.layers[first_attn_idx].layer_impl {
                LayerImpl::FullAttention(attn) => {
                    attn.rotary_emb.compute_cos_sin(position_ids, xs.dtype())?
                }
                _ => unreachable!(),
            }
        };

        let attention_mask = DeviceMappedMask::new(attention_mask.clone(), &*self.mapper)?;

        // Precompute deepstack index tensors once to avoid repeated CPU-GPU syncs
        let deepstack_indices = if let Some(visual_pos_masks) = visual_pos_masks {
            let mask_flat: Vec<f32> = visual_pos_masks
                .to_device(&self.device)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let indices: Vec<u32> = mask_flat
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .map(|(i, _)| i as u32)
                .collect();
            if indices.is_empty() {
                None
            } else {
                let hidden = xs.dim(candle_core::D::Minus1)?;
                let n = indices.len();
                let idx = Tensor::from_vec(indices, (n,), &self.device)?;
                let idx_expanded = idx.unsqueeze(1)?.repeat((1, hidden))?;
                Some((idx, idx_expanded))
            }
        } else {
            None
        };

        let mut rotary_by_device = HashMap::new();
        let mut indices_by_device = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            match &self.layer_types[i] {
                LayerType::FullAttention => {
                    if let Some(HybridLayerCache::Attention(kv_cache)) = hybrid_cache.get_mut(i) {
                        let cos_sin = rotary_for_layer(
                            &mut rotary_by_device,
                            &cos_sin,
                            &*self.mapper,
                            i,
                            xs.device(),
                        )?;
                        xs = layer
                            .forward_attention(
                                i,
                                &xs,
                                &attention_mask.get(xs.device()),
                                cos_sin,
                                kv_cache,
                                ctx.paged_layer(i),
                                ctx.flash_params(),
                            )
                            .map_err(|error| {
                                forward_error_context(
                                    error,
                                    i,
                                    Bf16ForwardStage::FullAttentionLayer,
                                    &xs,
                                )
                            })?;
                    }
                }
                LayerType::LinearAttention => {
                    if let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i) {
                        let recurrent_metadata = recurrent_metadata.as_ref().expect(
                            "checked above: linear-attention layers require recurrent metadata",
                        );
                        let indices = tensor_for_layer(
                            &mut indices_by_device,
                            recurrent_metadata.state_indices(),
                            &*self.mapper,
                            i,
                            pool.device(),
                        )?;

                        let conv_state = pool.gather_conv_state(indices)?;
                        let recurrent_state = pool.gather_recurrent_state(indices)?;

                        let mut gdn_cache = GdnLayerCache {
                            conv_state,
                            recurrent_state,
                        };

                        xs = layer
                            .forward_linear(i, &xs, &mut gdn_cache, recurrent_metadata.batch_kind())
                            .map_err(|error| {
                                forward_error_context(
                                    error,
                                    i,
                                    Bf16ForwardStage::LinearAttentionLayer,
                                    &xs,
                                )
                            })?;

                        pool.scatter_conv_state_with_host_indices(
                            indices,
                            recurrent_metadata.state_indices_host(),
                            &gdn_cache.conv_state,
                        )?;
                        pool.scatter_recurrent_state_with_host_indices(
                            indices,
                            recurrent_metadata.state_indices_host(),
                            &gdn_cache.recurrent_state,
                        )?;
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {i} is not recurrent for a linear-attention layer."
                        );
                    }
                }
            }

            // Integrate DeepStack visual features when provided
            if let (Some((idx, idx_expanded)), Some(deepstack)) =
                (&deepstack_indices, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = self.deepstack_process(xs, idx, idx_expanded, &deepstack[i])?;
                }
            }
        }
        let xs = self.mapper.cast_nm_device(&xs, false)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs).map_err(|error| {
            forward_error_context(
                error,
                self.layers.len().saturating_sub(1),
                Bf16ForwardStage::LmHead,
                &xs,
            )
        })
    }

    fn deepstack_process(
        &self,
        hidden_states: Tensor,
        idx: &Tensor,
        idx_expanded: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

        let (batch, seq, hidden) = hidden_states.dims3()?;
        let total = batch * seq;
        let hidden_flat = hidden_states.reshape((total, hidden))?;

        if idx.dim(0)? != visual_embeds.dim(0)? {
            candle_core::bail!(
                "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
                visual_embeds.dim(0)?,
                idx.dim(0)?
            );
        }

        let result = hidden_flat.scatter_add(idx_expanded, &visual_embeds, 0)?;
        result.reshape((batch, seq, hidden))
    }
}

#[cfg(test)]
mod bf16_layer_device_tests {
    use super::*;
    use crate::device_map::DeviceMapSetting;

    #[test]
    fn observation_contract_bf16_layer_error_context_preserves_cause_and_layout() -> Result<()> {
        let input = Tensor::zeros((1, 3, 2), DType::BF16, &Device::Cpu)?.transpose(1, 2)?;
        let error = forward_error_context(
            candle_core::Error::Msg("synthetic technical cause".into()),
            32,
            Bf16ForwardStage::MlpGate,
            &input,
        );
        let candle_core::Error::Context { inner, context } = error else {
            panic!("expected bounded context")
        };
        assert_eq!(inner.to_string(), "synthetic technical cause");
        assert_eq!(context.to_string(), "BF16_OP_CONTEXT{kind=layer;layer=32;stage=mlp_gate;device=cpu;ordinal=0;dtype=BF16;dims=1,2,3;strides=6,1,2}");
        Ok(())
    }

    #[test]
    fn observation_contract_bf16_layer_error_context_rejects_unbounded_metadata() -> Result<()> {
        let inputs = [
            Tensor::zeros((1, 1, 1, 1, 1), DType::BF16, &Device::Cpu)?,
            Tensor::zeros(1, DType::BF16, &Device::Cpu)?
                .broadcast_as(BF16_CONTEXT_MAX_INTEGER + 1)?,
        ];
        for input in &inputs {
            let error = forward_error_context(
                candle_core::Error::Msg("original cause".into()),
                0,
                Bf16ForwardStage::Gdn,
                input,
            );
            assert!(matches!(error, candle_core::Error::Msg(_)));
        }
        let error = forward_error_context(
            candle_core::Error::Msg("original cause".into()),
            BF16_CONTEXT_MAX_LAYER + 1,
            Bf16ForwardStage::Gdn,
            &Tensor::zeros(1, DType::BF16, &Device::Cpu)?,
        );
        assert!(matches!(error, candle_core::Error::Msg(_)));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    const DENSE_PROBE_K: usize = 5120;
    #[cfg(feature = "cuda")]
    const DENSE_PROBE_N: usize = 10240;

    #[cfg(feature = "cuda")]
    #[derive(Clone, Copy)]
    enum DenseProbeBackend {
        Production,
        Candle,
        LtWrongInit,
        LtBoundInit,
    }

    #[cfg(feature = "cuda")]
    struct DenseProbe {
        backend: DenseProbeBackend,
        backend_name: &'static str,
        m: usize,
        ordinal: usize,
    }

    #[cfg(feature = "cuda")]
    impl DenseProbe {
        fn from_env() -> Result<Option<Self>> {
            let backend = std::env::var("BF16_DENSE_PROBE_BACKEND");
            let m = std::env::var("BF16_DENSE_PROBE_M");
            let ordinal = std::env::var("BF16_DENSE_PROBE_DEVICE");
            if matches!(&backend, Err(std::env::VarError::NotPresent))
                && matches!(&m, Err(std::env::VarError::NotPresent))
                && matches!(&ordinal, Err(std::env::VarError::NotPresent))
            {
                return Ok(None);
            }
            let (backend, backend_name) = match backend.as_deref() {
                Ok("production") => (DenseProbeBackend::Production, "production"),
                Ok("candle") => (DenseProbeBackend::Candle, "candle"),
                Ok("lt_wrong_init") => (DenseProbeBackend::LtWrongInit, "lt_wrong_init"),
                Ok("lt_bound_init") => (DenseProbeBackend::LtBoundInit, "lt_bound_init"),
                _ => candle_core::bail!("Select a supported BF16_DENSE_PROBE_BACKEND"),
            };
            let m = match m.as_deref() {
                Ok("1") => 1,
                Ok("9") => 9,
                Ok("9586") => 9586,
                _ => candle_core::bail!("Select BF16_DENSE_PROBE_M from 1, 9, 9586"),
            };
            let ordinal = match ordinal.as_deref() {
                Ok("0") => 0,
                Ok("1") => 1,
                _ => candle_core::bail!("Select BF16_DENSE_PROBE_DEVICE from 0, 1"),
            };
            Ok(Some(Self {
                backend,
                backend_name,
                m,
                ordinal,
            }))
        }

        fn record(&self, stage: &'static str, route: &'static str, current: usize) {
            eprintln!(
                "BF16_DENSE_PROBE backend={} route={} stage={} device={} current_before={} m={} n={} k={} dtype=BF16",
                self.backend_name, route, stage, self.ordinal, current, self.m, DENSE_PROBE_N, DENSE_PROBE_K
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn bind_probe_device(device: &Device) -> Result<()> {
        let Device::Cuda(device) = device else {
            candle_core::bail!("Dense probe requires CUDA")
        };
        device
            .cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)
    }

    #[cfg(feature = "cuda")]
    fn assert_probe_current(device: &Device) -> Result<()> {
        use candle_core::cuda::cudarc::driver::result::ctx;
        let Device::Cuda(device) = device else {
            candle_core::bail!("Dense probe requires CUDA")
        };
        let current = ctx::get_current().map_err(candle_core::Error::wrap)?;
        assert_eq!(current, Some(device.cuda_stream().context().cu_ctx()));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires two CUDA devices and one selected case per fresh process"]
    fn bf16_layer_device_two_gpu_dense_probe() -> Result<()> {
        use mistralrs_quant::{
            cublaslt::{
                fused_batch_matmul, maybe_init_cublas_lt_wrapper, CublasLt, CUBLASLT_CONTROLLER,
            },
            QuantMethodConfig, UnquantLinear,
        };
        let Some(probe) = DenseProbe::from_env()? else {
            eprintln!("BF16_DENSE_PROBE stage=not_selected");
            return Ok(());
        };
        // A failing CUDA call ends this test process; controls run in separate invocations.
        let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
        let device = &devices[probe.ordinal];
        let opposite = 1 - probe.ordinal;
        let input = Tensor::ones((1, probe.m, DENSE_PROBE_K), DType::BF16, device)?;
        let weight = Tensor::ones((DENSE_PROBE_N, DENSE_PROBE_K), DType::BF16, device)?;
        let weight_batch = weight.broadcast_left(1)?;
        let linear = UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
            weight, None,
        )))?;

        let (local_handle, route) = match probe.backend {
            DenseProbeBackend::Production => {
                bind_probe_device(&devices[0])?;
                assert_probe_current(&devices[0])?;
                maybe_init_cublas_lt_wrapper(devices[0].clone());
                assert!(CUBLASLT_CONTROLLER.get_for_device(&devices[0]).is_some());
                assert!(CUBLASLT_CONTROLLER.get_for_device(&devices[1]).is_none());
                let route =
                    if mistralrs_quant::gemv::should_use_gemv(&input, &weight_batch.squeeze(0)?) {
                        "gemv"
                    } else if probe.ordinal == 0 {
                        "cublaslt"
                    } else {
                        "cublas"
                    };
                (None, route)
            }
            DenseProbeBackend::Candle => (None, "cublas"),
            DenseProbeBackend::LtWrongInit | DenseProbeBackend::LtBoundInit => {
                let current = match probe.backend {
                    DenseProbeBackend::LtWrongInit => opposite,
                    _ => probe.ordinal,
                };
                bind_probe_device(&devices[current])?;
                assert_probe_current(&devices[current])?;
                probe.record("initialize", "cublaslt", current);
                // The one-case process owns cleanup; Drop could mask an asynchronous CUDA error.
                (
                    Some(std::mem::ManuallyDrop::new(CublasLt::new(device)?)),
                    "cublaslt",
                )
            }
        };

        let opposite_allocation = Tensor::ones(1, DType::BF16, &devices[opposite])?;
        assert_probe_current(&devices[opposite])?;
        probe.record("forward", route, opposite);
        let output = match probe.backend {
            DenseProbeBackend::Production => linear.forward(&input),
            DenseProbeBackend::Candle => input.matmul(&weight_batch.t()?),
            DenseProbeBackend::LtWrongInit | DenseProbeBackend::LtBoundInit => {
                let handle: &CublasLt = local_handle.as_ref().expect("local handle case");
                fused_batch_matmul(
                    &input,
                    &weight_batch,
                    None,
                    None,
                    None,
                    None,
                    None,
                    handle.clone(),
                )
                .and_then(|output| output.t())
            }
        };
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                probe.record("forward_error", route, opposite);
                return Err(error);
            }
        };
        device.synchronize()?;
        assert_eq!(output.dims(), &[1, probe.m, DENSE_PROBE_N]);
        assert_eq!(output.dtype(), DType::BF16);
        assert!(output.device().same_device(device));
        // The exact all-ones dot product is representable in BF16 and rejects NaNs as well.
        assert_eq!(
            output
                .eq(DENSE_PROBE_K as f64)?
                .min_all()?
                .to_scalar::<u8>()?,
            1
        );
        device.synchronize()?;
        assert_eq!(opposite_allocation.to_vec1::<half::bf16>()?.len(), 1);
        probe.record("validated", route, opposite);
        Ok(())
    }

    #[test]
    fn bf16_layer_device_mapping_reuses_single_device_buffers() -> Result<()> {
        let device = Device::Cpu;
        let mapper = DeviceMapSetting::dummy().into_mapper(3, &device, None, &[])?;
        let indices = Tensor::from_vec(vec![2u32, 0], 2, &device)?;
        let mut mapped = HashMap::new();
        let first = tensor_for_layer(&mut mapped, &indices, &*mapper, 0, &device)?.id();
        let second = tensor_for_layer(&mut mapped, &indices, &*mapper, 2, &device)?.id();
        assert_eq!(first, second);
        assert_eq!(mapped.len(), 1);
        let pair = (
            Tensor::from_vec(vec![1f32, 2.], (1, 1, 2), &device)?,
            Tensor::from_vec(vec![3f32, 4.], (1, 1, 2), &device)?,
        );
        let mut rotary = HashMap::new();
        let first = rotary_for_layer(&mut rotary, &pair, &*mapper, 0, &device)?
            .0
            .id();
        let second = rotary_for_layer(&mut rotary, &pair, &*mapper, 2, &device)?
            .0
            .id();
        assert_eq!(first, second);
        assert_eq!(rotary.len(), 1);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()
    }

    #[cfg(feature = "cuda")]
    fn patterned(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (u16::try_from((i + salt) % 31).expect("small pattern") as f32 - 15.) / 100.)
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn run_gdn_pool(
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) -> Result<(Vec<f32>, Tensor, Tensor)> {
        use crate::cuda::gdn::{
            causal_conv1d_cuda, fused_decode_recurrence_cuda, prepare_recurrence_inputs_cuda,
            warp_gated_delta_rule_recurrence_cuda,
        };
        let device = conv_state.device();
        let input = Tensor::from_vec(patterned(512 * 65, 1), (1, 512, 65), device)?
            .to_dtype(DType::BF16)?;
        let weight =
            Tensor::from_vec(patterned(512 * 4, 2), (512, 4), device)?.to_dtype(DType::BF16)?;
        let b = Tensor::from_vec(patterned(130, 3), (1, 65, 2), device)?.to_dtype(DType::BF16)?;
        let a = Tensor::from_vec(patterned(130, 4), (1, 65, 2), device)?.to_dtype(DType::BF16)?;
        let a_log = Tensor::from_vec(vec![-0.1f32, 0.2], 2, device)?;
        let dt_bias = Tensor::from_vec(vec![-0.2f32, 0.1], 2, device)?;
        let mut state = recurrent_state.reshape((2, 128, 128))?;
        let (output, conv_state) = causal_conv1d_cuda(&input, &weight, &conv_state, 4, false)?;
        let mixed = output.transpose(1, 2)?.contiguous()?;
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &mixed, &b, &a, &a_log, &dt_bias, 1, 65, 1, 2, 128, 128,
        )?;
        let prefill = warp_gated_delta_rule_recurrence_cuda(&q, &k, &v, &g, &beta, &mut state)?;
        let token =
            Tensor::from_vec(patterned(512, 5), (1, 512, 1), device)?.to_dtype(DType::BF16)?;
        let (decode, conv_state) = causal_conv1d_cuda(&token, &weight, &conv_state, 4, true)?;
        let mixed = decode.transpose(1, 2)?.contiguous()?;
        let b = Tensor::from_vec(patterned(2, 6), (1, 1, 2), device)?.to_dtype(DType::BF16)?;
        let a = Tensor::from_vec(patterned(2, 7), (1, 1, 2), device)?.to_dtype(DType::BF16)?;
        let decode = fused_decode_recurrence_cuda(
            &mixed, &b, &a, &a_log, &dt_bias, &mut state, 1, 1, 2, 128, 128,
        )?;
        let mut values = flat(&prefill)?;
        values.extend(flat(&decode)?);
        Ok((values, conv_state, state.reshape((1, 2, 128, 128))?))
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires two CUDA devices"]
    fn bf16_layer_device_two_gpu_state_kernels_rotary_and_restore() -> Result<()> {
        use crate::device_map::{DeviceLayerMapMetadata, DeviceMapMetadata};
        use crate::kv_cache::RecurrentStateSnapshot;
        let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
        let mapper = DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
            DeviceLayerMapMetadata {
                ordinal: 0,
                layers: 2,
            },
            DeviceLayerMapMetadata {
                ordinal: 1,
                layers: 1,
            },
        ]))
        .into_mapper(3, &devices[0], None, &devices)?;
        let config = HybridCacheConfig {
            layer_types: vec![
                HybridLayerType::Recurrent,
                HybridLayerType::Attention,
                HybridLayerType::Recurrent,
            ],
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 512,
                conv_width: 4,
                state_dims: vec![2, 128, 128],
                recurrent_dtype: Some(DType::F32),
            },
        };
        let mut cache = HybridCache::new_with_layer_devices(config, DType::BF16, |layer| {
            mapper.device_for(layer, false).expect("mapped layer")
        })?;
        assert_eq!(cache.allocate_seq(), Some(0));
        let indices = Tensor::from_vec(vec![0u32], 1, &devices[0])?;
        let mut mapped = HashMap::new();
        let mut results = Vec::new();
        for (layer_idx, device_idx) in [(0, 0), (2, 1)] {
            let pool = cache
                .get_mut(layer_idx)
                .and_then(HybridLayerCache::as_recurrent_pool_mut)
                .expect("recurrent fixture layer");
            assert!(pool.device().same_device(&devices[device_idx]));
            let indices =
                tensor_for_layer(&mut mapped, &indices, &*mapper, layer_idx, pool.device())?;
            assert!(indices.device().same_device(pool.device()));
            let (mut values, conv_state, recurrent_state) = run_gdn_pool(
                pool.gather_conv_state(indices)?,
                pool.gather_recurrent_state(indices)?,
            )?;
            pool.scatter_conv_state_with_host_indices(indices, Some(&[0]), &conv_state)?;
            pool.scatter_recurrent_state_with_host_indices(indices, Some(&[0]), &recurrent_state)?;
            values.extend(flat(&pool.gather_conv_state(indices)?)?);
            values.extend(flat(&pool.gather_recurrent_state(indices)?)?);
            results.push(values);
        }
        assert_eq!(mapped.len(), 2);
        assert_eq!(results[0].len(), results[1].len());
        for (&reference, &split) in results[0].iter().zip(&results[1]) {
            assert!(
                reference.is_finite() && split.is_finite() && (reference - split).abs() <= 1e-5
            );
        }
        let snapshots: Vec<_> = cache
            .snapshot_recurrent_state(0)?
            .into_iter()
            .map(|snapshot| {
                Ok(RecurrentStateSnapshot {
                    conv_state: snapshot.conv_state.to_device(&Device::Cpu)?,
                    recurrent_state: snapshot.recurrent_state.to_device(&Device::Cpu)?,
                })
            })
            .collect::<Result<_>>()?;
        cache.reset_seq(0)?;
        for snapshot in cache.snapshot_recurrent_state(0)? {
            assert!(flat(&snapshot.conv_state)?.iter().all(|&value| value == 0.));
            assert!(flat(&snapshot.recurrent_state)?
                .iter()
                .all(|&value| value == 0.));
        }
        cache.restore_recurrent_state(0, &snapshots)?;
        for (idx, snapshot) in cache.snapshot_recurrent_state(0)?.iter().enumerate() {
            assert!(snapshot.conv_state.device().same_device(&devices[idx]));
            assert!(snapshot.recurrent_state.device().same_device(&devices[idx]));
            assert_eq!(
                flat(&snapshot.conv_state)?,
                flat(&snapshots[idx].conv_state)?
            );
            assert_eq!(
                flat(&snapshot.recurrent_state)?,
                flat(&snapshots[idx].recurrent_state)?
            );
        }
        let pair = (
            Tensor::from_vec(patterned(128, 8), (1, 1, 128), &devices[0])?.to_dtype(DType::BF16)?,
            Tensor::from_vec(patterned(128, 9), (1, 1, 128), &devices[0])?.to_dtype(DType::BF16)?,
        );
        let mut rotary = HashMap::new();
        let mapped_pair = rotary_for_layer(&mut rotary, &pair, &*mapper, 2, &devices[1])?;
        assert!(mapped_pair.0.device().same_device(&devices[1]));
        assert!(mapped_pair.1.device().same_device(&devices[1]));
        assert_eq!(flat(&mapped_pair.0)?, flat(&pair.0)?);
        assert_eq!(flat(&mapped_pair.1)?, flat(&pair.1)?);
        let returned = mapper.cast_nm_device(&mapped_pair.0, false)?;
        assert!(returned.device().same_device(&devices[0]));
        assert_eq!(flat(&returned)?, flat(&pair.0)?);
        Ok(())
    }
}

impl IsqModel for Qwen3_5TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_lm = uvb.pp("model").pp("language_model");
        uvb_lm.pp("embed_tokens").add(&self.embed_tokens);
        uvb_lm.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_lm.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("dt_bias", gdn.dt_bias.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("A_log", gdn.a_log.clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("norm")
                        .add_tensor("weight", gdn.norm.weight.clone());
                }
            }
        }

        uvb.to_safetensors()
    }
}
