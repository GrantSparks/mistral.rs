use std::{
    ops::Range,
    sync::{Arc, Weak},
};

use anyhow::{bail, Context, Result};
use candle_core::Tensor;

use crate::{
    paged_attention::block_hash::MultiModalFeature, sequence::Sequence, MistralRs, NormalRequest,
    Pipeline, Request, RequestMessage,
};

type LoadedPipeline = tokio::sync::Mutex<dyn Pipeline>;

/// One image's padding-token span and SHA-256 of its patch-major little-endian f32 values.
pub struct PreparedImageView {
    pub image_tokens: Range<usize>,
    pub pixel_sha256: [u8; 32],
}

#[doc(hidden)]
pub struct PreparedImagePrompt {
    pub(crate) prefix_token_ids: Vec<u32>,
    pub(crate) prompt: String,
    pub(crate) pixels: Tensor,
    pub(crate) grid: Tensor,
    pub(crate) features: Vec<MultiModalFeature>,
    pub(crate) views: Vec<PreparedImageView>,
}

impl PreparedImagePrompt {
    pub(crate) fn install(&self, sequence: &mut Sequence) {
        sequence.multimodal.cached_pixel_values = Some(self.pixels.clone());
        sequence.multimodal.cached_img_thw = Some(self.grid.clone());
        sequence.multimodal.rope_img_grid_thw = Some(self.grid.clone());
        sequence.multimodal.has_changed_prompt = true;
        sequence.set_mm_features(self.features.clone());
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct PreparedImageChatRequest {
    pub(crate) request: Box<NormalRequest>,
    pub(crate) prompt: Arc<PreparedImagePrompt>,
    pub(crate) pipeline: Weak<LoadedPipeline>,
    model_id: String,
}

impl PreparedImageChatRequest {
    pub fn prefix_token_ids(&self) -> &[u32] {
        &self.prompt.prefix_token_ids
    }

    pub fn views(&self) -> &[PreparedImageView] {
        &self.prompt.views
    }
}

fn local_preflight_available() -> bool {
    !crate::distributed::is_daemon()
        && !mistralrs_quant::distributed::use_nccl()
        && !cfg!(feature = "ring")
}

impl MistralRs {
    #[doc(hidden)]
    pub async fn prepare_image_chat(
        &self,
        request: Box<NormalRequest>,
    ) -> Result<PreparedImageChatRequest> {
        if !local_preflight_available() {
            bail!("prepared image chat requires a local engine");
        }
        if request.truncate_sequence
            || request.sampling_params.n_choices != 1
            || request.return_raw_logits
            || request.adapter.is_some()
            || request.tools.is_some()
            || request.tool_choice.is_some()
            || request.web_search_options.is_some()
            || request.enable_code_execution
            || request.enable_shell
            || request.session_id.is_some()
            || request.max_tool_rounds.is_some()
            || request.tool_dispatch_url.is_some()
            || request.files.is_some()
            || !request.input_files.is_empty()
        {
            bail!("prepared image chat requires a single choice without truncation or tools");
        }
        let model_id = self.resolve_alias_or_default(request.model_id.as_deref())?;
        let pipeline = self
            .engines
            .read()
            .map_err(|_| crate::MistralRsError::EnginePoisoned)?
            .get(&model_id)
            .with_context(|| format!("model {model_id} must be loaded for image preflight"))?
            .reboot_state
            .pipeline
            .clone();
        tokio::task::spawn_blocking(move || {
            let RequestMessage::MultimodalChat {
                images,
                audios,
                videos,
                messages,
                enable_thinking,
                reasoning_effort,
            } = &request.messages
            else {
                bail!("prepared image chat requires image messages");
            };
            if images.is_empty() || !audios.is_empty() || !videos.is_empty() {
                bail!("prepared image chat accepts only images and text");
            }
            let prompt = {
                let pipeline = pipeline.blocking_lock();
                let tokenizer = pipeline
                    .tokenizer()
                    .context("image preflight needs a tokenizer")?;
                let processor = pipeline.get_processor();
                let (tokens, _) = processor.process(
                    &*pipeline,
                    messages.clone(),
                    true,
                    true,
                    *enable_thinking,
                    *reasoning_effort,
                    Vec::new(),
                )?;
                processor.inputs_processor().prepare_image_prompt(
                    &tokenizer,
                    &tokens,
                    images,
                    &pipeline.device(),
                    pipeline.get_input_processor_config(),
                )?
            };
            Ok(PreparedImageChatRequest {
                request,
                prompt: Arc::new(prompt),
                pipeline: Arc::downgrade(&pipeline),
                model_id,
            })
        })
        .await?
    }

    #[doc(hidden)]
    pub async fn stream_prepared_image_chat(
        &self,
        prepared: PreparedImageChatRequest,
    ) -> Result<()> {
        if !local_preflight_available() {
            bail!("prepared image chat requires a local engine");
        }
        let sender = {
            let engines = self
                .engines
                .read()
                .map_err(|_| crate::MistralRsError::EnginePoisoned)?;
            let engine = engines
                .get(&prepared.model_id)
                .filter(|engine| {
                    !engine.is_finished()
                        && prepared
                            .pipeline
                            .ptr_eq(&Arc::downgrade(&engine.reboot_state.pipeline))
                })
                .context("the pipeline that prepared this image chat is no longer loaded")?;
            engine.sender.clone()
        };
        sender.send(Request::PreparedImageChat(prepared)).await?;
        Ok(())
    }
}
