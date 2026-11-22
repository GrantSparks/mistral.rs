use std::{
    ops::Range,
    sync::{Arc, Weak},
};

use anyhow::{bail, Context, Result};
use candle_core::Tensor;

use crate::{
    paged_attention::block_hash::MultiModalFeature, sequence::Sequence, MistralRs, ModelCategory,
    NormalRequest, Pipeline, Request, RequestMessage,
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

pub(crate) enum PreparedChatPrompt {
    Text { tokens: Vec<u32>, rendered: String },
    Image(PreparedImagePrompt),
}

impl PreparedChatPrompt {
    pub(crate) fn prompt(&self) -> (&[u32], &str) {
        match self {
            Self::Text { tokens, rendered } => (tokens, rendered),
            Self::Image(image) => (&image.prefix_token_ids, &image.prompt),
        }
    }

    pub(crate) fn install(&self, sequence: &mut Sequence) {
        if let Self::Image(image) = self {
            image.install(sequence);
        }
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct PreparedChatRequest {
    pub(crate) request: Box<NormalRequest>,
    pub(crate) prompt: Arc<PreparedChatPrompt>,
    pub(crate) pipeline: Weak<LoadedPipeline>,
    model_id: String,
}

impl PreparedChatRequest {
    pub fn prefix_token_ids(&self) -> &[u32] {
        self.prompt.prompt().0
    }

    pub fn views(&self) -> &[PreparedImageView] {
        match &*self.prompt {
            PreparedChatPrompt::Text { .. } => &[],
            PreparedChatPrompt::Image(image) => &image.views,
        }
    }
}

fn local_preflight_available() -> bool {
    !crate::distributed::is_daemon()
        && !mistralrs_quant::distributed::use_nccl()
        && !cfg!(feature = "ring")
}

impl MistralRs {
    #[doc(hidden)]
    pub async fn prepare_chat(&self, request: Box<NormalRequest>) -> Result<PreparedChatRequest> {
        if !local_preflight_available() {
            bail!("prepared chat requires a local engine");
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
            bail!("prepared chat requires a single choice without truncation or tools");
        }
        let model_id = self.resolve_alias_or_default(request.model_id.as_deref())?;
        let pipeline = self
            .engines
            .read()
            .map_err(|_| crate::MistralRsError::EnginePoisoned)?
            .get(&model_id)
            .with_context(|| format!("model {model_id} must be loaded for chat preflight"))?
            .reboot_state
            .pipeline
            .clone();
        tokio::task::spawn_blocking(move || {
            let prompt = {
                let pipeline = pipeline.blocking_lock();
                prepare_chat_prompt(&*pipeline, &request.messages)?
            };
            Ok(PreparedChatRequest {
                request,
                prompt: Arc::new(prompt),
                pipeline: Arc::downgrade(&pipeline),
                model_id,
            })
        })
        .await?
    }

    #[doc(hidden)]
    pub async fn stream_prepared_chat(&self, prepared: PreparedChatRequest) -> Result<()> {
        if !local_preflight_available() {
            bail!("prepared chat requires a local engine");
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
                .context("the pipeline that prepared this chat is no longer loaded")?;
            engine.sender.clone()
        };
        sender.send(Request::PreparedChat(prepared)).await?;
        Ok(())
    }
}

fn prepare_chat_prompt(
    pipeline: &dyn Pipeline,
    request: &RequestMessage,
) -> Result<PreparedChatPrompt> {
    if !matches!(
        pipeline.category(),
        ModelCategory::Text | ModelCategory::Multimodal { .. }
    ) || !pipeline
        .get_chat_template()
        .is_some_and(|template| template.has_chat_template())
    {
        bail!("prepared chat requires a text or multimodal pipeline with a chat template");
    }
    let (messages, enable_thinking, reasoning_effort, images) = match request {
        RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort,
        } => (messages, *enable_thinking, *reasoning_effort, None),
        RequestMessage::MultimodalChat {
            messages,
            enable_thinking,
            reasoning_effort,
            images,
            audios,
            videos,
        } if !images.is_empty() && audios.is_empty() && videos.is_empty() => {
            (messages, *enable_thinking, *reasoning_effort, Some(images))
        }
        _ => bail!("prepared chat accepts only text and image chat messages"),
    };
    let processor = pipeline.get_processor();
    let (tokens, rendered) = processor.process(
        pipeline,
        messages.clone(),
        true,
        true,
        enable_thinking,
        reasoning_effort,
        Vec::new(),
    )?;
    if let Some(images) = images {
        let tokenizer = pipeline
            .tokenizer()
            .context("image preflight needs a tokenizer")?;
        return Ok(PreparedChatPrompt::Image(
            processor.inputs_processor().prepare_image_prompt(
                &tokenizer,
                &tokens,
                images,
                &pipeline.device(),
                pipeline.get_input_processor_config(),
            )?,
        ));
    }
    Ok(PreparedChatPrompt::Text { tokens, rendered })
}

#[cfg(test)]
#[path = "image_preflight_tests.rs"]
mod tests;
