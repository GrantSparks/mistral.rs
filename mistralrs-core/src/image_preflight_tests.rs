use std::{
    any::Any,
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use ahash::AHashMap;
use candle_core::{DType, Device};
use either::Either;
use indexmap::IndexMap;
use rand_isaac::Isaac64Rng;
use tokenizers::{
    decoders::byte_level::ByteLevel, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
};
use tokio::sync::mpsc;

use super::*;
use crate::{
    device_map::DeviceMapper,
    paged_attention::{KvCacheLayout, ModelConfigMetadata},
    pipeline::{
        chat_template::{ChatTemplate, ChatTemplateValue},
        llg::build_llg_factory,
        sampling::sample_and_add_toks,
        text_models_inputs_processor::ModelInputs,
        AnyMoePipelineMixin, BasicProcessor, CacheManagerMixin, EitherCache, ForwardInputsResult,
        GeneralMetadata, InputsProcessor, IsqPipelineMixin, MessagesAction, MetadataMixin,
        Modalities, ModelKind, NormalCache, PreProcessingMixin, Processor, SupportedModality,
    },
    prefix_cacher::PrefixCacheManagerV2,
    DefaultSchedulerMethod, IsqType, MessageContent, MistralRsBuilder, ReasoningEffort, Response,
    SamplingParams, SchedulerConfig, Tool,
};

const CONTEXT_TOKENS: usize = 8;

#[derive(Default)]
struct CountingProcessor(AtomicUsize);

impl Processor for CountingProcessor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> Result<(Vec<u32>, String)> {
        self.0.fetch_add(1, Ordering::Relaxed);
        BasicProcessor.process(
            pipeline,
            messages,
            add_generation_prompt,
            add_special_tokens,
            enable_thinking,
            reasoning_effort,
            tools,
        )
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        BasicProcessor.inputs_processor()
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        BasicProcessor.get_special_tokens()
    }

    fn template_action(&self) -> MessagesAction {
        BasicProcessor.template_action()
    }
}

struct TextPipeline {
    processor: Arc<CountingProcessor>,
    tokenizer: Arc<Tokenizer>,
    template: Arc<ChatTemplate>,
    metadata: Arc<GeneralMetadata>,
    cache: EitherCache,
    forwarded: mpsc::UnboundedSender<Vec<u32>>,
}

impl MetadataMixin for TextPipeline {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }

    fn name(&self) -> String {
        "text-preflight-test".into()
    }

    fn reset_non_granular_state(&self) {}

    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }

    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

impl PreProcessingMixin for TextPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }

    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.template.clone())
    }

    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for TextPipeline {
    fn re_isq_model(&mut self, _: IsqType) -> Result<()> {
        unreachable!()
    }
}

impl CacheManagerMixin for TextPipeline {
    fn clone_in_cache(&self, _: &mut [&mut Sequence]) -> candle_core::Result<()> {
        Ok(())
    }

    fn clone_out_cache(&self, _: &mut [&mut Sequence]) {}

    fn set_none_cache(
        &self,
        _: &mut [&mut Sequence],
        _: bool,
        _: bool,
        _: bool,
    ) -> candle_core::Result<()> {
        Ok(())
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

impl AnyMoePipelineMixin for TextPipeline {}

#[async_trait::async_trait]
impl Pipeline for TextPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let inputs = inputs.downcast::<ModelInputs>().unwrap();
        let input_ids = inputs.input_ids.squeeze(0)?.to_vec1::<u32>()?;
        self.forwarded.send(input_ids).unwrap();
        let logits = Tensor::new(&[[[0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0]]], &Device::Cpu)?;
        Ok(ForwardInputsResult::CausalGeneration { logits })
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> candle_core::Result<()> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

async fn runner(
    processor: Arc<CountingProcessor>,
    forwarded: mpsc::UnboundedSender<Vec<u32>>,
) -> Arc<MistralRs> {
    let vocab = ["<unk>", "user", "hello", "assistant", "think"]
        .into_iter()
        .enumerate()
        .map(|(index, token)| (token.to_string(), index as u32))
        .collect::<AHashMap<_, _>>();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_special_tokens(&[AddedToken::from("<eos>", true)]);
    let eos = tokenizer.token_to_id("<eos>").unwrap();
    let mut template = ChatTemplate::default();
    template.chat_template = Some(ChatTemplateValue(Either::Left(
        "{% for message in messages %}{{message.role}} {{message.content}} {% endfor %}{% if add_generation_prompt %}assistant{% if enable_thinking %} think{% endif %}{% endif %}".into(),
    )));
    let model_metadata = ModelConfigMetadata {
        max_seq_len: CONTEXT_TOKENS,
        num_layers: 1,
        hidden_size: 1,
        num_kv_heads: 1,
        num_attn_heads: 1,
        sliding_window: None,
        k_head_dim: 1,
        v_head_dim: 1,
        kv_cache_layout: KvCacheLayout::StandardNoFlashInfer,
    };
    let metadata = GeneralMetadata {
        max_seq_len: CONTEXT_TOKENS,
        llg_factory: Some(build_llg_factory(tokenizer.clone()).unwrap()),
        no_kv_cache: true,
        no_prefix_cache: true,
        num_hidden_layers: 1,
        eos_tok: vec![eos],
        kind: ModelKind::Normal,
        is_xlora: false,
        activation_dtype: DType::F32,
        sliding_window: None,
        cache_config: None,
        cache_engine: None,
        model_metadata: Some(Arc::new(model_metadata)),
        modalities: Modalities {
            input: vec![SupportedModality::Text],
            output: vec![SupportedModality::Text],
        },
        loaded_for_uqff_write: false,
    };
    MistralRsBuilder::new(
        Arc::new(tokio::sync::Mutex::new(TextPipeline {
            processor,
            tokenizer: Arc::new(tokenizer),
            template: Arc::new(template),
            metadata: Arc::new(metadata),
            cache: EitherCache::Normal(NormalCache::new(1, CONTEXT_TOKENS)),
            forwarded,
        })),
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(NonZeroUsize::MIN),
        },
        false,
        None,
    )
    .build()
    .await
}

fn text_request(max_output: usize) -> (Box<NormalRequest>, mpsc::Receiver<Response>) {
    let (tx, rx) = mpsc::channel(4);
    let mut sampling = SamplingParams::deterministic();
    sampling.max_len = Some(max_output);
    let mut request = NormalRequest::new_simple(
        RequestMessage::Chat {
            messages: vec![IndexMap::from([
                ("role".into(), Either::Left("user".into())),
                ("content".into(), Either::Left("hello".into())),
            ])],
            enable_thinking: Some(true),
            reasoning_effort: None,
        },
        sampling,
        tx,
        0,
        None,
        None,
    );
    request.is_streaming = true;
    (Box::new(request), rx)
}

#[tokio::test]
async fn text_chat_preflight_dispatches_the_exact_prompt_without_rendering_again() -> Result<()> {
    let processor = Arc::new(CountingProcessor::default());
    let (sent, mut forwarded) = mpsc::unbounded_channel();
    let runner = runner(processor.clone(), sent).await;
    let (request, mut responses) = text_request(1);
    let prepared = runner.prepare_chat(request).await?;
    let expected = [1, 2, 3, 4];
    assert_eq!(prepared.prefix_token_ids(), expected);
    assert!(prepared.views().is_empty());
    assert!(forwarded.try_recv().is_err());
    assert_eq!(processor.0.load(Ordering::Relaxed), 1);

    runner.stream_prepared_chat(prepared).await?;
    let deadline = Duration::from_secs(5);
    assert_eq!(
        tokio::time::timeout(deadline, forwarded.recv()).await?,
        Some(expected.to_vec())
    );
    match tokio::time::timeout(deadline, responses.recv()).await? {
        Some(Response::Chunk(_)) => (),
        Some(Response::ModelError(message, _)) => bail!("mock model failed: {message}"),
        Some(Response::InternalError(error) | Response::ValidationError(error)) => {
            bail!("mock model failed: {error}");
        }
        _ => bail!("prepared chat did not retain chat streaming semantics"),
    }
    assert_eq!(processor.0.load(Ordering::Relaxed), 1);

    let (normal, _responses) = text_request(1);
    runner
        .get_sender(None)?
        .send(Request::Normal(normal))
        .await?;
    assert_eq!(
        tokio::time::timeout(deadline, forwarded.recv()).await?,
        Some(expected.to_vec())
    );
    assert_eq!(processor.0.load(Ordering::Relaxed), 2);

    let (over_budget, mut responses) = text_request(CONTEXT_TOKENS - expected.len() + 1);
    let prepared = runner.prepare_chat(over_budget).await?;
    assert_eq!(prepared.prefix_token_ids(), expected);
    runner.stream_prepared_chat(prepared).await?;
    assert!(
        matches!(tokio::time::timeout(deadline, responses.recv()).await?,
        Some(Response::ValidationError(error)) if error.to_string().contains("exceeds the model context"))
    );
    assert!(forwarded.try_recv().is_err());
    assert_eq!(processor.0.load(Ordering::Relaxed), 3);
    Ok(())
}
