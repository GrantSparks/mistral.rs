use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::{DynamicImage, Rgb, RgbImage};
use sha2::{Digest, Sha256};
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit, AddedToken,
    Tokenizer,
};

use crate::{
    paged_attention::block_hash::MultiModalFeature,
    pipeline::InputsProcessor,
    sampler::Sampler,
    sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer},
    vision_models::{preprocessor_config::PreProcessorConfig, ModelInputs},
};

use super::{Qwen3VLImageProcessor, Qwen3VLProcessor, Qwen3VLVisionSpecificArgs};

const PATCH_SIZE: u32 = 16;
const MERGE_SIZE: u32 = 2;
const TEMPORAL_PATCH_SIZE: u32 = 2;
const FIRST_WIDTH: u32 = 64;
const SECOND_WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const FIRST_VISUAL_TOKENS: usize = 4;
const SECOND_VISUAL_TOKENS: usize = 6;
const CHAT_START: &str = "<|im_start|>";
const CHAT_END: &str = "<|im_end|>";
const UNKNOWN: &str = "<unk>";
const TWO_VIEW_PROMPT: &str = concat!(
    "<|im_start|>user <|vision_start|><|image_pad|><|vision_end|> first <|im_end|>",
    "<|im_start|>assistant seen <|im_end|>",
    "<|im_start|>user <|vision_start|><|image_pad|><|vision_end|> second <|im_end|>",
    "<|im_start|>assistant",
);

fn tokenizer() -> Result<Arc<Tokenizer>> {
    let special_tokens = [
        CHAT_START,
        CHAT_END,
        Qwen3VLProcessor::IMAGE_PAD,
        Qwen3VLProcessor::VIDEO_PAD,
        Qwen3VLProcessor::VISION_START,
        Qwen3VLProcessor::VISION_END,
        Qwen3VLProcessor::PLACEHOLDER,
    ];
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [UNKNOWN, "user", "assistant", "first", "seen", "second"]
                    .into_iter()
                    .chain(special_tokens)
                    .zip(0..)
                    .map(|(token, id)| (token.to_string(), id))
                    .collect(),
            )
            .unk_token(UNKNOWN.into())
            .build()
            .map_err(anyhow::Error::msg)?,
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .add_special_tokens(&special_tokens.map(|token| AddedToken::from(token.to_string(), true)));
    Ok(Arc::new(tokenizer))
}

fn processor_config() -> Arc<PreProcessorConfig> {
    Arc::new(PreProcessorConfig {
        do_resize: Some(true),
        min_pixels: Some((FIRST_WIDTH * HEIGHT) as usize),
        max_pixels: Some((SECOND_WIDTH * HEIGHT) as usize),
        patch_size: Some(PATCH_SIZE as usize),
        merge_size: Some(MERGE_SIZE as usize),
        temporal_patch_size: Some(TEMPORAL_PATCH_SIZE as usize),
        ..Default::default()
    })
}

fn patterned_image(width: u32, seed: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, HEIGHT, |x, y| {
        Rgb([
            ((3 * x + 5 * y + seed) % 256) as u8,
            ((7 * x + 2 * y + 2 * seed) % 256) as u8,
            ((11 * x + 13 * y + 3 * seed) % 256) as u8,
        ])
    }))
}

fn fixture_sequence(
    tokens: Vec<u32>,
    prompt: String,
    images: Vec<DynamicImage>,
) -> Result<Sequence> {
    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        32,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )?;
    let (responder, _) = tokio::sync::mpsc::channel(1);
    Ok(Sequence::new_waiting(
        tokens,
        prompt,
        0,
        0,
        1,
        responder,
        sampler,
        vec![],
        vec![],
        None,
        false,
        false,
        Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        ))),
        0,
        0,
        SequenceRecognizer::None,
        None,
        None,
        Some(images),
        None,
        None,
        None,
        None,
        None,
        SeqStepType::PromptAndDecode,
        None,
        None,
        None,
        false,
        false,
        vec![],
        None,
    ))
}

fn float_bits(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn assert_features_match(left: &[MultiModalFeature], right: &[MultiModalFeature]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.kind, right.kind);
        assert_eq!(left.item_range, right.item_range);
        assert_eq!(left.hashes, right.hashes);
        assert_eq!(left.offset, right.offset);
        assert_eq!(left.length, right.length);
        assert_eq!(left.attention_policy, right.attention_policy);
        assert_eq!(left.splittable, right.splittable);
    }
}

fn model_inputs(
    processor: &Qwen3VLImageProcessor,
    tokenizer: Arc<Tokenizer>,
    config: Arc<PreProcessorConfig>,
    sequence: &mut Sequence,
) -> Result<Box<ModelInputs>> {
    processor
        .process_inputs(
            Some(tokenizer),
            &mut [sequence],
            true,
            false,
            &Device::Cpu,
            false,
            None,
            false,
            None,
            Some(config),
            None,
            None,
        )?
        .inputs
        .downcast()
        .ok()
        .context("Qwen processor did not produce vision model inputs")
}

#[test]
fn qwen_image_preflight_matches_ordinary_model_inputs() -> Result<()> {
    let processor = Qwen3VLImageProcessor { max_edge: None };
    let tokenizer = tokenizer()?;
    let config = processor_config();
    let photos = vec![
        patterned_image(FIRST_WIDTH, 19),
        patterned_image(SECOND_WIDTH, 67),
    ];
    let original = tokenizer
        .encode_fast(TWO_VIEW_PROMPT, false)
        .map_err(anyhow::Error::msg)?;
    assert!(!original
        .get_ids()
        .contains(&tokenizer.token_to_id(UNKNOWN).unwrap()));

    let prepared = processor.prepare_image_prompt(
        &tokenizer,
        original.get_ids(),
        &photos,
        &Device::Cpu,
        Some(config.clone()),
    )?;
    let pad = tokenizer.token_to_id(Qwen3VLProcessor::IMAGE_PAD).unwrap();
    let start = tokenizer
        .token_to_id(Qwen3VLProcessor::VISION_START)
        .unwrap();
    let end = tokenizer.token_to_id(Qwen3VLProcessor::VISION_END).unwrap();
    let mut lengths = [FIRST_VISUAL_TOKENS, SECOND_VISUAL_TOKENS].into_iter();
    let mut expected_ids = Vec::new();
    for &token in original.get_ids() {
        if token == pad {
            expected_ids.extend(std::iter::repeat_n(pad, lengths.next().unwrap()));
        } else {
            expected_ids.push(token);
        }
    }
    assert!(lengths.next().is_none());
    assert_eq!(prepared.prefix_token_ids, expected_ids);
    assert_eq!(
        prepared.grid.to_vec2::<u32>()?,
        vec![vec![1, 4, 4], vec![1, 4, 6]]
    );
    assert_eq!(prepared.views.len(), 2);
    let mut expected_offset = 0;
    for (view, tokens) in prepared
        .views
        .iter()
        .zip([FIRST_VISUAL_TOKENS, SECOND_VISUAL_TOKENS])
    {
        let span = &view.image_tokens;
        let patch_rows = tokens * MERGE_SIZE.pow(2) as usize;
        assert_eq!(span.len(), tokens);
        assert_eq!(prepared.prefix_token_ids[span.start - 1], start);
        assert_eq!(prepared.prefix_token_ids[span.end], end);
        assert!(prepared.prefix_token_ids[span.clone()]
            .iter()
            .all(|&id| id == pad));
        let pixels = prepared.pixels.narrow(0, expected_offset, patch_rows)?;
        assert_eq!(
            pixels.dims(),
            &[
                patch_rows,
                (3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE.pow(2)) as usize
            ]
        );
        let mut hash = Sha256::new();
        for value in float_bits(&pixels)? {
            hash.update(value.to_le_bytes());
        }
        let digest: [u8; 32] = hash.finalize().into();
        assert_eq!(view.pixel_sha256, digest);
        expected_offset += patch_rows;
    }
    assert_ne!(
        prepared.views[0].pixel_sha256,
        prepared.views[1].pixel_sha256
    );
    assert_eq!(expected_offset, prepared.pixels.dim(0)?);

    let mut ordinary = fixture_sequence(
        original.get_ids().to_vec(),
        TWO_VIEW_PROMPT.into(),
        photos.clone(),
    )?;
    processor.prepare_for_paged_prompt_planning(
        Some(tokenizer.clone()),
        &mut [&mut ordinary],
        &Device::Cpu,
        Some(config.clone()),
        None,
    )?;
    assert_eq!(ordinary.get_toks(), prepared.prefix_token_ids);
    assert_eq!(ordinary.get_initial_prompt(), prepared.prompt);
    assert_features_match(ordinary.mm_features(), &prepared.features);

    let mut admitted = fixture_sequence(
        prepared.prefix_token_ids.clone(),
        prepared.prompt.clone(),
        photos,
    )?;
    prepared.install(&mut admitted);
    assert!(admitted.multimodal.has_changed_prompt);
    assert_features_match(admitted.mm_features(), ordinary.mm_features());
    assert_eq!(
        float_bits(admitted.multimodal.cached_pixel_values.as_ref().unwrap())?,
        float_bits(ordinary.multimodal.cached_pixel_values.as_ref().unwrap())?
    );
    assert_eq!(
        admitted
            .multimodal
            .cached_img_thw
            .as_ref()
            .unwrap()
            .to_vec2::<u32>()?,
        ordinary
            .multimodal
            .cached_img_thw
            .as_ref()
            .unwrap()
            .to_vec2::<u32>()?
    );

    let admitted_inputs =
        model_inputs(&processor, tokenizer.clone(), config.clone(), &mut admitted)?;
    let ordinary_inputs = model_inputs(&processor, tokenizer, config, &mut ordinary)?;
    assert_eq!(
        admitted_inputs.input_ids.to_vec2::<u32>()?,
        ordinary_inputs.input_ids.to_vec2::<u32>()?
    );
    assert_eq!(
        float_bits(admitted_inputs.pixel_values.as_ref().unwrap())?,
        float_bits(ordinary_inputs.pixel_values.as_ref().unwrap())?
    );
    let admitted_args = admitted_inputs
        .model_specific_args
        .downcast::<Qwen3VLVisionSpecificArgs>()
        .unwrap();
    let ordinary_args = ordinary_inputs
        .model_specific_args
        .downcast::<Qwen3VLVisionSpecificArgs>()
        .unwrap();
    assert_eq!(
        admitted_args
            .image_grid_thw
            .as_ref()
            .unwrap()
            .to_vec2::<u32>()?,
        ordinary_args
            .image_grid_thw
            .as_ref()
            .unwrap()
            .to_vec2::<u32>()?
    );
    Ok(())
}

#[test]
fn qwen_image_preflight_rejects_missing_or_extra_placeholders() -> Result<()> {
    let processor = Qwen3VLImageProcessor { max_edge: None };
    let tokenizer = tokenizer()?;
    let config = processor_config();
    let photos = [
        patterned_image(FIRST_WIDTH, 19),
        patterned_image(SECOND_WIDTH, 67),
    ];
    for (prompt, images, expected) in [
        (
            "<|im_start|>user first <|im_end|><|im_start|>assistant",
            &photos[..],
            "0 placeholders",
        ),
        (TWO_VIEW_PROMPT, &photos[..1], "2 placeholders"),
    ] {
        let tokens = tokenizer
            .encode_fast(prompt, false)
            .map_err(anyhow::Error::msg)?;
        let error = processor
            .prepare_image_prompt(
                &tokenizer,
                tokens.get_ids(),
                images,
                &Device::Cpu,
                Some(config.clone()),
            )
            .err()
            .context("preflight accepted a mismatched image count")?;
        assert!(error.to_string().contains(expected), "{error:#}");
    }
    Ok(())
}
