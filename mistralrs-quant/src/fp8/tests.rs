use candle_core::WithDType;
use float8::F8E4M3;
use half::bf16;

fn reference_term(a: bf16, w: F8E4M3, scale: f32) -> f64 {
    // WithDType::to_f64 takes F8E4M3 by value and shadows float8's
    // borrowed inherent method in the pinned Candle implementation.
    // Explicit dispatch avoids that trait implementation's recursion.
    a.to_f64() * F8E4M3::to_f64(&w) * scale as f64
}

#[test]
fn fp8_library_cpu_reference_conversion() {
    fn require_dtype<T: WithDType>() {}
    require_dtype::<F8E4M3>();
    for (bits, expected) in [
        (0, 0.0),
        (128, -0.0),
        (1, 2f64.powi(-9)),
        (129, -2f64.powi(-9)),
        (56, 1.0),
        (184, -1.0),
        (126, 448.0),
        (254, -448.0),
    ] {
        assert_eq!(
            reference_term(bf16::ONE, F8E4M3::from_bits(bits), 1.0),
            expected
        );
    }
    assert_eq!(
        reference_term(bf16::from_f32(0.5), F8E4M3::from_bits(254), 1.125),
        -252.0
    );
    assert!(reference_term(bf16::ONE, F8E4M3::from_bits(127), 1.0).is_nan());
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
mod cuda {
    use super::*;
    use crate::blockwise_fp8::{
        fp8_blockwise_dequantize, fp8_blockwise_matmul, fp8_blockwise_quantize,
    };
    use candle_core::{cuda::cudarc::driver::sys, DType, Device, Result, Tensor};
    fn device() -> Result<Device> {
        let ordinal = std::env::var("FP8_GEMM_CUDA_DEVICE").unwrap_or_else(|_| "0".into());
        let ordinal = match ordinal.as_str() {
            "0" => 0,
            "1" => 1,
            _ => candle_core::bail!("fixture ordinal must be 0 or 1"),
        };
        eprintln!("fp8_gemm_phase stage=device_begin ordinal={ordinal}");
        let dev = Device::new_cuda(ordinal)?;
        eprintln!("fp8_gemm_phase stage=device_ready ordinal={ordinal}");
        Ok(dev)
    }

    fn gamma(ops: usize) -> f64 {
        // Full-ULP FP32 unit covers either rounding direction. Applies
        // only to normal finite intermediates; extremes are separate.
        let nu = ops as f64 * 2f64.powi(-23);
        nu / (1.0 - nu)
    }

    fn rounding_allowance(exact: f64, reduction_bound: f64) -> f64 {
        let magnitude = exact.abs() + reduction_bound;
        let exponent = if magnitude == 0.0 {
            -126
        } else {
            magnitude.log2().floor() as i32
        };
        2f64.powi(exponent.max(-126) - 8)
    }

    // Times complete production forwards, including dynamic activation
    // quantization and allocations. Weight quantization is startup only.
    fn library_timed(
        dev: &Device,
        mut forward: impl FnMut() -> Result<Tensor>,
    ) -> Result<(Vec<bf16>, f64)> {
        let Device::Cuda(cuda) = dev else {
            candle_core::bail!("CUDA required");
        };
        for _ in 0..3 {
            let _ = forward()?;
        }
        dev.synchronize()?;
        let stream = cuda.cuda_stream();
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        let mut last = None;
        for _ in 0..5 {
            last = Some(forward()?);
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        let ms = start.elapsed_ms(&end).map_err(candle_core::Error::wrap)? as f64 / 5.0;
        let out = last.ok_or_else(|| candle_core::Error::Msg("missing output".into()))?;
        Ok((out.flatten_all()?.to_vec1::<bf16>()?, ms))
    }

    fn library_operand(len: usize, width: usize, seed: usize, outlier: bool) -> Vec<bf16> {
        let mut squared = 0.0;
        let mut max_error = 0f64;
        let values = (0..len)
            .map(|i| {
                let h = i
                    .wrapping_mul(seed)
                    .wrapping_add((i / width).wrapping_mul(71));
                let value = if outlier && i == len / 2 {
                    if seed == 37 {
                        6.0
                    } else {
                        0.875
                    }
                } else {
                    ((h % 1009) as f64 - 504.0) / 1260.0
                };
                let bf = bf16::from_f64(value);
                let error = (bf.to_f64() - value).abs();
                max_error = max_error.max(error);
                squared += error * error;
                bf
            })
            .collect();
        eprintln!("fp8_library_source_bf16 seed={seed} elements={len} outlier={outlier} max_abs={max_error:.9e} rms={:.9e}", (squared / len as f64).sqrt());
        values
    }

    // Full small reference or 32 evenly distributed large output points.
    // Arithmetic correctness and additional quantization loss are separate.
    fn library_assess(
        shape: (usize, usize, usize),
        values: &[bf16],
        label: &str,
        mut reference: impl FnMut(usize, usize) -> (f64, f64),
        enforce: bool,
    ) {
        let (m, n, k) = shape;
        assert_eq!(values.len(), m * n);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{label}: nonfinite output"
        );
        let count = if m * n <= 1024 { m * n } else { 32 };
        let mut max_abs = 0f64;
        let mut max_rel = 0f64;
        let mut squared = 0f64;
        let mut max_bound = 0f64;
        let mut cancellation = 0;
        let mut bound_violations = 0;
        let mut first_bound_violation = None;
        let mut max_bound_excess = 0f64;
        for i in 0..count {
            let index = if count == m * n {
                i
            } else {
                i * (m * n - 1) / (count - 1)
            };
            let (exact, sum_abs) = reference(index / n, index % n);
            let error = (values[index].to_f64() - exact).abs();
            // FP32 reduction, products/scales and BF16 output rounding;
            // dequantize_w's FP32 operand rounding is covered as well.
            let reduction = gamma(2 * k + 8) * sum_abs;
            let bound = reduction + rounding_allowance(exact, reduction);
            max_bound = max_bound.max(bound);
            max_abs = max_abs.max(error);
            squared += error * error;
            if exact.abs() >= sum_abs * 2f64.powi(-10) && exact != 0.0 {
                max_rel = max_rel.max(error / exact.abs());
            } else {
                cancellation += 1;
            }
            if error > bound {
                bound_violations += 1;
                max_bound_excess = max_bound_excess.max(error - bound);
                if first_bound_violation.is_none() {
                    first_bound_violation = Some((index, exact, error, bound));
                }
            }
            if enforce {
                assert!(
                    error <= bound,
                    "{label} index={index} exact={exact} abs={error} bound={bound}"
                );
            }
        }
        eprintln!("fp8_library_accuracy backend={label} M={m} N={n} K={k} samples={count} enforced={enforce} max_abs={max_abs:.9e} rms={:.9e} max_rel_away_from_cancellation={max_rel:.9e} cancellation_samples={cancellation} max_bound={max_bound:.9e}", (squared / count as f64).sqrt());
        eprintln!("fp8_library_bound_diagnostic backend={label} fp32_single_round_bound_violations={bound_violations} max_bound_excess={max_bound_excess:.9e} first_violation_index_exact_error_bound={first_bound_violation:?}");
    }

    fn library_dot(a: &[f32], w: &[f32], k: usize, row: usize, col: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut abs = 0.0;
        for j in 0..k {
            let term = a[row * k + j] as f64 * w[col * k + j] as f64;
            sum += term;
            abs += term.abs();
        }
        (sum, abs)
    }

    fn library_observe(
        layer: &crate::FP8Linear,
        label: &str,
        probe_gpu_cast: bool,
    ) -> Result<Vec<f32>> {
        use crate::QuantMethod;
        if probe_gpu_cast {
            eprintln!("fp8_library_phase stage=gpu_dequantize_w_observer_begin operand={label}");
            match layer.dequantize_w() {
            Ok(_) => eprintln!("fp8_library_observer operand={label} production_gpu_dequantize_w=available"),
            Err(err) => eprintln!("fp8_library_observer operand={label} production_gpu_dequantize_w=failed error={err} expected_candle_symbol=cast_f8e4m3_f32 compiled_candle_symbol=cast_f8_e4m3_f32"),
        }
        }
        // Observe the actual constructor-owned FP8 tensor and reciprocal
        // scale. Only the observer moves to CPU; forwards remain unchanged.
        // Candle's CUDA cast requests cast_f8e4m3_f32, while pinned cast.cu
        // defines cast_f8_e4m3_f32. Preserve that production observer defect
        // above rather than fixing production code in a timing fixture.
        eprintln!("fp8_library_phase stage=cpu_stored_operand_observer_begin operand={label}");
        let weight = layer.lin.weight().to_device(&Device::Cpu)?;
        let scale = layer.dequant_w_scale.to_device(&Device::Cpu)?;
        let values = weight
            .to_dtype(DType::F32)?
            .broadcast_mul(&scale)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        eprintln!("fp8_library_phase stage=cpu_stored_operand_observer_done operand={label}");
        Ok(values)
    }

    fn library_compare(m: usize, n: usize, k: usize, outlier: bool) -> Result<()> {
        use crate::{FP8Linear, QuantMethod, QuantMethodConfig};
        let dev = device()?;
        let a = library_operand(m * k, k, 37, outlier);
        let w = library_operand(n * k, k, 53, outlier);
        let input = Tensor::from_slice(&a, (1, m, k), &dev)?;
        let weight = Tensor::from_slice(&w, (n, k), &dev)?;
        // Tensor handles share the immutable source; no FP8 artifact is
        // treated as a dense parent for any comparator.
        crate::cublaslt::maybe_init_cublas_lt_wrapper(input.device().to_owned());
        assert!(
            crate::cublaslt::CUBLASLT_CONTROLLER
                .get_for_device(&dev)
                .is_some(),
            "production FP8 cuBLASLt route unavailable; fallback is not a comparison"
        );
        let Device::Cuda(cuda) = &dev else {
            candle_core::bail!("CUDA required");
        };
        let stream = cuda.cuda_stream();
        dev.synchronize()?;
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        let fp8 = FP8Linear::new(QuantMethodConfig::FP8 {
            lin: candle_nn::Linear::new(weight.to_owned(), None),
            dtype: DType::F8E4M3,
        })?;
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        let startup_ms = start.elapsed_ms(&end).map_err(candle_core::Error::wrap)?;
        eprintln!("fp8_library_route M={m} N={n} K={k} route=FP8Linear_cublaslt weight_quantization_startup_ms={startup_ms:.6} outlier={outlier}");
        let (bf, bf_ms) = library_timed(&dev, || input.squeeze(0)?.matmul(&weight.t()?))?;
        let (qw, scales) = fp8_blockwise_quantize(&weight, vec![128, 128])?;
        let quant_block_w = qw.flatten_all()?.to_vec1::<F8E4M3>()?;
        let quant_block_s = scales.flatten_all()?.to_vec1::<f32>()?;
        let (scalar, scalar_ms) = library_timed(&dev, || {
            fp8_blockwise_matmul(&input.squeeze(0)?, &qw, &scales, &[128, 128])
        })?;
        eprintln!("fp8_library_phase stage=blockwise_dequant_bf16_timing_begin M={m} N={n} K={k}");
        let (dequant, dequant_ms) = library_timed(&dev, || {
            // Charge the maintained dequantizer, BF16 weight allocation, and
            // matmul on every forward. No persistent dense weight cache.
            let dense = fp8_blockwise_dequantize(&qw, &scales, vec![128, 128], DType::BF16)?;
            input.squeeze(0)?.matmul(&dense.t()?)
        })?;
        eprintln!("fp8_library_phase stage=blockwise_dequant_bf16_timing_done M={m} N={n} K={k}");
        let source_a = a.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let source_w = w.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let reduced_bf16 = candle_core::cuda_backend::gemm_reduced_precision_bf16();
        let reduced_f32 = candle_core::cuda_backend::gemm_reduced_precision_f32();
        eprintln!("fp8_library_candle_precision reduced_bf16={reduced_bf16} reduced_f32={reduced_f32} cublas_math_mode=default_unmodified bf16_reduction_policy=not_forced_to_fp32");
        // Candle 27f20fe requests COMPUTE_32F with BF16 output and default
        // cuBLAS math mode. COMPUTE_32F alone does not prohibit BF16 partial
        // reductions; DISALLOW_REDUCED_PRECISION_REDUCTION would be required.
        // https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior
        // This does not establish which algorithm produced the v1 mismatch.
        assert!(
            !reduced_bf16 && !reduced_f32,
            "fixture requires Candle's default compute settings"
        );
        if m * n <= 1024 {
            assert_eq!(
                input.flatten_all()?.to_vec1::<bf16>()?,
                a,
                "source activation upload"
            );
            assert_eq!(
                weight.flatten_all()?.to_vec1::<bf16>()?,
                w,
                "source weight upload"
            );
            // Same BF16 values lifted exactly to F32. Full F32 output avoids
            // mixed BF16 output reductions, then one explicit BF16 conversion
            // supplies a small sanity check for source/layout/reference math.
            // Its cost is excluded from every reported comparator timing.
            let sanity = input
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .matmul(&weight.to_dtype(DType::F32)?.t()?)?
                .to_dtype(DType::BF16)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            library_assess(
                (m, n, k),
                &sanity,
                "f32_source_then_bf16_reference_sanity",
                |r, c| library_dot(&source_a, &source_w, k, r, c),
                true,
            );
        }
        let block_reference = |r: usize, c: usize| {
            let mut sum = 0.0;
            let mut abs = 0.0;
            for j in 0..k {
                let term = reference_term(
                    a[r * k + j],
                    quant_block_w[c * k + j],
                    quant_block_s[c / 128 * k.div_ceil(128) + j / 128],
                );
                sum += term;
                abs += term.abs();
            }
            (sum, abs)
        };
        library_assess(
            (m, n, k),
            &scalar,
            "blockwise_scalar_vs_quantized",
            block_reference,
            true,
        );
        eprintln!("fp8_library_acceptance backend=blockwise_dequant_bf16_gemm status=arithmeticUnadmitted reason=bf16_weight_rounding_and_default_reduction_policy");
        library_assess(
            (m, n, k),
            &dequant,
            "blockwise_dequant_bf16_vs_actual_blockquantized",
            block_reference,
            false,
        );
        library_assess(
            (m, n, k),
            &dequant,
            "blockwise_dequant_bf16_vs_source",
            |r, c| library_dot(&source_a, &source_w, k, r, c),
            false,
        );
        library_assess(
            (m, n, k),
            &scalar,
            "blockwise_scalar_vs_source",
            |r, c| library_dot(&source_a, &source_w, k, r, c),
            false,
        );
        // Probe the exact requested workload before introducing padding.
        eprintln!("fp8_library_phase stage=production_forward_probe_begin M={m} N={n} K={k}");
        let needs_padding = match fp8.forward_raw(&input) {
            Ok(output) => {
                dev.synchronize()?;
                drop(output);
                eprintln!(
                    "fp8_library_phase stage=production_forward_probe_done M={m} N={n} K={k}"
                );
                false
            }
            Err(err) => {
                eprintln!("fp8_library_original_shape_error M={m} N={n} K={k} error={err}");
                true
            }
        };
        let padded_m = m.div_ceil(16) * 16;
        eprintln!("fp8_library_phase stage=production_forward_timing_begin M={m} N={n} K={k} padding={needs_padding}");
        let (fq, fq_ms) = if needs_padding {
            assert_ne!(
                m, padded_m,
                "already aligned shape failed; do not hide unsupported route"
            );
            library_timed(&dev, || {
                // Charge padding allocation/copy and output slicing to
                // every forward; zero padding preserves absmax/scales.
                let zero = Tensor::zeros((1, padded_m - m, k), DType::BF16, &dev)?;
                let padded = Tensor::cat(&[&input, &zero], 1)?;
                fp8.forward_raw(&padded)?.narrow(1, 0, m)?.contiguous()
            })?
        } else {
            library_timed(&dev, || fp8.forward_raw(&input))?
        };
        eprintln!("fp8_library_phase stage=production_forward_timing_done M={m} N={n} K={k}");
        let quant_w = library_observe(&fp8, "weight", m * n <= 1024 && !outlier)?;
        // Observe production activation quantization via the same public
        // constructor, rather than copying the private quantizer formula.
        let act = FP8Linear::new(QuantMethodConfig::FP8 {
            lin: candle_nn::Linear::new(input.squeeze(0)?, None),
            dtype: DType::F8E4M3,
        })?;
        let quant_a = library_observe(&act, "activation", false)?;
        // The unchanged FP32 bound remains a reported diagnostic while the
        // owner diagnoses near-cancellation accumulation discrepancies.
        eprintln!("fp8_library_acceptance backend=FP8Linear status=arithmeticUnadmitted reason=fp32_bound_discrepancy_under_investigation");
        library_assess(
            (m, n, k),
            &fq,
            "FP8Linear_vs_actual_quantized",
            |r, c| library_dot(&quant_a, &quant_w, k, r, c),
            false,
        );
        library_assess(
            (m, n, k),
            &fq,
            "FP8Linear_vs_source",
            |r, c| library_dot(&source_a, &source_w, k, r, c),
            false,
        );
        eprintln!("fp8_library_timing M={m} N={n} K={k} warmup=3 iterations=5 bf16_library_ms={bf_ms:.6} blockwise_scalar_ms={scalar_ms:.6} blockwise_dequant_plus_bf16_gemm_ms={dequant_ms:.6} blockwise_dequant_temporary_bf16_weight_bytes={} FP8Linear_dynamic_activation_plus_gemm_ms={fq_ms:.6} padding_and_slicing_charged={needs_padding} execution_M={} weight_quantization_excluded=true dynamic_activation_fp8_bytes={} output_bf16_bytes={} padding_bf16_bytes={} blockwise_scalar_output_allocation_charged=true known_tensor_bytes_only=true quantizer_reduction_scratch_bytes=unmeasured library_workspace_bytes=unknown", n*k*2, if needs_padding {padded_m} else {m}, if needs_padding {padded_m*k} else {m*k}, if needs_padding {padded_m*n*2} else {m*n*2}, if needs_padding {padded_m*k*2} else {0});
        // The default cuBLAS BF16 reduction policy does not guarantee the
        // FP32 + one final BF16 rounding model. Retain the exact v1 threshold
        // as an unadmitted diagnostic; no BF16 correctness claim follows.
        eprintln!("fp8_library_acceptance backend=bf16_default status=unadmitted reason=fp32_single_round_reduction_policy_unproven v1_small_bound_failure_retained=true");
        library_assess(
            (m, n, k),
            &bf,
            "bf16_default_unadmitted_vs_source",
            |r, c| library_dot(&source_a, &source_w, k, r, c),
            false,
        );
        Ok(())
    }

    fn production_wrapper(m: usize, n: usize, k: usize, batch: usize, bias: bool) -> Result<()> {
        use crate::{BlockwiseFP8Linear, QuantMethod, QuantMethodConfig};
        let dev = device()?;
        let a = library_operand(m * k, k, 37, true);
        let w = library_operand(n * k, k, 53, true);
        let input = Tensor::from_slice(&a, (batch, m / batch, k), &dev)?;
        let dense = Tensor::from_slice(&w, (n, k), &dev)?;
        let (qw, scales) = fp8_blockwise_quantize(&dense, vec![128, 128])?;
        let bias = if bias {
            Some(Tensor::from_vec(
                (0..n)
                    .map(|i| bf16::from_f32((i % 17) as f32 / 64.0))
                    .collect::<Vec<_>>(),
                n,
                &dev,
            )?)
        } else {
            None
        };
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight: qw.to_owned(),
            weight_scale_inv: scales.to_owned(),
            bias: bias.as_ref().cloned(),
            dequant_dtype: DType::BF16,
            weight_block_size: vec![128, 128],
        })?;
        let original = crate::serving_observation::counters();
        let probe = layer.forward_raw(&input)?;
        assert_eq!(probe.dims(), &[batch, m / batch, n]);
        let after = crate::serving_observation::counters();
        assert!(
            crate::serving_observation::enabled(),
            "wrapper route evidence requires observations"
        );
        let selected = m >= 128 && n * k >= 1_048_576;
        assert_eq!(
            after[0] - original[0],
            usize::from(!selected),
            "native route counter"
        );
        assert_eq!(
            after[1] - original[1],
            usize::from(selected),
            "dequant route counter"
        );
        let add_bias = |out: Tensor| -> Result<Tensor> {
            match &bias {
                Some(b) => out.broadcast_add(b),
                None => Ok(out),
            }
        };
        let (native, native_ms) = library_timed(&dev, || {
            add_bias(fp8_blockwise_matmul(
                &input.reshape((m, k))?,
                &qw,
                &scales,
                &[128, 128],
            )?)
        })?;
        let (dequant, dequant_ms) = library_timed(&dev, || {
            let w = fp8_blockwise_dequantize(&qw, &scales, vec![128, 128], DType::BF16)?;
            add_bias(input.reshape((m, k))?.matmul(&w.t()?)?)
        })?;
        let (actual, wrapper_ms) = library_timed(&dev, || layer.forward_raw(&input))?;
        let expected = if selected { &dequant } else { &native };
        assert_eq!(
            &actual, expected,
            "production route must match the maintained comparator exactly, bias once"
        );
        assert!(actual.iter().all(|v| v.is_finite()));
        let q = qw.flatten_all()?.to_vec1::<F8E4M3>()?;
        let scale = scales.flatten_all()?.to_vec1::<f32>()?;
        let bias_host = bias.as_ref().map(|b| b.to_vec1::<bf16>()).transpose()?;
        library_assess(
            (m, n, k),
            &actual,
            "production_blockwise_wrapper_vs_stored_operands",
            |r, c| {
                let mut sum = 0.0;
                let mut abs = 0.0;
                for j in 0..k {
                    let term = reference_term(
                        a[r * k + j],
                        q[c * k + j],
                        scale[c / 128 * k.div_ceil(128) + j / 128],
                    );
                    sum += term;
                    abs += term.abs();
                }
                if let Some(b) = &bias_host {
                    sum += b[c].to_f64();
                    abs += b[c].to_f64().abs();
                }
                (sum, abs)
            },
            false,
        );
        eprintln!("fp8_library_wrapper M={m} N={n} K={k} batch={batch} bias={} selected_dequant={selected} warmup=3 iterations=5 native_ms={native_ms:.6} dequant_bf16_gemm_ms={dequant_ms:.6} actual_wrapper_ms={wrapper_ms:.6} transient_bf16_weight_bytes={} exact_comparator_match=true arithmetic_admission=unproven library_workspace_bytes=unknown",bias.is_some(),n*k*2);
        Ok(())
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_wrapper_small() -> Result<()> {
        production_wrapper(32, 145, 257, 2, true)?;
        production_wrapper(128, 145, 257, 2, true)
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_wrapper_crossover() -> Result<()> {
        for m in [1, 32, 128] {
            production_wrapper(m, 5120, 17408, 1, false)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_wrapper_mlp_up() -> Result<()> {
        production_wrapper(1394, 17408, 5120, 1, false)
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_wrapper_mlp_down() -> Result<()> {
        production_wrapper(1394, 5120, 17408, 1, false)
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_small() -> Result<()> {
        library_compare(17, 32, 48, false)?;
        library_compare(17, 32, 48, true)
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_mlp_up() -> Result<()> {
        library_compare(1394, 17408, 5120, true)
    }

    #[test]
    #[ignore = "real CUDA; owner-held GPU leases and bounded process required"]
    fn fp8_library_cuda_mlp_down() -> Result<()> {
        library_compare(1394, 5120, 17408, true)
    }
}
