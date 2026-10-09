---
title: FP8 prefill library candidate
description: Measured maintained-library route for large dense BF16 prefill.
---

The candidate dense `BlockwiseFP8Linear` CUDA forward selects the maintained blockwise dequantizer plus `UnquantLinear` for BF16 inputs with at least 128 flattened rows, at least 1,048,576 weight elements, and BF16 dequantization dtype. It flattens leading dimensions to rank two, uses Candle cuBLAS matmul, applies bias once, and restores the original leading dimensions. Smaller dense rows or weights, single-token decode, and other dtypes retain the existing native path. Indexed MoE dispatch is unchanged.

FP8 storage, block scales, artifact formats, and loading contracts remain unchanged. Every selected forward materializes a transient BF16 weight tensor of `N * K * 2` bytes; it does not retain a dense weight cache. Dispatch records the existing FP8 dequantization counter. CUDA library workspace memory remains unmeasured.

The size policy conservatively excludes the measured tiny `M=128, N=145, K=257` regression while retaining the dominant model shapes. It does not claim an optimal crossover.

This selection is an inference candidate from the measured maintained-library comparisons, not a whole-model performance acceptance. The ignored wrapper fixtures compare the actual production route against the same maintained comparator, assert exact output agreement and route counters, and report original-operand numerical discrepancies separately. Default BF16 library reduction policy and additional BF16 weight rounding remain explicit arithmetic-admission limitations. No custom CUDA kernel, activation quantizer, or environment opt-in is introduced.

Use `make test-fp8-library` for portable selector boundaries. The external GPU owner wrapper runs `make test-fp8-library-cuda` with `FP8_LIBRARY_CUDA_FILTER` set to `fp8_library_cuda_wrapper_small`, `fp8_library_cuda_wrapper_crossover`, `fp8_library_cuda_wrapper_mlp_up`, or `fp8_library_cuda_wrapper_mlp_down`. Crossover rows are 1, 32, and 128; dominant model rows are 1394. CUDA-event timings use three warmups and five forwards and charge dequantization and output allocation on every call.
