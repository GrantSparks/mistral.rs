#![allow(clippy::cast_possible_truncation)]

use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::DType;

#[cfg(feature = "cuda")]
fn ensure_same_device(reference: &Tensor, inputs: &[(&str, &Tensor)]) -> Result<()> {
    for &(name, tensor) in inputs {
        if !reference.device().same_device(tensor.device()) {
            candle_core::bail!(
                "GDN CUDA {name} device {:?} does not match input device {:?}",
                tensor.device().location(),
                reference.device().location()
            );
        }
    }
    Ok(())
}

/// CUDA-accelerated gated delta rule recurrence.
///
/// Inputs (all contiguous, f32):
///   q, k: [BH, S, K]  v: [BH, S, V]  g, beta: [BH, S]
///   state: [BH, K, V] (mutated in place)
///
/// Returns: output [BH, S, V]
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    ensure_same_device(
        q,
        &[
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ],
    )?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let dev = q.device().as_cuda_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q must be a cuda tensor"),
    };
    let q_offset = q_l.start_offset();

    let (k_s, k_l) = k.storage_and_layout();
    let k_s = match &*k_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k must be a cuda tensor"),
    };
    let k_offset = k_l.start_offset();

    let (v_s, v_l) = v.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("v must be a cuda tensor"),
    };
    let v_offset = v_l.start_offset();

    let (g_s, g_l) = g.storage_and_layout();
    let g_s = match &*g_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("g must be a cuda tensor"),
    };
    let g_offset = g_l.start_offset();

    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_s = match &*beta_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("beta must be a cuda tensor"),
    };
    let beta_offset = beta_l.start_offset();

    let (state_s, state_l) = state.storage_and_layout();
    let state_s = match &*state_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("state must be a cuda tensor"),
    };
    let state_offset = state_l.start_offset();

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;

    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        crate::cuda::ffi::gated_delta_rule_recurrence(
            q_s.slice(q_offset..).device_ptr(q_s.stream()).0 as *const f32,
            k_s.slice(k_offset..).device_ptr(k_s.stream()).0 as *const f32,
            v_s.slice(v_offset..).device_ptr(v_s.stream()).0 as *const f32,
            g_s.slice(g_offset..).device_ptr(g_s.stream()).0 as *const f32,
            beta_s.slice(beta_offset..).device_ptr(beta_s.stream()).0 as *const f32,
            state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
            output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
            bh as i32,
            seq_len as i32,
            k_dim as i32,
            v_dim as i32,
            stream,
        );
    }

    // The kernel wrote state in-place via the raw pointer; rewrap
    // (state tensor's underlying CudaSlice was modified directly)

    let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (bh, seq_len, v_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn gated_delta_rule_recurrence_cuda(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    candle_core::bail!("gated_delta_rule_recurrence_cuda requires the cuda feature")
}

/// CUDA-accelerated chunked gated delta rule recurrence (prefill optimization).
///
/// Processes prefill tokens in 64-token chunks instead of one at a time.
/// Same interface as `gated_delta_rule_recurrence_cuda`.
///
/// Inputs (all contiguous, f32):
///   q, k: [BH, S, K]  v: [BH, S, V]  g, beta: [BH, S]
///   state: [BH, K, V] (mutated in place)
///
/// Returns: output [BH, S, V]
#[cfg(feature = "cuda")]
pub fn chunked_gated_delta_rule_recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    ensure_same_device(
        q,
        &[
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ],
    )?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let dev = q.device().as_cuda_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q must be a cuda tensor"),
    };
    let q_offset = q_l.start_offset();

    let (k_s, k_l) = k.storage_and_layout();
    let k_s = match &*k_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k must be a cuda tensor"),
    };
    let k_offset = k_l.start_offset();

    let (v_s, v_l) = v.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("v must be a cuda tensor"),
    };
    let v_offset = v_l.start_offset();

    let (g_s, g_l) = g.storage_and_layout();
    let g_s = match &*g_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("g must be a cuda tensor"),
    };
    let g_offset = g_l.start_offset();

    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_s = match &*beta_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("beta must be a cuda tensor"),
    };
    let beta_offset = beta_l.start_offset();

    let (state_s, state_l) = state.storage_and_layout();
    let state_s = match &*state_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("state must be a cuda tensor"),
    };
    let state_offset = state_l.start_offset();

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;

    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        crate::cuda::ffi::chunked_gated_delta_rule_recurrence(
            q_s.slice(q_offset..).device_ptr(q_s.stream()).0 as *const f32,
            k_s.slice(k_offset..).device_ptr(k_s.stream()).0 as *const f32,
            v_s.slice(v_offset..).device_ptr(v_s.stream()).0 as *const f32,
            g_s.slice(g_offset..).device_ptr(g_s.stream()).0 as *const f32,
            beta_s.slice(beta_offset..).device_ptr(beta_s.stream()).0 as *const f32,
            state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
            output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
            bh as i32,
            seq_len as i32,
            k_dim as i32,
            v_dim as i32,
            stream,
        );
    }

    let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (bh, seq_len, v_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn chunked_gated_delta_rule_recurrence_cuda(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    candle_core::bail!("chunked_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
pub fn warp_gated_delta_rule_recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    ensure_same_device(
        q,
        &[
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ],
    )?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let dev = q.device().as_cuda_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q must be a cuda tensor"),
    };
    let q_offset = q_l.start_offset();

    let (k_s, k_l) = k.storage_and_layout();
    let k_s = match &*k_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k must be a cuda tensor"),
    };
    let k_offset = k_l.start_offset();

    let (v_s, v_l) = v.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("v must be a cuda tensor"),
    };
    let v_offset = v_l.start_offset();

    let (g_s, g_l) = g.storage_and_layout();
    let g_s = match &*g_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("g must be a cuda tensor"),
    };
    let g_offset = g_l.start_offset();

    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_s = match &*beta_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("beta must be a cuda tensor"),
    };
    let beta_offset = beta_l.start_offset();

    let (state_s, state_l) = state.storage_and_layout();
    let state_s = match &*state_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("state must be a cuda tensor"),
    };
    let state_offset = state_l.start_offset();

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        crate::cuda::ffi::warp_gated_delta_rule_recurrence(
            q_s.slice(q_offset..).device_ptr(q_s.stream()).0 as *const f32,
            k_s.slice(k_offset..).device_ptr(k_s.stream()).0 as *const f32,
            v_s.slice(v_offset..).device_ptr(v_s.stream()).0 as *const f32,
            g_s.slice(g_offset..).device_ptr(g_s.stream()).0 as *const f32,
            beta_s.slice(beta_offset..).device_ptr(beta_s.stream()).0 as *const f32,
            state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
            output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
            bh as i32,
            seq_len as i32,
            k_dim as i32,
            v_dim as i32,
            stream,
        );
    }

    let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (bh, seq_len, v_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn warp_gated_delta_rule_recurrence_cuda(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    candle_core::bail!("warp_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

/// CUDA-accelerated causal conv1d (both update and full paths).
///
/// For update (is_update=true):
///   x: [B, conv_dim, 1]  weight: [conv_dim, kernel_size]
///   conv_state: [B, conv_dim, kernel_size] (mutated in place for update)
///   Returns: (output [B, conv_dim, 1], updated conv_state)
///
/// For full (is_update=false):
///   x: [B, conv_dim, S]  weight: [conv_dim, kernel_size]
///   conv_state: [B, conv_dim, kernel_size] carried in from the preceding
///     chunk and used as the left context (all-zeros for a fresh sequence)
///   Returns: (output [B, conv_dim, S], new conv_state [B, conv_dim, kernel_size])
#[cfg(feature = "cuda")]
pub fn causal_conv1d_cuda(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    is_update: bool,
) -> Result<(Tensor, Tensor)> {
    ensure_same_device(x, &[("weight", weight), ("conv_state", conv_state)])?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;
    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        x: &Tensor,
        weight: &Tensor,
        conv_state: &Tensor,
        kernel_size: usize,
        is_update: bool,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let dev = x.device().as_cuda_device()?;
        let (batch_size, conv_dim, seq_len) = x.dims3()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("x must be a cuda tensor"),
        };
        let x_offset = x_l.start_offset();

        let (w_s, w_l) = weight.storage_and_layout();
        let w_s = match &*w_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let w_offset = w_l.start_offset();

        let stream = dev.cuda_stream().cu_stream() as i64;

        if is_update {
            // Clone conv_state so the kernel can mutate it in place
            let conv_state_new = conv_state.clone();

            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim) }?;

            // Scope the borrow of conv_state_new so we can move it later
            {
                let (cs_s, cs_l) = conv_state_new.storage_and_layout();
                let cs_s = match &*cs_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("conv_state must be a cuda tensor"),
                };
                let cs_offset = cs_l.start_offset();

                unsafe {
                    crate::cuda::ffi::causal_conv1d_update(
                        x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                        w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                        cs_s.slice(cs_offset..).device_ptr(cs_s.stream()).0 as *mut c_void,
                        output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                        batch_size as i32,
                        conv_dim as i32,
                        kernel_size as i32,
                        dtype_code,
                        stream,
                    );
                }
            }

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, conv_dim, 1usize),
            ));

            Ok((output, conv_state_new))
        } else {
            // Full path: seed the left context from the incoming conv_state and
            // write the new state into a fresh buffer. The incoming state is a
            // distinct allocation from cs_buf, so the kernel's restrict
            // pointers do not alias.
            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * seq_len) }?;
            let cs_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * kernel_size) }?;

            let (cs_in_storage, cs_in_l) = conv_state.storage_and_layout();
            let cs_in_s = match &*cs_in_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("conv_state must be a cuda tensor"),
            };
            let cs_in_offset = cs_in_l.start_offset();

            unsafe {
                crate::cuda::ffi::causal_conv1d_full(
                    x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                    w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                    cs_in_s.slice(cs_in_offset..).device_ptr(cs_in_s.stream()).0 as *const c_void,
                    cs_buf.device_ptr(cs_buf.stream()).0 as *mut c_void,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                    batch_size as i32,
                    conv_dim as i32,
                    seq_len as i32,
                    kernel_size as i32,
                    dtype_code,
                    stream,
                );
            }

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, conv_dim, seq_len),
            ));

            let cs_storage = candle::CudaStorage::wrap_cuda_slice(cs_buf, dev.clone());
            let new_conv_state = Tensor::from((
                candle::Storage::Cuda(cs_storage),
                (batch_size, conv_dim, kernel_size),
            ));

            Ok((output, new_conv_state))
        }
    }

    match x.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(x, weight, conv_state, kernel_size, is_update, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(x, weight, conv_state, kernel_size, is_update, 1),
        other => candle_core::bail!("causal_conv1d_cuda only supports f16/bf16, got {:?}", other),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn causal_conv1d_cuda(
    _x: &Tensor,
    _weight: &Tensor,
    _conv_state: &Tensor,
    _kernel_size: usize,
    _is_update: bool,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("causal_conv1d_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn prepare_recurrence_inputs_cuda(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    ensure_same_device(
        mixed_qkv,
        &[("b", b), ("a", a), ("a_log", a_log), ("dt_bias", dt_bias)],
    )?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        mixed_qkv: &Tensor,
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        batch_size: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dev = mixed_qkv.device().as_cuda_device()?;

        let (mixed_s, mixed_l) = mixed_qkv.storage_and_layout();
        let mixed_s = match &*mixed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("mixed_qkv must be a cuda tensor"),
        };
        let mixed_offset = mixed_l.start_offset();

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let bh = batch_size * num_v_heads;
        let q_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_k_dim) }?;
        let k_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_k_dim) }?;
        let v_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_v_dim) }?;
        let g_buf = unsafe { dev.alloc::<f32>(bh * seq_len) }?;
        let beta_buf = unsafe { dev.alloc::<f32>(bh * seq_len) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_prepare_recurrence(
                mixed_s.slice(mixed_offset..).device_ptr(mixed_s.stream()).0 as *const c_void,
                b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                q_buf.device_ptr(q_buf.stream()).0 as *mut f32,
                k_buf.device_ptr(k_buf.stream()).0 as *mut f32,
                v_buf.device_ptr(v_buf.stream()).0 as *mut f32,
                g_buf.device_ptr(g_buf.stream()).0 as *mut f32,
                beta_buf.device_ptr(beta_buf.stream()).0 as *mut f32,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                dtype_code,
                stream,
            );
        }

        let q = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(q_buf, dev.clone())),
            (bh, seq_len, head_k_dim),
        ));
        let k = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(k_buf, dev.clone())),
            (bh, seq_len, head_k_dim),
        ));
        let v = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(v_buf, dev.clone())),
            (bh, seq_len, head_v_dim),
        ));
        let g = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(g_buf, dev.clone())),
            (bh, seq_len),
        ));
        let beta = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(beta_buf, dev.clone())),
            (bh, seq_len),
        ));

        Ok((q, k, v, g, beta))
    }

    match mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            batch_size,
            seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            0,
        ),
        DType::BF16 => cuda_fwd::<half::bf16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            batch_size,
            seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            1,
        ),
        other => candle_core::bail!(
            "prepare_recurrence_inputs_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused, clippy::too_many_arguments)]
pub fn prepare_recurrence_inputs_cuda(
    _mixed_qkv: &Tensor,
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
    _batch_size: usize,
    _seq_len: usize,
    _num_k_heads: usize,
    _num_v_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    candle_core::bail!("prepare_recurrence_inputs_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_recurrence_cuda(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    batch_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<Tensor> {
    ensure_same_device(
        mixed_qkv,
        &[
            ("b", b),
            ("a", a),
            ("a_log", a_log),
            ("dt_bias", dt_bias),
            ("state", state),
        ],
    )?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        mixed_qkv: &Tensor,
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        batch_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let dev = mixed_qkv.device().as_cuda_device()?;

        let (mixed_s, mixed_l) = mixed_qkv.storage_and_layout();
        let mixed_s = match &*mixed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("mixed_qkv must be a cuda tensor"),
        };
        let mixed_offset = mixed_l.start_offset();

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let (state_s, state_l) = state.storage_and_layout();
        let state_s = match &*state_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("state must be a cuda tensor"),
        };
        let state_offset = state_l.start_offset();

        let bh = batch_size * num_v_heads;
        let output_buf = unsafe { dev.alloc::<f32>(bh * head_v_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_decode_recurrence(
                mixed_s.slice(mixed_offset..).device_ptr(mixed_s.stream()).0 as *const c_void,
                b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
                output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
                batch_size as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                dtype_code,
                stream,
            );
        }

        Ok(Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(
                output_buf,
                dev.clone(),
            )),
            (bh, 1, head_v_dim),
        )))
    }

    match mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            0,
        ),
        DType::BF16 => cuda_fwd::<half::bf16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            1,
        ),
        other => candle_core::bail!(
            "fused_decode_recurrence_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused, clippy::too_many_arguments)]
pub fn fused_decode_recurrence_cuda(
    _mixed_qkv: &Tensor,
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
    _state: &mut Tensor,
    _batch_size: usize,
    _num_k_heads: usize,
    _num_v_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
) -> Result<Tensor> {
    candle_core::bail!("fused_decode_recurrence_cuda requires the cuda feature")
}

/// CUDA-accelerated fused GDN gating computation.
///
/// Computes: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
#[cfg(feature = "cuda")]
pub fn rmsnorm_gated_cuda(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    ensure_same_device(x, &[("gate", gate), ("weight", weight)])?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        x: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f64,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let gate = gate.contiguous()?;
        let weight = weight.contiguous()?;
        let (rows, hidden_dim) = x.dims2()?;
        let dev = x.device().as_cuda_device()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("x must be a cuda tensor"),
        };
        let x_offset = x_l.start_offset();

        let (gate_s, gate_l) = gate.storage_and_layout();
        let gate_s = match &*gate_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("gate must be a cuda tensor"),
        };
        let gate_offset = gate_l.start_offset();

        let (weight_s, weight_l) = weight.storage_and_layout();
        let weight_s = match &*weight_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let weight_offset = weight_l.start_offset();

        let output_buf = unsafe { dev.alloc::<T>(rows * hidden_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_rmsnorm_gated(
                x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                gate_s.slice(gate_offset..).device_ptr(gate_s.stream()).0 as *const c_void,
                weight_s
                    .slice(weight_offset..)
                    .device_ptr(weight_s.stream())
                    .0 as *const c_void,
                output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                rows as i32,
                hidden_dim as i32,
                eps as f32,
                dtype_code,
                stream,
            );
        }

        let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
        Ok(Tensor::from((
            candle::Storage::Cuda(output_storage),
            (rows, hidden_dim),
        )))
    }

    match x.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(x, gate, weight, eps, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(x, gate, weight, eps, 1),
        other => candle_core::bail!("rmsnorm_gated_cuda only supports f16/bf16, got {:?}", other),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn rmsnorm_gated_cuda(
    _x: &Tensor,
    _gate: &Tensor,
    _weight: &Tensor,
    _eps: f64,
) -> Result<Tensor> {
    candle_core::bail!("rmsnorm_gated_cuda requires the cuda feature")
}

/// b, a: [total_elements] in f16/bf16
/// a_log, dt_bias: [num_heads] in f32
///
/// Returns: (beta, g) in original dtype
#[cfg(feature = "cuda")]
pub fn fused_gdn_gating_cuda(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    ensure_same_device(b, &[("a", a), ("a_log", a_log), ("dt_bias", dt_bias)])?;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let total_elements = b.elem_count();
        let num_heads = a_log.elem_count();
        let shape = b.shape().clone();
        let dev = b.device().as_cuda_device()?;

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let beta_buf = unsafe { dev.alloc::<T>(total_elements) }?;
        let g_buf = unsafe { dev.alloc::<T>(total_elements) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::fused_gdn_gating(
                b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                beta_buf.device_ptr(beta_buf.stream()).0 as *mut c_void,
                g_buf.device_ptr(g_buf.stream()).0 as *mut c_void,
                total_elements as i32,
                num_heads as i32,
                dtype_code,
                stream,
            );
        }

        let beta_storage = candle::CudaStorage::wrap_cuda_slice(beta_buf, dev.clone());
        let beta = Tensor::from((candle::Storage::Cuda(beta_storage), shape.clone()));

        let g_storage = candle::CudaStorage::wrap_cuda_slice(g_buf, dev.clone());
        let g = Tensor::from((candle::Storage::Cuda(g_storage), shape));

        Ok((beta, g))
    }

    match b.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(b, a, a_log, dt_bias, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(b, a, a_log, dt_bias, 1),
        other => candle_core::bail!(
            "fused_gdn_gating_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn fused_gdn_gating_cuda(
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("fused_gdn_gating_cuda requires the cuda feature")
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires two CUDA devices"]
    fn bf16_layer_device_launch_rejects_foreign_tensors() -> Result<()> {
        let first = Device::new_cuda(0)?;
        let second = Device::new_cuda(1)?;
        let local = Tensor::ones(1, DType::F32, &second)?;
        let mut foreign = Tensor::zeros(1, DType::F32, &first)?;
        for error in [
            gated_delta_rule_recurrence_cuda(&local, &local, &local, &local, &local, &mut foreign)
                .expect_err("foreign scalar state"),
            chunked_gated_delta_rule_recurrence_cuda(
                &local,
                &local,
                &local,
                &local,
                &local,
                &mut foreign,
            )
            .expect_err("foreign chunked state"),
            warp_gated_delta_rule_recurrence_cuda(
                &local,
                &local,
                &local,
                &local,
                &local,
                &mut foreign,
            )
            .expect_err("foreign warp state"),
            causal_conv1d_cuda(&local, &local, &foreign, 4, false).expect_err("foreign conv state"),
            prepare_recurrence_inputs_cuda(
                &local, &local, &local, &foreign, &local, 1, 1, 1, 1, 128, 128,
            )
            .expect_err("foreign gating weight"),
            fused_decode_recurrence_cuda(
                &local,
                &local,
                &local,
                &local,
                &local,
                &mut foreign,
                1,
                1,
                1,
                128,
                128,
            )
            .expect_err("foreign decode state"),
            rmsnorm_gated_cuda(&local, &foreign, &local, 1e-6).expect_err("foreign norm gate"),
            fused_gdn_gating_cuda(&local, &local, &foreign, &local).expect_err("foreign a_log"),
        ] {
            let message = error.to_string();
            assert!(message.contains("GDN CUDA"), "{message}");
            assert!(message.contains("does not match input device"), "{message}");
        }
        second.synchronize()?;
        assert_eq!(local.to_vec1::<f32>()?, vec![1.]);
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct RecurrenceCase {
        bh: usize,
        seq_len: usize,
        k_dim: usize,
        v_dim: usize,
    }

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as u16 as f32;
                ((x / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn tensor2(data: Vec<f32>, shape: (usize, usize), dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, dev)
    }

    fn tensor3(data: Vec<f32>, shape: (usize, usize, usize), dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, dev)
    }

    fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> (f32, usize, f32, f32) {
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut lhs_at_max = 0.0f32;
        let mut rhs_at_max = 0.0f32;
        for (idx, (&left, &right)) in lhs.iter().zip(rhs).enumerate() {
            let diff = (left - right).abs();
            if diff > max_diff || diff.is_nan() {
                max_diff = diff;
                max_idx = idx;
                lhs_at_max = left;
                rhs_at_max = right;
            }
        }
        (max_diff, max_idx, lhs_at_max, rhs_at_max)
    }

    fn assert_close(label: &str, lhs: &[f32], rhs: &[f32], tol: f32) {
        let lhs_nan = lhs.iter().filter(|x| x.is_nan()).count();
        let rhs_nan = rhs.iter().filter(|x| x.is_nan()).count();
        let (max_diff, max_idx, lhs_at_max, rhs_at_max) = max_abs_diff(lhs, rhs);
        assert!(
            lhs_nan == 0 && rhs_nan == 0 && max_diff <= tol,
            "{label}: max_diff={max_diff} at {max_idx}, lhs={lhs_at_max}, rhs={rhs_at_max}, lhs_nan={lhs_nan}, rhs_nan={rhs_nan}"
        );
    }

    fn run_case(case: RecurrenceCase, dev: &Device) -> Result<()> {
        let q = tensor3(
            patterned(case.bh * case.seq_len * case.k_dim, 1, 0.02, 0.0),
            (case.bh, case.seq_len, case.k_dim),
            dev,
        )?;
        let k = tensor3(
            patterned(case.bh * case.seq_len * case.k_dim, 2, 0.02, 0.0),
            (case.bh, case.seq_len, case.k_dim),
            dev,
        )?;
        let v = tensor3(
            patterned(case.bh * case.seq_len * case.v_dim, 3, 0.05, 0.0),
            (case.bh, case.seq_len, case.v_dim),
            dev,
        )?;
        let g = tensor2(
            patterned(case.bh * case.seq_len, 4, 0.03, -0.08),
            (case.bh, case.seq_len),
            dev,
        )?;
        let beta = tensor2(
            patterned(case.bh * case.seq_len, 5, 0.15, 0.5),
            (case.bh, case.seq_len),
            dev,
        )?;
        let state = patterned(case.bh * case.k_dim * case.v_dim, 6, 0.01, 0.0);

        let mut state_scalar = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
        let scalar = gated_delta_rule_recurrence_cuda(&q, &k, &v, &g, &beta, &mut state_scalar)?;
        let mut state_chunked = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
        let chunked =
            chunked_gated_delta_rule_recurrence_cuda(&q, &k, &v, &g, &beta, &mut state_chunked)?;
        let mut state_warp = tensor3(state, (case.bh, case.k_dim, case.v_dim), dev)?;
        let warp = warp_gated_delta_rule_recurrence_cuda(&q, &k, &v, &g, &beta, &mut state_warp)?;

        let scalar_flat = flat(&scalar)?;
        let scalar_state_flat = flat(&state_scalar)?;
        let chunked_flat = flat(&chunked)?;
        let chunked_state_flat = flat(&state_chunked)?;
        let warp_flat = flat(&warp)?;
        let warp_state_flat = flat(&state_warp)?;

        let name = format!(
            "bh={},seq={},k={},v={}",
            case.bh, case.seq_len, case.k_dim, case.v_dim
        );
        assert_close(
            &format!("{name} chunked output"),
            &scalar_flat,
            &chunked_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} chunked state"),
            &scalar_state_flat,
            &chunked_state_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} warp output"),
            &scalar_flat,
            &warp_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} warp state"),
            &scalar_state_flat,
            &warp_state_flat,
            3.0e-4,
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct ConvCase {
        batch_size: usize,
        conv_dim: usize,
        kernel_size: usize,
        tokens: usize,
        split: usize,
        state_scale: f32,
    }

    /// Continuation equals stepwise: run `split` tokens through the
    /// single-token update kernel, then feed the state it leaves behind to the
    /// multi-token kernel for the rest. Outputs and the final state must match
    /// decoding every token one at a time.
    ///
    /// Equality is exact, not approximate. Both kernels accumulate the same
    /// `kernel_size` products in the same order from the same stored values,
    /// one thread per (channel, position) with no cross-thread reduction and no
    /// atomics, so there is no launch-order nondeterminism for a tolerance to
    /// absorb -- any difference here is a difference in what the paths compute.
    fn run_conv_case(case: ConvCase, dtype: DType, dev: &Device) -> Result<()> {
        let ConvCase {
            batch_size,
            conv_dim,
            kernel_size,
            tokens,
            split,
            state_scale,
        } = case;

        let x = tensor3(
            patterned(batch_size * conv_dim * tokens, 21, 0.08, 0.01),
            (batch_size, conv_dim, tokens),
            dev,
        )?
        .to_dtype(dtype)?;
        let weight = tensor2(
            patterned(conv_dim * kernel_size, 22, 0.05, -0.01),
            (conv_dim, kernel_size),
            dev,
        )?
        .to_dtype(dtype)?;
        let initial_state = tensor3(
            patterned(batch_size * conv_dim * kernel_size, 23, state_scale, 0.0),
            (batch_size, conv_dim, kernel_size),
            dev,
        )?
        .to_dtype(dtype)?;

        // The update kernel mutates the state it is handed, so the snapshot has
        // to be an independent allocation rather than a view.
        let mut state = initial_state.copy()?;
        let mut snapshot = None;
        let mut stepwise = Vec::with_capacity(tokens);
        for t in 0..tokens {
            if t == split {
                snapshot = Some(state.copy()?);
            }
            let token = x.narrow(2, t, 1)?.contiguous()?;
            let (out, next) = causal_conv1d_cuda(&token, &weight, &state, kernel_size, true)?;
            stepwise.push(out);
            state = next;
        }
        let snapshot = snapshot.expect("split must be < tokens");

        let tail = x.narrow(2, split, tokens - split)?.contiguous()?;
        let (full_out, full_state) =
            causal_conv1d_cuda(&tail, &weight, &snapshot, kernel_size, false)?;

        let expected_out = Tensor::cat(&stepwise[split..], 2)?;
        let name = format!(
            "{dtype:?} b={batch_size},c={conv_dim},k={kernel_size},n={tokens},split={split},scale={state_scale}"
        );
        assert_close(
            &format!("{name} continuation output"),
            &flat(&full_out.to_dtype(DType::F32)?)?,
            &flat(&expected_out.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            &format!("{name} continuation state"),
            &flat(&full_state.to_dtype(DType::F32)?)?,
            &flat(&state.to_dtype(DType::F32)?)?,
            0.0,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn conv_continuation_matches_stepwise_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for case in [
                // split=0 from an all-zero state is the fresh-sequence case:
                // seeding from zeros is zero-padding, so this is also the proof
                // that a cold prefill still computes what it used to.
                ConvCase {
                    batch_size: 1,
                    conv_dim: 512,
                    kernel_size: 4,
                    tokens: 9,
                    split: 0,
                    state_scale: 0.0,
                },
                // split=0 carrying real state: the whole chunk is a resume.
                ConvCase {
                    batch_size: 2,
                    conv_dim: 512,
                    kernel_size: 4,
                    tokens: 9,
                    split: 0,
                    state_scale: 0.03,
                },
                // splits shorter than, equal to, and past the kernel.
                ConvCase {
                    batch_size: 2,
                    conv_dim: 512,
                    kernel_size: 4,
                    tokens: 9,
                    split: 2,
                    state_scale: 0.03,
                },
                ConvCase {
                    batch_size: 2,
                    conv_dim: 512,
                    kernel_size: 4,
                    tokens: 9,
                    split: 4,
                    state_scale: 0.03,
                },
                ConvCase {
                    batch_size: 2,
                    conv_dim: 512,
                    kernel_size: 4,
                    tokens: 9,
                    split: 7,
                    state_scale: 0.03,
                },
                // A tail shorter than the kernel: the saved state is then part
                // prior context, part new chunk.
                ConvCase {
                    batch_size: 1,
                    conv_dim: 4096,
                    kernel_size: 4,
                    tokens: 5,
                    split: 3,
                    state_scale: 0.03,
                },
                // The production shape: Qwen3-Next carries conv_dim 4096 with a
                // kernel of 4 across every GDN layer.
                ConvCase {
                    batch_size: 1,
                    conv_dim: 4096,
                    kernel_size: 4,
                    tokens: 33,
                    split: 16,
                    state_scale: 0.03,
                },
            ] {
                run_conv_case(case, dtype, &dev)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn warp_recurrence_matches_scalar_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for case in [
            RecurrenceCase {
                bh: 1,
                seq_len: 1,
                k_dim: 64,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 1,
                seq_len: 65,
                k_dim: 64,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 2,
                seq_len: 128,
                k_dim: 128,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 8,
                seq_len: 256,
                k_dim: 128,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 32,
                seq_len: 512,
                k_dim: 128,
                v_dim: 128,
            },
        ] {
            run_case(case, &dev)?;
        }
        Ok(())
    }
}
