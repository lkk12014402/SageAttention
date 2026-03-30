"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
SageAttention3 — Python API for Blackwell FP4 Microscaling Attention
---------------------------------------------------------------------------
Reference: SageAttention3: Microscaling FP4 Attention for Inference and An
           Exploration of 8-Bit Training (NeurIPS 2025 / arXiv:2505.11594)

Pipeline overview (corresponds to Figure 1 / Section 3 of the paper):

  Step 0 — Smooth K / Smooth Q  (Section 3.1)
    K  ← K - mean_over_tokens(K)              (per-head, per-dim channel)
    Q  ← Q - mean_over_128-token-blocks(Q)    (per-block, per-dim channel)
    ΔS ← Q_mean @ K^T                         (correction term for smoothing)

  Step 1 — FP4 microscaling quantisation  (Section 3.2)
    For each tensor X ∈ {Q, K, V}:
      • Divide every contiguous 16 elements along the head-dim axis into a
        "micro-group" (1×16 granularity, matching Blackwell hardware).
      • scale_group = max(|x_i for i in group|) / 6.0  → stored as FP8 e4m3
      • x_fp4_i     = round_e2m1(x_i / scale_group)   → stored as 4-bit e2m1

  Step 2 — Blocked-scaled FP4 attention kernel  (Section 3.3 / Algorithm 1)
    For each (m_block, head, batch) tile:
      S  = Q_fp4 @ K_fp4^T  (FP4×FP4 → FP32 via Blackwell SM120 MMA atom)
         + ΔS                (add the smoothing correction in FP32)
      P  = online_softmax(S * softmax_scale)    (FP32 running max/sum)
      P  → two-level quantisation:
           Level 1 (per-token):  fp8_scale_p  = max_per_row(|P|) * (1/(448*6))
           Level 2 (microscale): fp4_p_i      = round_e2m1(P_i / (fp8_scale_p * 6))
      O  = P_fp4 @ V_fp4^T  (FP4×FP4 → FP32)
    Finalise: O ← O / row_sum

Usage example:
  from sageattn3 import sageattn3_blackwell

  # q, k, v: torch.Tensor of shape [batch, heads, seqlen, head_dim]
  #           dtype must be torch.float16 or torch.bfloat16
  #           head_dim must be 64 or 128
  o = sageattn3_blackwell(q, k, v, is_causal=False)
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          # pointer to input Q:  [B, H, L, D]
    q_out_ptr,      # pointer to output Q_smoothed: [B, H, L, D]
    qm_out_ptr,     # pointer to group-mean output Q_mean: [B, H, num_groups, D]
    B, H, L, D: tl.constexpr,    # batch, heads, seq-len, head-dim
    stride_qb, stride_qh, stride_ql, stride_qd,  # strides for Q
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  # strides for Q_mean
    GROUP_SIZE: tl.constexpr      # number of tokens per group (128)
):
    """
    Triton kernel: per-block Q smoothing.

    Corresponds to the "Smooth Q" pre-processing step in SageAttention3
    Section 3.1.  Each Triton program handles one (batch, head, group) triple.

    Grid dimension mapping:
      axis 0 → program_id(0) = batch index (pid_b)
      axis 1 → program_id(1) = head  index (pid_h)
      axis 2 → program_id(2) = group index along the sequence dimension
                               (each group spans GROUP_SIZE = 128 consecutive
                                tokens)

    Algorithm (for one program):
      q_group  = Q[b, h, group_start : group_start+GROUP_SIZE, :]   shape [G, D]
      qm_group = mean(q_group, axis=0)                               shape [D]
      q_out    = q_group - qm_group                                  (centred)
      qm_out   = qm_group                                            (saved for ΔS)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    # First token index in this group along the sequence dimension
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)  # [GROUP_SIZE]
    
    # 2-D offset into Q:  q[offsets, :] for this (batch, head)
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)  # [GROUP_SIZE, D]
    
    # Compute the mean over the GROUP_SIZE tokens → shape [D]
    # This is q_mean_g in the paper's notation: q̄_g = (1/G) Σ_i q_i
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    # Subtract group mean: q̃_i = q_i - q̄_g  (Step 1 of Smooth Q)
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    # Store the group mean for later use in computing ΔS = q̄_g @ K^T
    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    """
    Apply per-block Q smoothing using the Triton kernel above.

    Args:
        q: Q tensor of shape [B, H, L, D], dtype float16 or bfloat16.
           L must be divisible by GROUP_SIZE=128.

    Returns:
        q_out: smoothed Q of the same shape [B, H, L, D].
        qm:    block-wise mean of shape [B, H, L//128, D], used later to
               compute the ΔS correction term.

    This corresponds to the per-block variant of "Smooth Q" in SageAttention3
    Section 3.1.  The kernel is launched with a 3-D grid:
        grid = (B, H, L // 128)
    """
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    # Each program handles one (batch, head, group) combination
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    """
    Pre-process Q, K, V before FP4 quantisation.

    This implements Step 0 of the SageAttention3 pipeline (Section 3.1):
      1. Smooth K:  K ← K - mean_over_tokens(K)
         Removes the per-channel (head-dim) DC offset of K so its distribution
         is centred at zero — prerequisite for accurate FP4 quantisation.

      2. Pad sequences to a multiple of 128 tokens (required by the Blackwell
         tile shape kBlockN = 128).

      3. Smooth Q (two variants):
         a) per_block_mean=True  → use the Triton kernel group_mean_kernel,
            which computes a separate mean per 128-token group of Q.
         b) per_block_mean=False → use a single global mean across all tokens.

      4. Compute ΔS = Q_mean @ K^T  (FP32).
         Because Q̃ = Q - Q_mean, the true score Q @ K^T equals
         Q̃ @ K^T + ΔS.  ΔS is computed in higher precision and added back
         inside the CUDA attention kernel.

    Args:
        q:              [B, H, L, D], float16 or bfloat16.
        k:              [B, H, S, D], float16 or bfloat16.
        v:              [B, H, S, D], float16 or bfloat16.
        per_block_mean: whether to use per-128-token group mean for Q
                        (True by default for accuracy).

    Returns:
        q:       smoothed & padded Q,       [B, H, ceil(L/128)*128, D].
        k:       smoothed & padded K,       [B, H, ceil(S/128)*128, D].
        v:       padded V,                  [B, H, ceil(S/128)*128, D].
        delta_s: correction term (FP32),    [B, H, L//128, ceil(S/128)*128].
    """

    def pad_128(x):
        # Pad the sequence dimension (dim=2) to the next multiple of 128
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    # Step 1: Smooth K — subtract global per-head token mean
    # k.mean(dim=-2, keepdim=True) → shape [B, H, 1, D]
    k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_128(x), [q, k, v])

    # Step 2: Smooth Q and extract the group mean used for ΔS
    if per_block_mean:
        # Use Triton kernel: separate mean per 128-token group
        q, qm = triton_group_mean(q)
    else:
        # Single global mean across all tokens
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm

    # Step 3: ΔS = Q_mean @ K^T   shape [B, H, num_groups, seqlen_k_padded]
    # Cast to FP32 to preserve precision of the correction term.
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 microscaling quantisation for Q (and K in non-permuted form).

    Corresponds to the "FP4 microscaling" step in SageAttention3 Section 3.2.
    Each consecutive group of 16 head-dim elements is quantised independently:
      scale = max(|x_i|, i in group) / 6.0   (stored as FP8 e4m3)
      x_fp4 = round_e2m1(x_i / scale)         (stored as 4-bit e2m1)

    Args:
        x: input tensor [B, H, N, D], float16 or bfloat16.
           N = number of tokens, D = head_dim (must be divisible by 16).

    Returns:
        packed_fp4: packed FP4 data,   [B, H, N, D//2],  dtype=uint8.
                    Two e2m1 values packed per byte.
        fp8_scale:  FP8 e4m3 scales,   [B, H, N, D//16], dtype=float8_e4m3fn.
                    One scale per 1×16 micro-group.

    Layout note: the scale values are stored in the "blockscaled" interleaved
    format required by the Blackwell SM120 MMA atom
    (see blockscaled_layout.h for details).
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # Calls scaled_fp4_quant_kernel in fp4_quantization_4d.cu
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 microscaling quantisation for K with token-index permutation.

    Same quantisation algorithm as scale_and_quant_fp4, but the output token
    indices are permuted according to the pattern expected by the Blackwell
    FP4 attention kernel for operand B (K matrix).  The permutation reorders
    token rows so that the 32-row TMA tile loaded by the kernel is laid out
    contiguously in the blockscaled interleaved format.

    Permutation pattern (for groups of 32 tokens):
      [0,1,8,9,16,17,24,25, 2,3,10,11,18,19,26,27, 4,5,12,13,..., 6,7,14,15,22,23,30,31]

    Args:
        x: input K tensor [B, H, N, D], float16 or bfloat16.

    Returns:
        packed_fp4: packed FP4 data (permuted),  [B, H, N, D//2],  uint8.
        fp8_scale:  FP8 e4m3 scales (permuted),  [B, H, N, D//16], float8_e4m3fn.
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # Calls scaled_fp4_quant_kernel<permute=true> in fp4_quantization_4d.cu
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 microscaling quantisation for V, transposed to [D, N] layout.

    V is stored transposed (head_dim-major) so that PV can be computed as:
      O = P @ V   where P is [M, N] and V^T is [D, N],
    which is equivalent to O = P @ V with V in its original [N, D] layout,
    but in the kernel the inner-product dimension is N (tokens) rather than D.

    The microscaling here groups 16 consecutive *sequence* positions (after
    transposition) into one FP4 group, so the 1×16 granularity is along the
    N dimension of the transposed V^T.

    Args:
        x: input V tensor [B, H, N, D], float16 or bfloat16.

    Returns:
        packed_fp4: packed FP4 data,  [B, H, D, N//2],  uint8.
                    Layout is [B, H, head_dim, seqlen//2].
        fp8_scale:  FP8 e4m3 scales, [B, H, D, N//16], float8_e4m3fn.
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    # Output is transposed: [B, H, D, N//2] instead of [B, H, N, D//2]
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # Calls scaled_fp4_quant_trans_kernel in fp4_quantization_4d.cu
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    """
    Launch the Blackwell FP4 attention CUDA kernel.

    This function calls fp4attn_cuda.fwd which wraps the CUTLASS-based kernel
    defined in sageattn3/blackwell/kernel_ws.h.  The kernel implements
    Algorithm 1 of SageAttention3:

      For each output tile (m_block, head, batch):
        1. Load Q_fp4, SFQ (FP8 scales for Q) from SMEM via TMA.
        2. For each KV block (n_block, from max to 0):
           a. Load K_fp4, SFK, ΔS block via TMA.
           b. Compute S = Q_fp4 ⊗ K_fp4 + ΔS   (⊗ = FP4 block-scaled MMA)
           c. Mask out-of-range positions (causal or padding).
           d. Online softmax update: m_new, l_new, scores_scale.
           e. Quantise P tile to FP4 using two-level quantisation
              (per-token FP8 normalisation → per-group FP4 microscaling).
           f. Load V^T_fp4, SFVT.
           g. Compute O += P_fp4 ⊗ V^T_fp4   (FP4 block-scaled MMA).
        3. Rescale accumulated O by 1 / row_sum.
        4. Write O tile back to global memory via TMA.

    Args:
        qlist:        (packed_fp4_Q, fp8_scale_Q) from scale_and_quant_fp4.
        klist:        (packed_fp4_K, fp8_scale_K) from scale_and_quant_fp4_permute.
        vlist:        (packed_fp4_V, fp8_scale_V) from scale_and_quant_fp4_transpose.
        delta_s:      ΔS correction term [B, H, num_groups, KL_padded], float32.
        KL:           actual (unpadded) key/value sequence length.
        is_causal:    if True apply causal (lower-triangular) mask.
        per_block_mean: must match what was used in preprocess_qkv.
        is_bf16:      True if output should be bfloat16; False for float16.

    Returns:
        A tuple (O, ...) where O is the attention output [B, H, QL_padded, D].
        (additional elements may include LSE; use [0] to get O.)
    """
    # softmax_scale = 1 / sqrt(head_dim)
    # qlist[0].shape[-1] is D//2 (packed), so actual D = shape[-1]*2
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    """
    Drop-in replacement for scaled_dot_product_attention using SageAttention3.

    Implements the full FP4 microscaling attention pipeline on NVIDIA Blackwell
    (SM120) GPUs as described in SageAttention3 (NeurIPS 2025, arXiv:2505.11594).

    Full pipeline:
      1. preprocess_qkv — Smooth K, Smooth Q, compute ΔS, pad to 128-multiples.
      2. scale_and_quant_fp4        — quantise Q to FP4 (plain layout).
      3. scale_and_quant_fp4_permute — quantise K to FP4 (permuted layout for MMA).
      4. scale_and_quant_fp4_transpose — quantise V to FP4 (transposed layout).
      5. blockscaled_fp4_attn       — run the Blackwell FP4 CUTLASS kernel.
      6. Slice output back to the original query sequence length QL.

    Args:
        q:              Query  tensor [B, H, QL, D], float16 or bfloat16.
        k:              Key    tensor [B, H, KL, D], float16 or bfloat16.
        v:              Value  tensor [B, H, KL, D], float16 or bfloat16.
        attn_mask:      Not currently used (reserved for future support).
        is_causal:      Apply causal mask (lower-triangular).  Default False.
        per_block_mean: Use per-128-token group mean for Q smoothing.
                        True is recommended for accuracy.
        **kwargs:       Extra arguments (ignored; present for API compatibility).

    Returns:
        o:  Attention output tensor [B, H, QL, D], same dtype as q.

    Constraints:
        - head_dim D must be 64 or 128 (falls back to sdpa for D >= 256).
        - GPU must be NVIDIA Blackwell (SM120, e.g. RTX 5090 / B100 / GB200).
        - Sequence lengths QL, KL are automatically padded to a multiple of 128.

    Example:
        from sageattn3 import sageattn3_blackwell
        import torch

        B, H, L, D = 2, 16, 4096, 128
        q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, L, D, device='cuda', dtype=torch.float16)
        o = sageattn3_blackwell(q, k, v, is_causal=True)
        # o has shape [B, H, L, D], dtype float16
    """
    # Fall back to standard SDPA for unsupported head dims
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)

    # Save the original (unpadded) query sequence length for output slicing
    QL = q.size(2)
    # Save the original key sequence length for the causal mask inside the kernel
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16

    # Step 0: Smooth Q/K and compute ΔS.  q, k, v are padded to multiples of 128.
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)

    # Step 1: FP4 microscaling quantisation
    qlist_from_cuda = scale_and_quant_fp4(q)              # plain layout for Q
    klist_from_cuda = scale_and_quant_fp4_permute(k)      # permuted layout for K
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)    # transposed layout for V

    # Step 2: Run the CUTLASS FP4 attention kernel
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda, 
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16
    )[0][:, :, :QL, :].contiguous()  # slice back to original QL, drop padding
    return o_fp4