/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 * softmax_fused.h — Fused online softmax + P two-level FP4 quantisation
 * ---------------------------------------------------------------------------
 * This file implements SoftmaxFused, the most mathematically dense part of
 * SageAttention3.  It fuses three operations into a single pass over the
 * attention score tile:
 *
 *  (A) Online softmax (running maximum / sum updates per row)
 *      Standard FlashAttention-style algorithm (Dao et al., 2022):
 *        m_new = max(m_old, row_max(S_tile))
 *        l_new = exp(m_old - m_new) * l_old + sum(exp(S_tile - m_new))
 *
 *  (B) Per-token normalisation (Level 1 of two-level P quantisation, §3.4)
 *      To represent P ∈ [0, 1] in FP4 e2m1, we need to rescale it into
 *      the FP4 representable range.  The maximum representable FP4 e2m1
 *      value is 6.0, and the maximum FP8 e4m3 scale value is 448.0, so
 *      the combined range is 448 × 6 = 2688.  Therefore:
 *        fp8_scalexfp4_scale = 1 / (448 × 6)   ← normalisation factor
 *      For each row:
 *        fp8_scale_p = max(|P_row|) * fp8_scalexfp4_scale
 *                   (stored as FP8 e4m3, absorbs the per-token magnitude)
 *
 *  (C) Per-group microscaling (Level 2, §3.2)
 *      Within each 1×16 group of P values, the microscale absorbs any
 *      remaining magnitude variation:
 *        fp4_scale_g = max(|P_group|) / fp8_scale_p   ← stored as FP8 e4m3
 *        p_fp4_i     = round_e2m1(P_i / (fp8_scale_p × fp4_scale_g))
 *
 * Why 448 × 6?
 *   - 448 is the maximum finite value of FP8 e4m3fn
 *     (exponent bias 7, max exp = 14, max mantissa = 1.110₂ → 448)
 *   - 6.0 is the maximum finite value of FP4 e2m1
 *     (exponent bias 1, max exp = 2, max mantissa = 1.1₂ → 6.0)
 *   - Together they cover the full range of P ∈ [0, 1] with minimal
 *     clipping, since P_max = 1.0 << 448 × 6.
 *
 * The log2 representation:
 *   All exp/max computations use base-2 logarithms for efficiency.
 *     ptx_exp2(x) ≡ 2^x
 *   softmax_scale_log2 = log2(softmax_scale) = -0.5 × log2(D)
 *
 * Template parameter Rows:
 *   Number of score rows owned by this thread per MMA tile.
 *   For kBlockM=128 with the SM120 atom:  Rows = 2*(2*128/NumMmaThreads)
 */
#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Rows>
struct SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum;       // l: running row sum of exp values,  shape [Rows]
    TensorT row_max;       // m: running row maximum of S values, shape [Rows]
    TensorT scores_scale;  // exp(m_prev - m_new): correction factor for O rescaling

    // -------------------------------------------------------------------------
    // Level 1 normalisation constant (per-token):
    //   fp8_scalexfp4_scale = 1 / (448 × 6) ≈ 3.72e-4
    // This maps P ∈ [0,1] into the combined FP8×FP4 representable range.
    // See §3.4 of the SageAttention3 paper.
    // -------------------------------------------------------------------------
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    // Precomputed log2(fp8_scalexfp4_scale) = log2(1/(448*6)) ≈ -11.39
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
    // log2(fp4_max_value) = log2(6.0) ≈ 2.585  (max FP4 e2m1 value)
    // Used when computing AbsMaxP: AbsMaxP /= 6 to get the FP8 group scale
    static constexpr float fp4_scale_log2 = -2.584962500721156f; // log2f(fp4_scale)
    // Number of threads that participate in each row reduction (warp-level exchange)
    static constexpr int RowReductionThr = 4;

    CUTLASS_DEVICE SoftmaxFused(){};

    // -------------------------------------------------------------------------
    // online_softmax_with_quant
    //
    // This is the core function, called once per K-block tile.
    // It performs (A) + (B) from the file header in a single pass.
    //
    // Template parameters:
    //   FirstTile — true for the first K-block (initialises m, l, scores_scale)
    //   InfCheck  — true if we may have -inf entries (causal masking edge tiles)
    //
    // Arguments:
    //   acc     — FP32 attention score tile S[m_tile, n_tile]; updated in-place
    //             to hold the normalised-and-scaled P tile.
    //   AbsMaxP — per-group absolute maxima for P, shape [Rows, n_groups].
    //             Used to compute the per-group FP8 microscale for P.
    //   softmax_scale_log2 — log2(1/sqrt(D))
    //
    // After this function:
    //   acc     contains P values scaled by fp8_scalexfp4_scale/AbsMaxP
    //           (i.e. values in [−1, 1] ready for FP4 rounding)
    //   AbsMaxP contains the per-group FP8 scale factors for P (in [0, 6])
    // -------------------------------------------------------------------------
    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc, 
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        // Reinterpret acc into two views:
        //   reduction_view  — for warp-level max/sum reductions
        //   conversion_view — for element-wise operations (exp, scale, quantise)
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));
        
        if constexpr (FirstTile) {
            // Initialise running statistics for the first tile
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            // ----------------------------------------------------------------
            // Step A1: compute row maximum over the current S tile
            //   For each row mi, iterate over all (ei, ni) entries and find max.
            //   __shfl_xor_sync with mask=1 exchanges values between pairs of
            //   adjacent threads that own the same logical row (8-element groups).
            // ----------------------------------------------------------------
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    // Exchange local max between thread pairs sharing the same row group
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1);
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    // Update the running row maximum (across all ni groups)
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                // Exchange row_max across a quad (4 threads, mask=2) to fully
                // reduce the row maximum across all threads owning this row
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // ----------------------------------------------------------------
                // Step A2: compute max_scaled = m * softmax_scale_log2 + offset
                //   In log2 domain: exp2(S * scale - max_scaled) = exp(S * scale - m)
                //   where max_scaled incorporates fp8_scalexfp4_scale_log2 so that
                //   the resulting P values are pre-divided by (448*6) — fitting
                //   into the [0, 1/(448*6)] range before quantisation.
                // ----------------------------------------------------------------
                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // Step A3: compute exp2(S * scale - max_scaled) → P normalised values
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }

                // ----------------------------------------------------------------
                // Step B: per-group AbsMaxP normalisation
                //   AbsMaxP(mi, sfi) currently holds the raw maximum of |S| for
                //   group sfi.  We compute:
                //     AbsMaxP = exp2(max_S_group * scale - max_scaled + log2(6))
                //             = (max_S_group / row_max) * 6.0
                //   This gives the per-group FP8 scale that, when divided into the
                //   corresponding P values, yields FP4 e2m1 in range [-6, 6].
                //   Adding fp4_scale_log2 = log2(6) shifts the scale by ×6 so that
                //   each group's max maps to exactly the FP4 maximum of 6.0.
                // ----------------------------------------------------------------
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }
            // Accumulate row sums (l) from normalised P values
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            // ----------------------------------------------------------------
            // Subsequent tiles: update m and l using the recurrence relations
            //   scores_max_prev = m_old
            //   m_new = max(m_old, row_max(S_new_tile))
            //   scores_scale(mi) = exp(m_old - m_new)     ← O rescaling factor
            //   l_new = l_old * scores_scale + sum_row(exp(S_new))
            // ----------------------------------------------------------------
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);  // save m_old
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                // scores_scale = exp2((m_old - m_new) * softmax_scale_log2)
                // Used later in rescale_o to correct the previous O partial sum
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                // Rescale running sum: l_new = l_old * scores_scale
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                // Compute exp(S_new - m_new) for all entries and accumulate sum
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                // Update per-group P scale factors
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }
        }

        // ----------------------------------------------------------------
        // Step C: divide P values by per-group AbsMaxP scale
        //   After this division, each group of 16 P values is in [-1, 1]
        //   (relative to the group maximum), ready for round_e2m1() in the
        //   quantize() lambda in mainloop_tma_ws.h.
        //   The actual FP4 magnitude is recovered during PV MMA by multiplying
        //   by the FP8 group scale stored in AbsMaxP.
        // ----------------------------------------------------------------
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
        }
    }

    // -----------------------------------------------------------------------
    // finalize: apply the final 1/l normalisation to the accumulated O tensor.
    //
    // After all K-blocks have been processed, the Consumer has:
    //   O_acc ≈ Σ_j (P_j * V_j)   (unnormalised)
    //   row_sum ≈ l = Σ_j Σ_n exp(S_j_n - m)
    //
    // We finish the softmax normalisation:
    //   O = O_acc / l
    //
    // The row_sum is first reduced across the RowReductionThr=4 threads
    // that cooperatively own the same logical row via __shfl_xor_sync.
    // -----------------------------------------------------------------------
    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            // Reduce row_sum across RowReductionThr=4 threads (i=1,2,then2-fold exchange)
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            // Guard against division by zero (can happen for fully-masked rows)
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            // Multiply each element of row mi in O by inv_sum
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) *= inv_sum;
             }
        }
    }

    // -----------------------------------------------------------------------
    // rescale_o: merge two partial O accumulators after an m update.
    //
    // When the running maximum m increases from m_old to m_new:
    //   O_new = O_old * scores_scale + O_tmp
    // where:
    //   O_old    = accumulated O from previous K-blocks
    //   O_tmp    = O from the current K-block (just computed by PV MMA)
    //   scores_scale = exp(m_old - m_new)
    //
    // This is the standard FlashAttention online update for the output.
    // -----------------------------------------------------------------------
    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                // O_merged = O_old * exp(m_old - m_new) + O_new_tile
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }

    }


};
} // namespace flash