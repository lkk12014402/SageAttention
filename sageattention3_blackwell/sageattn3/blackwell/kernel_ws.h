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
 * kernel_ws.h — Top-level CUDA kernel for Blackwell FP4 warp-specialised attention
 * ---------------------------------------------------------------------------
 * This file defines compute_attn_ws, the CUDA kernel implementing SageAttention3
 * Algorithm 1 using a warp-specialised (WS) producer/consumer pipeline.
 *
 * Warp-group specialisation (Blackwell warpgroup model):
 *   The kernel uses kNWarps warp-groups, split into three roles:
 *
 *   WarpGroupRole::Producer   (warp-group 0, 4 warps = 128 threads)
 *     ProducerWarpRole::Mainloop  (warp 0 in the group)
 *       → Asynchronously loads Q, K, V, SFQ, SFK, SFVt, ΔS from global memory
 *         into SMEM via TMA (Tensor Memory Accelerator).  Only one lane (thread)
 *         actually issues TMA copies; the rest do nothing.
 *     ProducerWarpRole::Epilogue  (warp 1 in the group)
 *       → After Consumer computes O, writes O from SMEM to global memory via TMA.
 *     Warps 2–3:  idle (allocation padding for register budgeting).
 *
 *   WarpGroupRole::Consumer0 / Consumer1  (warp-groups 1 and 2)
 *     Each consumer warp-group handles a separate attention tile in an interleaved
 *     fashion (double-buffered consumers allow the two groups to overlap MMA work).
 *     Each consumer:
 *       1. Issues FP4 QK matmul (tiled_mma_qk) over all K blocks.
 *       2. Applies online softmax with P two-level quantisation.
 *       3. Issues FP4 PV matmul (tiled_mma_pv).
 *       4. Finalises O = acc / row_sum.
 *
 * Grid mapping:
 *   Each CTA (thread block) processes one (m_block, head, batch) triplet.
 *   The TileScheduler maps blockIdx → (m_block, bidh, bidb).
 *   Sequence dimension:   m_block = blockIdx.x (via scheduler)
 *   Head dimension:       bidh    = blockIdx.y (via scheduler)
 *   Batch dimension:      bidb    = blockIdx.z (via scheduler)
 *
 * Register budgeting:
 *   Producer warp-group deallocates 24 registers to leave more for consumers.
 *   Each consumer warp-group allocates 232 registers for the large MMA accumulators.
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "params.h"
#include "utils.h"
#include "tile_scheduler.h"
#include "mainloop_tma_ws.h"
#include "epilogue_tma_ws.h"
#include "named_barrier.h"
#include "softmax_fused.h"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                    ) {

    using Element = typename Ktraits::Element;       // FP4 e2m1
    using ElementAccum = typename Ktraits::ElementAccum;  // FP32
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;  // (kBlockM, kBlockN, kHeadDim)
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    // Number of MMA threads (Consumer0 + Consumer1 combined)
    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMmaQK{});
    // Number of threads in the Producer warp-group
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;


    enum class WarpGroupRole {
        Producer = 0,
        Consumer0 = 1,
        Consumer1 = 2
    };
    enum class ProducerWarpRole {
        Mainloop = 0,
        Epilogue = 1,
        Warp2 = 2,
        Warp3 = 3
    };

    extern __shared__ char shared_memory[];
    // Cast raw SMEM to the structured SharedStorage defined in kernel_traits.h
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    // elect_one_sync() returns 1 for exactly one thread per warp (the "elected lane")
    int const lane_predicate = cute::elect_one_sync();
    // Canonical warp and warp-group indices (0-based)
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    // Thread index within the current warp-group (0–127)
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    // Warp index within the current warp-group (0–3)
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index

    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    pipeline_params_v.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = warp_group_thread_idx == 0;
    pipeline_params_v.num_consumers = NumMmaThreads;

    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumMmaThreads;

    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_params_q.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipelineQ::ThreadCategory::Producer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumMmaThreads;

    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    uint32_t epilogue_barrier_group_size_list[2] = {cutlass::NumThreadsPerWarp, NumMmaThreads};
    typename EpilogueBarrier::Params params_epilogue_barrier;
    params_epilogue_barrier.group_id = (warp_group_role == WarpGroupRole::Producer);
    params_epilogue_barrier.group_size_list = epilogue_barrier_group_size_list;
    EpilogueBarrier barrier_o(shared_storage.barrier_o, params_epilogue_barrier);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;
    __syncthreads();

    if (warp_group_role == WarpGroupRole::Producer) {
        // Producer warp-group: donate registers to consumers and handle data loading
        cutlass::arch::warpgroup_reg_dealloc<24>();
        TileScheduler scheduler;
        
        if (producer_warp_role == ProducerWarpRole::Mainloop) {  // Load Q, K, V
            // TMA pipeline write-state (tracks which SMEM slot to write next)
            PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();
            
        int work_idx = 0;
            // Iterate over assigned work tiles; each tile covers one (m_block, head, batch) triplet
            for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
                int tile_count_semaphore = 0;
                // Load Q (once per tile) and all K/V blocks (one per pipeline stage)
                collective_mainloop.load(mainloop_params, scheduler_params, 
                                         pipeline_q, pipeline_k, pipeline_v, 
                                         smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v,
                                         shared_storage, work_tile_info, work_idx, tile_count_semaphore);
            }
            // Signal to consumers that no more data is coming
            collective_mainloop.load_tail(pipeline_q, pipeline_k, pipeline_v, 
                                          smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v);
        } else if (producer_warp_role == ProducerWarpRole::Epilogue) {
            // Epilogue warp: after Consumer finishes computing O, write it back to GMEM
            for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
                barrier_o.wait();  // Wait until Consumer has written O to SMEM
                // Issue TMA store: copy O from SMEM to global memory
                collective_epilogue.tma_store(shared_storage, epilogue_params, work_tile_info, scheduler_params, threadIdx.x);
                collective_epilogue.store_tail();  // Wait for TMA store to complete
                barrier_o.arrive();  // Signal Consumer that SMEM is free for next tile
            }
            
        }
    } else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
        // Consumer warp-groups: allocate extra registers for large MMA accumulators
        cutlass::arch::warpgroup_reg_alloc<232>();
        typename Ktraits::TiledMmaPV tiled_mma_pv;
        TileScheduler scheduler{};
        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        PipelineStateQ smem_pipe_read_q;

        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
            // tOrO: output accumulator tensor (FP32), shape (kBlockM, kHeadDim)
            // Partitioned according to the TiledMmaPV mapping for this thread
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_MNK{}));
            // SoftmaxFused maintains the online running max (m) and sum (l) per row,
            // and fuses the P two-level quantisation into the softmax pass.
            flash::SoftmaxFused<2 * (2 * kBlockM / NumMmaThreads)> softmax_fused;
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;  // tile index, head, batch

            // Number of K blocks to process (seqlen_k / kBlockN, capped by causal mask)
            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
                collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord);
                continue;
            }

            // Run the main attention compute loop:
            //   FP4 QK matmul → online softmax + P quant → FP4 PV matmul
            // (thread_idx = threadIdx.x - NumCopyThreads offsets past Producer threads)
            collective_mainloop.mma(mainloop_params, pipeline_q, pipeline_k, pipeline_v, smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax_fused, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block, shared_storage);
            // Sync with Epilogue warp before writing O tile to SMEM
            barrier_o.wait();
            // Convert FP32 accumulator to FP16/BF16 and write O tile to SMEM via STSM
            collective_epilogue.mma_store(shared_storage, tiled_mma_pv, tOrO, threadIdx.x - NumCopyThreads); 
            // Signal Epilogue warp that O is ready in SMEM
            barrier_o.arrive();
            ++work_idx;
        }
    }
}

} // namespace flash
