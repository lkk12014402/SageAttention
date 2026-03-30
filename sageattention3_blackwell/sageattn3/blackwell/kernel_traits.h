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
 * kernel_traits.h — Compile-time kernel configuration for Blackwell FP4 attention
 * ---------------------------------------------------------------------------
 * This file defines Flash_fwd_kernel_traits, a compile-time traits struct that
 * centralises ALL shape / type / layout constants used by the FP4 attention kernel.
 * By changing the template parameters you can change:
 *   - kBlockM  : number of query tokens processed per CTA (64 or 128)
 *   - kBlockN  : number of key/value tokens per pipeline stage (typically 128)
 *   - kHeadDim : head dimension D (64 or 128)
 *   - kStages  : pipeline depth for TMA double/triple buffering
 *   - kClusterM: CTA cluster size along the M (query) dimension
 *   - BlockMean: use per-block vs global Smooth-Q mean
 *
 * FP4 element types (SageAttention3 §3.2):
 *   Element   = float_e2m1_t   — 4-bit FP with 2-bit exponent, 1-bit mantissa (NV FP4)
 *   ElementSF = float_ue4m3_t  — 8-bit unsigned FP scale factor (FP8 e4m3)
 *
 * Blackwell MMA atom (SageAttention3 §3.3):
 *   SM120_16x32x64_TN_VS_NVFP4
 *   → 16 rows (M), 32 cols (N), 64 K-dimension (depth) per atom
 *   → Block-scaled: each atom reads 64/16 = 4 FP8 scale factors per row of A
 *     and 64/16 = 4 FP8 scale factors per col of B.
 *   → Accumulator is FP32.
 *
 * How to modify group size:
 *   The microscaling group size is fixed at SFVectorSize=16 elements (1×16 groups).
 *   To change it, update SFVectorSize and ensure blockscaled_layout.h / the hardware
 *   MMA atom support the new size.  The Blackwell hardware only supports 1×16 groups
 *   in its NVFP4 tensor core instruction.
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "blockscaled_layout.h"
#include "cute_extension.h"
#include "named_barrier.h"
using namespace cute;

template <
    int kStages,     // TMA pipeline depth (number of K/V tiles in flight simultaneously)
    int EpiStages,   // Epilogue pipeline stages (usually 1)
    typename Element,        // FP4 data type (float_e2m1_t)
    typename ElementSF,      // FP8 scale factor type (float_ue4m3_t)
    typename OutputType,     // Output accumulator type (float16 or bfloat16)
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV
>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0>{
    
    // Shared memory for Q tile (kBlockM × kHeadDim), FP4 packed
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    // Shared memory for K tiles (kBlockN × kHeadDim × kStages), FP4 packed
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    // FP8 scale factors for Q  (kBlockM × kHeadDim/16)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
    // FP8 scale factors for K  (kBlockN × kHeadDim/16 × kStages)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
    // FP8 scale factors for V^T (kHeadDim × kBlockN/16 × kStages)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;
    // ΔS correction tiles (FP32), shape (kBlockM × kBlockN × kStages)
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;
    // V^T tiles (kHeadDim × kBlockN × kStages), FP4 packed
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    // Output tile O (kBlockM × kHeadDim), FP16/BF16
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    
    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };
  };

template <
    int kHeadDim_,   // D: head dimension, must be 64 or 128
    int kBlockM_,    // Tile size along Q sequence dimension (64 or 128 tokens)
    int kBlockN_,    // Tile size along KV sequence dimension (pipeline stage size)
    int kStages_,    // TMA pipeline depth (number of simultaneous K/V tiles)
    int kClusterM_,  // CTA cluster size along M (for distributed attention, usually 1)
    bool BlockMean_, // Use per-block Smooth-Q mean (true) or global mean (false)
    typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>, 
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits {
    static constexpr int kBlockM = kBlockM_;   // Query tile M
    static constexpr int kBlockN = kBlockN_;   // KV tile N
    static constexpr int kHeadDim = kHeadDim_; // Head dimension D
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;      // Always enabled for FP4 accuracy
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == 64 || kBlockM == 128);
    // 3 warp-groups: 1 Producer + 2 Consumers (Consumer0, Consumer1)
    // Each warp-group has 4 warps of 32 threads = 128 threads
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;  // 384 or 256
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;
    // Number of FP8 scale factors per row of Q/K (one per 16-element micro-group)
    static constexpr int NumSFQK = kHeadDim / 16;
    // Number of FP8 scale factors per column of P (one per 16-token group for V)
    static constexpr int NumSFPV = kBlockN / 16;

    // -----------------------------------------------------------------------
    // Element types
    // -----------------------------------------------------------------------
    using ElementSF = cutlass::float_ue4m3_t;  // FP8 e4m3 unsigned scale factors
    using Element   = cutlass::float_e2m1_t;   // NV FP4 e2m1 data elements
    using ElementAccum = float;                 // FP32 accumulator for MMA
    using ElementOut = ElementOut_;             // Output: float16 or bfloat16
    using index_t = int64_t;
    // Micro-group size: 16 elements share one FP8 scale (§3.2 of the paper)
    static constexpr auto SFVectorSize = 16;

    // -----------------------------------------------------------------------
    // Tile shape: (kBlockM, kBlockN, kHeadDim)
    // Corresponds to (M-tile, N-tile, K-tile) in GEMM notation.
    // -----------------------------------------------------------------------
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
    // PermTileM: the permutation tile along M, capped at 128
    using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;
    
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    // -----------------------------------------------------------------------
    // Atom layout: how many MMA atoms tile the warp-group
    //   kBlockM=128 → 8 atoms along M  (8×16 = 128 rows)
    //   kBlockM= 64 → 4 atoms along M  (4×16 =  64 rows)
    // -----------------------------------------------------------------------
    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>
                                            >;

    // -----------------------------------------------------------------------
    // Tiled MMA for QK^T (FP4×FP4 → FP32)
    //   SM120_16x32x64_TN_VS_NVFP4:
    //     M=16 rows of Q,  N=32 cols of K,  K=64 depth (inner product)
    //     _TN = A is row-major, B is col-major (transposed)
    //     _VS = "vector-scaled" i.e. block-scaled with FP8 scale factors
    //     _NVFP4 = NVIDIA FP4 e2m1 data
    // -----------------------------------------------------------------------
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
      ));
    
    // Tiled MMA for P×V^T (FP4×FP4 → FP32)
    // Same atom; output tile is (kBlockM × kHeadDim)
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
      ));
    
    // Number of FP8 scale factors consumed per MMA atom along the K dimension:
    //   = atom_K / SFVectorSize = 64 / 16 = 4
    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    // TMA copy descriptors (Tensor Memory Accelerator, Blackwell SM90+)
    using GmemTiledCopy   = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    // -----------------------------------------------------------------------
    // Shared memory layouts for Q, K, V, V^T, ΔS
    // The sm120_rr_smem_selector chooses the optimal swizzled layout for FP4
    // data that avoids bank conflicts during LDSM (load-shared-to-register)
    // -----------------------------------------------------------------------
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());
    // Q SMEM: kBlockM × kHeadDim  (no pipeline stages; Q is loaded once)
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    // K SMEM: kBlockN × kHeadDim × kStages  (multi-stage pipeline for K)
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // V^T SMEM: kBlockN × kHeadDim × kStages (V stored transposed)
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // V^T alternative layout (head_dim × kBlockN × kStages)
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));
    // ΔS SMEM: kBlockM × kBlockN × kStages  (FP32 correction term)
    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS = 
        decltype(tile_to_shape(SmemLayoutAtomDS{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // Copy atoms for loading Q/K/V and scale factors from SMEM to registers
    using SmemCopyAtomQ  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    // -----------------------------------------------------------------------
    // Block-scaled layout helpers (defines how FP8 scales map to MMA operands)
    // SFVectorSize=16 means one scale covers a 1×16 vector of FP4 elements,
    // matching the Blackwell hardware constraint.
    // -----------------------------------------------------------------------
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF  = typename BlkScaledConfig::LayoutSF;
    using SfAtom    = typename BlkScaledConfig::SfAtom;
    // SMEM layouts for Q/K/V scale factors, deduced from the MMA tile shapes
    using SmemLayoutAtomSFQ  = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK  = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV  = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));

    // LayoutSFP: layout for scale factors of P (the attention probability matrix)
    // P has shape (kBlockM × kBlockN); its scale factors are per-token (one FP8
    // value per row of 16 elements), arranged for direct consumption by the MMA atom.
    using LayoutSFP = decltype(
      make_layout(
          make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
          make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
      )
    );
    // LayoutP: layout for quantised P values (FP4 e2m1, organised for MMA operand A)
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );

    // SFQ, SFK, SFV SMEM layouts (with pipeline stage dimension appended)
    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
      ));
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))
      ));
    using SmemLayoutSFV = decltype(make_layout(
        append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))
      ));
    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))
      ));

    // Output SMEM layout (FP16/BF16, swizzled for bank-conflict-free STSM stores)
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));

    // Aggregate shared storage type that fits all of the above into SMEM
    using SharedStorage = SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS, 
        SmemLayoutO, SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

    // Pipeline types for TMA async loads (one stage for Q, kStages for K/V)
    using MainloopPipeline  = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState     = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ   = typename MainloopPipelineQ::Params;
    using PipelineStateQ    = typename cutlass::PipelineState<1>;
    // Epilogue barrier synchronises Producer warp with Consumer warp-groups
    // for writing the final O tile
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

