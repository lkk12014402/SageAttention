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
 * fp4_quantization_4d.cu — FP4 microscaling quantisation kernels
 * ---------------------------------------------------------------------------
 * Implements the FP4 microscaling quantisation step from SageAttention3 §3.2.
 *
 * Three kernels are provided, all performing the same mathematical operation
 * but with different output layouts required by the Blackwell attention kernel:
 *
 *   scaled_fp4_quant_kernel<permute=false>  → scaled_fp4_quant()
 *     Plain layout: output[B, H, N, D//2], scale[B, H, N, D//16]
 *     Used for Q.
 *
 *   scaled_fp4_quant_kernel<permute=true>   → scaled_fp4_quant_permute()
 *     Permuted layout: token indices within each 32-row group are reordered
 *     so that the blockscaled interleaved format required for MMA operand B
 *     (K matrix) is correct.
 *     Used for K.
 *
 *   scaled_fp4_quant_trans_kernel           → scaled_fp4_quant_trans()
 *     Transposed layout: output[B, H, D, N//2], scale[B, H, D, N//16]
 *     V is stored head_dim-major so that P @ V^T can be computed with the
 *     standard FP4 tensor core where the inner product runs over tokens (N).
 *     Used for V.
 *
 * Microscaling algorithm (per group of 16 elements along head_dim):
 *   1. Load 16 elements from input  (as 8×half2 packed vectors)
 *   2. Compute group maximum: s = max(|x_i|, i=0..15)
 *   3. Scale factor: SFValue = s / 6.0   (6.0 = max FP4 e2m1 value)
 *   4. Round-trip SFValue through FP8 e4m3 (store then reload) to simulate
 *      the precision loss of storing the scale as FP8.
 *   5. Divide each element by SFValue: y_i = x_i / SFValue
 *   6. Convert each y_i to FP4 e2m1 using the PTX instruction:
 *        cvt.rn.satfinite.e2m1x2.f32   (converts two FP32 values to 2×FP4)
 *   7. Pack 8 FP4 values into a 32-bit uint32 (two per byte)
 *   8. Store packed FP4 data and FP8 scale factor in the blockscaled layout.
 *
 * Blockscaled scale factor storage layout:
 *   Scale factors must be stored in the specific interleaved format consumed
 *   by the SM120 blockscaled MMA atom.  The offset formula:
 *     offset = (col_id_local / 4) * 256 + (col_id_local % 4)
 *            + (token_id_local / 16) * 4 + (token_id_local % 16) * 16
 *   groups 64 tokens × 1 scale into a 256-byte tile, interleaved such that
 *   each warp can load exactly one scale per thread in a single LDSM.
 *
 * CVT_FP4_ELTS_PER_THREAD = 16:
 *   Each thread processes 16 elements, which equals the group size.
 *   Therefore one thread computes exactly one FP8 scale factor.
 */
#include <torch/all.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                \
  if (pytorch_dtype == at::ScalarType::Half) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (pytorch_dtype == at::ScalarType::BFloat16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;    \
    TORCH_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)     \
  TORCH_CHECK(x.dtype() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)    \
  TORCH_CHECK(x.dim() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.stride(-1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")

constexpr int CVT_FP4_ELTS_PER_THREAD = 16;

// -------------------------------------------------------------------------
// fp32_vec_to_e2m1: convert 8 float32 values (as 4×float2) to 8 FP4 e2m1
// values packed into a single uint32 (2 FP4 values per byte).
//
// Uses the PTX instruction cvt.rn.satfinite.e2m1x2.f32 which converts two
// FP32 inputs to two FP4 e2m1 values and packs them into a byte.  Four such
// conversions produce one 32-bit word.
//
// Note: __CUDA_ARCH__ >= 1000 is Blackwell (SM100+). On older GPUs this
// returns 0 (not supported).
// -------------------------------------------------------------------------
inline __device__ uint32_t fp32_vec_to_e2m1(float2 *array) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Define a 32 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[8];
};

template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
__global__ void scaled_fp4_quant_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_seq_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_seq_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  // -----------------------------------------------------------------------
  // Thread/block mapping:
  //   blockIdx.y → batch index (bidb)
  //   blockIdx.z → head  index (bidh)
  //   blockIdx.x → token-block index  (covers BLOCK_SIZE tokens per block)
  //
  // Within the block:
  //   Each group of NUM_THREADS_PER_TOKEN consecutive threads handles
  //   one token (all head_dim elements).
  //   threadIdx.x / NUM_THREADS_PER_TOKEN → local token within the block
  //   threadIdx.x % NUM_THREADS_PER_TOKEN → which D-slice of the token
  //     Each D-slice is CVT_FP4_ELTS_PER_THREAD=16 elements, i.e. one
  //     micro-group (the unit of FP4 microscaling).
  // -----------------------------------------------------------------------
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Number of threads responsible for a single token's head_dim elements
  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;

  // Global token index for this thread (before permutation)
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;
  
  int load_token_id;
  if constexpr (!permute) {
    // Plain (non-permuted) layout: load token in natural order
    load_token_id = token_id;
  } else {
    // Permuted layout for K: reorder tokens within each 32-row sub-block so
    // that the blockscaled K tensor is in the correct interleaved format for
    // the SM120 MMA atom's B operand.
    // Permutation within each group of 32 tokens:
    //   [0,1,8,9,16,17,24,25, 2,3,10,11,18,19,26,27, 4,5,12,13,..., 6,7,14,15,22,23,30,31]
    int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
    int local_token_id_residue = local_token_id % 32;
    load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 + 
                    ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PackedVec in_vec;
  
  // Zero-pad the vector (handles out-of-bounds tokens with all zeros)
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (load_token_id < num_tokens) {
    // Load CVT_FP4_ELTS_PER_THREAD=16 elements (as 8×half2) for this micro-group
    // This corresponds to 1×16 micro-group in head_dim dimension (§3.2 of paper)
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input + // batch dim
                                          head_id * stride_h_input +   // head dim
                                          load_token_id * stride_seq_input + // seq dim
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim (16 elements)
  }

  // -----------------------------------------------------------------------
  // Step 1: compute group maximum  max(|x_i|, i=0..15)
  //
  // We use __habs2/__hmax2 to process two half-precision values at a time.
  // The loop reduces across the 8 half2 values in in_vec.
  // -----------------------------------------------------------------------
  // calculate max of every consecutive 16 elements
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    // When each thread only holds 8 elements, two adjacent threads share
    // one 16-element micro-group → exchange max via shuffle
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  // Reduce the half2 to a single float maximum
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // -----------------------------------------------------------------------
  // Step 2: compute FP8 e4m3 scale factor
  //   SFValue = vecMax / 6.0   (6.0 = max representable FP4 e2m1 value)
  //
  // Round-trip through FP8 e4m3 to simulate the precision of storing the
  // scale.  This ensures that the reconstruction error at decode time
  // exactly matches what the hardware will compute.
  // -----------------------------------------------------------------------
  // scaling factor
  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);  // encode to FP8
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));            // decode back to float

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // -----------------------------------------------------------------------
  // Step 3: divide elements by scale factor, convert to float2 for PTX
  //   y_i = x_i / SFValue   → each y_i ∈ [-6, 6]
  // -----------------------------------------------------------------------
  // convert input to float2 and apply scale
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  // -----------------------------------------------------------------------
  // Step 4: convert to FP4 e2m1 using PTX cvt.rn.satfinite.e2m1x2.f32
  //   Two FP32 → one byte (two FP4 packed, lower nibble = first value)
  //   8 FP4 values → 4 bytes = 1 uint32 per call to fp32_vec_to_e2m1
  // -----------------------------------------------------------------------
  // convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  // -----------------------------------------------------------------------
  // Step 5: store packed FP4 data
  // -----------------------------------------------------------------------
  // save, do not check range
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }
  
  // -----------------------------------------------------------------------
  // Step 6: store FP8 scale factor in the blockscaled interleaved format.
  //
  // The SM120 MMA atom requires scale factors in a specific interleaved
  // layout where groups of 64 tokens × 1 scale occupy 256 bytes.
  // The offset formula maps (token_id_local, col_id_local) to this layout:
  //   offset = (col_id_local / 4) * 256  ← which 256-byte group (D-major)
  //          + (col_id_local % 4)         ← position within the 4 columns
  //          + (token_id_local / 16) * 4  ← which row-group of 16 tokens
  //          + (token_id_local % 16) * 16 ← position within the 16-token group
  //
  // token_id_local is the token index within the current 64-token super-block
  // (since we process BLOCK_SIZE tokens per CUDA block, token_id % 64 gives
  // the position within a 64-row tile).
  // -----------------------------------------------------------------------
  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf + (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, typename T>
__global__ void scaled_fp4_quant_trans_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_d_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_d_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  // -----------------------------------------------------------------------
  // Thread/block mapping for transposed V quantisation:
  //   blockIdx.y → batch index (bidb)
  //   blockIdx.z → head  index (bidh)
  //   blockIdx.x → token-block index (covers BLOCK_SIZE tokens)
  //
  // This kernel transposes the token and head_dim axes of V, so the output
  // layout is [B, H, D, N//2] instead of [B, H, N, D//2].
  //
  // Two thread roles within a block:
  //   - NUM_THREADS_PER_TOKEN threads read from the same token (across D)
  //   - NUM_THREADS_PER_SEQ   threads read from the same D position (across N)
  //
  // The transposition is performed via shared memory (SMEM):
  //   1. Load in natural [N, D] order into SMEM.
  //   2. __syncthreads() to ensure all data is visible.
  //   3. Reload in transposed [D, N] order from SMEM.
  // -----------------------------------------------------------------------
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
  // Number of threads needed to cover BLOCK_SIZE tokens when each thread reads CVT_FP4_ELTS_PER_THREAD seq elements
  constexpr uint32_t NUM_THREADS_PER_SEQ = BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD;

  // Load one token per NUM_THREADS_PER_TOKEN threads (natural order)
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  PackedVec in_vec;
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  
  if (token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(input + 
                                          batch_id * stride_bz_input + // batch dim
                                          head_id * stride_h_input +   // head dim
                                          token_id * stride_seq_input + // seq dim
                                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim
  }

  // -----------------------------------------------------------------------
  // Transposition via SMEM:
  //   Write in_vec into shared_input in row-major [N, D] order.
  //   After __syncthreads, reload 16 elements in column-major [D, N] order.
  //   This effectively transposes the tile.
  // -----------------------------------------------------------------------
  // transpose
  __shared__ T shared_input[BLOCK_SIZE * head_dim];
  reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // After transposition: thread's data is now CVT_FP4_ELTS_PER_THREAD consecutive
    // elements along the sequence dimension for a fixed head_dim position.
    // The formula reads: element at (d, n) = shared_input[d + n * head_dim]
    in_vec.elts[i].x = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i) * head_dim];
    in_vec.elts[i].y = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i + 1) * head_dim];
  }

  // The same microscaling steps as the non-transposed kernel:
  // compute max of the 16-element group (now along the seq dimension)
  // calculate max of every consecutive 16 elements
  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) { // shuffle across two threads
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  // scaling factor: SFValue = max / 6.0 → stored as FP8 e4m3
  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  // convert input to float2 and apply scale
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  // convert to e2m1
  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  // Store packed FP4 in transposed [D, N//2] layout
  // save
  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + 
                                batch_id * stride_bz_output +
                                head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  // Store FP8 scale factors in blockscaled layout (same formula as non-transposed,
  // but now row = head_dim index and col = token group index)
  uint8_t *output_sf_save_base = output_sf + 
                                batch_id * stride_bz_output_sf +
                                head_id * stride_h_output_sf +
                                (threadIdx.x / NUM_THREADS_PER_SEQ / 64) * 64 * stride_d_output_sf;
  uint32_t row_id_local = (threadIdx.x / NUM_THREADS_PER_SEQ) % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + threadIdx.x % NUM_THREADS_PER_SEQ;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                            (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + (threadIdx.x % NUM_THREADS_PER_SEQ) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                              (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

void scaled_fp4_quant(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = 128;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, num_tokens, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_tokens, num_heads, head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, num_tokens, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, num_tokens, head_dim / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_permute(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = 128;

  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads, head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr int BLOCK_SIZE = 128;
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, true, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_seq_output,
              stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout) {
  constexpr int BLOCK_SIZE = 128;
  
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(output_sf);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_DTYPE(output, at::ScalarType::Byte);
  CHECK_DTYPE(output_sf, at::ScalarType::Float8_e4m3fn);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input; 
  int stride_d_output, stride_d_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_d_output = output.stride(1);
    stride_d_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, head_dim, num_heads, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_d_output = output.stride(2);
    stride_d_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_SHAPE(output, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_SHAPE(output_sf, batch_size, num_heads, head_dim, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  }

  auto input_dtype = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_trans_kernel<HEAD_DIM, BLOCK_SIZE, c_type>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<c_type*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<uint8_t*>(output_sf.data_ptr()),
              batch_size, num_heads, num_tokens,
              stride_bz_input, stride_h_input, stride_seq_input,
              stride_bz_output, stride_h_output, stride_d_output,
              stride_bz_output_sf, stride_h_output_sf, stride_d_output_sf);
    });
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_fp4_quant", &scaled_fp4_quant);
  m.def("scaled_fp4_quant_permute", &scaled_fp4_quant_permute);
  m.def("scaled_fp4_quant_trans", &scaled_fp4_quant_trans);
}