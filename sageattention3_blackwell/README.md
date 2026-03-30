# SageAttention3
<!-- We are continuously updating more features. You could **Star** and **Watch** our repository to stay updated.

--- -->
This repository provides the official implementation of SageAttention3

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training**  
Paper: https://arxiv.org/abs/2505.11594  
Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen

# Limitaitions:
Currently, SageAttention3 works well for: 
1. Video generation models: CogVideoX-2B, HunyuanVideo, Mochi.
2. Almost all image generation models, including Flux and Stable-Diffusion3.5.

**Note: SageAttention3 does not guarantee lossless acceleration for all models. For other video generation models, we recommend selectively using SageAttention2++ in certain layers or timesteps.**  

For example:  
- Apply **SageAttention2++** only at the **first and last timesteps**,  
- Use **SageAttention3** for all the others.  

This hybrid approach may achieve **lossless acceleration**.  

## Installation
### Base environment
+ `python>=3.13`   , `torch>=2.8.0`, `CUDA >=12.8`

### Install Package

To use SageAttention3, please **compile from source**:
```
git clone https://github.com/thu-ml/SageAttention
cd SageAttention/sageattention3_blackwell 
python setup.py install
```


## How to Use
```python
from sageattn3 import sageattn3_blackwell
attn_output = sageattn3_blackwell(q, k, v, is_causal=False)
```
+ `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` 
+ `is_causal` determines the use of a causal mask.

## Code Guide: Annotated Source Files

The source files in this directory contain **detailed tutorial-style comments**
that map every kernel instruction back to the SageAttention3 paper (NeurIPS 2025 /
arXiv:2505.11594).  If you are reading this code for the first time, we recommend
starting in this order:

| File | What it explains | Paper section |
|------|-----------------|---------------|
| [`sageattn3/api.py`](sageattn3/api.py) | End-to-end Python pipeline: Smooth Q/K → FP4 quant → CUDA attention kernel → output | §3.1–3.4 |
| [`sageattn3/quantization/fp4_quantization_4d.cu`](sageattn3/quantization/fp4_quantization_4d.cu) | FP4 microscaling quantisation kernel (1×16 groups, FP8 scale, blockscaled layout) | §3.2 |
| [`sageattn3/blackwell/kernel_traits.h`](sageattn3/blackwell/kernel_traits.h) | All compile-time constants: tile sizes, MMA atom, element types, SMEM layouts | §3.3 |
| [`sageattn3/blackwell/softmax_fused.h`](sageattn3/blackwell/softmax_fused.h) | Online softmax (m, l updates) fused with P two-level quantisation | §3.4 |
| [`sageattn3/blackwell/mainloop_tma_ws.h`](sageattn3/blackwell/mainloop_tma_ws.h) | Main compute loop: FP4 QK MMA, P quant, FP4 PV MMA, ΔS correction | §3.1, §3.3, Algorithm 1 |
| [`sageattn3/blackwell/kernel_ws.h`](sageattn3/blackwell/kernel_ws.h) | Top-level CUDA kernel; warp-group specialisation (Producer/Consumer) | §3.3 |
| [`sageattn3/blackwell/epilogue_tma_ws.h`](sageattn3/blackwell/epilogue_tma_ws.h) | Output epilogue: FP32→FP16/BF16 conversion and TMA store | §3.3 |

### How to modify the group size
The FP4 microscaling group size is currently **16 elements** (matching the
Blackwell SM120 hardware constraint).  To experiment with a different group
size you would need to:
1. Change `SFVectorSize` in `kernel_traits.h`.
2. Update the scale-factor offset formula in `fp4_quantization_4d.cu`.
3. Ensure the new size is supported by the MMA atom and update
   `blockscaled_layout.h` accordingly.

### How to try a different P scaling strategy
The two-level P quantisation uses `448 × 6` as the combined normalisation
constant (see `softmax_fused.h`).  The constant `448` comes from the maximum
FP8 e4m3 value, and `6` from the maximum FP4 e2m1 value.  To try a different
strategy:
- Modify `fp8_scalexfp4_scale` and `fp8_scalexfp4_scale_log2` in `softmax_fused.h`.
- If you want per-row (not per-group) scaling, the AbsMaxP tensor shape and
  the `quantize` lambda in `mainloop_tma_ws.h` would need adjustment.

## Performance
### Speed of Kernels
![Speed on RTX5090](../assets/sage3_speed.png)

### Video and Image Generation Examples
![Image Examples](../assets/sage3_result.png)



## Citation
**If you use this code or find our work valuable, please cite:**
```
@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
@inproceedings{zhang2024sageattention2,
  title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
  author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
@article{zhang2025sageattention2++,
  title={Sageattention2++: A more efficient implementation of sageattention2},
  author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.21136},
  year={2025}
}
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```
