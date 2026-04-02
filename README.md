<h1 align="center"> <p>SGL</p></h1>

<p align="center">
  <picture>
    <img width="20%" alt="SGL" src="./logo.png">
  </picture>
</p>




The official implementation of "2025 CVPR A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for accelerating Large VLMs".

> Wangbo Zhao<sup>1</sup>, Yizeng Han<sup>2</sup>,  Jiasheng Tang<sup>2,3</sup>, Zhikai Li<sup>1</sup>, Yibing Song<sup>2,3</sup>, Kai Wang<sup>1</sup>, Zhangyang Wang<sup>4</sup>, Yang You<sup>1</sup>
>
> <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com/?language=zh), <sup>3</sup>Hupan Lab, <sup>4</sup>[The University of Texas at Austin](https://www.tsinghua.edu.cn/)
>
>  [Paper](https://arxiv.org/abs/2412.03324)


## Update
2025.02.27 SGL has been accepted to CVPR 2025.

## 💥 Overview
![20241230195723](https://github.com/user-attachments/assets/e244efd4-4136-4402-856f-95e87e33d408)

(a) Small VLM-guided visual token pruning in a large VLM (SGP). We update a global attention map aggregated from all layer of a small VLM. This global attention map is used to rank visual tokens and guide the visual token pruning in a large VLM. 

(b) Aggregation of attention maps in SGP. We aggregate the attention score of visual tokens received from prompt tokens and generated tokens across all heads and layers in the small LM. Higher scores indicate greater significance. 

(c) Inference with Small VLM Early Exiting (SEE). When the early exiting decision score from the small VLM is sufficient, the larger VLM will not be invoked.

## 🔨 Usage


1. Please refer to the documentation of [InternVL](https://github.com/OpenGVLab/InternVL) to set up the environment and prepare the data for evaluation.

2. We take `bash textvqa2B-26B.sh` as an example, which takes InternVL2-2B as the small model to accelerate the large model InternVL2-26B.

3. `--large_model_prune_layer` controls where pruning is applied in the large model.

4. `--small_model_attention_layer_range` controls which decoder layers from the small model contribute to the visual-token importance map.
   Use `all` to match the original repo behavior.
   Use `1-8`, `9-16`, `17-24` for early/middle/late analysis on InternVL2-2B.
   Use `1-10`, `11-20`, `21-30` only when your small model has at least 30 decoder layers.

Example runs:

```bash
# Original behavior: aggregate all small-model layers
bash textvqa2B-26B.sh

# Early layers only
SMALL_ATTENTION_LAYER_RANGE=1-8 bash textvqa2B-26B.sh

# Middle layers only
SMALL_ATTENTION_LAYER_RANGE=9-16 bash textvqa2B-26B.sh

# Late layers only
SMALL_ATTENTION_LAYER_RANGE=17-24 bash textvqa2B-26B.sh
```

The evaluation outputs are now separated by both prune setting and small-model attention range, so different runs do not overwrite each other.

## GitHub And Colab

To upload this project to your own GitHub repository:

```bash
git init
git add .
git commit -m "Add small-model attention layer range experiments"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

This repository now includes a `.gitignore` so local datasets, checkpoints, and result files are not uploaded by default.

For Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/<your-username>/<your-repo>.git
%cd /content/<your-repo>
!pip install -U pip
!pip install torch torchvision transformers accelerate sentencepiece timm einops decord pillow
```

Store datasets and checkpoints in Google Drive, then run:

```bash
%env SMALL_CHECKPOINT=/content/drive/MyDrive/model_ckpts/InternVL2-2B
%env LARGE_CHECKPOINT=/content/drive/MyDrive/model_ckpts/InternVL2-26B
%env SMALL_ATTENTION_LAYER_RANGE=1-8
%env LOAD_FLAGS=--load-in-4bit
!bash colab_textvqa_single_gpu.sh
```

Notes:

- Use `torchrun --nproc_per_node=1` on Colab, not the 8-GPU launcher.
- `InternVL2-26B` is usually too large for standard free Colab in full precision. In practice you will likely need `--load-in-4bit`, a high-memory GPU, or a smaller large model such as `InternVL2-8B`.
- For a 24-layer small model such as `InternVL2-2B`, use ranges like `1-8`, `9-16`, and `17-24`.








## 🤔 Citation
If you found our work useful, please consider citing us.
```
@article{zhao2024stitch,
  title={A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for accelerating Large VLMs},
  author={Zhao, Wangbo and Han, Yizeng and Tang, Jiasheng and Li, Zhikai and Song, Yibing and Wang, Kai and Wang, Zhangyang and You, Yang},
  journal={arXiv preprint arXiv:2412.03324},
  year={2024}
}
```

## 🙏 Acknowledgement
SGL is built with reference to the code of the following projects: [InternVL](https://github.com/OpenGVLab/InternVL), [FastV](https://github.com/pkunlp-icler/FastV), [QWen2-VL](https://github.com/QwenLM/Qwen2-VL), and [LLaVa-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/).

## ☎️ Contact
🔥🔥🔥 If you are interested in this work and hope to cooperate with us, please drop an email to wangbo.zhao96@gmail.com 🔥🔥🔥
