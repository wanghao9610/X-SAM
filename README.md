<div align="center">
<h1>✨X-SAM✨</h1>
<h3>From Segment Anything to Any Segmentation</h3>

[Hao Wang](https://wanghao9610.github.io)<sup>1,2</sup>, [Limeng Qiao](https://scholar.google.com/citations?user=3PFZAg0AAAAJ&hl=en)<sup>3</sup>, [Zequn Jie](https://scholar.google.com/citations?user=4sKGNB0AAAAJ&hl)<sup>3</sup>, [Zhijian Huang](https://zhijian11.github.io/)<sup>1</sup>, [Chengjian Feng](https://fcjian.github.io/)<sup>3</sup>,

[Qingfang Zheng](https://openreview.net/profile?id=%7EZheng_Qingfang1)<sup>2</sup>, [Lin Ma](https://forestlinma.com/)<sup>3</sup>, [Xiangyuan Lan](https://scholar.google.com/citations?user=c3iwWRcAAAAJ&hl)<sup>2</sup><sup>:email:</sup>, [Xiaodan Liang](https://scholar.google.com/citations?user=voxznZAAAAAJ&hl)<sup>1</sup><sup>:email:</sup>

<sup>1</sup> Sun Yat-sen University, <sup>2</sup> Peng Cheng Laboratory, <sup>3</sup> Meituan Inc.

<sup>:email:</sup> Corresponding author
</div>

<div align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href='https://wanghao9610.github.io/X-SAM/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/🌐_Project-Webpage-green?style=flat&logoColor=white' alt='webpage'>
  </a>
  <a href="https://arxiv.org/abs/2508.04655" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2508.04655-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/hao9610/X-SAM' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/HuggingFace-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/wanghao9610/X-SAM" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://47.115.200.157:7861" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=yellow' alt='Demo'>
  </a>
</div>

## :eyes: Notice

> **Note:** X-SAM is under active development, and we will continue to update the code and documentation. Please check [TODO](#white_check_mark-todo) to get our development schedule.

> **Reproducing AAAI 2026 results:** To reproduce the results reported in our AAAI 2026 submission, please switch to the [`AAAI26_Archive`](https://github.com/wanghao9610/X-SAM/tree/AAAI26_Archive) branch:
> ```bash
> git clone -b AAAI26_Archive https://github.com/wanghao9610/X-SAM.git
> ```

We strongly recommend that everyone uses **English** to communicate in issues. This helps developers from around the world discuss, share experiences, and answer questions together.

*If you have any questions or would like to collaborate, please feel free to open an issue or reach out to me at `wanghao9610@gmail.com`.*

## :boom: Updates

- **`2026-07-03`**: ✨ We update the codebase to align with [X2SAM](https://github.com/wanghao9610/X2SAM), along with a new training recipe based on Qwen3-VL.
- **`2026-06-18`**: 🎉🎉🎉 Congratulations! 🎉🎉🎉 X2SAM has been accepted by ECCV 2026!
- **`2026-04-28`**: We release [X2SAM](https://github.com/wanghao9610/X2SAM), a new project for any segmentation in images and videos. X-SAM training configs are also supported in X2SAM — welcome to try it!
- **`2026-01-29`**: We update the [camera-ready version](https://arxiv.org/abs/2508.04655) of our paper and uploaded the official [paper poster](https://github.com/user-attachments/files/24925624/AAAI2026_X-SAM_Poster.pdf) for AAAI 2026.
- **`2025-11-21`**: We release the code for X-SAM with [Qwen3-4B-Instruct-2507](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py) and [Qwen3-1.7B](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_1x7b_wothinking_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py). We will release the weights soon.
- **`2025-11-19`**: We release the code for [Training X-SAM](#5-training). Welcome to try it! If you have any questions, please feel free to open an issue.
- **`2025-11-08`**: Congratulations! 🎉🎉🎉 X-SAM has been accepted by AAAI 2026! We will release all the code in the coming week!
- **`2025-09-28`**: We update the [Local Demo](#local-demo) inference script — you can run local inference instead of on the Web Demo.
- **`2025-08-11`**: Thanks for your great attention to our work! We have deployed another [Online Demo2](http://121.43.252.12:7862). You can also try it if [Online Demo1](http://47.115.200.157:7861) is not available.
- **`2025-08-11`**: We released the effective code for [Evaluation on Segmentation Benchmarks](#6-evaluation). We have updated all code except for [Training X-SAM](#5-training).
- **`2025-08-10`**: We released the detailed instructions for [Demo Deployment](#computer-demo).
- **`2025-08-08`**: We released the simple code for [Evaluation on VLM Benchmarks](#6-evaluation).
- **`2025-08-06`**: We are excited to publish the [Technical Report](https://arxiv.org/pdf/2508.04655), please check it out for more technical details.
- **`2025-08-05`**: We provided the [Model Weights](https://huggingface.co/hao9610/X-SAM) on HuggingFace🤗.
- **`2025-07-26`**: We deployed the [Online Demo](http://47.115.200.157:7861), you can try it now!

## :rocket: Highlights

This repository provides the official PyTorch implementation, pre-trained models, training, evaluation, visualization, and demo code of X-SAM:

* X-SAM introduces a unified multimodal large language model (MLLM) framework, extending the segmentation paradigm from *segment anything* to *any segmentation*, thereby enhancing pixel-level perceptual understanding.

* X-SAM proposes a novel Visual GrounDed (VGD) segmentation task, which segments all instance objects using interactive visual prompts, empowering the model with visually grounded, pixel-wise interpretative capabilities.

* X-SAM presents a unified training strategy that enables co-training across multiple datasets. Experimental results demonstrate that X-SAM achieves state-of-the-art performance on various image segmentation benchmarks, highlighting its efficiency in multimodal, pixel-level visual understanding.

✨ This repository provides unified and effective code for training, evaluation, and visualization of segmentation MLLMs. We hope this repository will promote further research on MLLMs. For image-and-video any-segmentation, please also check out our follow-up project [X2SAM](https://github.com/wanghao9610/X2SAM).

## :book: Table of Contents
- [Abstract](#bookmark-abstract)
- [Overview](#mag-overview)
- [Benchmarks](#bar_chart-benchmarks)
- [Quickstart](#checkered_flag-quickstart)
- [Demo](#computer-demo)
- [TODO](#white_check_mark-todo)
- [Acknowledge](#pray-acknowledge)
- [Citation](#pushpin-citation)

## :bookmark: Abstract

Large Language Models (LLMs) demonstrate strong capabilities in broad knowledge representation, yet they are inherently deficient in pixel-level perceptual understanding. Although the Segment Anything Model (SAM) represents a significant advancement in visual-prompt-driven image segmentation, it exhibits notable limitations in multi-mask prediction and category-specific segmentation tasks, and it cannot integrate all segmentation tasks within a unified model architecture. To address these limitations, we present X-SAM, a streamlined Multimodal Large Language Model (MLLM) framework that extends the segmentation paradigm from *segment anything* to *any segmentation*. Specifically, we introduce a novel unified framework that enables more advanced pixel-level perceptual comprehension for MLLMs. Furthermore, we propose a new segmentation task, termed Visual GrounDed (VGD) segmentation, which segments all instance objects with interactive visual prompts and empowers MLLMs with visual grounded, pixel-wise interpretative capabilities. To enable effective training on diverse data sources, we present a unified training strategy that supports co-training across multiple datasets. Experimental results demonstrate that X-SAM achieves state-of-the-art performance on a wide range of image segmentation benchmarks, highlighting its efficiency for multimodal, pixel-level visual understanding.

## :mag: Overview

<div align="left">
  <img src="docs/srcs/images/framework.png" width="800" alt="X-SAM Overview">
  <p><em>Figure 1: Overview of X-SAM. The Vision Encoder extracts global visual representations, while the Mask Encoder captures fine-grained visual features. The Large Language Model generates the language response and produces the latent condition embedding, which guides the Mask Decoder in generating the segmentation mask.</em></p>
</div>

## :bar_chart: Benchmarks

<div align="left">
  <img src="docs/srcs/images/overall.png" width="800" alt="Overall Performance">
  <p><em>Table 1: Comparison of state-of-the-art segmentation methods across image segmentation benchmarks.</em></p>
</div>

👉 **More benchmark results can be found in [benchmarks.md](docs/mds/benchmarks.md).**

👉 **To reproduce these AAAI 2026 submission results**, use the [`AAAI26_Archive`](https://github.com/wanghao9610/X-SAM/tree/AAAI26_Archive) branch (see [Notice](#eyes-notice)).

## :checkered_flag: Quickstart

### 1. Structure

We provide a detailed project structure for X-SAM. Please follow this structure to organize the project.

<details open>
<summary><b>📁 Project</b></summary>

```bash
X-SAM
├── datas
│   ├── img_chat
│   ├── img_gcgseg
│   ├── img_genseg
│   ├── img_intseg
│   ├── img_ovseg
│   ├── img_reaseg
│   ├── img_refseg
│   ├── img_vgdseg
│   └── LMUData
├── inits
│   ├── huggingface
│   ├── mask2former-swin-large-coco-panoptic
│   ├── Phi-3-mini-4k-instruct
│   ├── sam-vit-large
│   └── X-SAM
├── xsam
│   ├── requirements
│   └── xsam
│       ├── configs
│       ├── dataset
│       ├── demo
│       ├── engine
│       ├── evaluation
│       ├── model
│       ├── structures
│       ├── tools
│       └── utils
├── wkdrs
│   ├── s1_train
│   │   └── ...
│   ├── s2_train
│   │   └── ...
│   ├── s3_train
│   │   └── ...
│   └── ...
...
```

</details>

### 2. Environment

#### Basic setup

```bash
# 1) Clone X-SAM and enter project home directory
git clone https://github.com/wanghao9610/X-SAM.git
cd X-SAM
export PROJ_HOME="$(realpath ./)"
export PYTHONPATH="$PROJ_HOME/xsam:$PYTHONPATH"

# 2) Create and activate conda environment
conda create -n xsam python=3.10 -y
conda activate xsam

# 3) Install X-SAM dependencies
cd "$PROJ_HOME/xsam"
pip install -r requirements/runtime.txt
pip install -r requirements/deepspeed.txt
pip install -r requirements/xsam.txt

# 4) Compile Deformable-Attention
cd "$PROJ_HOME/xsam/xsam/model/ops"
bash make.sh

# 5) Setup .env (machine-local paths for tooling; .env is gitignored)
cd "$PROJ_HOME"
cp -n .env.example .env
# CONDA_HOME  — conda installation root
# PYTHON_HOME — active env root (xsam); basename is the env name
sed -i "s|YOUR_CONDA_HOME|$(conda info --base)|; s|YOUR_PYTHON_HOME|${CONDA_PREFIX}|" .env
```

#### Optional: set CUDA_HOME

```bash
export CUDA_HOME="your_cuda12.4_path"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo -e "CUDA version:\n$(nvcc -V)"
```

#### Optional: install gcc 11

```bash
conda install gcc=11 gxx=11 -c conda-forge -y
```

#### Optional: install VLMEvalKit

```bash
cd "$PROJ_HOME"
git clone -b v0.3rc1 https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### 3. Dataset

Please refer to [datasets.md](docs/mds/datasets.md) for detailed instructions on data preparation.

### 4. Model

Please refer to [models.md](docs/mds/models.md) for detailed instructions on model preparation.

### 5. Training

We provide a comprehensive script that covers the entire pipeline, including training, evaluation, and visualization. For detailed instructions, please refer to [gpu_run.sh](runs/gpu_run.sh).

```bash
cd "$PROJ_HOME"

# Distributed training across multiple nodes.
# Set NODE_RANK to specify the rank (ID) of each node in distributed training.
# MASTER_ADDR and MASTER_PORT should be set to the IP address and port of your master node.
# Execute the following commands on every machine, updating NODE_RANK for each node accordingly.

# 1) Stage 1: Segmentor Fine-tuning
NUM_NODES=1 NODE_RANK=0 GPU_PER_NODE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 \
  bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s1_train/xsam_sam_vit_large_m2f_e36_gpu16.py \
  "train"

# 2) Stage 2: Alignment Pre-training (Phi-3 example)
NUM_NODES=1 NODE_RANK=0 GPU_PER_NODE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 \
  bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s2_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_e1_gpu16.py \
  "train"

# 3) Stage 3: Mixed Fine-tuning (Phi-3 example)
NUM_NODES=1 NODE_RANK=0 GPU_PER_NODE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 \
  bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  "train segeval vlmeval visualize"
```

<details close>
<summary><b>More Stage-3 configs</b></summary>

| Backbone | Config |
|----------|--------|
| Phi-3-mini-4k-instruct | [xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py](xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py) |
| Qwen3-4B-Instruct-2507 | [xsam_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py) |
| Qwen3-1.7B | [xsam_qwen3_1x7b_wothinking_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_1x7b_wothinking_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py) |
| Qwen3-VL-4B-Instruct (LoRA) | [xsam_qwen3_vl_4b_instruct_sam_vit_large_m2f_e1_gpu16_lora.py](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_vl_4b_instruct_sam_vit_large_m2f_e1_gpu16_lora.py) |
| Qwen3-VL-2B-Instruct (LoRA) | [xsam_qwen3_vl_2b_instruct_sam_vit_large_m2f_e1_gpu16_lora.py](xsam/xsam/configs/xsam/s3_train/xsam_qwen3_vl_2b_instruct_sam_vit_large_m2f_e1_gpu16_lora.py) |

</details>

### 6. Evaluation

#### Segmentation Benchmarks

```bash
cd "$PROJ_HOME"
bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  "segeval"
```

#### VLM Benchmarks

```bash
cd "$PROJ_HOME"
bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  "vlmeval"
```

### 7. Visualization

```bash
cd "$PROJ_HOME"
bash runs/gpu_run.sh \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  "visualize"
```

### 8. Tools

<details close>
<summary><b>Dataset Exploration</b></summary>

We provide a tool for dataset exploration. You can use it to explore the dataset and get visualizations of the samples.

```bash
cd "$PROJ_HOME"
python xsam/xsam/tools/explore.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --output-dir "wkdrs/dataset_exploration" \
  --subset train \
  --max-samples 100
```

</details>

<details close>
<summary><b>Model Conversion</b></summary>

We provide a tool for model conversion. You can use it to convert the checkpoint to the Hugging Face format.

```bash
cd "$PROJ_HOME"
python xsam/xsam/tools/model_tools/pth_to_hf.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  "wkdrs/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16" \
  --pth_model latest
```

</details>

## :computer: Demo

### Local Demo

<details open>
<summary><b>🏞️ Inference</b></summary>

```bash
cd "$PROJ_HOME"
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name TASK_NAME \
  --image INPUT_IMAGE/INPUT_DIR \
  --prompt INPUT_PROMPT \
  --vprompt-masks INPUT_VPROMPT_MASKS
```

</details>

<details close>
<summary><b>🏞️ Examples</b></summary>

```bash
# Example: img_chat
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_chat \
  --image xsam/xsam/demo/sample.jpg \
  --prompt "What is unusal about this image?"

# Example: img_genseg
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_genseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs" \
  --prompt "ins: person, bird, boat; sem: water, sky"

# Example: img_refseg
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_refseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs" \
  --prompt "the ironing man"

# Example: img_reaseg
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_reaseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs" \
  --prompt "What can be used to warm clothes?"

# Example: img_gcgseg
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_gcgseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs"

# Example: img_intseg (requires visual prompt masks)
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_intseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs" \
  --vprompt-masks PATH_TO_VPROMPT_MASK.png

# Example: img_vgdseg (requires visual prompt masks)
python xsam/xsam/demo/demo.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --task-name img_vgdseg \
  --image xsam/xsam/demo/sample.jpg \
  --output-dir "wkdrs/demo_outputs" \
  --vprompt-masks PATH_TO_VPROMPT_MASK1.png PATH_TO_VPROMPT_MASK2.png
```

</details>

### Web Demo

<details open>
<summary><b>🛠️ Deployment</b></summary>

```bash
cd "$PROJ_HOME"
python xsam/xsam/demo/app.py \
  xsam/xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py \
  --pth_model "inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin" \
  --log-dir "wkdrs/app_logs" \
  --seed 0 \
  --port 7860
```

Then, you can access the demo website at `http://localhost:7860`.

<!-- <div align="center">
  <img src="docs/srcs/images/app.png" width="800" alt="Gradio Demo">
</div>
</details> -->

## :white_check_mark: TODO

- [x] Release the [Online Demo](http://47.115.200.157:7861).
- [x] Release the [Model Weights](https://huggingface.co/hao9610/X-SAM).
- [x] Release the [Technical Report](https://arxiv.org/abs/2508.04655).
- [x] Release the code for [Evaluation on VLM Benchmarks](#6-evaluation).
- [x] Release the code for [Demo Deployment](#computer-demo).
- [x] Release the code for [Evaluation on Segmentation Benchmarks](#6-evaluation).
- [x] Release the code for [Training X-SAM](#5-training).
- [x] Release the code and weight for X-SAM with Qwen3.
- [x] Release the code and weight for X-SAM with Qwen3-VL.
- [ ] Release the inference and demo code supporting transformers.
- [ ] Release the code and instructions for training with Ascend NPU.

## :pray: Acknowledge

This project has referenced some excellent open-sourced repos ([xtuner](https://github.com/InternLM/xtuner), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [Sa2VA](https://github.com/magic-research/Sa2VA)). Thanks for their wonderful works and contributions to the community!

## :pushpin: Citation

If you find X-SAM and X2SAM are helpful for your research or applications, please consider giving us a star 🌟 and citing the following papers by the following BibTex entry.

```bibtex
@inproceedings{wang2026xsam,
  title={X-SAM: From segment anything to any segmentation},
  author={Wang, Hao and Qiao, Limeng and Jie, Zequn and Huang, Zhijian and Feng, Chengjian and Zheng, Qingfang and Ma, Lin and Lan, Xiangyuan and Liang, Xiaodan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={26187--26196},
  year={2026}
}

@article{wang2026x2sam,
  title={X2SAM: Any Segmentation in Images and Videos},
  author={Wang, Hao and Qiao, Limeng and Zhang, Chi and Wan, Guanglu and Ma, Lin and Lan, Xiangyuan and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2605.00891},
  year={2026}
}
```
