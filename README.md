<div align="center">
<h1>X-SAM </h1>
<h3>From Segment Anything to Any Segmentation</h3>

[Hao Wang](https://github.com/wanghao9610)<sup>1,2</sup>,[Limeng Qiao](https://scholar.google.com/citations?user=3PFZAg0AAAAJ&hl=en)<sup>3</sup>,[Zequn Jie](https://scholar.google.com/citations?user=4sKGNB0AAAAJ&hl)<sup>3</sup>, [Zhijian Huang](https://zhijian11.github.io/)<sup>1</sup>, [Chengjian Feng](https://fcjian.github.io/)<sup>3</sup>, 

[Qingfang Zheng](https://openreview.net/profile?id=%7EZheng_Qingfang1)<sup>1</sup>, [Lin Ma](https://forestlinma.com/)<sup>3</sup>, [Xiangyuan Lan](https://scholar.google.com/citations?user=c3iwWRcAAAAJ&hl)<sup>2</sup><sup>:email:</sup>, [Xiaodan Liang](https://scholar.google.com/citations?user=voxznZAAAAAJ&hl)<sup>1,2</sup><sup>:email:</sup>

<sup>1</sup> Sun Yat-sen University, <sup>2</sup> Pengcheng Lab, <sup>3</sup> Meituan Inc

<sup>:email:</sup> Corresponding author.
</div>

<div align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-paper_id-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/wanghao9610/X-SAM" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://47.115.200.157:7861" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://wanghao9610.github.io/X-SAM/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
</div>

## :fire: Updates
- **`2025-07-24`**: We release the [Demo](http://47.115.200.157:7861) of X-SAM.

## :rocket: Introduction
This project provides the official PyTorch implementation of X-SAM.

* X-SAM is novel unified segmentation MLLMs, which offers superior performance on all image segmentation benchmarks.

* X-SAM integrates the SAM into MLLMs via a unified formulation adapted to all image segmentation, extending the SAM's capability from *segment anything* to *any segmentation*.

* X-SAM co-trains on multi data sources via a effective multi-stage training strategy, achieving the robust performance across all tasks.

This project provides awesome code for segmentation MLLMs:
* Training code for segmentation MLLMs.
* Evaluation code for all image segmentation benchmarks.
* Visualization code for segmentation MLLMs.
* Training code for LLaVA-based MLLMs (based on [XTuner](https://github.com/InternLM/xtuner)).
* Evaluation code for all VLM benchmarks (based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)).

*If you have any questions, please feel free to open an issue.*

## :page_facing_up: Abstract

The Segment Anything Model (SAM) has emerged as a pivotal advancement in computer vision, particularly within the context of visual-prompt-driven segmentation. However, SAM is constrained by intrinsic limitations in multi-mask prediction and category-specific image segmentation tasks. Concurrently, Large Language Models (LLMs) have exhibited remarkable proficiency in comprehensive knowledge representation across a wide range of domains, yet they inherently lack the capacity for pixel-level perceptual understanding. To bridge these complementary gaps, we present X-SAM, a streamlined Multimodal Large Language Model (MLLM) framework that seamlessly integrates SAM with LLMs, thereby extending SAM's capabilities from *segment anything* to *any segmentation*. Specifically, we introduce a novel approach for integrating SAM with MLLMs, which facilitates more advanced dense, pixel-level perceptual comprehension within MLLMs. Furthermore, we propose a new segmentation paradigm, termed **V**isual **G**roun**D**ed (VGD) segmentation, which empowers MLLMs with visual grounded, pixel-wise interpretative capabilities. To enable effective training of MLLMs on diverse data sources, we devise a unified training strategy that supports co-training across multiple datasets. Experimental results demonstrate that X-SAM achieves state-of-the-art performance on a wide range of image segmentation benchmarks, highlighting its efficiency for multimodal pixel-level visual understanding.

## :mag: Overview


<img src="docs/images/xsam_framework.png">


## :bar_chart: Benchmark Results

Please refer to the [benchmark results](docs/benchmark_results.md) for more details.

## :white_check_mark: TODO
- [x] Release the [Demo](http://47.115.200.157:7861).
- [ ] Release the [weight](https://huggingface.co/wanghao9610/X-SAM).
- [ ] Release the code and instructions for demo.
- [ ] Release the code for evaluation on all segmentation benchmarks.
- [ ] Release the code for evaluation on all VLM Benchmarks.
- [ ] Release the code for training LLaVA-based MLLMs.
- [ ] Release the code for training X-SAM (More than 500 🌟).

## :blush: Acknowledge

This project has referenced some excellent open-sourced repositories: [XTuner](https://github.com/InternLM/xtuner), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [Sa2VA](https://github.com/magic-research/Sa2VA). Thanks for their wonderful works and contributions to the community.

## :pushpin: Citation
If you find X-SAM helpful for your research or applications, please consider giving us a star 🌟 and citing it using the following BibTeX entry.

```bibtex

```
