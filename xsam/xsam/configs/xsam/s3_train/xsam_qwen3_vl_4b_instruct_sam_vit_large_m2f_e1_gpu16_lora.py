from copy import deepcopy
from itertools import chain
from os import getenv

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from xsam.dataset import (
    ConcatDataset,
    ImgChatDataset,
    ImgGCGSegDataset,
    ImgGenSegDataset,
    ImgGRefSegDataset,
    ImgIntSegDataset,
    ImgOVSegDataset,
    ImgReaSegDataset,
    ImgRefSegDataset,
    ImgVGDSegDataset,
)
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    img_chat_map_fn,
    img_gcgseg_map_fn,
    img_genseg_map_fn,
    img_intseg_map_fn,
    img_ovseg_map_fn,
    img_reaseg_map_fn,
    img_refseg_map_fn,
    img_vgdseg_map_fn,
    template_map_fn_factory,
)
from xsam.dataset.process_fns import (
    img_gcgseg_postprocess_fn,
    img_genseg_postprocess_fn,
    img_grefseg_postprocess_fn,
    img_intseg_postprocess_fn,
    img_ovseg_postprocess_fn,
    img_reaseg_postprocess_fn,
    img_refseg_postprocess_fn,
    img_vgdseg_postprocess_fn,
    process_map_fn_factory,
)
from xsam.dataset.processors import Qwen3VLImageProcessor, SamImageProcessor
from xsam.dataset.samplers import CustomBatchSampler, SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, GenerationChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop, ValLoop
from xsam.evaluation.evaluators import (
    ImgGCGSegEvaluator,
    ImgGenSegEvaluator,
    ImgIntSegEvaluator,
    ImgOVSegEvaluator,
    ImgReaSegEvaluator,
    ImgRefSegEvaluator,
    ImgVGDSegEvaluator,
)
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
from xsam.model.utils import frame_transpose_temporal_process_fn, temporal_process_fn_factory
from xsam.model.vlms.qwen3vl import Qwen3VLForConditionalGeneration
from xsam.utils.template import PROMPT_TEMPLATE
from xsam.utils.visualize import Visualizer

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./wkdrs/")

# Model
vlm_name_or_path = init_dir + "Qwen3-VL-4B-Instruct"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Specify the pretrained pth
s1_pretrained_pth = work_dir + "s1_train/xsam_sam_vit_large_m2f_e36_gpu16/pytorch_model.bin"

# Prompt
prompt_template = PROMPT_TEMPLATE.qwen3_vl_instruct
max_length = int(262144 - 40**2 - 1024)

# Scheduler & Optimizer
batch_size = 4  # per_device
batch_mult = 1  # batch_mult for image datasets
mega_batch_mult = 1  # mega_batch_mult for dataloader
oversample_mult = 1.0  # oversample_mult for datasets
oversample_ratio = 0.1  # oversample ratio for datasets
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate
val_interval = 2000
val_sample = -1  # Maximum samples per dataset during training evaluation; <=0 means full dataset, which will be very slow

# Logging
log_interval = 10

# Inference
inf_interval = 2000
use_infer = True  # False for disable inference during training, True for enable inference during training
use_eval = True  # False for disable evaluation during training, True for enable evaluation during training
output_ids_with_output = True  # False for predict mode when inference, True for tensor mode when evaluation

SYSTEM = ""
generation_images = [
    code_dir + "xsam/configs/xsam/samples/sample.jpg",
]
generation_inputs = {
    "chat": "What is unusal about this image?",
    "genseg": "Can you generate segmentation masks for this image based on the specified categories: <p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, <p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, <p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, <p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, <p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, <p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, <p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, <p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, <p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, <p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, <p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, <p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, <p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, <p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, <p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, <p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, <p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, <p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, <p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, <p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, <p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, <p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, <p>curtain</p>, <p>door</p>, <p>floor wood</p>, <p>flower</p>, <p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, <p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, <p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, <p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, <p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall brick</p>, <p>wall stone</p>, <p>wall tile</p>, <p>wall wood</p>, <p>water</p>, <p>window blind</p>, <p>window</p>, <p>tree</p>, <p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, <p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, <p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, <p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p>? Please output the segmentation mask.",
    "refseg": "Can you segment <p>the ironing man</p> in this image? Please output the corresponding segmentation mask.",
    "reaseg": "<p>What can be used to warm clothes?</p> Please output the corresponding segmentation mask.",
    "gcgseg": "Can you provide a brief description of the this image? Respond with interleaved segmentation masks for the corresponding phrases.",
    "intseg": "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "vgdseg": "Can you segment the image based on the following regions: <p><region></p>, <p><region></p>? Please output the segmentation mask.",
}
vprompt_masks = {
    "chat": (None,),
    "genseg": (None,),
    "refseg": (None,),
    "reaseg": (None,),
    "gcgseg": (None,),
    "intseg": (code_dir + "xsam/configs/xsam/samples/vpmasks/img_vpmask0.png",),
    "vgdseg": (
        code_dir + "xsam/configs/xsam/samples/vpmasks/img_vpmask0.png",
        code_dir + "xsam/configs/xsam/samples/vpmasks/img_vpmask1.png",
    ),
}
postprocess_fn = {
    "chat": None,
    "genseg": img_genseg_postprocess_fn,
    "refseg": img_refseg_postprocess_fn,
    "reaseg": img_reaseg_postprocess_fn,
    "gcgseg": img_gcgseg_postprocess_fn,
    "intseg": img_intseg_postprocess_fn,
    "vgdseg": img_vgdseg_postprocess_fn,
}

#######################################################################
#            PART 2  Model & Tokenizer &  Processor              #
#######################################################################
special_tokens = ["<SEG>", "<p>", "</p>"]    # special tokens
image_token = "<|vision_start|><image><|vision_end|>"  # Qwen3-VL vision token wrapper
cond_type = "phrase"  # "phrase" or "cls" or "all"
ignore_value = 255  # value for ignored mask
ignore_label = -100  # label for ignored class
background_label = -1  # label for background class
ptoken_shift = 0  # 0 for no shift(include <p> and </p>), 1 for shift by one token(exclude <p> and </p>)
expand2square = False  # whether to expand the image to a square
use_placeholder = True  # whether to use placeholder for the image, especially for QwenVL series
sampler_input_feat = "extra_pixel_values"  # "pixel_values" or "extra_pixel_values"
sampler_pooling_mode = "mean"  # "mean" or "max"
sampler_pooling_kernel_size = 4
sampler_pooling_output_size = None
num_class = 128  # num_class for vocabulary based tasks
extra_num_class = 8  # num_class for non-vocabulary based tasks

train_tasks = ["chat", "genseg", "refseg", "reaseg", "gcgseg", "vgdseg"]
infer_tasks = ["chat", "genseg", "refseg", "reaseg", "gcgseg", "intseg", "vgdseg"]
eval_tasks = ["genseg", "ovseg", "refseg", "reaseg", "gcgseg", "vgdseg", "intseg"]
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=vlm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=Qwen3VLImageProcessor.from_pretrained,
    pretrained_model_name_or_path=vlm_name_or_path,
    trust_remote_code=True,
)

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_vlm=False,
    freeze_llm=False,
    freeze_vision_encoder=True,
    freeze_mask_encoder=False,
    use_dual_encoder=False,
    use_vision_sampler=True,
    extra_temporal_process_fn=dict(type=temporal_process_fn_factory, fn=frame_transpose_temporal_process_fn),
    connector_type="conv",
    cond_type=cond_type,
    extra_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat=sampler_input_feat,
    sampler_pooling_mode=sampler_pooling_mode,
    sampler_pooling_kernel_size=sampler_pooling_kernel_size,
    sampler_pooling_output_size=sampler_pooling_output_size,
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    tokenizer=tokenizer,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    postprocess_fn=img_genseg_postprocess_fn,
    vlm=dict(
        type=Qwen3VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=vlm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ),
    vlm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=None,
        exclude_modules=r".*visual.*",
        task_type="CAUSAL_LM",
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                ignore_value=ignore_value,
                ignore_label=ignore_label,
                background_label=background_label,
                attn_implementation="compiled",
                use_text_cross_attn=False,
                use_zero_init=False,
                head_cls_type="learn",
                use_repeat_cond=False,
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        use_memory=False,
        init_decoder=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
img_chat_data_root = data_dir + "img_chat/"
img_genseg_data_root = data_dir + "img_genseg/"
img_ovseg_data_root = data_dir + "img_ovseg/"
img_refseg_data_root = data_dir + "img_refseg/"
img_reaseg_data_root = data_dir + "img_reaseg/"
img_gcgseg_data_root = data_dir + "img_gcgseg/"
img_intseg_data_root = data_dir + "img_intseg/"
img_vgdseg_data_root = data_dir + "img_vgdseg/"

img_chat_llava_dataset = dict(
    type=ImgChatDataset,
    data_path=img_chat_data_root + "llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=img_chat_data_root + "llava/llava_images",
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    task_name="img_chat",
    data_name="img_chat_llava",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_chat_map_fn,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    preprocess_text_data=False,
    batch_mult=batch_mult,
)

img_genseg_coco_dataset = dict(
    type=ImgGenSegDataset,
    data_path=img_genseg_data_root + "coco2017/annotations/panoptic_train2017.json",
    image_folder=img_genseg_data_root + "coco2017/train2017",
    pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_train2017",
    tokenizer=tokenizer,
    task_name="img_genseg",
    data_name="img_genseg_coco_panoptic_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_genseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    num_class=num_class,
    max_length=max_length,
    use_full_cat=False,
    use_variant_cat=True,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_refseg_refcoco_dataset = dict(
    type=ImgRefSegDataset,
    data_root=img_refseg_data_root + "refcocos",
    image_folder=img_refseg_data_root + "refcocos/images/train2014",
    dataset="refcoco",
    data_split="train",
    tokenizer=tokenizer,
    task_name="img_refseg",
    data_name="img_refseg_refcoco_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=img_refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_refseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_refseg_refcocop_dataset = dict(
    type=ImgRefSegDataset,
    data_root=img_refseg_data_root + "refcocos",
    image_folder=img_refseg_data_root + "refcocos/images/train2014",
    dataset="refcocop",
    data_split="train",
    tokenizer=tokenizer,
    task_name="img_refseg",
    data_name="img_refseg_refcocop_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=img_refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_refseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_refseg_refcocog_dataset = dict(
    type=ImgRefSegDataset,
    data_root=img_refseg_data_root + "refcocos",
    image_folder=img_refseg_data_root + "refcocos/images/train2014",
    dataset="refcocog",
    data_split="train",
    tokenizer=tokenizer,
    task_name="img_refseg",
    data_name="img_refseg_refcocog_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=img_refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_refseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_reaseg_lisa_dataset = dict(
    type=ImgReaSegDataset,
    data_root=img_reaseg_data_root + "lisa",
    image_folder=img_reaseg_data_root + "lisa/train",
    explain_path=img_reaseg_data_root + "lisa/explanatory/train.json",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="img_reaseg",
    data_name="img_reaseg_lisa_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=img_reaseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_reaseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_gcgseg_grandf_dataset = dict(
    type=ImgGCGSegDataset,
    data_path=img_gcgseg_data_root + "grand_f/annotations/train/GranDf_HA_GCG_train.json",
    data_root=img_gcgseg_data_root,
    image_folder=img_gcgseg_data_root + "grand_f/images/GranDf_HA_images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="img_gcgseg",
    data_name="img_gcgseg_grandf_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_gcgseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_gcgseg_refcocog_dataset = dict(
    type=ImgGCGSegDataset,
    data_path=img_gcgseg_data_root + "grand_f/annotations/train/RefCOCOg_GCG_train.json",
    data_root=img_gcgseg_data_root,
    image_folder=img_gcgseg_data_root + "grand_f/images/coco2014/train2014",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="img_gcgseg",
    data_name="img_gcgseg_refcocog_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_gcgseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_gcgseg_psg_dataset = dict(
    type=ImgGCGSegDataset,
    data_path=img_gcgseg_data_root + "grand_f/annotations/train/OpenPsgGCG_train.json",
    data_root=img_gcgseg_data_root,
    image_folder=img_gcgseg_data_root + "grand_f/images/coco2017",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="img_gcgseg",
    data_name="img_gcgseg_psg_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_gcgseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_gcgseg_flickr_dataset = dict(
    type=ImgGCGSegDataset,
    data_path=img_gcgseg_data_root + "grand_f/annotations/train/flickr_mergedGT_GCG_train.json",
    data_root=img_gcgseg_data_root,
    image_folder=img_gcgseg_data_root + "grand_f/images/flickr30k/images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="img_gcgseg",
    data_name="img_gcgseg_flickr_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_gcgseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    num_class=extra_num_class,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

img_vgdseg_coco_dataset = dict(
    type=ImgVGDSegDataset,
    source_data_path=img_vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_train2017.json",
    data_path=img_vgdseg_data_root + "coco_vgd/annotations/img_vgdseg_coco_train.json",
    image_folder=img_vgdseg_data_root + "coco_vgd/coco2017/train2017",
    tokenizer=tokenizer,
    data_mode="train",
    task_name="img_vgdseg",
    data_name="img_vgdseg_coco_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=img_vgdseg_map_fn,
        cond_type=cond_type,
        image_token=image_token,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_negative_sample=False,
    num_class=extra_num_class,
    max_length=max_length,
    expand2square=expand2square,
    use_placeholder=use_placeholder,
    ignore_value=ignore_value,
    ignore_label=ignore_label,
    background_label=background_label,
    ptoken_shift=ptoken_shift,
    batch_mult=batch_mult,
)

image_train_datasets = {
    "chat": [img_chat_llava_dataset],
    "genseg": [img_genseg_coco_dataset],
    "refseg": [
        img_refseg_refcoco_dataset,
        img_refseg_refcocop_dataset,
        img_refseg_refcocog_dataset,
    ],
    "reaseg": [img_reaseg_lisa_dataset],
    "vgdseg": [img_vgdseg_coco_dataset],
    "gcgseg": [
        img_gcgseg_grandf_dataset,
        img_gcgseg_refcocog_dataset,
        img_gcgseg_psg_dataset,
        img_gcgseg_flickr_dataset,
    ],
}

train_datasets = list(chain(*[image_train_datasets.get(task, []) for task in train_tasks]))
generation_inputs = [generation_inputs[task] for task in infer_tasks]
vprompt_masks = [vprompt_masks[task] for task in infer_tasks]
postprocess_fn = [postprocess_fn[task] for task in infer_tasks]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=dict(
        type=ConcatDataset,
        oversample_ratio=oversample_ratio,
        oversample_mult=oversample_mult,
        datasets=train_datasets,
    ),
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        per_device_batch_size=batch_size,
        accumulative_counts=accumulative_counts,
        mega_batch_mult=mega_batch_mult,
    ),
    batch_sampler=dict(type=CustomBatchSampler, drop_last=False),
    collate_fn=dict(type=xsam_collate_fn),
)

val_datasets = {
    "genseg": [
        dict(
            type=ImgGenSegDataset,
            data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
            image_folder=img_genseg_data_root + "coco2017/val2017",
            pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
            sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_genseg",
            data_name="img_genseg_coco_panoptic_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            output_ids_with_output=output_ids_with_output,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_panoptic",
                threshold=0.0,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_genseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory,
                template=prompt_template,
                output_suffix=output_ids_with_output,
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGenSegDataset,
            data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
            image_folder=img_genseg_data_root + "coco2017/val2017",
            pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
            sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_genseg",
            data_name="img_genseg_coco_panoptic_semantic_val",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_genseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_semantic",
            ),
            template_map_fn=dict(
                type=template_map_fn_factory,
                template=prompt_template,
                output_suffix=output_ids_with_output,
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGenSegDataset,
            data_path=img_genseg_data_root + "coco2017/annotations/instances_val2017.json",
            image_folder=img_genseg_data_root + "coco2017/val2017",
            task_name="img_genseg",
            data_name="img_genseg_coco_instance_val",
            data_mode="eval",
            tokenizer=tokenizer,
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_instance",
                threshold=0.0,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_genseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory,
                template=prompt_template,
                output_suffix=output_ids_with_output,
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "ovseg": [
        dict(
            type=ImgOVSegDataset,
            data_path=img_ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
            image_folder=img_ovseg_data_root + "ade20k/images/validation",
            pan_segmap_folder=img_ovseg_data_root + "ade20k/ade20k_panoptic_val",
            sem_segmap_folder=img_ovseg_data_root + "ade20k/annotations_detectron2/validation",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_ovseg",
            data_name="img_ovseg_ade20k_panoptic_val",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_ovseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            postprocess_fn=dict(
                type=process_map_fn_factory, fn=img_ovseg_postprocess_fn, threshold=0.0, task_name="img_ovseg_panoptic"
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgOVSegDataset,
            data_path=img_ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
            image_folder=img_ovseg_data_root + "ade20k/images/validation",
            pan_segmap_folder=img_ovseg_data_root + "ade20k/ade20k_panoptic_val",
            sem_segmap_folder=img_ovseg_data_root + "ade20k/annotations_detectron2/validation",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_ovseg",
            data_name="img_ovseg_ade20k_panoptic_semantic_val",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_ovseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            postprocess_fn=dict(
                type=process_map_fn_factory, fn=img_ovseg_postprocess_fn, task_name="img_ovseg_semantic"
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgOVSegDataset,
            data_path=img_ovseg_data_root + "ade20k/ade20k_instance_val.json",
            image_folder=img_ovseg_data_root + "ade20k/images/validation",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_ovseg",
            data_name="img_ovseg_ade20k_instance_val",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_ovseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            postprocess_fn=dict(
                type=process_map_fn_factory, fn=img_ovseg_postprocess_fn, task_name="img_ovseg_instance", threshold=0.0
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=num_class,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "refseg": [
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcoco",
            data_split="val",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcoco_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcoco",
            data_split="testA",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcoco_testA",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcoco",
            data_split="testB",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcoco_testB",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            num_class=1,
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcocop",
            data_split="val",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcocop_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcocop",
            data_split="testA",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcocop_testA",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcocop",
            data_split="testB",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcocop_testB",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcocog",
            data_split="val",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcocog_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="refcocog",
            data_split="test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_refcocog_test",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=img_refseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="grefcoco",
            data_split="val",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_grefcoco_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_grefseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="grefcoco",
            data_split="testA",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_grefcoco_testA",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_grefseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGRefSegDataset,
            data_root=img_refseg_data_root + "refcocos",
            image_folder=img_refseg_data_root + "refcocos/images/train2014",
            dataset="grefcoco",
            data_split="testB",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_refseg",
            data_name="img_refseg_grefcoco_testB",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_grefseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=cond_type,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            num_class=1,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "reaseg": [
        dict(
            type=ImgReaSegDataset,
            data_root=img_reaseg_data_root + "lisa",
            image_folder=img_reaseg_data_root + "lisa/val",
            data_split="val",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_reaseg",
            data_name="img_reaseg_lisa_val",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_reaseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_reaseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_variant_cat=True,
            use_random_cat=True,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgReaSegDataset,
            data_root=img_reaseg_data_root + "lisa",
            image_folder=img_reaseg_data_root + "lisa/test",
            data_split="test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_reaseg",
            data_name="img_reaseg_lisa_test_short",
            query_type="phrase",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_reaseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_reaseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_variant_cat=True,
            use_random_cat=True,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgReaSegDataset,
            data_root=img_reaseg_data_root + "lisa",
            image_folder=img_reaseg_data_root + "lisa/test",
            data_split="test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_reaseg",
            data_name="img_reaseg_lisa_test_long",
            query_type="sentence",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_reaseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_reaseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_variant_cat=True,
            use_random_cat=True,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgReaSegDataset,
            data_root=img_reaseg_data_root + "lisa",
            image_folder=img_reaseg_data_root + "lisa/test",
            data_split="test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_reaseg",
            data_name="img_reaseg_lisa_test_all",
            query_type="all",
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            output_ids_with_output=output_ids_with_output,
            image_processor=image_processor,
            postprocess_fn=img_reaseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_reaseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_variant_cat=True,
            use_random_cat=True,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "gcgseg": [
        dict(
            type=ImgGCGSegDataset,
            data_path=img_gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_mask_gt.json",
            cap_data_path=img_gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_caption_gt.json",
            data_root=img_gcgseg_data_root,
            image_folder=img_gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_gcgseg",
            data_name="img_gcgseg_val",
            output_ids_with_output=False,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            postprocess_fn=img_gcgseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_gcgseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgGCGSegDataset,
            data_path=img_gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_mask_gt.json",
            cap_data_path=img_gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_caption_gt.json",
            data_root=img_gcgseg_data_root,
            image_folder=img_gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
            data_mode="eval",
            tokenizer=tokenizer,
            task_name="img_gcgseg",
            data_name="img_gcgseg_test",
            output_ids_with_output=False,
            cond_type=cond_type,
            special_tokens=special_tokens,
            image_processor=image_processor,
            extra_image_processor=extra_image_processor,
            postprocess_fn=img_gcgseg_postprocess_fn,
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_gcgseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "vgdseg": [
        dict(
            type=ImgVGDSegDataset,
            source_data_path=img_vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
            data_path=img_vgdseg_data_root + "coco_vgd/annotations/img_vgdseg_coco_val.json",
            image_folder=img_vgdseg_data_root + "coco_vgd/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_vgdseg",
            data_name="img_vgdseg_coco_point_val",
            data_mode="eval",
            visual_prompt_type="point_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_vgdseg_postprocess_fn,
                threshold=0.0,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_vgdseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_negative_sample=False,
            num_class=16,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgVGDSegDataset,
            source_data_path=img_vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
            data_path=img_vgdseg_data_root + "coco_vgd/annotations/img_vgdseg_coco_val.json",
            image_folder=img_vgdseg_data_root + "coco_vgd/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_vgdseg",
            data_name="img_vgdseg_coco_scribble_val",
            data_mode="eval",
            visual_prompt_type="scribble_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_vgdseg_postprocess_fn,
                threshold=0.0,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_vgdseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_negative_sample=False,
            num_class=16,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgVGDSegDataset,
            source_data_path=img_vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
            data_path=img_vgdseg_data_root + "coco_vgd/annotations/img_vgdseg_coco_val.json",
            image_folder=img_vgdseg_data_root + "coco_vgd/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_vgdseg",
            data_name="img_vgdseg_coco_box_val",
            data_mode="eval",
            visual_prompt_type="box_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_vgdseg_postprocess_fn,
                threshold=0.0,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_vgdseg_map_fn,
                cond_type=cond_type,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_negative_sample=False,
            num_class=16,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgVGDSegDataset,
            source_data_path=img_vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
            data_path=img_vgdseg_data_root + "coco_vgd/annotations/img_vgdseg_coco_val.json",
            image_folder=img_vgdseg_data_root + "coco_vgd/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_vgdseg",
            data_name="img_vgdseg_coco_mask_val",
            data_mode="eval",
            visual_prompt_type="mask_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_vgdseg_postprocess_fn,
                threshold=0.0,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_vgdseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            use_negative_sample=False,
            num_class=16,
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
    "intseg": [
        dict(
            type=ImgIntSegDataset,
            source_data_path=img_intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
            data_path=img_intseg_data_root + "coco_int/annotations/img_intseg_coco_val.json",
            image_folder=img_intseg_data_root + "coco_int/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_intseg",
            data_name="img_intseg_coco_point_val",
            data_mode="eval",
            visual_prompt_type="point_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_intseg_postprocess_fn,
                threshold=0.5,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_intseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgIntSegDataset,
            source_data_path=img_intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
            data_path=img_intseg_data_root + "coco_int/annotations/img_intseg_coco_val.json",
            image_folder=img_intseg_data_root + "coco_int/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_intseg",
            data_name="img_intseg_coco_scribble_val",
            data_mode="eval",
            visual_prompt_type="scribble_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_intseg_postprocess_fn,
                threshold=0.5,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_intseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgIntSegDataset,
            source_data_path=img_intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
            data_path=img_intseg_data_root + "coco_int/annotations/img_intseg_coco_val.json",
            image_folder=img_intseg_data_root + "coco_int/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_intseg",
            data_name="img_intseg_coco_box_val",
            data_mode="eval",
            visual_prompt_type="box_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_intseg_postprocess_fn,
                threshold=0.5,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_intseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
        dict(
            type=ImgIntSegDataset,
            source_data_path=img_intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
            data_path=img_intseg_data_root + "coco_int/annotations/img_intseg_coco_val.json",
            image_folder=img_intseg_data_root + "coco_int/coco2017/val2017",
            tokenizer=tokenizer,
            task_name="img_intseg",
            data_name="img_intseg_coco_mask_val",
            data_mode="eval",
            visual_prompt_type="mask_visual_prompt",
            output_ids_with_output=output_ids_with_output,
            cond_type=cond_type,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            image_processor=image_processor,
            postprocess_fn=dict(
                type=process_map_fn_factory,
                fn=img_intseg_postprocess_fn,
                threshold=0.5,
                return_contiguous_labels=True,
            ),
            dataset_map_fn=dict(
                type=dataset_map_fn_factory,
                fn=img_intseg_map_fn,
                cond_type=cond_type,
                image_token=image_token,
            ),
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
            ),
            max_length=max_length,
            expand2square=expand2square,
            use_placeholder=use_placeholder,
            ignore_value=ignore_value,
            background_label=background_label,
        ),
    ],
}

val_evaluators = {
    "genseg": [
        dict(
            type=ImgGenSegEvaluator,
            data_name="img_genseg_coco_panoptic_val",
            distributed=True,
            support_loading=False,
        ),
        dict(
            type=ImgGenSegEvaluator,
            data_name="img_genseg_coco_panoptic_semantic_val",
            distributed=True,
            support_loading=False,
        ),
        dict(
            type=ImgGenSegEvaluator,
            data_name="img_genseg_coco_instance_val",
            distributed=True,
        ),
    ],
    "ovseg": [
        dict(
            type=ImgOVSegEvaluator,
            data_name="img_ovseg_ade20k_panoptic_val",
            distributed=True,
            support_loading=False,
        ),
        dict(
            type=ImgOVSegEvaluator,
            data_name="img_ovseg_ade20k_panoptic_semantic_val",
            distributed=True,
            support_loading=False,
        ),
        dict(
            type=ImgOVSegEvaluator,
            data_name="img_ovseg_ade20k_instance_val",
            distributed=True,
        ),
    ],
    "refseg": [
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcoco_val",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcoco_testA",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcoco_testB",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcocop_val",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcocop_testA",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcocop_testB",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcocog_val",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_refcocog_test",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_grefcoco_val",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_grefcoco_testA",
        ),
        dict(
            type=ImgRefSegEvaluator,
            distributed=True,
            data_name="img_refseg_grefcoco_testB",
        ),
    ],
    "reaseg": [
        dict(
            type=ImgReaSegEvaluator,
            distributed=True,
            data_name="img_reaseg_lisa_val",
        ),
        dict(
            type=ImgReaSegEvaluator,
            distributed=True,
            data_name="img_reaseg_lisa_test_short",
        ),
        dict(
            type=ImgReaSegEvaluator,
            distributed=True,
            data_name="img_reaseg_lisa_test_long",
        ),
        dict(
            type=ImgReaSegEvaluator,
            distributed=True,
            data_name="img_reaseg_lisa_test_all",
        ),
    ],
    "gcgseg": [
        dict(
            type=ImgGCGSegEvaluator,
            distributed=True,
            data_name="img_gcgseg_val",
            evaluation_metrics=["miou", "map", "caption"],
        ),
        dict(
            type=ImgGCGSegEvaluator,
            distributed=True,
            data_name="img_gcgseg_test",
            evaluation_metrics=["miou", "map", "caption"],
        ),
    ],
    "vgdseg": [
        dict(
            type=ImgVGDSegEvaluator,
            distributed=True,
            support_loading=False,
            data_name="img_vgdseg_coco_point_val",
        ),
        dict(
            type=ImgVGDSegEvaluator,
            distributed=True,
            support_loading=False,
            data_name="img_vgdseg_coco_scribble_val",
        ),
        dict(
            type=ImgVGDSegEvaluator,
            distributed=True,
            support_loading=False,
            data_name="img_vgdseg_coco_box_val",
        ),
        dict(
            type=ImgVGDSegEvaluator,
            distributed=True,
            support_loading=False,
            data_name="img_vgdseg_coco_mask_val",
        ),
    ],
    "intseg": [
        dict(
            type=ImgIntSegEvaluator,
            distributed=True,
            data_name="img_intseg_coco_point_val",
        ),
        dict(
            type=ImgIntSegEvaluator,
            distributed=True,
            data_name="img_intseg_coco_scribble_val",
        ),
        dict(
            type=ImgIntSegEvaluator,
            distributed=True,
            data_name="img_intseg_coco_box_val",
        ),
        dict(
            type=ImgIntSegEvaluator,
            distributed=True,
            data_name="img_intseg_coco_mask_val",
        ),
    ],
}

val_datasets = list(chain(*[val_datasets.get(task, []) for task in eval_tasks]))
val_evaluator = (
    list(chain(*[val_evaluators.get(task, []) for task in eval_tasks]))
    if use_eval
    else None
)

val_dataloader = (
    dict(
        batch_size=1,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        dataset=dict(type=ConcatDataset, datasets=val_datasets, max_sample=val_sample),
        sampler=dict(type=DefaultSampler, shuffle=False),
        collate_fn=dict(type=xsam_collate_fn),
    )
    if use_eval
    else None
)

vis_datasets = deepcopy(val_datasets)
for dataset in vis_datasets:
    if dataset["task_name"] in ["img_genseg", "img_ovseg", "img_vgdseg", "img_intseg"]:
        dataset["postprocess_fn"]["threshold"] = 0.5  # type: ignore

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        # Avoid adding tied/shared parameters (e.g., embedding <-> lm_head) multiple times
        # when traversing complex HF modules
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "vision_encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_begin=val_interval, val_interval=val_interval)
val_cfg = (
    dict(type=ValLoop, dataloader=val_dataloader, evaluator=val_evaluator)
    if use_eval
    else None
)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = dict(
    type=Visualizer,
    scale=1.0,
    font_size_scale=1.0,
)

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["vlm", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    *(
        [
            dict(
                type=GenerationChatHook,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                image_processor=image_processor,
                image_token=image_token,
                expand2square=expand2square,
                use_placeholder=use_placeholder,
                image_postprocess_fns=postprocess_fn,
                extra_image_processor=extra_image_processor,
                visualizer=visualizer,
                every_n_iters=inf_interval,
                image_inputs=generation_inputs,
                image_task_names=infer_tasks,
                generation_images=generation_images,
                image_vprompt_masks=vprompt_masks,
                system=SYSTEM,
                prompt_template=prompt_template,
            )
        ]
        if use_infer
        else []
    ),
    dict(type=PTCheckpointHook, clean_pth=False),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=log_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)
