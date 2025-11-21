from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import (
    ConcatDataset,
    GCGSegDataset,
    GenSegDataset,
    ImgConvDataset,
    IntSegDataset,
    OVSegDataset,
    ReaSegDataset,
    RefSegDataset,
    VGDSegDataset,
)
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    gcgseg_map_fn,
    genseg_map_fn,
    imgconv_map_fn,
    intseg_map_fn,
    ovseg_map_fn,
    reaseg_map_fn,
    refseg_map_fn,
    template_map_fn_factory,
    vgdseg_map_fn,
)
from xsam.dataset.process_fns import (
    gcgseg_postprocess_fn,
    genseg_postprocess_fn,
    intseg_postprocess_fn,
    ovseg_postprocess_fn,
    process_map_fn_factory,
    reaseg_postprocess_fn,
    refseg_postprocess_fn,
    vgdseg_postprocess_fn,
)
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.evaluation.evaluators import (
    GCGSegEvaluator,
    GenSegEvaluator,
    IntSegEvaluator,
    OVSegEvaluator,
    ReaSegEvaluator,
    RefSegEvaluator,
    VGDSegEvaluator,
)
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
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
llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip2-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Specify the pretrained pth
# Case1: Comment the following for training from scratch
s1_pretrained_pth = work_dir + "s1_seg_finetune/xsam_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = (
    work_dir
    + "s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain/pytorch_model.bin"
)  # noqa: E501

# Case2: Uncomment the following for evaluating from our pretrained model
# s1_pretrained_pth = None
# s2_pretrained_pth = None

# Prompt
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 2
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = 10

# Evaluate the generation performance during the training
evaluation_freq = 2000
SYSTEM = ""
evaluation_images = [
    code_dir + "xsam/configs/xsam/images/imgconv.jpg",
    code_dir + "xsam/configs/xsam/images/genseg.jpg",
    code_dir + "xsam/configs/xsam/images/refseg.jpg",
    code_dir + "xsam/configs/xsam/images/reaseg.jpg",
    code_dir + "xsam/configs/xsam/images/gcgseg.jpg",
    code_dir + "xsam/configs/xsam/images/intseg.jpg",
    code_dir + "xsam/configs/xsam/images/intseg.jpg",
    code_dir + "xsam/configs/xsam/images/intseg.jpg",
    code_dir + "xsam/configs/xsam/images/intseg.jpg",
    code_dir + "xsam/configs/xsam/images/vgdseg.jpg",
    code_dir + "xsam/configs/xsam/images/vgdseg.jpg",
    code_dir + "xsam/configs/xsam/images/vgdseg.jpg",
    code_dir + "xsam/configs/xsam/images/vgdseg.jpg",
    code_dir + "xsam/configs/xsam/images/vgdseg.jpg",
]
evaluation_inputs = [
    "Can you describe this image in detail? Please elaborate in your response.",
    "Can you generate segmentation masks for this image based on the specified categories: <p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, <p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, <p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, <p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, <p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, <p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, <p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, <p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, <p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, <p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, <p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, <p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, <p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, <p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, <p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, <p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, <p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, <p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, <p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, <p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, <p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, <p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, <p>curtain</p>, <p>door</p>, <p>floor wood</p>, <p>flower</p>, <p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, <p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, <p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, <p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, <p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall brick</p>, <p>wall stone</p>, <p>wall tile</p>, <p>wall wood</p>, <p>water</p>, <p>window blind</p>, <p>window</p>, <p>tree</p>, <p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, <p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, <p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, <p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p>? Please output the segmentation mask.",
    "Can you segment <p>the women with red coat</p> in this image? Please output the corresponding segmentation mask.",
    "<p>when enjoying an ice cream sundae, what can we use to scoop up the whipped cream and place it on top of the ice cream?</p> Please output the corresponding segmentation mask.",
    "Can you provide a brief description of the this image? Respond with interleaved segmentation masks for the corresponding phrases.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>, <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>, <p><region></p>? Please output the segmentation mask.",
]
vprompt_masks = [
    (None,),
    (None,),
    (None,),
    (None,),
    (None,),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/intseg_point0.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/intseg_scribble1.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/intseg_box0.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/intseg_mask1.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_point0.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_scribble1.png",),
    (code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_box0.png",),
    (
        code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_point0.png",
        code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_scribble1.png",
    ),
    (
        code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_box0.png",
        code_dir + "xsam/configs/xsam/images/vprompt_masks/vgdseg_point1.png",
    ),
]

#######################################################################
#            PART 2  Model & Tokenizer &  Processor              #
#######################################################################
# TODO: add special tokens via import from xsam.utils
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"  # "phrase" "cls" "all"
ignore_label = 255
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
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
    freeze_llm=False,
    freeze_visual_encoder=False,
    freeze_segmentor_encoder=False,
    use_dual_encoder=True,
    use_vision_sampler=True,
    connector_type="conv",
    cond_type=cond_type,
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=genseg_postprocess_fn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
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
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=True,
        open_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
imgconv_data_root = data_dir + "imgconv_data/"
genseg_data_root = data_dir + "genseg_data/"
ovseg_data_root = data_dir + "ovseg_data/"
refseg_data_root = data_dir + "refseg_data/"
reaseg_data_root = data_dir + "reaseg_data/"
gcgseg_data_root = data_dir + "gcgseg_data/"
intseg_data_root = data_dir + "intseg_data/"
vgdseg_data_root = data_dir + "vgdseg_data/"

llava_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=imgconv_data_root + "llava/llava_images",
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    task_name="imgconv",
    data_name="llava_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=imgconv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=False,
)

coco_genseg_dataset = dict(
    type=GenSegDataset,
    data_path=genseg_data_root + "coco2017/annotations/panoptic_train2017.json",
    image_folder=genseg_data_root + "coco2017/train2017",
    panseg_map_folder=genseg_data_root + "coco2017/panoptic_train2017",
    tokenizer=tokenizer,
    task_name="genseg",
    data_name="coco_panoptic_genseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=genseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    use_variant_cat=True,
    pad_image_to_square=False,
)

refcoco_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcoco",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcoco_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

refcocop_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcoco+",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcoco+_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

refcocog_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcocog",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcocog_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

lisa_reaseg_dataset = dict(
    type=ReaSegDataset,
    data_root=reaseg_data_root + "lisa",
    image_folder=reaseg_data_root + "lisa/train",
    explain_path=reaseg_data_root + "lisa/explanatory/train.json",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="lisa_reaseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reaseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reaseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

grandf_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/GranDf_HA_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="grandf_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

refcocog_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/RefCOCOg_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/coco2014/train2014",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="refcocog_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

psg_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/OpenPsgGCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/coco2017",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="psg_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

flickr_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/flickr_mergedGT_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/flickr30k/images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="flickr_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

coco_vgdseg_dataset = dict(
    type=VGDSegDataset,
    source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_train2017.json",
    data_path=vgdseg_data_root + "coco_vgd/annotations/coco_vgdseg_train.json",
    image_folder=vgdseg_data_root + "coco_vgd/coco2017/train2017",
    tokenizer=tokenizer,
    data_mode="train",
    task_name="vgdseg",
    data_name="coco_vgdseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=vgdseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_negative_sample=True,
    num_sample=5,
    max_length=max_length,
    pad_image_to_square=False,
)

train_datasets = dict(
    type=ConcatDataset,
    oversample_ratio=0.1,
    datasets=[
        llava_imgconv_dataset,
        coco_genseg_dataset,
        refcoco_refseg_dataset,
        refcocop_refseg_dataset,
        refcocog_refseg_dataset,
        lisa_reaseg_dataset,
        grandf_gcgseg_dataset,
        refcocog_gcgseg_dataset,
        psg_gcgseg_dataset,
        flickr_gcgseg_dataset,
        coco_vgdseg_dataset,
    ],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_datasets,
    persistent_workers=True,
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

# False for predict mode, True for tensor mode
output_ids_with_output = True
val_datasets = [
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        panseg_map_folder=genseg_data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=genseg_data_root + "coco2017/panoptic_semseg_val2017",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="coco_panoptic_genseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        output_ids_with_output=output_ids_with_output,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="panoptic_genseg",
            threshold=0.0,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        panseg_map_folder=genseg_data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=genseg_data_root + "coco2017/panoptic_semseg_val2017",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="coco_panoptic_semantic_genseg",  # semantic genseg shared with panoptic annotation
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="semantic_genseg",
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        task_name="genseg",
        data_name="coco_instance_genseg",
        data_mode="eval",
        tokenizer=tokenizer,
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="instance_genseg",
            threshold=0.0,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        panseg_map_folder=ovseg_data_root + "ade20k/ade20k_panoptic_val",
        semseg_map_folder=ovseg_data_root + "ade20k/annotations_detectron2/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_panoptic_ovseg",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory, fn=ovseg_postprocess_fn, threshold=0.0, task_name="panoptic_ovseg"
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        panseg_map_folder=ovseg_data_root + "ade20k/ade20k_panoptic_val",
        semseg_map_folder=ovseg_data_root + "ade20k/annotations_detectron2/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_panoptic_semantic_ovseg",  # semantic ovseg shared with panoptic annotation
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(type=process_map_fn_factory, fn=ovseg_postprocess_fn, task_name="semantic_ovseg"),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_instance_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_instance_ovseg",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory, fn=ovseg_postprocess_fn, task_name="instance_ovseg", threshold=0.0
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=None,
        label_file=ovseg_data_root + "pascal_ctx/labels/pascal_ctx59.txt",
        image_folder=ovseg_data_root + "pascal_ctx/images/validation",
        semseg_map_folder=ovseg_data_root + "pascal_ctx/annotations_ctx59/validation",
        label_shift=1,
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="pc59_semantic_ovseg",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(type=process_map_fn_factory, fn=ovseg_postprocess_fn, task_name="semantic_ovseg"),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="testA",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_testA_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="testB",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_testB_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="testA",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_testA_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="testB",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_testB_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcocog",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcocog_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcocog",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcocog_test_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/val",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="val_reaseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_all_reaseg",
        query_type="all",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_sentence_reaseg",
        query_type="sentence",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_phrase_reaseg",
        query_type="phrase",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=GCGSegDataset,
        data_path=gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_mask_gt.json",
        cap_data_path=gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_caption_gt.json",
        data_root=gcgseg_data_root,
        image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="gcgseg",
        data_name="val_gcgseg",
        output_ids_with_output=False,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=gcgseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=gcgseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=GCGSegDataset,
        data_path=gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_mask_gt.json",
        cap_data_path=gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_caption_gt.json",
        data_root=gcgseg_data_root,
        image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="gcgseg",
        data_name="test_gcgseg",
        output_ids_with_output=False,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=gcgseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=gcgseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="point_vgdseg",
        data_mode="eval",
        visual_prompt_type="point_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_sample=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="scribble_vgdseg",
        data_mode="eval",
        visual_prompt_type="scribble_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_sample=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="box_vgdseg",
        data_mode="eval",
        visual_prompt_type="box_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_sample=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="mask_vgdseg",
        data_mode="eval",
        visual_prompt_type="mask_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_sample=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="point_intseg",
        data_mode="eval",
        visual_prompt_type="point_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="scribble_intseg",
        data_mode="eval",
        visual_prompt_type="scribble_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="box_intseg",
        data_mode="eval",
        visual_prompt_type="box_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="mask_intseg",
        data_mode="eval",
        visual_prompt_type="mask_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
]

val_evaluators = [
    dict(
        type=GenSegEvaluator,
        distributed=True,
        data_name="coco_panoptic_genseg",
    ),
    dict(
        type=GenSegEvaluator,
        data_name="coco_panoptic_semantic_genseg",
        distributed=True,
    ),
    dict(
        type=GenSegEvaluator,
        data_name="coco_instance_genseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_panoptic_ovseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_panoptic_semantic_ovseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_instance_ovseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="pc59_semantic_ovseg",
        distributed=True,
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_testA_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_testB_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_testA_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_testB_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcocog_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcocog_test_refseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="val_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_all_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_sentence_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_phrase_reaseg",
    ),
    dict(
        type=GCGSegEvaluator,
        distributed=True,
        data_name="val_gcgseg",
    ),
    dict(
        type=GCGSegEvaluator,
        distributed=True,
        data_name="test_gcgseg",
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="point_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="scribble_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="box_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="mask_vgdseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="point_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="scribble_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="box_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="mask_intseg",
        distributed=True,
    ),
]

vis_datasets = val_datasets

vis_datasets = deepcopy(val_datasets)
for dataset in vis_datasets:
    if dataset["task_name"] in ["genseg", "ovseg", "vgdseg", "intseg"]:
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
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
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
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

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
        module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        image_processor=image_processor,
        postprocess_fns=[
            None,
            genseg_postprocess_fn,
            refseg_postprocess_fn,
            reaseg_postprocess_fn,
            gcgseg_postprocess_fn,
            intseg_postprocess_fn,
            intseg_postprocess_fn,
            intseg_postprocess_fn,
            intseg_postprocess_fn,
            vgdseg_postprocess_fn,
            vgdseg_postprocess_fn,
            vgdseg_postprocess_fn,
            vgdseg_postprocess_fn,
            vgdseg_postprocess_fn,
        ],
        extra_image_processor=extra_image_processor,
        visualizer=visualizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        vprompt_masks=vprompt_masks,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
    dict(type=PTCheckpointHook, clean_pth=False),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
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
