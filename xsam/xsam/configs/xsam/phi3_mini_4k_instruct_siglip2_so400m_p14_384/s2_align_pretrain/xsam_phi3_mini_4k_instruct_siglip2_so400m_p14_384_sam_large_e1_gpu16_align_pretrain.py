from os import getenv

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import ImageConvDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import image_conv_map_fn, template_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners import TrainLoop
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.sam import SamModel

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

# Specify the pretrained pth
s1_pretrained_pth = work_dir + "s1_seg_finetune/xsam_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin"

# Data
data_root = data_dir + "img_conv_data/"
data_path = data_root + "LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
image_folder = data_root + "LLaVA-Pretrain/images"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 4
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW
lr = 1e-3
betas = (0.9, 0.999)
weight_decay = 0
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
evaluation_images = code_dir + "xsam/configs/xsam/images/llava_imgconv.jpg"
evaluation_inputs = ["Can you describe this image in detail? Please elaborate in your response."]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
ignore_label = 255
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
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
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=True,
    use_dual_encoder=True,
    s1_pretrained_pth=s1_pretrained_pth,
    tokenizer=tokenizer,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,  # from transformers
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
        ),
        torch_dtype=torch.bfloat16,
        drop_decoder=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_imgconv_dataset = dict(
    type=ImageConvDataset,
    data_path=data_path,
    tokenizer=tokenizer,
    image_folder=image_folder,
    task_name="imgconv",
    data_name="llava_imgconv",
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=image_conv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=False,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_imgconv_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=xsam_collate_fn),
)

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
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "segmentor.encoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
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

# set visualizer
visualizer = None

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
