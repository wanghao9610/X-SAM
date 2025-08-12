from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from torch.optim import AdamW
from xtuner.dataset.samplers import LengthGroupedSampler

from xsam.dataset import GenericSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import generic_seg_postprocess_fn, process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners import TrainLoop
from xsam.evaluation.evaluators import GenericSegEvaluator
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
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
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Data
data_root = data_dir + "gen_seg_data/"
data_path = data_root + "coco2017/annotations/panoptic_train2017.json"
image_folder = data_root + "coco2017/train2017"
panseg_map_folder = data_root + "coco2017/panoptic_train2017"

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 1
dataloader_num_workers = 8
max_epochs = 36
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 0.01  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = 10

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# TODO: add special tokens via import from xsam.utils

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_segmentor_encoder=False,
    use_activation_checkpointing=False,
    postprocess_fn=generic_seg_postprocess_fn,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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
        close_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
        "do_crop": True,
        "crop_size": {"height": 1024, "width": 1024},
    }
)

pannoptic_genseg_dataset = dict(
    type=GenericSegDataset,
    data_path=data_path,
    image_folder=image_folder,
    panseg_map_folder=panseg_map_folder,
    extra_image_processor=train_extra_image_processor,
    task_name="genseg",
    data_name="panoptic_genseg",
    pad_image_to_square=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=pannoptic_genseg_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
        mega_batch_mult=1,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

val_datasets = [
    dict(
        type=GenericSegDataset,
        data_path=data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        panseg_map_folder=data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=data_root + "coco2017/panoptic_semseg_val2017",
        task_name="genseg",
        data_name="panoptic_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=generic_seg_postprocess_fn, task_name="panoptic_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=GenericSegDataset,
        data_path=data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        panseg_map_folder=data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=data_root + "coco2017/panoptic_semseg_val2017",
        task_name="genseg",
        data_name="panoptic_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=generic_seg_postprocess_fn, task_name="semantic_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=GenericSegDataset,
        data_path=data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        task_name="genseg",
        data_name="instance_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=generic_seg_postprocess_fn, task_name="instance_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
]

val_evaluators = [
    dict(
        type=GenericSegEvaluator,
        data_name="panoptic_genseg",
        distributed=True,
    ),
    dict(
        type=GenericSegEvaluator,
        data_name="semantic_genseg",
        distributed=True,
    ),
    dict(
        type=GenericSegEvaluator,
        data_name="instance_genseg",
        distributed=True,
    ),
]

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
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
        type=MultiStepLR,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 30],
        gamma=0.1,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = None

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "connector", "segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
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

find_unused_parameters = True
