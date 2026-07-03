from copy import deepcopy
from os import getenv

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from xsam.dataset import ConcatDataset, ImgGenSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import img_genseg_postprocess_fn, process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import LengthGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop, ValLoop
from xsam.evaluation.evaluators import ImgGenSegEvaluator
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
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Data
img_genseg_data_root = data_dir + "img_genseg/"

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
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

# Evaluate
val_interval = 2000

# Logging
log_interval = 10

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
use_binary_cls = False

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_mask_encoder=False,
    use_activation_checkpointing=False,
    postprocess_fn=img_genseg_postprocess_fn,
    connector_type="conv",
    extra_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                attn_implementation="compiled",
                head_cls_type="linear",
                num_labels=133,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        use_prompt_encoder=False,
        use_memory=False,
        init_decoder=True,
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

coco_panoptic_genseg_dataset = dict(
    type=ImgGenSegDataset,
    data_path=img_genseg_data_root + "coco2017/annotations/panoptic_train2017.json",
    image_folder=img_genseg_data_root + "coco2017/train2017",
    pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_train2017",
    extra_image_processor=train_extra_image_processor,
    task_name="img_genseg",
    data_name="img_genseg_coco_panoptic_train",
    expand2square=True,
    use_binary_cls=use_binary_cls,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=coco_panoptic_genseg_dataset,
    persistent_workers=False,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

val_datasets = [
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
        sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
        task_name="img_genseg",
        data_name="img_genseg_coco_panoptic_val",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="panoptic_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
        sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
        task_name="img_genseg",
        data_name="img_genseg_coco_panoptic_semantic_val",  # semantic genseg shared with panoptic annotation
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="semantic_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        task_name="img_genseg",
        data_name="img_genseg_coco_instance_val",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="instance_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=dict(type=ConcatDataset, datasets=val_datasets),
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=xsam_collate_fn),
)

val_evaluators = [
    dict(
        type=ImgGenSegEvaluator,
        data_name="img_genseg_coco_panoptic_val",
        distributed=True,
    ),
    dict(
        type=ImgGenSegEvaluator,
        data_name="img_genseg_coco_panoptic_semantic_val",  # semantic genseg shared with panoptic annotation
        distributed=True,
    ),
    dict(
        type=ImgGenSegEvaluator,
        data_name="img_genseg_coco_instance_val",
        distributed=True,
    ),
]

vis_datasets = val_datasets

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
    constructor="CustomOptimWrapperConstructor",
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
    paramwise_cfg=dict(
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

# learning policy
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
val_cfg = dict(type=ValLoop, dataloader=val_dataloader, evaluator=val_evaluator)

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
        module_names=["segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
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

find_unused_parameters = True
