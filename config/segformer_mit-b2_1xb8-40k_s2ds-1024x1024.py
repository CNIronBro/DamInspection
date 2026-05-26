norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (1024, 1024)
data_root = '/root/Dataset/s2ds'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0)

# SegFormer MiT-B2: load pretrained weights from Crack500 fine-tuning
checkpoint = '/hy-tmp/log/segformer/crack500/20260120_100821/iter_38000.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        # Load from Crack500 fine-tuned checkpoint (transfer learning)
        # Use prefix='backbone.' to extract only backbone weights from full model checkpoint
        # This skips decode_head (2 classes) and lets it initialize randomly for new task (7 classes)
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,  # S2DS: background/crack/spalling/corrosion/efflorescence/vegetation/control_point
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                # Class weights: follow s2ds config to handle class imbalance
                # Order: background / crack / spalling / corrosion / efflorescence / vegetation / control_point
                class_weight=[0.2, 5.0, 1.0, 1.0, 1.5, 1.0, 1.5],
                loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# --------------------
# Data / Pipeline (S2DS: RGBA color map -> label)
# --------------------
_s2ds_color_map = {
    (0, 0, 0): 0,           # background
    (255, 255, 255): 1,     # crack
    (255, 0, 0): 2,         # spalling
    (255, 255, 0): 3,       # corrosion
    (0, 255, 255): 4,       # efflorescence
    (0, 255, 0): 5,         # vegetation
    (0, 0, 255): 6,         # control_point
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # Convert RGBA color mask to discrete class IDs (ignore alpha channel)
    dict(type='ColorMapToLabel', color_map=_s2ds_color_map),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.75, 1.5),
        keep_ratio=True),
    # Prioritize cropping patches containing crack(=1) to increase crack samples in batch
    dict(
        type='ClassBalancedRandomCrop',
        crop_size=crop_size,
        class_ids=(1,),
        num_retry=10,
        cat_max_ratio=0.75,
        ignore_index=255),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ColorMapToLabel', color_map=_s2ds_color_map),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,  # Reduced from 8 to 4 due to GPU memory constraints with SegFormer-B2 @ 1024x1024
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='S2DSDataset',
        data_root=data_root,
        data_prefix=dict(img_path='train_img', seg_map_path='train_lab'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='S2DSDataset',
        data_root=data_root,
        data_prefix=dict(img_path='val_img', seg_map_path='val_lab'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='S2DSDataset',
        data_root=data_root,
        data_prefix=dict(img_path='test_img', seg_map_path='test_lab'),
        pipeline=test_pipeline))

# --------------------
# Evaluation
# --------------------
val_evaluator = [
    dict(
        type='IoUMetric',
        iou_metrics=['mIoU', 'mDice', 'mFscore'],
        beta=1,
        classwise_results=True),
]
test_evaluator = val_evaluator

# --------------------
# Runtime
# --------------------
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = None
tta_model = dict(type='SegTTAModel')

# --------------------
# Schedule (iter-based)
# --------------------
max_iters = 40000
interval = 2000

# Optimizer: AdamW (follow SegFormer standard config)
# For transfer learning, use slightly lower lr than standard training
optimizer = dict(
    type='AdamW',
    lr=0.00003,  # Lower lr for transfer learning (half of standard 0.00006)
    betas=(0.9, 0.999),
    weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),  # Head uses 10x lr for faster adaptation
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1000,
        end=max_iters,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=interval,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=3407)

