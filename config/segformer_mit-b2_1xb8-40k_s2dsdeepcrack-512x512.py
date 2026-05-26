# segformer_mit-b2_1xb8-40k_s2dsdeepcrack-512x512_v2.py
# Based on your current mixed-dataset config, with requested updates.
# Source configs: s2ds (loss/class_weight, pipeline order) :contentReference[oaicite:4]{index=4}
#                 s2dsdeepcrack (base) :contentReference[oaicite:5]{index=5}

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

# Merged dataset root (contains train_img/train_lab, val_img/val_lab, test_img/test_lab)
data_root = '/root/work/s2ds_deepcrack'

# Important: labels contain ignore_index=255, so pad seg with 255 (not 0)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# SegFormer MiT-B2 pretrained backbone
checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b2_20220624-66e8bf70.pth'
)

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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
decode_head=dict(
    type='SegformerHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=7,
    norm_cfg=norm_cfg,
    align_corners=False,
    ignore_index=255,   # 关键：让 head 知道 255 是忽略像素
    loss_decode=[
        dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[0.1, 1.3132369189474946, 0.21899774795658566, 0.7030975114131134, 1.0821013568259492,
                          0.4572359271226918, 1.9071834366745792],
            loss_weight=1.0),
        dict(
            type='DiceLoss',
            loss_weight=1.0),
    ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# --------------------
# Data / Pipeline
# --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # (3) keep_ratio=False (you observed better results for DeepCrack)
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.75, 1.5),
        keep_ratio=False),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, seg=255)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# (2) Align test_pipeline order with S2DS: load annotations before resize
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='S2DSDeepCrackDataset',
        data_root=data_root,
        data_prefix=dict(img_path='train_img', seg_map_path='train_lab'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='S2DSDeepCrackDataset',
        data_root=data_root,
        data_prefix=dict(img_path='val_img', seg_map_path='val_lab'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='S2DSDeepCrackDataset',
        data_root=data_root,
        data_prefix=dict(img_path='test_img', seg_map_path='test_lab'),
        pipeline=test_pipeline))

# --------------------
# Evaluation
# --------------------
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    beta=1,
    classwise_results=True)
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

# --------------------
# Schedule (iter-based)
# --------------------
max_iters = 40000
interval = 2000

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
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

randomness = dict(seed=304)
