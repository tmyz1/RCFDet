_base_ = '../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py'

custom_imports = dict(
    imports=['mmdet.datasets.m3fd_fusion'],
    allow_failed_imports=False)

dataset_type = 'M3FDFusionDataset'
data_root = r'/root/autodl-tmp/M3FD/cocodataset'
fusion_img_prefix = r'/root/autodl-tmp/M3FD_ffusion/cocodataset/images'
swin_pretrained = r'/root/autodl-tmp/weight/swin_tiny_224.pth'

# AutoDL example:
# data_root = r'/root/autodl-tmp/M3FD/cocodataset'
# fusion_img_prefix = (
#     r'/root/autodl-tmp/FSATFusion-main/FSATFusion-main'
#     r'/fusion results/M3FD_FFusion')
# swin_pretrained = r'/root/autodl-tmp/weight/swin_tiny_224.pth'

num_classes = 6
max_epochs = 150
backend_args = None

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        _delete_=True,
        type='swintransformer',
        pretrain_img_size=384,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        pretrained=swin_pretrained),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img=fusion_img_prefix),
        fusion_suffix='_FSATFusion',
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img=fusion_img_prefix),
        fusion_suffix='_FSATFusion',
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/test.json',
    metric='bbox',
    format_only=False,
    outfile_prefix=r'/root/autodl-tmp/Swin-Deformable-output/ffusion/test',
    backend_args=backend_args)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=1.0,
        begin=0,
        end=20,
        by_epoch=True),
    dict(
        type='ReduceOnPlateauLR',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=6,
        factor=0.1,
        threshold=1e-3,
        threshold_rule='rel',
        min_value=1e-6,
        eps=1e-8,
        begin=20,
        by_epoch=True,
        verbose=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50))

randomness = dict(seed=66, deterministic=False, diff_rank_seed=True)

load_from = r'/root/autodl-tmp/weight/dino.pth'
resume = False
work_dir = r'/root/autodl-tmp/Swin-Deformable-output/ffusion'
auto_scale_lr = dict(base_batch_size=4, enable=True)
