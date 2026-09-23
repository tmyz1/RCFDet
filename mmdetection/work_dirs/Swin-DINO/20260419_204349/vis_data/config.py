auto_scale_lr = dict(base_batch_size=4, enable=True)
backend_args = None
data_root = 'E:\\my_data\\M3FD\\cocodataset'
dataset_type = 'M3FDDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'E:\\reproduce\\weights\\DINO\\dino.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 150
model = dict(
    as_two_stage=True,
    backbone=dict(
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        embed_dims=96,
        in_channels=3,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        pretrained='E:\\reproduce\\weights\\SwinTransformer\\swin_big_224.pth',
        qkv_bias=True,
        type='SwinNet',
        window_size=7),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=6,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='RGBTDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        kernel_size=1,
        out_channels=256,
        reduction=8,
        type='swin_deformable_neck'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        end_factor=1.0,
        start_factor=1.0,
        type='LinearLR'),
    dict(
        begin=20,
        by_epoch=True,
        eps=1e-08,
        factor=0.1,
        min_value=1e-06,
        monitor='coco/bbox_mAP',
        patience=6,
        rule='greater',
        threshold=0.001,
        threshold_rule='rel',
        type='ReduceOnPlateauLR',
        verbose=True),
]
randomness = dict(deterministic=False, diff_rank_seed=True, seed=66)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    collate_fn='mmdetection.mmdet.utils.collate_rgbt.collate_rgbt',
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/infrared'),
        data_root='E:\\my_data\\M3FD\\cocodataset',
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                384,
                384,
            ), type='ResizeRGBT'),
            dict(type='PackDetRGBTInputs'),
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/root/autodl-tmp/M3FD/cocodataset/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    outfile_prefix='E:\\pycharm\\project\\work_dirs\\model\\3.24',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        384,
        384,
    ), type='ResizeRGBT'),
    dict(type='PackDetRGBTInputs'),
]
train_cfg = dict(max_epochs=150, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    collate_fn='mmdetection.mmdet.utils.collate_rgbt.collate_rgbt',
    dataset=dict(
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/infrared'),
        data_root='E:\\my_data\\M3FD\\cocodataset',
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                384,
                384,
            ), type='ResizeRGBT'),
            dict(prob=0.5, type='RandomFlipRGBT'),
            dict(prob=0.5, type='RandomVerticalFlipRGBT'),
            dict(prob=0.3, type='RandomCutoutRGBT'),
            dict(type='PackDetRGBTInputs'),
        ],
        type='M3FDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        384,
        384,
    ), type='ResizeRGBT'),
    dict(prob=0.5, type='RandomFlipRGBT'),
    dict(prob=0.5, type='RandomVerticalFlipRGBT'),
    dict(prob=0.3, type='RandomCutoutRGBT'),
    dict(type='PackDetRGBTInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    collate_fn='mmdetection.mmdet.utils.collate_rgbt.collate_rgbt',
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/infrared'),
        data_root='E:\\my_data\\M3FD\\cocodataset',
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                384,
                384,
            ), type='ResizeRGBT'),
            dict(type='PackDetRGBTInputs'),
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='E:\\my_data\\M3FD\\cocodataset\\annotations\\test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\Swin-DINO'
work_dirs = 'E:\\pycharm\\project\\work_dirs\\model\\3.24'
