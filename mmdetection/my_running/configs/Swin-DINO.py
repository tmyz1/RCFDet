from mmdetection.mmdet.utils.collate_rgbt import collate_rgbt
auto_scale_lr = dict(base_batch_size=4, enable=True)#自动学习率缩放 配置
backend_args = None
randomness = dict(
    seed=66,
    deterministic=False,
    diff_rank_seed=True
)
#需要应用的自定义类
#数据集根目录
#data_root = r'/root/autodl-tmp/M3FD/cocodataset'
data_root = r"D:\data\M3FD\cocodataset"
dataset_type = 'M3FDDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        type='CheckpointHook',
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
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
#load_from = r'/root/autodl-tmp/weight/dino.pth'
#load_from = r'/root/autodl-tmp/Swin-Deformable-output/train/epoch_42.pth'
load_from = r"D:\pycharm work\weight\dino.pth"
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
#整体模型
model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    #图像，数据标准化模块
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
    #特征提取模块
    backbone=dict(
        type='SwinNet',
        in_channels=3,
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        #pretrained=r'/root/autodl-tmp/weight/swin_tiny_224.pth',
        pretrained=r"D:\pycharm work\weight\swin_tiny_224.pth"
    ),
    #Swin-Transformer转DINO对齐模块
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768
        ],
        kernel_size=1,
        out_channels=256,
        reduction=8,
        type='swin_deformable_neck'),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=6,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    #噪声模块
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR
#训练的轮数
max_epochs=150
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=1.0,
        begin=0,
        end=20,
        by_epoch=True
    ),
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
        verbose=True
    )
]

resume = False#是否从断点恢复训练
test_cfg = dict(type='TestLoop')
#构建test_dataloader
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=r'annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/infrared'),
        #data_root=r'/root/autodl-tmp/M3FD/cocodataset',
        data_root =r"D:\data\M3FD\cocodataset",
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='ResizeRGBT', scale=(384,384), keep_ratio=False),
            dict(type='PackDetRGBTInputs')
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=2,
    collate_fn=collate_rgbt,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    #ann_file=r'/root/autodl-tmp/M3FD/cocodataset/annotations/test.json',
    ann_file = r"D:\data\M3FD\cocodataset\annotations\test.json",
    backend_args=None,
    format_only=False,# 仅保存结果，不计算指标（若需同时计算指标，设为False）
    metric='bbox',
    type='CocoMetric',
    #outfile_prefix=r'/root/autodl-tmp/Swin-Deformable-output/test',
    outfile_prefix = r"D:\pycharm work\Swin-Deformable output",
)
test_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeRGBT', scale=(384,384), keep_ratio=False),
    dict(type='PackDetRGBTInputs')
]
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        #data_root=r'/root/autodl-tmp/M3FD/cocodataset',
        data_root =r"D:\data\M3FD\cocodataset",
        ann_file=r'annotations/train.json',
        data_prefix=dict(img='images/infrared'),
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='ResizeRGBT', scale=(384,384), keep_ratio=False),
            dict(type='RandomFlipRGBT', prob=0.5),#水平翻转
            dict(type='RandomVerticalFlipRGBT', prob=0.5),#垂直翻转
            # dict(type='RandomBrightnessContrastRGBT',prob=0.3),#光照增强
            # dict(type='RandomBlurRGBT',prob=0.3),#随机模糊
            dict(type='RandomCutoutRGBT',prob=0.3),#随机遮盖
            dict(type='PackDetRGBTInputs')
        ],
    ),
    drop_last=False,
    num_workers=2,
    collate_fn=collate_rgbt,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)
train_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeRGBT', scale=(384,384), keep_ratio=False),
    dict(type='RandomFlipRGBT', prob=0.5),
    dict(type='RandomVerticalFlipRGBT', prob=0.5),#垂直翻转
    dict(type='RandomCutoutRGBT',prob=0.3),#随机遮盖
    dict(type='PackDetRGBTInputs')
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=r'annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/infrared'),
        #data_root=r'/root/autodl-tmp/M3FD/cocodataset',
        data_root =r"D:\data\M3FD\cocodataset",
        pipeline=[
            dict(type='LoadRGBTImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='ResizeRGBT', scale=(384,384), keep_ratio=False),
            dict(type='PackDetRGBTInputs')
        ],
        test_mode=True,
        type='M3FDDataset'),
    drop_last=False,
    num_workers=2,
    collate_fn=collate_rgbt,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
val_evaluator = dict(
    #ann_file=r'/root/autodl-tmp/M3FD/cocodataset/annotations/test.json',
    ann_file = r"D:\data\M3FD\cocodataset\annotations\test.json",
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
#work_dir = r'/root/autodl-tmp/Swin-Deformable-output/train'
work_dirs = "D:\pycharm work\Swin-Deformable output"