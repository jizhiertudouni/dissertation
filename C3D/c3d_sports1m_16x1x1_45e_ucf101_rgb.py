model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=2,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
dataset_type = 'VideoDataset'
data_root = '/nobackup/parkinson/crop/'
data_root_val = '/nobackup/parkinson/crop/'
split = 1
ann_file_train = '/nobackup/parkinson/crop/train_set.txt'
ann_file_val = '/nobackup/parkinson/crop/val_set.txt'
ann_file_test = '/nobackup/parkinson/crop/test_set.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=1),
    dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=30)]),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=30,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='/nobackup/parkinson/crop/train_set.txt',
        data_prefix='/nobackup/parkinson/crop/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='RandomCrop', size=112),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='VideoDataset',
        ann_file='/nobackup/parkinson/crop/val_set.txt',
        data_prefix='/nobackup/parkinson/crop/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='VideoDataset',
        ann_file='/nobackup/parkinson/crop/test_set.txt',
        data_prefix='/nobackup/parkinson/crop/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
optimizer = dict(
    type='AdamW',
    lr=0.00023,
    betas=(0.9, 0.999),
    weight_decay=0.08,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5)
total_epochs = 30
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/nobackup/parkinson/result731/C3d'
load_from = '/nobackup/parkinson/checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth'
resume_from = None
workflow = [('train', 1)]
seed = 0
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
