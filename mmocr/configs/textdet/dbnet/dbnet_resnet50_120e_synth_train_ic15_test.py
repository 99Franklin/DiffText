_base_ = [
    'dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = None

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

_base_.optim_wrapper.optimizer.lr = 0.002
_base_.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=20)
param_scheduler = [
    dict(type='LinearLR', end=10, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=10, end=120),
]
# param_scheduler = [
#     dict(type='LinearLR', end=100, start_factor=0.001),
#     dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
# ]
