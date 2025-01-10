_base_ = [
    '_base_fcenet_resnet50_fpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_sgd_base.py',
    '../_base_/schedules/schedule_sgd_100k.py',
]

optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
# optim_wrapper = dict(optimizer=dict(lr=5e-4, weight_decay=5e-4))
# optim_wrapper = dict(optimizer=dict(lr=2.5e-4, weight_decay=5e-4))
# train_cfg = dict(max_epochs=1500)
# learning policy
# param_scheduler = [
#     dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
# ]

# dataset settings
# icdar2015_textdet_train = _base_.icdar2015_textdet_train
# icdar2015_textdet_test = _base_.icdar2015_textdet_test
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test.pipeline = _base_.test_pipeline

# icdar2015_textdet_train = _base_.icdar2013_textdet_train
# icdar2015_textdet_test = _base_.icdar2013_textdet_test
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test.pipeline = _base_.test_pipeline

icdar2015_textdet_train = _base_.totaltext_textdet_train
icdar2015_textdet_test = _base_.totaltext_textdet_test
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test.pipeline = _base_.test_pipeline

# icdar2015_textdet_train = _base_.ctw1500_textdet_train
# icdar2015_textdet_test = _base_.ctw1500_textdet_test
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
)
