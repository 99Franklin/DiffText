_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_adam_600e.py',
    '../_base_/schedules/schedule_sgd_100k.py',
    '_base_panet_resnet18_fpem-ffm.py',
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
)

# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_test = _base_.icdar2015_textdet_test
# pipeline settings
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test.pipeline = _base_.test_pipeline

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000)

param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, by_epoch=False, end=50000),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=16)

# train_dataloader = dict(
#     batch_size=64,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=icdar2015_textdet_train)
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=icdar2015_textdet_test)
# test_dataloader = val_dataloader
#
# val_evaluator = dict(
#     type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
# test_evaluator = val_evaluator
#
# auto_scale_lr = dict(base_batch_size=64)
