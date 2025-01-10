_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    # '../_base_/datasets/synthtext.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/datasets/icdar2017.py',
    '../_base_/schedules/schedule_sgd_100k.py',
]

# dataset settings
synthtext_textdet_train = _base_.icdar2015_textdet_train
synthtext_textdet_train.pipeline = _base_.train_pipeline
synthtext_textdet_test = _base_.icdar2015_textdet_test
synthtext_textdet_test.pipeline = _base_.test_pipeline

# synthtext_textdet_train = _base_.icdar2013_textdet_train
# synthtext_textdet_train.pipeline = _base_.train_pipeline
# synthtext_textdet_test = _base_.icdar2013_textdet_test
# synthtext_textdet_test.pipeline = _base_.test_pipeline

# synthtext_textdet_train = _base_.totaltext_textdet_train
# synthtext_textdet_train.pipeline = _base_.train_pipeline
# synthtext_textdet_test = _base_.totaltext_textdet_test
# synthtext_textdet_test.pipeline = _base_.test_pipeline

# synthtext_textdet_train = _base_.ctw1500_textdet_train
# synthtext_textdet_train.pipeline = _base_.train_pipeline
# synthtext_textdet_test = _base_.ctw1500_textdet_test
# synthtext_textdet_test.pipeline = _base_.test_pipeline

# synthtext_textdet_train = _base_.icdar2017_textdet_train
# synthtext_textdet_train.pipeline = _base_.train_pipeline
# synthtext_textdet_test = _base_.icdar2017_textdet_test
# synthtext_textdet_test.pipeline = _base_.test_pipeline

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001),
#     loss_scale='dynamic')

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synthtext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=synthtext_textdet_test)

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
)

auto_scale_lr = dict(base_batch_size=16)
