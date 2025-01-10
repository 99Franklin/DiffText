_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    # '../_base_/datasets/icdar2013.py',
    # '../_base_/datasets/icdar2015.py',
    # '../_base_/datasets/totaltext.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# TODO: Replace the link
# load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnet/tmp_1.0_pretrain/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-ed322016.pth'  # noqa
# load_from = None  # noqa
# load_from = "output/new_10k_synthtext/epoch_1.pth"  # noqa
# load_from = "output/new_10k_VISD/epoch_1.pth"
load_from = "output/new_SD_base/epoch_1.pth"
# load_from = "output/new_5k_unreal_text/epoch_1.pth"
# load_from = "output/new_10k_synthtext3d_repeat_1/iter_100000.pth"
# load_from = "output/new_10k_curve_synthtext/epoch_1.pth"

# load_from = "output/new_SD_dual_text/epoch_1.pth"
# load_from = "output/new_SD_text_vae_text_glyph/epoch_1.pth"

# dataset settings
# icdar2015_textdet_train = _base_.icdar2015_textdet_train
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test = _base_.icdar2015_textdet_test
# icdar2015_textdet_test.pipeline = _base_.test_pipeline
icdar2015_textdet_train = _base_.ctw1500_textdet_train
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test = _base_.ctw1500_textdet_test
icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
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

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))

# optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3.3e-5, betas=(0.9, 0.999), eps=1e-8,
#                                                          weight_decay=0.0001))

auto_scale_lr = dict(base_batch_size=16)
