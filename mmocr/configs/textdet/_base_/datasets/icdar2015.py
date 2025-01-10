icdar2015_textdet_data_root = 'data/icdar2015'

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root=icdar2015_textdet_data_root,
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/my_synth_data",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_synth_data_with_large_num_with_rec",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_with_large_num_with_v2_rec",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/10k_filter_synthtext",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/10k_synthtext",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/VISD/10K",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_text_glyph_90_conf_35_angle",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_text_vae_text_glyph_90_conf_35_angle",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# chosen one
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_16_repeat_35_angle_90_conf",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_dual_text_90_conf_35_angle",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_only_text_vae_90_conf_35_angle",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_no_rec_35_angle",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_no_crop",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_no_crop_no_rec",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_with_noise_gaussian_1",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_with_noise_gaussian_2",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_with_noise_salt_and_pepper_0.05",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/SD_with_noise_mix",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/10k_unrealtext",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/5k_unrealtext",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/0812_SD_with_gaussian_noise",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/10k_curve_synthtext",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/10k_synthtext3d",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/1006_14k",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root="data/1006_14k_curve",
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)

icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root="data/1006_10k_curve",
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
