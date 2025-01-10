icdar2017_textdet_data_root = 'data/mlt2017'

icdar2017_textdet_train = dict(
    type='OCRDataset',
    data_root=icdar2017_textdet_data_root,
    ann_file='textdet_test.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2017_textdet_test = dict(
    type='OCRDataset',
    data_root=icdar2017_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
