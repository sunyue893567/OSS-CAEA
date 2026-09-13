# Load a Cityscapes/GTA OSS-CAEA checkpoint; vocabulary buffers are recomputed.
_base_ = ['../_base_/ov_datasets/dg_citys2deliver_512x512.py',
          '../_base_/default_runtime.py', '../_base_/models/oss_caea.py']
model = dict(backbone=dict(class_json='open_vocab/deliver.json'),
             decode_head=dict(num_classes=25, loss_cls=dict(class_weight=[1.]*25+[.1])))
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
