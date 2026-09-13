_base_ = ['../_base_/datasets/dg_citys2acdc_512x512.py', '../_base_/default_runtime.py', '../_base_/models/oss_caea.py']
# 80k iterations; two per GPU gives total batch 8 on four GPUs.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(
    type='OptimWrapper', constructor='PEFTOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=.05),
    paramwise_cfg=dict(norm_decay_mult=0., custom_keys={
        'query_embed': dict(decay_mult=0.), 'query_feat': dict(decay_mult=0.),
        'level_embed': dict(decay_mult=0.)}), clip_grad=dict(max_norm=.1, norm_type=2))
param_scheduler = [dict(type='MultiStepLR', begin=0, end=80000, by_epoch=False,
                        milestones=[64000,68000], gamma=.1)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='SegVisualizationHook'))
