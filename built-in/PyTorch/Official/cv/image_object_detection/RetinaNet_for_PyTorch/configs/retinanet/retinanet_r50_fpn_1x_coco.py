_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='NpuFusedSGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
log_config = dict( # config to register logger hook
    interval=50, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook')
    ])

dist_params = dict(backend='hccl')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8
)