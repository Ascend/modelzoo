from yacs.config import CfgNode as CN

cfg = CN()

cfg.mode = 'train'

# ---------------------------------------------------------------------------- #
# Model (common)
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.name = 'EDVR'
cfg.model.scale = 4
cfg.model.num_frames = 5
# Options for the input dimension
# 4: 4D tensor, with shape [b*frames, h, w, c], used when model frozen
# 5: 5D tensor, with shape [b, frames, h, w, c]
cfg.model.input_format_dimension = 5
cfg.model.convert_output_to_uint8 = False

# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #
cfg.data = CN()
cfg.data.data_dir = 'data/reds'
cfg.data.train_set = 'train.json'
cfg.data.eval_set = 'val.json'
cfg.data.train_batch_size = 4
cfg.data.train_raw_size = [180, 320]
cfg.data.train_in_size = [64, 64]
cfg.data.train_data_queue_size = 64
cfg.data.num_threads = 1
cfg.data.eval_batch_size = 1
cfg.data.eval_raw_size = [180, 320]
cfg.data.eval_in_size = [180, 320]

cfg.data.eval_in_patch = False
cfg.data.eval_pad_size = 32
cfg.data.read_mode = 'python'  # ['tf', 'python']

cfg.data.noise = CN()
cfg.data.noise.noise_type = 'clean'  # see ascendcv.dataloader.noise.py for more options
cfg.data.noise.random_seed = None

# gaussian noise params
cfg.data.noise.mean = 0.
cfg.data.noise.std = 0.05
# salt-pepper noise params
cfg.data.noise.amount = 0.005
cfg.data.noise.salt_ratio = 0.5
# speckle noise params => no options
# gaussian process noise params
cfg.data.noise.min_std = 0.01
cfg.data.noise.max_std = 0.1

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
cfg.solver = CN()

cfg.solver.optimizer = CN()
cfg.solver.optimizer.type = 'Adam'

cfg.solver.lr_schedule = CN()
cfg.solver.lr_schedule.type = 'CosineRestart'
cfg.solver.lr_schedule.base_lr = 4e-4
cfg.solver.lr_schedule.total_steps = [150000, 150000, 150000, 150000]
cfg.solver.lr_schedule.restart_weights = [1, 0.5, 0.5, 0.5]
cfg.solver.lr_schedule.min_lr = 1e-7

cfg.solver.mix_precision = False
cfg.solver.loss_scale = 'off'
cfg.solver.xla = False
cfg.solver.checkpoint_interval = 500
cfg.solver.print_interval = 20


# ---------------------------------------------------------------------------- #
# EDVR
# ---------------------------------------------------------------------------- #
cfg.edvr = CN()
cfg.edvr.with_tsa = True
cfg.edvr.mid_channels = 64
cfg.edvr.num_groups = 1
cfg.edvr.num_deform_groups = 8
cfg.edvr.num_blocks_extraction = 5
cfg.edvr.num_blocks_reconstruction = 10
cfg.edvr.loss_type = 'l2'       # ['marginal l1', 'l1', 'l2', 'charbonnier']
cfg.edvr.loss_reduction = 'mean'
cfg.edvr.loss_margin = 1e-6
cfg.edvr.dcn_version = 'v2'
cfg.edvr.impl = 'npu'
cfg.edvr.upsampling = 'bilinear'   # ['bilinear', 'bicubic']

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
cfg.device = 'gpu'
cfg.device_ids = [0]
cfg.rank_size = 1
cfg.root_rank = 0

cfg.output_dir = 'outputs/edvr'
cfg.inference_result_dir = 'test'
cfg.checkpoint = ''
cfg.continue_training = False
cfg.random_seed = 20210126

cfg.writer_num_threads = 8
# -1 for infinite queue size. Consider a finite one when the output image is large.
cfg.writer_queue_size = -1
