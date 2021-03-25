from .dufvsr import DUF
from .dufvsr_p3d import DUFwP3D
from .edvr import EDVR


def build_model(cfg):
    kwargs = dict(
        model_name=cfg.model.name,
        scale=cfg.model.scale,
        num_frames=cfg.model.num_frames,
        data_dir=cfg.data.data_dir,
        output_dir=cfg.output_dir,
        solver=cfg.solver,
        device=cfg.device,
        is_distributed=cfg.rank_size > 1,
        cfg=cfg)

    if cfg.mode == 'train':
        kwargs['batch_size'] = cfg.data.train_batch_size
        kwargs['raw_size'] = cfg.data.train_raw_size
        kwargs['in_size'] = cfg.data.train_in_size
        kwargs['set_file'] = cfg.data.train_set
        kwargs['is_train'] = True
        kwargs['checkpoint'] = ''
    else:
        eval_in_h = cfg.data.eval_in_size[0] + (cfg.data.eval_pad_size*2 if cfg.data.eval_in_patch else 0)
        eval_in_w = cfg.data.eval_in_size[1] + (cfg.data.eval_pad_size*2 if cfg.data.eval_in_patch else 0)
        kwargs['batch_size'] = cfg.data.eval_batch_size
        kwargs['raw_size'] = cfg.data.eval_raw_size
        kwargs['in_size'] = [eval_in_h, eval_in_w]
        kwargs['set_file'] = cfg.data.eval_set
        kwargs['is_train'] = False
        kwargs['checkpoint'] = cfg.checkpoint

    if cfg.model.name == 'DUF':
        model = DUF(**kwargs)
    elif cfg.model.name == 'DUF_P3D':
        model = DUFwP3D(**kwargs)
    elif cfg.model.name == 'EDVR':
        model = EDVR(**kwargs)
    else:
        raise NotImplementedError
    return model
