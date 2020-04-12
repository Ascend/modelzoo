from functools import partial

# from mmcv.runner import get_dist_info
from ..utils import get_dist_info
from .sampler import GroupSampler, DistributedGroupSampler, DistributedSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_sampler(dataset,
                  imgs_per_gpu,
                  workers_per_gpu,
                  num_gpus=1,
                  world_size=1,
                  rank=0,
                  dist=True,
                  **kwargs):
    shuffle = kwargs.get('shuffle', False)
    if dist:
        # rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            print("before sample: word_size:{},local_rank:{}".format(world_size, rank))
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
            print("sample: word_size:{},local_rank:{}".format(world_size, rank))

    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None

    return sampler
