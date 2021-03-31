from hccl.manage.api import create_group  
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id


def mix_parallel_init(args):
    #混合并行相关初始化
    global_world_size = get_rank_size()
    global_rank = get_rank_id()
    print('1111111111111111111111111111111111111111')
    print('global_world_size',global_world_size)
    print('global_rank', global_rank)
    model_parallel_size = args['model_parallel_dim']
    data_parallel_size = int(global_world_size / model_parallel_size)
    

    for i in range(model_parallel_size):
        ranks = [num for num in range(i, global_world_size, model_parallel_size)]
        if i == (global_rank % model_parallel_size):
            create_group("DATA_PARALLEL_GROUP", data_parallel_size, ranks)
 
    for i in range(data_parallel_size):
        ranks = [num for num in range(i * model_parallel_size, (i + 1) * model_parallel_size)]
        if i == (global_rank // model_parallel_size):
            create_group("MODEL_PARALLEL_GROUP", model_parallel_size, ranks)

def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    MODEL_PARALLEL_GROUP  = "MODEL_PARALLEL_GROUP"
    return MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    DATA_PARALLEL_GROUP = "DATA_PARALLEL_GROUP"
    return DATA_PARALLEL_GROUP

def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    print('model_parallel_size',get_rank_size(group=get_model_parallel_group()))
    return get_rank_size(group=get_model_parallel_group())

def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    print('model_parallel_rank',get_rank_id(group=get_model_parallel_group()))
    return get_rank_id(group=get_model_parallel_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    print('data_parallel_size',get_rank_size(group=get_data_parallel_group()))
    return get_rank_size(group=get_data_parallel_group())

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    print('data_parallel_rank',get_rank_id(group=get_data_parallel_group()))
    return get_rank_id(group=get_data_parallel_group())
