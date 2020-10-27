from mindspore.train.serialization import load_checkpoint, load_param_into_net

def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """
    incremental_learning or not
    """
    param_dict = load_checkpoint(pretrain_ckpt_path)
    load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False