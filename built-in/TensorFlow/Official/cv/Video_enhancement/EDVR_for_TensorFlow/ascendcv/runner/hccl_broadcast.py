import tensorflow as tf
from npu_bridge.hccl import hccl_ops


def broadcast_global_variables(root_rank, index):
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            outputs = hccl_ops.broadcast(tensor=[var], root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    return tf.group(op_list)
