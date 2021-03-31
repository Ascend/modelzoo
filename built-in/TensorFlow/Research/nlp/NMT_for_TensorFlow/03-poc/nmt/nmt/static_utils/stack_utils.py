import tensorflow as tf

num_splits = 1
each_splits = 32

def npu_unstack(inputs, axis=0): # default is unstack along the 0-th dim
    unstack_stage_1 = tf.split(inputs, num_splits, axis=axis )

    outputs = []
    for split_part in range(num_splits):
        unstack_part = tf.unstack(unstack_stage_1[split_part], axis=axis)
        for j in range(each_splits):
          outputs.append(unstack_part[j])
    return outputs


def npu_stack(inputs, axis=0):  # default is stack along the 0-th dim
    tmp_bi_outputs = []
    for i in range(num_splits):
        start_index = i*each_splits
        end_index = (i+1)*each_splits
        outputs_part = inputs[start_index:end_index]
        outputs_stack_part = tf.stack(outputs_part, axis=axis)
        tmp_bi_outputs.append(outputs_stack_part)
         
    bi_outputs = tf.concat(tmp_bi_outputs, axis)
    return bi_outputs
