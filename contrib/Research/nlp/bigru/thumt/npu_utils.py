import tensorflow as tf

NPU_MACHINE = True
if NPU_MACHINE:
    # 导入TFAdapter插件库
    import npu_bridge 
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    from npu_bridge.estimator.npu_ops import dropout as ops_dropout

    def npu_config(param=None):
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision") # 对conv2d和matmulv2不支持
        custom_op.parameter_map["use_off_line"].b = True # 必须显式关闭remap式开启，在昇腾AI处理器执行训练
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
        return config
 
    def init_npu():
        """Initialize npu manually.
        Returns:
        `init_sess` npu  init session config.
        `npu_init` npu  init ops.
        """
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision") # 对conv2d和matmulv2不支持
        custom_op.parameter_map["use_off_line"].b = True # 必须显式开启，在昇腾AI处理器执行训练
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
        init_sess = tf.Session(config=config)

        npu_init = npu_ops.initialize_system()  
        init_sess.run(npu_init)

        return init_sess  
else:
    from tensorflow.nn import dropout as ops_dropout
    def npu_config(param=None):
        return None


def restore_from_save(checkpoint_file=None):
    if checkpoint_file:
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, checkpoint_file)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

def table_initiailization():
    return tf.tables_initializer()

# TF session config
def old_session_config(params, distribute=None):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)

    if distribute is not None:
        if distribute.is_distributed_training_mode():
            config.gpu_options.visible_device_list = str(distribute.local_rank())
        elif params.device_list:
            device_str = ",".join([str(i) for i in params.device_list])
            config.gpu_options.visible_device_list = device_str

    return config

def cpu_tensor_test(ts, feed_dict=None):
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        if feed_dict is None:
            output = sess.run(ts)
        else:
            output = sess.run(ts, feed_dict=feed_dict)  
        print(output)
        return output

def npu_tensor_test(ts):
    with tf.Session(config=npu_config()) as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        print(sess.run(ts))
      

