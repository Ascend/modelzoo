import tensorflow as tf
import sys
import os
import logging
import time
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
import os
#os.system('dot -T png -o ' + png_file + ' ' + dot_file)

from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

import megatron_config
from mde.distribute.mix_parallel_init import mix_parallel_init


#NPU 分布式初始化
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关

init_sess = tf.Session(config=config)
init_sess.run(npu_int)

#混合并行相关初始化
mix_parallel_init(megatron_config.megatron_config())  

from data_loader import DataLoader
from mde.training import optimizer
from mde.distribute.mix_parallel_init import get_data_parallel_world_size,get_data_parallel_group,get_model_parallel_rank
from gpt import megatron


sys.path.append("./mde")


def get_learning_rate(config): 
    global_step = tf.train.get_global_step()
    warmup_steps = config['warmup_steps']
    total_steps = config['training_steps']
    lr_warmup_mode = config['lr_warmup_mode']
    lr_decay_mode = config['lr_decay_mode']
    warmup_lr = 0.0
    max_lr = config['learning_rate']
    end_lr = 0.0
    cd_alpha = float(end_lr)/float(max_lr) 

    with tf.device('/cpu:0'):  # Allow fallback to CPU if no GPU support for these ops
        learning_rate = tf.cond(global_step < warmup_steps,
                                lambda: lr_warmup(lr_warmup_mode, warmup_lr, global_step, warmup_steps,
                                                        max_lr),
                                lambda: lr_decay(max_lr, end_lr, lr_decay_mode, warmup_steps, total_steps, global_step,
                                                        cd_alpha))

        learning_rate = tf.identity(learning_rate, 'learning_rate')
    return learning_rate


#learning rate decay
def lr_decay(max_lr, end_lr, lr_decay_mode, warmup_steps, total_steps, global_step,cd_alpha):
    if lr_decay_mode == 'constant':
        learning_rate = tf.cast(max_lr, tf.float32)

    elif lr_decay_mode == 'cosine':
        learning_rate = tf.train.cosine_decay(max_lr,
                                              global_step - warmup_steps,
                                              decay_steps=total_steps - warmup_steps,
                                              alpha=cd_alpha) 
    elif lr_decay_mode == 'linear':
        learning_rate = decay_linear(max_lr, end_lr, warmup_steps, total_steps, global_step)

    else:
        raise ValueError('Invalid type of lr_decay_mode')
    return learning_rate

def decay_linear( lr_start, lr_end, it_start, it_end, global_step ):
    down_steps = it_end - it_start
    down_range = lr_start - lr_end 
    down_per_step = float( down_range ) / float( down_steps )
    res = tf.subtract( tf.cast(lr_start, tf.float32),  tf.multiply( tf.cast(down_per_step, tf.float32), tf.subtract(tf.cast(global_step, tf.float32), tf.cast(it_start, tf.float32) )) )
    return res


#learning rate warm up
def lr_warmup(lr_warmup_mode, warmup_lr, global_step, warmup_steps, warmup_end_lr):
    if lr_warmup_mode == 'linear':
        learning_rate = linear_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr)
    elif lr_warmup_mode == 'cosine':
        learning_rate = cos_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr)
    else:
        raise ValueError('Invalid type of lr_warmup_mode')
    return learning_rate


def linear_warmup(warmup_lr, global_step, warmup_steps, warmup_end_lr):
    from tensorflow.python.ops import math_ops
    p = tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    diff = math_ops.subtract(warmup_end_lr, warmup_lr)
    res = math_ops.add(warmup_lr, math_ops.multiply(diff, p))
    return res

def cos_warmup( warmup_lr, global_step, warmup_steps, warmup_end_lr ):
    PI = 3.14159265359
    diff = tf.subtract( warmup_end_lr, warmup_lr )
    ang = PI +  PI * ( tf.cast( global_step, tf.float32 ) / tf.cast( warmup_steps,tf.float32 ))
    offset = diff * 0.5 * ( 1.0 + tf.math.cos( ang ) )
    res =  tf.add( warmup_lr, offset )
    return res


def _get_custom_getter():
  """Returns a custom getter that this class's methods must be called under.
  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:
  ```python
  network = ConvNetBuilder(...)
  with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```
  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

  def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']

    if requested_dtype == tf.float16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    if cast_to_float16:
      var = math_ops.cast(var, tf.float16)
    return var

  return inner_custom_getter


def _fp16_getter():
  def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    # cast_to_float16 = False
    # requested_dtype = kwargs['dtype']

    # if requested_dtype == tf.float16:
    #   # Only change the variable dtype if doing so does not decrease variable
    #   # precision.
    #   kwargs['dtype'] = tf.float32
    #   cast_to_float16 = True
    var = getter(*args, **kwargs)
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    # if cast_to_float16:
    #   var = math_ops.cast(var, tf.float16)
    return var

  return inner_custom_getter

@tf_contextlib.contextmanager
def float16_scope(dtype,use_smdp_optimizer):
  """Scope class for float16 variables so that the model uses custom getter.
  This enables variables to be read as float16 type when using get_variable.
  """

  with variable_scope.variable_scope(
      '', custom_getter=_fp16_getter() if use_smdp_optimizer else _get_custom_getter(),
          dtype=dtype) as varscope:
    yield varscope


class PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i + 1])
            
class LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, config, warmup_steps=5):
  #  def __init__(self, global_batch_size, num_records, display_every=10, logger=None):
        self.global_batch_size = config['batch_size'] * get_data_parallel_world_size()
        self.warmup_steps = warmup_steps
        self.iter_times = []
        self.num_records = config['num_training_samples']
        self.display_every = config['display_every']
        self.logger = get_logger(config['log_name'], config['log_dir'])
        rank0log(self.logger, 'PY' + str(sys.version) + 'TF' + str(tf.__version__))


    def after_create_session(self, session, coord):
        rank0log(self.logger, 'Step   Epoch   Speed   Loss   FinLoss   LR')
        self.elapsed_secs = 0.
        self.count = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs( 
            fetches=[tf.train.get_global_step(), 'loss:0', 'total_loss:0', 'learning_rate:0'])
#                     'loss:0', 'loss:0', 'learning_rate:0'])

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        self.iter_times.append(batch_time)
        self.elapsed_secs += batch_time
        self.count += 1
        global_step, loss, total_loss, lr = run_values.results
        if global_step == 1 or global_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            epoch = global_step * self.global_batch_size / self.num_records

            self.logger.info('step:%6i  epoch:%5.1f  bps:%7.1f  loss:%6.6f  total_loss:%6.6f  lr:%9.7f' %
                             (global_step, epoch, img_per_sec, loss, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0

    def get_average_speed(self):
        avg_time = np.mean(self.iter_times[self.warmup_steps:])
        speed = self.global_batch_size / avg_time
        return speed



def rank0log(logger, *args, **kwargs):
    if logger: 
        logger.info(''.join([str(x) for x in list(args)]))
    else:
        print(*args, **kwargs)


def get_logger(log_name, log_dir):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # INFO, ERROR
    # file handler which logs debug messages
    if not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            # if log_dir is common for multiple ranks like on nfs
            pass
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # add formatter to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(fh)        

    return logger

def get_estimator_model_func(features, labels, mode, params):
    params['hidden_size'] = params['n_embd']

    with float16_scope(dtype=params['dtype'], 
                        use_smdp_optimizer=(params['optimizer'] == 'smdp_adamw')): #如果使用smdp_optimizer，则直接使用纯fp16的scope,smdp内部实现混合精度
        output = megatron(features=features, params=params, past=None, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.cast(output, tf.float32)  

    #out = tf.Print(out,[out],message='out:', summarize=600)
    #方案1 使用sparse_softmax_cross_entropy_with_logits
    #loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    #方案2 手动onehot
    
    one_hot_labels = tf.one_hot(labels, depth=params['n_vocab'], dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss_batch = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])


    #loss mask
    eod_token = tf.constant(0, tf.int32)         #########################eod_token的ID是0
    loss_mask = tf.not_equal(features,eod_token)
    loss_mask = tf.cast(loss_mask, dtype = tf.float32)

    predict_label = tf.math.argmax(logits, axis=2)
    predict_label = predict_label * tf.cast(loss_mask, dtype = tf.int64)
    with tf.control_dependencies([predict_label]):
      loss_batch = loss_mask * loss_batch
    loss = tf.reduce_sum(loss_batch)/tf.reduce_sum(loss_mask)
    loss = tf.identity(loss, name='loss')
    total_loss = tf.identity(loss, name='total_loss') 

    with tf.device('/cpu:0'):
        #learning_rate = get_learning_rate(params)
        learning_rate = 0.0000001
    learning_rate = tf.identity(learning_rate, name='learning_rate')
    
    optimzier_used = optimizer.Optimizer(params) 
    opt = optimzier_used.get_optimizer(learning_rate)

    # opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
    with tf.control_dependencies(update_ops):
        with tf.name_scope('loss_scale'):
            loss_scale = tf.constant(1024.0, tf.float32)
            loss_scale = tf.identity(loss_scale, name='loss_scale')
            scaled_grads_and_vars = opt.compute_gradients(total_loss * loss_scale)

            if params['optimizer'] == 'smdp_adamw':#smdp_opt保存的var都是fp16的，计算出来的grad也是fp16的，在apply之前需要cast成fp32
                scaled_grads_and_vars = [(tf.cast(g,tf.float32), v) for g,v in scaled_grads_and_vars]
                
            grad_var_list = [ (g/loss_scale, v)  for g,v in scaled_grads_and_vars ]
            
            if get_data_parallel_world_size() > 1:
                #手动实现数据并行维度的梯度汇聚
                averaged_grad_var_list = [ (average_grad(g), v)  for g,v in grad_var_list ]
            else:
                averaged_grad_var_list = grad_var_list
            #-------------------

            train_op = opt.apply_gradients( averaged_grad_var_list, global_step = tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def average_grad(grad):
    return hccl_ops.allreduce(grad, "sum", fusion=2, fusion_id=1, group=get_data_parallel_group())/float(get_data_parallel_world_size())

def main():
    # #NPU 分布式初始化
    # npu_int = npu_ops.initialize_system()
    # npu_shutdown = npu_ops.shutdown_system()

    # config = tf.ConfigProto()
    # custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name =  "NpuOptimizer"
    # custom_op.parameter_map["use_off_line"].b = True
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关

    # init_sess = tf.Session(config=config)
    # init_sess.run(npu_int)

    #训练流程
    gpu_thread_count = 2
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    params = megatron_config.megatron_config()
    dataset = DataLoader(params)

    #混合并行相关初始化
    # mix_parallel_init(params)

    session_config = tf.ConfigProto(
    inter_op_parallelism_threads=10,
    intra_op_parallelism_threads=10,
    allow_soft_placement=True,)

    graph_mem = 1024 * 1024 * 1024 * 13
    var_mem = 1024 * 1024 * 1024 * 18

    run_config = NPURunConfig(hcom_parallel=False, 
                            #precision_mode='allow_mix_precision',   #注释掉，手动实现混合精度
                            enable_data_pre_proc=True, 
                            save_checkpoints_steps=params['save_checkpoints_steps'] if get_model_parallel_rank()==0 else 0 ,
                            keep_checkpoint_max=1,
                            session_config=session_config, 
                            model_dir = params['log_dir'], 
                            iterations_per_loop=1,
                            graph_memory_max_size = graph_mem,
                            variable_memory_max_size = var_mem,
                            horovod_mode =True)  #关闭NPUEstimator的自动broadcast,不确定这样是否正确

    classifier =NPUEstimator(
        model_fn= get_estimator_model_func,
        params = params, 
        config= run_config)

    training_hooks = [PrefillStagingAreasHook(), LogSessionRunHook(params)]


    classifier.train( input_fn=lambda:dataset.get_train_input_fn(),
                               max_steps = 10,  #params['nstep'],
                               hooks = training_hooks)


    #NPU 分布式关闭
    init_sess.run(npu_shutdown)
    init_sess.close()

if __name__ == '__main__':
    main()
