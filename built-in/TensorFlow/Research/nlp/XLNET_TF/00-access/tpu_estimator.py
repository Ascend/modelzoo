
'TPUEstimator class.'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import collections
import copy
import os
import signal
import sys
import threading
import time
import numpy as np
import six
from six.moves import queue as Queue
from six.moves import xrange
from tensorflow.contrib.tpu.proto import compilation_result_pb2 as tpu_compilation_result
from tensorflow.contrib.tpu.python.tpu import tensor_tracer
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import error_handling
from tensorflow.contrib.tpu.python.tpu import session_support
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_context
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import training_loop
from tensorflow.contrib.tpu.python.tpu import util as util_lib
from tensorflow.contrib.training.python.training import hparam
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export_output as export_output_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2 as contrib_summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import function_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
_INITIAL_LOSS = 10000000.0
_ZERO_LOSS = 0.0
_TPU_ESTIMATOR = 'custom_tpu_estimator'
_ITERATIONS_PER_LOOP_VAR = 'iterations_per_loop'
_BATCH_SIZE_KEY = 'batch_size'
_CTX_KEY = 'context'
_USE_TPU_KEY = 'use_tpu'
_CROSS_REPLICA_SUM_OP = 'CrossReplicaSum'
_ONE_GIGABYTE = ((1024 * 1024) * 1024)
_TPU_ENQUEUE_OPS = '_tpu_enqueue_ops'
_TPU_TRAIN_OP = '_tpu_train_op'
_REWRITE_FOR_INFERENCE_MODE = '_rewrite_for_inference'
_RESERVED_PARAMS_KEYS = [_BATCH_SIZE_KEY, _CTX_KEY]
_WRAP_INPUT_FN_INTO_WHILE_LOOP = False
ops.register_proto_function('{}_{}'.format(_TPU_ESTIMATOR, _ITERATIONS_PER_LOOP_VAR), proto_type=variable_pb2.VariableDef, to_proto=resource_variable_ops._to_proto_fn, from_proto=resource_variable_ops._from_proto_fn)

def _is_iterable(obj):
    'A Python 2 and 3 compatible util to check whether `obj` is iterable.'
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def _create_global_step(graph):
    graph = (graph or ops.get_default_graph())
    if (training.get_global_step(graph) is not None):
        raise ValueError('"global_step" already exists.')
    with graph.as_default() as g, g.name_scope(None):
        return variable_scope.get_variable(ops.GraphKeys.GLOBAL_STEP, shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, use_resource=True, collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])

def _create_or_get_iterations_per_loop():
    'Creates or gets the iterations_per_loop variable.\n\n  In TPUEstimator, the user provided computation, the model_fn, is wrapped\n  inside a tf.while_loop for peak performance. The iterations of the loop are\n  specified by this variable, which adjusts its value on the CPU after each TPU\n  program execution and before the next TPU execution.\n\n  The purpose of using a variable, rather then a constant, is to allow\n  TPUEstimator adapt the TPU training iterations according to the final steps\n  specified by users. For example, if the user sets the iterations_per_loop as 4\n  in TPUConfig and steps as 10 in TPUEstimator.train(), the iterations_per_loop\n  variable will have the following value before each TPU training.\n\n      - 1-th TPU execution: iterations_per_loop = 4\n      - 2-th TPU execution: iterations_per_loop = 4\n      - 3-th TPU execution: iterations_per_loop = 2\n\n  As model_fn increases the global step once per train_op invocation, the global\n  step is 10 after all TPU executions, matching the steps=10 inputs passed in by\n  users.\n\n  Returns:\n    A TF non-trainable resource variable.\n\n  Raises:\n    RuntimeError: If multi iterations_per_loop variables were found.\n  '
    graph = ops.get_default_graph()
    collection_name = '{}_{}'.format(_TPU_ESTIMATOR, _ITERATIONS_PER_LOOP_VAR)
    iter_vars = graph.get_collection(collection_name)
    if (len(iter_vars) == 1):
        return iter_vars[0]
    elif (len(iter_vars) > 1):
        raise RuntimeError('Multiple iterations_per_loop_var in collection.')
    with ops.colocate_with(training_util.get_global_step()):
        with variable_scope.variable_scope(_TPU_ESTIMATOR, reuse=variable_scope.AUTO_REUSE):
            return variable_scope.get_variable(_ITERATIONS_PER_LOOP_VAR, initializer=init_ops.zeros_initializer(), shape=[], dtype=dtypes.int32, trainable=False, collections=[collection_name, ops.GraphKeys.LOCAL_VARIABLES], use_resource=True)

def _sync_variables_ops(ctx):
    'Create varriables synchronization ops.\n\n  Gets the variables back from TPU nodes. This means the variables updated\n  by TPU will now be *synced* to host memory.\n  In BROADCAST mode, we skip this sync since the variables are ususally too\n  big to transmit via RPC.\n\n  Args:\n    ctx: A `_InternalTPUContext` instance with mode.\n\n  Returns:\n    A list of sync ops.\n  '
    if (not ctx.is_input_broadcast_with_iterators()):
        return [array_ops.check_numerics(v.read_value(), ('Gradient for %s is NaN' % v.name)).op for v in variables.trainable_variables()]
    else:
        return [control_flow_ops.no_op()]

def _increase_eval_step_op(iterations_per_loop):
    'Returns an op to increase the eval step for TPU evaluation.\n\n  Args:\n    iterations_per_loop: Tensor. The number of eval steps running in TPU system\n      before returning to CPU host for each `Session.run`.\n\n  Returns:\n    An operation\n  '
    eval_step = evaluation._get_or_create_eval_step()
    return state_ops.assign_add(eval_step, math_ops.cast((iterations_per_loop - 1), dtype=eval_step.dtype), use_locking=True)

def _extract_key_names(tensor_or_dict):
    if isinstance(tensor_or_dict, dict):
        return sorted(tensor_or_dict.keys())
    return []

class _SIGNAL(object):
    'Signal used to control the thread of infeed/outfeed.\n\n  All preserved signals must be negative numbers. Positive numbers are used to\n  indicate the number of iterations for next training/evaluation loop.\n  '
    NEXT_BATCH = (- 1)
    STOP = (- 2)

class TPUEstimatorSpec(model_fn_lib._TPUEstimatorSpec):
    'Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.\n\n  See `EstimatorSpec` for `mode`, `predictions`, `loss`, `train_op`, and\n  `export_outputs`.\n\n  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where\n  `metric_fn` runs on CPU to generate metrics and `tensors` represents the\n  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.\n  To be precise, TPU evaluation expects a slightly different signature from the\n  `tf.estimator.Estimator`. While `EstimatorSpec.eval_metric_ops` expects a\n  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.\n  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The\n  `tensors` usually specify the model logits, which are transferred back from\n  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch\n  size is the first dimension. Once all tensors are available at CPU host from\n  all shards, they are concatenated (on CPU) and passed as positional arguments\n  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is\n  a dict. `metric_fn` takes the `tensors` and returns a dict from metric string\n  name to the result of calling a metric function, namely a `(metric_tensor,\n  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the\n  `eval_metrics`.\n\n  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This\n  function should not capture any Tensors in `model_fn`.\n\n  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`\n  to pass to that function and returns a list of Tensors. `host_call` currently\n  works for train() and evaluate(). The Tensors returned by the function is\n  executed on the CPU on every step, so there is communication overhead when\n  sending tensors from TPU to CPU. To reduce the overhead, try reducing the\n  size of the tensors. The `tensors` are concatenated along their major (batch)\n  dimension, and so must be >= rank 1. The `host_call` is useful for writing\n  summaries with `tf.contrib.summary.create_file_writer`.\n  '

    def __new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metrics=None, export_outputs=None, scaffold_fn=None, host_call=None, training_hooks=None, evaluation_hooks=None, prediction_hooks=None):
        'Creates a validated `TPUEstimatorSpec` instance.'
        host_calls = {}
        if (eval_metrics is not None):
            host_calls['eval_metrics'] = eval_metrics
        if (host_call is not None):
            host_calls['host_call'] = host_call
        _OutfeedHostCall.validate(host_calls)
        training_hooks = tuple((training_hooks or []))
        evaluation_hooks = tuple((evaluation_hooks or []))
        prediction_hooks = tuple((prediction_hooks or []))
        for hook in ((training_hooks + evaluation_hooks) + prediction_hooks):
            if (not isinstance(hook, session_run_hook.SessionRunHook)):
                raise TypeError('All hooks must be SessionRunHook instances, given: {}'.format(hook))
        return super(TPUEstimatorSpec, cls).__new__(cls, mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metrics=eval_metrics, export_outputs=export_outputs, scaffold_fn=scaffold_fn, host_call=host_call, training_hooks=training_hooks, evaluation_hooks=evaluation_hooks, prediction_hooks=prediction_hooks)

    def as_estimator_spec(self):
        'Creates an equivalent `EstimatorSpec` used by CPU train/eval.'
        host_calls = {}
        if (self.eval_metrics is not None):
            host_calls['eval_metrics'] = self.eval_metrics
        if (self.host_call is not None):
            host_calls['host_call'] = self.host_call
        host_call_ret = _OutfeedHostCall.create_cpu_hostcall(host_calls)
        eval_metric_ops = None
        if (self.eval_metrics is not None):
            eval_metric_ops = host_call_ret['eval_metrics']
        hooks = None
        if (self.host_call is not None):
            hooks = [_OutfeedHostCallHook(host_call_ret['host_call'])]
        if tensor_tracer.TensorTracer.is_enabled():
            tt = tensor_tracer.TensorTracer()
            tracing_calls = tt.trace_cpu(ops.get_default_graph())
            tracing_call_ret = _OutfeedHostCall.create_cpu_hostcall(tracing_calls)
            tracing_functions = tracing_call_ret.values()
            if tracing_functions:
                if hooks:
                    hooks.extend([_OutfeedHostCallHook(tracing_functions)])
                else:
                    hooks = [_OutfeedHostCallHook(tracing_functions)]
        hooks = tuple((hooks or []))
        scaffold = (self.scaffold_fn() if self.scaffold_fn else None)
        return model_fn_lib.EstimatorSpec(mode=self.mode, predictions=self.predictions, loss=self.loss, train_op=self.train_op, eval_metric_ops=eval_metric_ops, export_outputs=self.export_outputs, scaffold=scaffold, training_hooks=(self.training_hooks + hooks), evaluation_hooks=(self.evaluation_hooks + hooks), prediction_hooks=(self.prediction_hooks + hooks))

class _OpQueueContext(object):
    'Manages work queue and thread for a infeed/outfeed thread.'

    def __init__(self, name, target, args):
        self._name = name
        self._queue = Queue.Queue()
        args = ((self,) + args)
        self._thread = threading.Thread(name=name, target=target, args=args)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._queue.put(_SIGNAL.STOP)

    def send_next_batch_signal(self, iterations):
        self._queue.put(iterations)

    def read_iteration_counts(self):
        while True:
            iterations = self._queue.get(block=True)
            logging.debug('%s read iterations %s', self._name, iterations)
            if (iterations == _SIGNAL.STOP):
                logging.info('%s received shutdown signal, stopping.', self._name)
                return
            (yield iterations)

    def join(self):
        logging.info('Shutting down %s thread.', self._name)
        self.stop()
        self._thread.join()

class _OpSignalOnceQueueContext(_OpQueueContext):
    'Manages work queue and thread for a infeed/outfeed thread.\n\n  This subclass only signals once.\n  '

    def __init__(self, name, target, args):
        super(_OpSignalOnceQueueContext, self).__init__(name, target, args)
        self._has_signaled = False

    def send_next_batch_signal(self, iterations):
        if (not self._has_signaled):
            self._queue.put(iterations)
            self._has_signaled = True

class TPUInfeedOutfeedSessionHook(session_run_hook.SessionRunHook):
    'A Session hook setting up the TPU initialization, infeed, and outfeed.\n\n  This hook does two major things:\n  1. initialize and shutdown TPU system.\n  2. launch and join the threads for infeed enqueue and (optional) outfeed\n     dequeue.\n  '

    def __init__(self, ctx, enqueue_ops, dequeue_ops, tpu_compile_op, run_infeed_loop_on_coordinator=True, rendezvous=None, master=None, session_config=None):
        self._master_job = ctx.master_job
        self._enqueue_ops = enqueue_ops
        self._dequeue_ops = dequeue_ops
        self._rendezvous = rendezvous
        self._master = master
        self._session_config = session_config
        self._run_infeed_loop_on_coordinator = run_infeed_loop_on_coordinator
        self._initial_infeed_sleep_secs = ctx.config.tpu_config.initial_infeed_sleep_secs
        self._feed_error = None
        self._finished = False
        self._should_initialize_tpu = True
        self._tpu_compile_op = tpu_compile_op

    def begin(self):
        logging.info('TPU job name %s', self._master_job)
        self._iterations_per_loop_var = _create_or_get_iterations_per_loop()
        self._init_ops = []
        if self._should_initialize_tpu:
            self._finalize_ops = [tpu.shutdown_system(job=self._master_job)]
        else:
            self._finalize_ops = []
        summary_writer_init_ops = contrib_summary.summary_writer_initializer_op()
        self._init_ops.extend(summary_writer_init_ops)
        for op in summary_writer_init_ops:
            self._finalize_ops.append(contrib_summary.flush(writer=op.inputs[0]))

    def _run_infeed(self, queue_ctx, session):
        logging.info('Starting infeed thread controller.')
        if self._initial_infeed_sleep_secs:
            logging.info('Infeed thread sleeping for %d seconds.', self._initial_infeed_sleep_secs)
            time.sleep(self._initial_infeed_sleep_secs)
            logging.info('Infeed thread starting after sleep')
        with self._rendezvous.catch_errors(source='infeed', session=session):
            if self._run_infeed_loop_on_coordinator:
                for (count, steps) in enumerate(queue_ctx.read_iteration_counts()):
                    for i in xrange(steps):
                        logging.debug('Infeed enqueue for iteration (%d, %d)', count, i)
                        session.run(self._enqueue_ops)
            else:
                for _ in queue_ctx.read_iteration_counts():
                    session.run(self._enqueue_ops)
            logging.info('Infeed thread finished, shutting down.')

    def _run_outfeed(self, queue_ctx, session):
        logging.info('Starting outfeed thread controller.')
        with self._rendezvous.catch_errors(source='outfeed', session=session):
            for (count, steps) in enumerate(queue_ctx.read_iteration_counts()):
                for i in xrange(steps):
                    logging.debug('Outfeed dequeue for iteration (%d, %d)', count, i)
                    session.run(self._dequeue_ops)
            logging.info('Outfeed thread finished, shutting down.')

    def _create_infeed_controller(self, name, target, args):
        return _OpQueueContext(name=name, target=target, args=args)

    def _assertCompilationSucceeded(self, result, coord):
        proto = tpu_compilation_result.CompilationResultProto()
        proto.ParseFromString(result)
        if proto.status_error_message:
            logging.error('Compilation failed: {}'.format(proto.status_error_message))
            coord.request_stop()
        else:
            logging.info('Compilation succeeded')

    def after_create_session(self, session, coord):
        if self._should_initialize_tpu:
            logging.info('Init TPU system')
            start = time.time()
            with ops.Graph().as_default():
                with tf_session.Session(self._master, config=npu_config_proto(config_proto=self._session_config)) as sess:
                    sess.run(tpu.initialize_system(job=self._master_job))
            logging.info('Initialized TPU in %d seconds', (time.time() - start))
        session.run(self._init_ops, options=config_pb2.RunOptions(timeout_in_ms=((5 * 60) * 1000)))
        if (os.environ.get('TPU_SPLIT_COMPILE_AND_EXECUTE', '') == '1'):
            logging.info('Compiling user program: this may take a while...')
            self._assertCompilationSucceeded(session.run(self._tpu_compile_op), coord)
        self._infeed_controller = self._create_infeed_controller(name='InfeedController', target=self._run_infeed, args=(session,))
        self._outfeed_controller = _OpQueueContext(name='OutfeedController', target=self._run_outfeed, args=(session,))
        watchdog_timeout = int(os.environ.get('TF_TPU_WATCHDOG_TIMEOUT', '0'))
        if (watchdog_timeout > 0):
            session_support.start_worker_watchdog(session, shutdown_timeout=watchdog_timeout)

    def before_run(self, run_context):
        self._feed_error = None
        iterations = run_context.session.run(self._iterations_per_loop_var)
        logging.info('Enqueue next (%d) batch(es) of data to infeed.', iterations)
        self._infeed_controller.send_next_batch_signal(iterations)
        logging.info('Dequeue next (%d) batch(es) of data from outfeed.', iterations)
        self._outfeed_controller.send_next_batch_signal(iterations)

    def end(self, session):
        self._finished = True
        logging.info('Stop infeed thread controller')
        self._infeed_controller.join()
        self._rendezvous.record_done('infeed')
        logging.info('Stop output thread controller')
        self._outfeed_controller.join()
        self._rendezvous.record_done('outfeed')
        logging.info('Shutdown TPU system.')
        session.run(self._finalize_ops)

class TPUInfeedOutfeedSessionHookForPrediction(TPUInfeedOutfeedSessionHook):

    def __init__(self, ctx, enqueue_ops, dequeue_ops, tpu_compile_op, rendezvous=None, master=None, session_config=None):
        super(TPUInfeedOutfeedSessionHookForPrediction, self).__init__(ctx, enqueue_ops, dequeue_ops, tpu_compile_op=tpu_compile_op, run_infeed_loop_on_coordinator=False, rendezvous=rendezvous, master=master, session_config=session_config)

    def _create_infeed_controller(self, name, target, args):
        return _OpSignalOnceQueueContext(name=name, target=target, args=args)

class _TPUStopAtStepHook(session_run_hook.SessionRunHook):
    'Hook that requests stop at a specified step.\n\n  This hook is similar to the `session_run_hook._StopAfterNEvalsHook` with\n  following differences for TPU training:\n\n  1. This hook sets the variable for iterations_per_loop, which is used by\n     `TPUInfeedOutfeedSessionHook` to control the iterations for infeed/outfeed.\n     As the hook execution order is not guaranteed, the variable update is\n     handled in `after_create_session` and `after_run` as\n     `TPUInfeedOutfeedSessionHook` reads the variable value in `before_run`.\n\n  2. For each training loop (session.run), the global step could be increased\n     multiple times on TPU. The global step tensor value will be explicitly read\n     again in `after_run` to ensure the latest value is retrieved to avoid race\n     condition.\n  '

    def __init__(self, iterations, num_steps=None, last_step=None):
        'Initializes a `StopAtStepHook`.\n\n    Args:\n      iterations: The number of iterations to run optimizer per training loop.\n      num_steps: Number of steps to execute.\n      last_step: Step after which to stop.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        if ((num_steps is None) and (last_step is None)):
            raise ValueError('One of num_steps or last_step must be specified.')
        if ((num_steps is not None) and (last_step is not None)):
            raise ValueError('Only one of num_steps or last_step can be specified.')
        self._num_steps = num_steps
        self._last_step = last_step
        self._iterations = iterations

    def _next_iterations(self, global_step, last_step):
        gap = (last_step - global_step)
        return min(gap, self._iterations)

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if (self._global_step_tensor is None):
            raise RuntimeError('Global step should be created.')
        self._iterations_per_loop_var = _create_or_get_iterations_per_loop()

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        if (self._last_step is None):
            self._last_step = (global_step + self._num_steps)
        iterations = self._next_iterations(global_step, self._last_step)
        self._iterations_per_loop_var.load(iterations, session=session)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._global_step_tensor)
        if (global_step >= self._last_step):
            run_context.request_stop()
        else:
            iterations = self._next_iterations(global_step, self._last_step)
            self._iterations_per_loop_var.load(iterations, session=run_context.session)

class _SetEvalIterationsHook(session_run_hook.SessionRunHook):
    'Hook that requests stop at a specified step.'

    def __init__(self, num_steps):
        'Initializes a `_SetEvalIterationsHook`.\n\n    Args:\n      num_steps: Number of steps to execute.\n    '
        self._num_steps = num_steps

    def begin(self):
        self._iterations_per_loop_var = _create_or_get_iterations_per_loop()

    def after_create_session(self, session, coord):
        self._iterations_per_loop_var.load(self._num_steps, session=session)

class _StoppingPredictHook(session_run_hook.SessionRunHook):
    'Hook that requests stop according to the stopping signal in prediction.'

    def __init__(self, scalar_stopping_signal):
        self._scalar_stopping_signal = scalar_stopping_signal

    def begin(self):
        self._iterations_per_loop_var = _create_or_get_iterations_per_loop()

    def after_create_session(self, session, coord):
        self._iterations_per_loop_var.load(1, session=session)

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(self._scalar_stopping_signal)

    def after_run(self, run_context, run_values):
        _ = run_context
        scalar_stopping_signal = run_values.results
        if _StopSignals.should_stop(scalar_stopping_signal):
            raise errors.OutOfRangeError(None, None, 'Stopped by stopping signal.')

def generate_per_core_enqueue_ops_fn_for_host(ctx, input_fn, inputs_structure_recorder, host_device, host_id):
    'Generates infeed enqueue ops for per-core input_fn on a single host.'
    captured_infeed_queue = _CapturedObject()
    tpu_ordinal_function_impl = ctx.tpu_ordinal_function(host_id)

    def enqueue_ops_fn():
        'A fn returns enqueue_ops.'
        num_cores_per_host = ctx.num_of_cores_per_host
        per_host_sharded_inputs = []
        for core_ordinal in range(num_cores_per_host):
            with ops.name_scope(('ordinal_%d' % core_ordinal)):
                user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=host_device, invocation_index=((host_id * ctx.num_of_cores_per_host) + core_ordinal))
                inputs = _Inputs.from_input_fn(input_fn(user_context))
                if inputs.is_dataset:
                    raise TypeError('`input_fn` returning `Dataset`  is not yet supported in per-Core input pipeline deployment yet. Please set TPUConfig.per_host_input_for_training to True or return `features` and `labels` from `input_fn`')
                (features, labels) = inputs.features_and_labels()
                inputs_structure_recorder.validate_and_record_structure(features, labels)
                flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels)
                per_host_sharded_inputs.append(flattened_inputs)
        infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]))
        captured_infeed_queue.capture(infeed_queue)
        per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_function_impl)
        return per_host_enqueue_ops
    return (enqueue_ops_fn, captured_infeed_queue)

def generate_per_host_enqueue_ops_fn_for_host(ctx, input_fn, inputs_structure_recorder, batch_axis, device, host_id):
    'Generates infeed enqueue ops for per-host input_fn on a single host.'
    captured_infeed_queue = _CapturedObject()
    dataset_initializer = None
    with ops.device(device):
        user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=device, invocation_index=host_id)
        inputs = _Inputs.from_input_fn(input_fn(user_context))
        is_dataset = inputs.is_dataset
        if (ctx.mode == model_fn_lib.ModeKeys.PREDICT):
            if (not is_dataset):
                raise TypeError('For mode PREDICT, `input_fn` must return `Dataset` instead of `features` and `labels`.')
            if (batch_axis is not None):
                raise TypeError('For mode PREDICT, batch_axis is not supported yet.')
            inputs = _InputsWithStoppingSignals(dataset=inputs.dataset, batch_size=ctx.batch_size_for_input_fn, add_padding=True)
        if is_dataset:
            dataset_initializer = inputs.dataset_initializer()
        tpu_ordinal_function_impl = ctx.tpu_ordinal_function(host_id)

    def enqueue_ops_fn():
        'A Fn returning the TPU infeed enqueue ops.\n\n    By providing as a Fn, it can be invoked inside the tf.while_loop such that\n    the input pipeline for multiple iterations can be executed by one\n    Session.run call.\n\n    Returns:\n      list of dict of ops.\n    '
        with ops.device(device):
            num_of_replicas_per_host = ctx.num_of_replicas_per_host
            (features, labels) = inputs.features_and_labels()
            signals = inputs.signals()
            inputs_structure_recorder.validate_and_record_structure(features, labels)
            unsharded_tensor_list = inputs_structure_recorder.flatten_features_and_labels(features, labels, signals)
            infeed_queue = tpu_feed.InfeedQueue(tuple_types=[t.dtype for t in unsharded_tensor_list], tuple_shapes=[t.shape for t in unsharded_tensor_list], shard_dimensions=batch_axis)
            captured_infeed_queue.capture(infeed_queue)
            infeed_queue.set_number_of_shards(num_of_replicas_per_host)
            per_host_enqueue_ops = infeed_queue.split_inputs_and_generate_enqueue_ops(unsharded_tensor_list, placement_function=(lambda x: device), tpu_ordinal_function=tpu_ordinal_function_impl)
            if (signals is None):
                return per_host_enqueue_ops
            else:
                return {'ops': per_host_enqueue_ops, 'signals': signals}
    return (enqueue_ops_fn, captured_infeed_queue, dataset_initializer)

def generate_per_host_v2_enqueue_ops_fn_for_host(ctx, input_fn, inputs_structure_recorder, device, host_id):
    'Generates infeed enqueue ops for per-host input_fn on a single host.'
    captured_infeed_queue = _CapturedObject()
    dataset_initializer = None
    with ops.device(device):
        user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=device, invocation_index=host_id)
        inputs = _Inputs.from_input_fn(input_fn(user_context))
        is_dataset = inputs.is_dataset
        if (not is_dataset):
            raise TypeError('`input_fn` must return a `Dataset` for the PER_HOST_V2 input pipeline configuration.')
        if (ctx.mode == model_fn_lib.ModeKeys.PREDICT):
            inputs = _InputsWithStoppingSignals(dataset=inputs.dataset, batch_size=ctx.batch_size_for_input_fn, add_padding=True, num_invocations_per_step=ctx.num_of_replicas_per_host)
        dataset_initializer = inputs.dataset_initializer()
        tpu_ordinal_function_impl = ctx.tpu_ordinal_function(host_id)

    def enqueue_ops_fn():
        'Generates the per_host enqueue ops.'
        control_deps = []
        per_host_sharded_inputs = []
        num_replicas_per_host = ctx.num_of_replicas_per_host
        cached_signals = None
        with ops.device(device):
            if (not inputs.is_dataset):
                raise TypeError('`input_fn` must return a `Dataset` for this mode.')
            for _ in range(num_replicas_per_host):
                with ops.control_dependencies(control_deps):
                    (features, labels) = inputs.features_and_labels()
                    signals = inputs.signals()
                    if cached_signals:
                        signals['stopping'] = cached_signals['stopping']
                    else:
                        cached_signals = signals
                inputs_structure_recorder.validate_and_record_structure(features, labels)
                flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels, signals)
                control_deps.extend(flattened_inputs)
                per_host_sharded_inputs.append(flattened_inputs)
            if inputs_structure_recorder.flattened_input_dims:
                input_partition_dims = inputs_structure_recorder.flattened_input_dims
                if signals:
                    input_partition_dims += ([None] * len(signals))
                infeed_queue = tpu_feed._PartitionedInfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]), host_id=host_id, input_partition_dims=input_partition_dims, device_assignment=ctx.device_assignment)
                per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs)
            else:
                infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]))
                per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_function_impl)
            captured_infeed_queue.capture(infeed_queue)
        if (signals is None):
            return per_host_enqueue_ops
        else:
            return {'ops': per_host_enqueue_ops, 'signals': signals}
    return (enqueue_ops_fn, captured_infeed_queue, dataset_initializer)

def generate_broadcast_enqueue_ops_fn(ctx, input_fn, inputs_structure_recorder, num_hosts):
    'Generates infeed enqueue ops for one input_fn on all the hosts.'
    captured_infeed_queue = _CapturedObject()
    dataset_initializer = None
    device_0 = ctx.tpu_host_placement_function(host_id=0)
    with ops.device(device_0):
        user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=device_0, invocation_index=0)
        inputs = _Inputs.from_input_fn(input_fn(user_context))
        is_dataset = inputs.is_dataset
        if (ctx.mode == model_fn_lib.ModeKeys.PREDICT):
            if (not is_dataset):
                raise TypeError('For mode PREDICT, `input_fn` must return `Dataset` instead of `features` and `labels`.')
            inputs = _InputsWithStoppingSignals(dataset=inputs.dataset, batch_size=ctx.batch_size_for_input_fn, add_padding=True)
        if is_dataset:
            dataset_initializer = inputs.dataset_initializer()
        num_replicas_per_host = ctx.num_of_replicas_per_host

    def tpu_ordinal_function_impl(replica_id):
        if ctx.device_assignment:
            return ctx.device_assignment.tpu_ordinal(replica=replica_id)
        else:
            return (replica_id % num_replicas_per_host)

    def device_function_impl(replica_id):
        return ctx.tpu_host_placement_function(replica_id=replica_id)

    def enqueue_ops_fn():
        'Generates enqueue ops for all the hosts.'
        broadcasted_inputs = []
        flattened_inputs = None
        signals = None
        for host_id in xrange(num_hosts):
            with ops.device(ctx.tpu_host_placement_function(host_id=host_id)):
                for _ in xrange(ctx.num_of_replicas_per_host):
                    if (flattened_inputs is None):
                        (features, labels) = inputs.features_and_labels()
                        signals = inputs.signals()
                        inputs_structure_recorder.validate_and_record_structure(features, labels)
                        flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels, signals)
                    broadcasted_inputs.append(flattened_inputs)
        infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(broadcasted_inputs[0]))
        captured_infeed_queue.capture(infeed_queue)
        enqueue_ops = infeed_queue.generate_enqueue_ops(broadcasted_inputs, tpu_ordinal_function=tpu_ordinal_function_impl, placement_function=device_function_impl)
        if (signals is None):
            return enqueue_ops
        else:
            return {'ops': enqueue_ops, 'signals': signals}
    return (enqueue_ops_fn, captured_infeed_queue, dataset_initializer)

class _InputPipeline(object):
    '`_InputPipeline` handles invoking `input_fn` and piping to infeed queue.\n\n  `_InputPipeline` abstracts the per-core/per-host `input_fn` invocation from\n  call site.  To be precise, based on the configuration in\n  `_InternalTPUContext`,  it invokes `input_fn` for all cores (usually\n  multi-host TPU training) or for one host (usually for single-host TPU\n  evaluation), and sends all `features` and `labels` returned by `input_fn` to\n  TPU infeed. For per-core invocation, `features` and `labels` are piped to\n  infeed directly, one tuple for each core. For per-host invocation,  `features`\n  and `labels` are split at host (with respect to `batch_axis`) and piped to all\n  cores accordingly.\n\n  In addition, flatten/unflatten are handled by `_InputPipeline` also.  Model\n  inputs returned by the `input_fn` can have one of the following forms:\n  1. features\n  2. (features, labels)\n  3. ((arbitrarily nested structure of features), labels)\n\n  Internally, form 1 is reformed to `(features, None)` as features and labels\n  are passed separately to underlying methods. For TPU training, TPUEstimator\n  may expect multiple `features` and `labels` tuples one for each core.\n\n  TPUEstimator allows various different structures for inputs (namely `features`\n  and `labels`).  Both `features` and `labels` can be any nested sturcture\n  supported by TF nest (namely, dict, tuples, namedtuples or any nested\n  structure of such of Tensors).  `labels` could be `None` as well.\n\n  These are flattened before they are passed to the infeed/outfeed library\n  as that expectes flattend lists.\n  '

    class InputsStructureRecorder(object):
        'The recorder to record inputs structure.'

        def __init__(self, input_partition_dims=None):
            self._feature_structure = {}
            self._flattened_input_dims = None
            if input_partition_dims:
                assert (len(input_partition_dims) <= 2), 'must have 1 or 2 elements.'
                if (len(input_partition_dims) == 2):
                    (self._feature_dims, self._label_dims) = input_partition_dims
                else:
                    self._feature_dims = input_partition_dims[0]
                    self._label_dims = None
                assert (self._feature_dims is not None), 'input_partition_dims[0] must not be None'
            else:
                self._feature_dims = None
                self._label_dims = None
            self._initialized = False

        @property
        def flattened_input_dims(self):
            assert self._initialized, 'InputsStructureRecorder is not initialized.'
            return self._flattened_input_dims

        def has_labels(self):
            return ('labels' in self._feature_structure)

        def _flatten_input_dims(self, feature_dims, feature_dims_names, label_dims, label_dims_names, label_names, has_labels):
            'Flatten input dims with the same order as flattened input tensors.'
            flattened_input_dims = []
            if feature_dims_names:
                flattened_input_dims.extend([feature_dims[name] for name in feature_dims_names])
            else:
                flattened_input_dims.append(feature_dims)
            if label_dims_names:
                flattened_input_dims.extend([label_dims[name] for name in label_dims_names])
            else:
                if label_names:
                    num_tensors_in_label = len(label_names)
                else:
                    num_tensors_in_label = int(has_labels)
                flattened_input_dims.extend(([label_dims] * num_tensors_in_label))
            return flattened_input_dims

        def validate_and_record_structure(self, features, labels):
            'Validates and records the structure of `features` and `labels`.'
            has_labels = (labels is not None)
            feature_names = _extract_key_names(features)
            label_names = _extract_key_names(labels)
            if (not self._initialized):
                self._initialized = True
                if (self._feature_dims is not None):
                    feature_dims_names = _extract_key_names(self._feature_dims)
                    if (feature_dims_names != feature_names):
                        raise ValueError('TPUConfig.input_partition_dims[0] mismatched feature keys. Expected {}, got {}'.format(feature_names, feature_dims_names))
                    label_dims_names = _extract_key_names(self._label_dims)
                    if ((self._label_dims is not None) and (label_dims_names != label_names)):
                        raise ValueError('TPUConfig.input_partition_dims[1] mismatched label keys. Expected {}, got {}'.format(label_names, label_dims_names))
                    self._flattened_input_dims = self._flatten_input_dims(self._feature_dims, feature_dims_names, self._label_dims, label_dims_names, label_names, has_labels)

        def flatten_features_and_labels(self, features, labels, signals=None):
            'Flattens the `features` and `labels` to a single tensor list.'
            self._feature_structure['features'] = features
            if (labels is not None):
                self._feature_structure['labels'] = labels
            if (signals is not None):
                self._feature_structure['signals'] = signals
            return data_nest.flatten(self._feature_structure)

        def unflatten_features_and_labels(self, flattened_inputs):
            'Restores the flattened inputs to original features and labels form.\n\n      Args:\n        flattened_inputs: Flattened inputs for each shard.\n\n      Returns:\n        A tuple of (`features`, `labels`), where `labels` could be None.\n        Each one, if present, should have identical structure (single tensor vs\n        dict) as the one returned by input_fn.\n\n      Raises:\n        ValueError: If the number of expected tensors from `flattened_inputs`\n          mismatches the recorded structure.\n      '
            unflattened_inputs = data_nest.pack_sequence_as(self._feature_structure, flattened_inputs)
            return _Inputs(unflattened_inputs['features'], unflattened_inputs.get('labels'), signals=unflattened_inputs.get('signals'))

    def __init__(self, input_fn, batch_axis, ctx):
        'Constructor.\n\n    Args:\n      input_fn: input fn for train or eval.\n      batch_axis: A python tuple of int values describing how each tensor\n        produced by the Estimator `input_fn` should be split across the TPU\n        compute shards.\n      ctx: A `_InternalTPUContext` instance with mode.\n\n    Raises:\n      ValueError: If both `sharded_features` and `num_cores` are `None`.\n    '
        self._inputs_structure_recorder = _InputPipeline.InputsStructureRecorder(ctx.input_partition_dims)
        self._sharded_per_core = ctx.is_input_sharded_per_core()
        self._input_fn = input_fn
        self._infeed_queue = None
        self._ctx = ctx
        self._batch_axis = batch_axis

    def generate_infeed_enqueue_ops_and_dequeue_fn(self):
        'Generates infeed enqueue ops and dequeue_fn.'
        (enqueue_ops, all_hooks, run_infeed_loop_on_coordinator) = self._invoke_input_fn_and_record_structure()
        self._validate_input_pipeline()

        def dequeue_fn():
            'dequeue_fn is used by TPU to retrieve the tensors.'
            values = self._infeed_queue.generate_dequeue_op(tpu_device=0)
            return self._inputs_structure_recorder.unflatten_features_and_labels(values)
        return (enqueue_ops, dequeue_fn, all_hooks, run_infeed_loop_on_coordinator)

    def _invoke_input_fn_and_record_structure(self):
        'Deploys the input pipeline and record input structure.'
        enqueue_ops = []
        infeed_queues = []
        all_dataset_initializers = []
        num_hosts = self._ctx.num_hosts
        tpu_host_placement_fn = self._ctx.tpu_host_placement_function
        run_infeed_loop_on_coordinator = True
        if self._sharded_per_core:
            for host_id in range(num_hosts):
                host_device = tpu_host_placement_fn(host_id=host_id)
                with ops.device(host_device):
                    with ops.name_scope(('input_pipeline_task%d' % host_id)):
                        (enqueue_ops_fn, captured_infeed_queue) = generate_per_core_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id)
                        if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
                            run_infeed_loop_on_coordinator = False
                            enqueue_ops.append(_wrap_computation_in_while_loop(device=host_device, op_fn=enqueue_ops_fn))
                        else:
                            enqueue_ops.append(enqueue_ops_fn())
                        infeed_queues.append(captured_infeed_queue.get())
        elif self._ctx.is_input_broadcast_with_iterators():
            host_device = tpu_host_placement_fn(host_id=0)
            (enqueue_ops_fn, captured_infeed_queue, dataset_initializer) = generate_broadcast_enqueue_ops_fn(self._ctx, self._input_fn, self._inputs_structure_recorder, num_hosts)
            if dataset_initializer:
                all_dataset_initializers.append(dataset_initializer)
                run_infeed_loop_on_coordinator = False
                wrap_fn = (_wrap_computation_in_while_loop if (self._ctx.mode != model_fn_lib.ModeKeys.PREDICT) else _wrap_computation_in_while_loop_with_stopping_signals)
                enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
            else:
                enqueue_ops.append(enqueue_ops_fn())
            infeed_queues.append(captured_infeed_queue.get())
        else:
            for host_id in range(num_hosts):
                host_device = tpu_host_placement_fn(host_id=host_id)
                with ops.device(host_device):
                    with ops.name_scope(('input_pipeline_task%d' % host_id)):
                        if self._ctx.is_input_per_host_with_iterators():
                            (enqueue_ops_fn, captured_infeed_queue, dataset_initializer) = generate_per_host_v2_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id)
                        else:
                            (enqueue_ops_fn, captured_infeed_queue, dataset_initializer) = generate_per_host_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, self._batch_axis, host_device, host_id)
                        if dataset_initializer:
                            all_dataset_initializers.append(dataset_initializer)
                            run_infeed_loop_on_coordinator = False
                            wrap_fn = (_wrap_computation_in_while_loop if (self._ctx.mode != model_fn_lib.ModeKeys.PREDICT) else _wrap_computation_in_while_loop_with_stopping_signals)
                            enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
                        else:
                            enqueue_ops.append(enqueue_ops_fn())
                        infeed_queues.append(captured_infeed_queue.get())
        self._infeed_queue = infeed_queues[0]
        return (enqueue_ops, [util_lib.MultiHostDatasetInitializerHook(all_dataset_initializers)], run_infeed_loop_on_coordinator)

    def _validate_input_pipeline(self):
        'Validates the input pipeline.\n\n    Perform some sanity checks to log user friendly information. We should\n    error out to give users better error message. But, if\n    _WRAP_INPUT_FN_INTO_WHILE_LOOP is False (legacy behavior), we cannot break\n    user code, so, log a warning.\n\n    Raises:\n      RuntimeError: If the validation failed.\n    '
        if ops.get_default_graph().get_collection(ops.GraphKeys.QUEUE_RUNNERS):
            err_msg = 'Input pipeline contains one or more QueueRunners. It could be slow and not scalable. Please consider converting your input pipeline to use `tf.data` instead (see https://www.tensorflow.org/guide/datasets for instructions.'
            if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
                raise RuntimeError(err_msg)
            else:
                logging.warn(err_msg)

class _ModelFnWrapper(object):
    'A `model_fn` wrapper.\n\n  This makes calling model_fn on CPU and TPU easier and more consistent and\n  performs necessary check and mutation required by TPU training and evaluation.\n\n  In addition, this wrapper manages converting the `model_fn` to a single TPU\n  train and eval step.\n  '

    def __init__(self, model_fn, train_cache_fn, eval_cache_fn, config, params, ctx):
        self._model_fn = model_fn
        self._train_cache_fn = train_cache_fn
        self._eval_cache_fn = eval_cache_fn
        self._config = config
        self._params = params
        self._ctx = ctx

    def call_without_tpu(self, features, labels, is_export_mode):
        return self._call_model_fn(features, labels, is_export_mode=is_export_mode)

    def convert_to_single_tpu_train_step(self, dequeue_fn):
        'Converts user provided model_fn` as a single train step on TPU.\n\n    The user provided `model_fn` takes input tuple\n    (features, labels) and produces the EstimatorSpec with train_op and loss for\n    train `mode`. This usually represents a single train computation on CPU.\n\n    For TPU training, a train (computation) step is first wrapped in a\n    tf.while_loop control flow to repeat for many times and then replicated to\n    all TPU shards. Besides the input should be taken from TPU infeed rather\n    than input pipeline (input_fn) directly. To fit TPU loop and replicate\n    pattern, the original train computation should be reformed, which is the\n    returned `train_step`.\n\n    Args:\n      dequeue_fn: The function to retrieve inputs, features and labels, from TPU\n        infeed dequeue channel.\n\n    Returns:\n      A tuple of train_fn, host_calls, and captured scaffold_fn. The train_fn\n      representing the train step for TPU.\n    '
        host_call = _OutfeedHostCall(self._ctx)
        captured_scaffold_fn = _CapturedObject()
        captured_training_hooks = _CapturedObject()

        def train_step(loss, *cache):
            'Training step function for use inside a while loop.'
            del loss
            inputs = dequeue_fn()
            (features, labels) = inputs.features_and_labels()
            estimator_spec = self._verify_estimator_spec(self._call_model_fn(features, labels, cache=cache))
            '\n        `cache` consists of a list of tensors, potentially empty (of length 0)\n      '
            cache = estimator_spec.cache
            (loss, train_op) = (estimator_spec.loss, estimator_spec.train_op)
            if isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec):
                captured_scaffold_fn.capture(estimator_spec.scaffold_fn)
            else:
                captured_scaffold_fn.capture(None)
            captured_training_hooks.capture(estimator_spec.training_hooks)
            tracing_ops = []
            if tensor_tracer.TensorTracer.is_enabled():
                tt = tensor_tracer.TensorTracer()
                (loss, tracing_ops) = tt.trace_tpu(ops.get_default_graph(), loss, self._ctx.num_replicas)
            with ops.control_dependencies(([train_op] + tracing_ops)):
                host_call_outfeed_ops = []
                if (isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec) and (estimator_spec.host_call is not None)):
                    host_call.record({'host_call': estimator_spec.host_call})
                    host_call_outfeed_ops = host_call.create_enqueue_op()
                with ops.control_dependencies(host_call_outfeed_ops):
                    return ([array_ops.identity(loss)] + cache)
        return (train_step, host_call, captured_scaffold_fn, captured_training_hooks)

    def convert_to_single_tpu_eval_step(self, dequeue_fn):
        'Converts user provided model_fn` as a single eval step on TPU.\n\n    Similar to training, the user provided `model_fn` takes input tuple\n    (features, labels) and produces the TPUEstimatorSpec with eval_metrics for\n    eval `mode`. This usually represents a single evaluation computation on CPU.\n\n    For TPU evaluation, a eval (computation) step is first wrapped in a\n    tf.while_loop control flow to repeat for many times and then replicated to\n    all TPU shards. Besides the input and output are slightly different. Input,\n    features and labels, should be taken from TPU infeed rather than input\n    pipeline (input_fn) directly. Output is managed in two stages.  First, the\n    model outputs as the result of evaluation computation, usually model logits,\n    should be transferred from TPU system to CPU. Then, all model outputs are\n    concatenated first on CPU and sent to the metric_fn for metrics computation.\n    To fit TPU evaluation pattern, the original eval computation should be\n    reformed, which is the returned `eval_step`.\n\n    Args:\n      dequeue_fn: The function to retrieve inputs, features and labels, from TPU\n        infeed dequeue channel.\n\n    Returns:\n      A tuple of eval_fn, host_calls, and captured scaffold_fn. The eval_fn\n      representing the eval step for TPU.\n    '
        host_calls = _OutfeedHostCall(self._ctx)
        captured_scaffold_fn = _CapturedObject()
        captured_eval_hooks = _CapturedObject()

        def eval_step(total_loss, *cache):
            'Evaluation step function for use inside a while loop.'
            inputs = dequeue_fn()
            (features, labels) = inputs.features_and_labels()
            tpu_estimator_spec = self._call_model_fn(features, labels, cache=cache)
            if (not isinstance(tpu_estimator_spec, model_fn_lib._TPUEstimatorSpec)):
                raise RuntimeError('estimator_spec used by TPU evaluation must have type`TPUEstimatorSpec`. Got {}'.format(type(tpu_estimator_spec)))
            cache = tpu_estimator_spec.cache
            loss = tpu_estimator_spec.loss
            captured_scaffold_fn.capture(tpu_estimator_spec.scaffold_fn)
            captured_eval_hooks.capture(tpu_estimator_spec.evaluation_hooks)
            to_record = {}
            if tpu_estimator_spec.eval_metrics:
                to_record['eval_metrics'] = tpu_estimator_spec.eval_metrics
            if (tpu_estimator_spec.host_call is not None):
                to_record['host_call'] = tpu_estimator_spec.host_call
            host_calls.record(to_record)
            with ops.control_dependencies(host_calls.create_enqueue_op()):
                return ([math_ops.add(total_loss, loss)] + cache)
        return (eval_step, host_calls, captured_scaffold_fn, captured_eval_hooks)

    def convert_to_single_tpu_predict_step(self, dequeue_fn):
        'Converts user provided model_fn` as a single predict step on TPU.\n\n    Args:\n      dequeue_fn: The function to retrieve inputs, features and labels, from TPU\n        infeed dequeue channel.\n\n    Returns:\n      A tuple of predict_fn, host_calls, and captured scaffold_fn. The\n      predict_fn representing the predict step for TPU.\n    '
        host_calls = _OutfeedHostCall(self._ctx)
        captured_scaffold_fn = _CapturedObject()
        captured_predict_hooks = _CapturedObject()

        def predict_step(unused_scalar_stopping_signal):
            'Evaluation step function for use inside a while loop.'
            inputs = dequeue_fn()
            (features, labels) = inputs.features_and_labels()
            stopping_signals = inputs.signals()
            assert (stopping_signals is not None), 'Internal Error: `signals` is missing.'
            tpu_estimator_spec = self._call_model_fn(features, labels, is_export_mode=False)
            if (not isinstance(tpu_estimator_spec, model_fn_lib._TPUEstimatorSpec)):
                raise RuntimeError('estimator_spec used by TPU prediction must have type`TPUEstimatorSpec`. Got {}'.format(type(tpu_estimator_spec)))
            self._verify_tpu_spec_predictions(tpu_estimator_spec.predictions)
            captured_scaffold_fn.capture(tpu_estimator_spec.scaffold_fn)
            captured_predict_hooks.capture(tpu_estimator_spec.prediction_hooks)
            to_record = {}
            identity_fn = (lambda **kwargs: kwargs)
            to_record['predictions'] = [identity_fn, tpu_estimator_spec.predictions]
            to_record['signals'] = [identity_fn, stopping_signals]
            if (tpu_estimator_spec.host_call is not None):
                to_record['host_call'] = tpu_estimator_spec.host_call
            host_calls.record(to_record)
            with ops.control_dependencies(host_calls.create_enqueue_op()):
                return _StopSignals.as_scalar_stopping_signal(stopping_signals)
        return (predict_step, host_calls, captured_scaffold_fn, captured_predict_hooks)

    def _verify_tpu_spec_predictions(self, predictions):
        'Validates TPUEstimatorSpec.predictions dict.'
        if (not isinstance(predictions, dict)):
            raise TypeError('TPUEstimatorSpec.predictions must be dict of Tensors.')
        for (key, tensor) in predictions.items():
            if (tensor.shape.dims[0].value is None):
                raise ValueError('The tensor with key ({}) in TPUEstimatorSpec.predictions has dynamic shape (should be static). Tensor: {}'.format(key, tensor))
        return predictions

    def _validate_model_features_and_labels(self, features, labels, is_export_mode):
        'Validates that the features and labels for the model function are valid.\n\n    A valid features/labels object is the one with:\n    - Type: A tensor or any nested structure of tensors supported by TF nest,\n        namely nested dictionary, tuple, namedtuple, or sequence of tensors.\n    - Static shape if is_export_mode is False.\n\n    Args:\n      features: the features that would be input to the model function.\n      labels: the labels that would be input to the model function.\n      is_export_mode: boolean value specifying if in export mode.\n\n    Raises:\n      TypeError: If features/labels are not of the correct type.\n      ValueError: If features/labels have dynamic shape.\n    '

        def validate(obj, obj_name):
            'Helper validate function.'
            if (is_export_mode or self._ctx.is_running_on_cpu(is_export_mode)):
                return
            if isinstance(obj, ops.Tensor):
                if (not obj.get_shape().is_fully_defined()):
                    raise ValueError('The {} to the model returned by input_fn must have static shape. Tensor: {}'.format(obj_name, obj))
            else:
                for tensor in data_nest.flatten(obj):
                    if (not tensor.get_shape().is_fully_defined()):
                        raise ValueError('The {} to the model returned by input_fn must have static shape. Tensor: {}'.format(obj_name, tensor))
        validate(features, 'features')
        if (labels is not None):
            validate(labels, 'labels')

    def _call_model_fn(self, features, labels, cache=None, is_export_mode=False):
        'Calls the model_fn with required parameters.'
        self._validate_model_features_and_labels(features, labels, is_export_mode)
        model_fn_args = function_utils.fn_args(self._model_fn)
        kwargs = {}
        config = copy.deepcopy(self._config)
        params = copy.deepcopy(self._params)
        if ('labels' in model_fn_args):
            kwargs['labels'] = labels
        elif (labels is not None):
            raise ValueError('model_fn does not take labels, but input_fn returns labels.')
        if ('mode' in model_fn_args):
            kwargs['mode'] = self._ctx.mode
        if ('config' in model_fn_args):
            kwargs['config'] = config
        if ('params' in model_fn_args):
            kwargs['params'] = params
        if (cache is not None):
            params['cache'] = cache
        if ('params' not in model_fn_args):
            raise ValueError("model_fn ({}) does not include params argument, required by TPUEstimator to pass batch size as params['batch_size']".format(self._model_fn))
        if is_export_mode:
            batch_size_for_model_fn = None
        else:
            batch_size_for_model_fn = self._ctx.batch_size_for_model_fn
        if (batch_size_for_model_fn is not None):
            _add_item_to_params(params, _BATCH_SIZE_KEY, batch_size_for_model_fn)
        running_on_cpu = self._ctx.is_running_on_cpu(is_export_mode)
        _add_item_to_params(params, _USE_TPU_KEY, (not running_on_cpu))
        if (not running_on_cpu):
            user_context = tpu_context.TPUContext(internal_ctx=self._ctx, call_from_input_fn=False)
            _add_item_to_params(params, _CTX_KEY, user_context)
        estimator_spec = self._model_fn(features=features, **kwargs)
        if (running_on_cpu and isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec)):
            return estimator_spec.as_estimator_spec()
        else:
            return estimator_spec

    def _verify_estimator_spec(self, estimator_spec):
        'Validates the estimator_spec.'
        if isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec):
            return estimator_spec
        err_msg = '{} returned by EstimatorSpec is not supported in TPUEstimator.'
        if estimator_spec.training_chief_hooks:
            raise ValueError(((err_msg.format('training_chief_hooks') + 'If you want') + ' to pass training hooks, please pass via training_hooks.'))
        if estimator_spec.scaffold:
            logging.warning('EstimatorSpec.Scaffold is ignored by TPU train/eval. Please use TPUEstimatorSpec.')
        return estimator_spec

class _OutfeedHostCall(object):
    'Support for `eval_metrics` and `host_call` in TPUEstimatorSpec.'

    def __init__(self, ctx):
        self._ctx = ctx
        self._names = []
        self._host_fns = {}
        self._tensor_keys = collections.defaultdict(list)
        self._tensors = collections.defaultdict(list)
        self._tensor_dtypes = collections.defaultdict(list)
        self._tensor_shapes = collections.defaultdict(list)

    @staticmethod
    def validate(host_calls):
        'Validates the `eval_metrics` and `host_call` in `TPUEstimatorSpec`.'
        for (name, host_call) in host_calls.items():
            if (not isinstance(host_call, (tuple, list))):
                raise ValueError('{} should be tuple or list'.format(name))
            if (len(host_call) != 2):
                raise ValueError('{} should have two elements.'.format(name))
            if (not callable(host_call[0])):
                raise TypeError('{}[0] should be callable.'.format(name))
            if (not isinstance(host_call[1], (tuple, list, dict))):
                raise ValueError('{}[1] should be tuple or list, or dict.'.format(name))
            if isinstance(host_call[1], (tuple, list)):
                fullargspec = tf_inspect.getfullargspec(host_call[0])
                fn_args = function_utils.fn_args(host_call[0])
                if ((fullargspec.varargs is None) and (len(host_call[1]) != len(fn_args))):
                    raise RuntimeError('In TPUEstimatorSpec.{}, length of tensors {} does not match method args of the function, which takes {}.'.format(name, len(host_call[1]), len(fn_args)))

    @staticmethod
    def create_cpu_hostcall(host_calls):
        'Runs on the host_call on CPU instead of TPU when use_tpu=False.'
        _OutfeedHostCall.validate(host_calls)
        ret = {}
        for (name, host_call) in host_calls.items():
            (host_fn, tensors) = host_call
            if isinstance(tensors, (tuple, list)):
                ret[name] = host_fn(*tensors)
            else:
                try:
                    ret[name] = host_fn(**tensors)
                except TypeError as e:
                    logging.warning("Exception while calling %s: %s. It is likely the tensors (%s[1]) do not match the function's arguments", name, e, name)
                    raise e
        return ret

    def record(self, host_calls):
        'Records the host_call structure.'
        for (name, host_call) in host_calls.items():
            (host_fn, tensor_list_or_dict) = host_call
            self._names.append(name)
            self._host_fns[name] = host_fn
            if isinstance(tensor_list_or_dict, dict):
                for (key, tensor) in six.iteritems(tensor_list_or_dict):
                    self._tensor_keys[name].append(key)
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)
            else:
                self._tensor_keys[name] = None
                for tensor in tensor_list_or_dict:
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)

    def create_enqueue_op(self):
        'Create the op to enqueue the recorded host_calls.\n\n    Returns:\n      A list of enqueue ops, which is empty if there are no host calls.\n    '
        if (not self._names):
            return []
        tensors = []
        for name in self._names:
            tensors.extend(self._tensors[name])
        with ops.device(tpu.core(0)):
            return [tpu_ops.outfeed_enqueue_tuple(tensors)]

    def create_tpu_hostcall(self):
        'Sends the tensors through outfeed and runs the host_fn on CPU.\n\n    The tensors are concatenated along dimension 0 to form a global tensor\n    across all shards. The concatenated function is passed to the host_fn and\n    executed on the first host.\n\n    Returns:\n      A dictionary mapping name to the return type of the host_call by that\n      name.\n\n    Raises:\n      RuntimeError: If outfeed tensor is scalar.\n    '
        if (not self._names):
            return {}
        ret = {}
        dequeue_ops = []
        tensor_dtypes = []
        tensor_shapes = []
        for name in self._names:
            for _ in self._tensors[name]:
                dequeue_ops.append([])
            for dtype in self._tensor_dtypes[name]:
                tensor_dtypes.append(dtype)
            for shape in self._tensor_shapes[name]:
                tensor_shapes.append(shape)
        for i in xrange(self._ctx.num_replicas):
            (host_device, ordinal_id) = self._ctx.device_for_replica(i)
            with ops.device(host_device):
                outfeed_tensors = tpu_ops.outfeed_dequeue_tuple(dtypes=tensor_dtypes, shapes=tensor_shapes, device_ordinal=ordinal_id)
                for (j, item) in enumerate(outfeed_tensors):
                    dequeue_ops[j].append(item)
        dequeue_ops_by_name = {}
        pos = 0
        for name in self._names:
            dequeue_ops_by_name[name] = dequeue_ops[pos:(pos + len(self._tensors[name]))]
            pos += len(self._tensors[name])
        with ops.device(self._ctx.tpu_host_placement_function(replica_id=0)):
            for name in self._names:
                dequeue_ops = dequeue_ops_by_name[name]
                for (i, item) in enumerate(dequeue_ops):
                    if (dequeue_ops[i][0].shape.ndims == 0):
                        raise RuntimeError('All tensors outfed from TPU should preserve batch size dimension, but got scalar {}'.format(dequeue_ops[i][0]))
                    dequeue_ops[i] = array_ops.concat(dequeue_ops[i], axis=0)
                if (self._tensor_keys[name] is not None):
                    dequeue_ops = dict(zip(self._tensor_keys[name], dequeue_ops))
                    try:
                        ret[name] = self._host_fns[name](**dequeue_ops)
                    except TypeError as e:
                        logging.warning("Exception while calling %s: %s. It is likely the tensors (%s[1]) do not match the function's arguments", name, e, name)
                        raise e
                else:
                    ret[name] = self._host_fns[name](*dequeue_ops)
        return ret

class _OutfeedHostCallHook(session_run_hook.SessionRunHook):
    'Hook to run host calls when use_tpu=False.'

    def __init__(self, tensors):
        self._tensors = tensors

    def begin(self):
        self._init_ops = contrib_summary.summary_writer_initializer_op()
        self._finalize_ops = []
        for op in self._init_ops:
            self._finalize_ops.append(contrib_summary.flush(writer=op.inputs[0]))

    def after_create_session(self, session, coord):
        session.run(self._init_ops)

    def before_run(self, run_context):
        return basic_session_run_hooks.SessionRunArgs(self._tensors)

    def end(self, session):
        session.run(self._finalize_ops)

class ExamplesPerSecondHook(basic_session_run_hooks.StepCounterHook):
    'Calculate and report global_step/sec and examples/sec during runtime.'

    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None):
        self._batch_size = batch_size
        super(ExamplesPerSecondHook, self).__init__(every_n_steps=every_n_steps, every_n_secs=every_n_secs, output_dir=output_dir, summary_writer=summary_writer)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        global_step_per_sec = (elapsed_steps / elapsed_time)
        examples_per_sec = (self._batch_size * global_step_per_sec)
        if (self._summary_writer is not None):
            global_step_summary = Summary(value=[Summary.Value(tag='global_step/sec', simple_value=global_step_per_sec)])
            example_summary = Summary(value=[Summary.Value(tag='examples/sec', simple_value=examples_per_sec)])
            self._summary_writer.add_summary(global_step_summary, global_step)
            self._summary_writer.add_summary(example_summary, global_step)
        logging.info('global_step/sec: %g', global_step_per_sec)
        logging.info('examples/sec: %g', examples_per_sec)

class InstallSignalHandlerHook(session_run_hook.SessionRunHook):
    'Change SIGINT (CTRL^C) handler to force quit the process.\n\n  The default behavior often results in hanging processes.\n  The original handler is restored after training/evaluation.\n  '

    def __init__(self):
        self._signal_fn = signal.getsignal(signal.SIGINT)

    def before_run(self, run_context):
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def end(self, session):
        signal.signal(signal.SIGINT, self._signal_fn)

class TPUEstimator(estimator_lib.Estimator):
    "Estimator with TPU support.\n\n  TPUEstimator also supports training on CPU and GPU. You don't need to define\n  a separate `tf.estimator.Estimator`.\n\n  TPUEstimator handles many of the details of running on TPU devices, such as\n  replicating inputs and models for each core, and returning to host\n  periodically to run hooks.\n\n  TPUEstimator transforms a global batch size in params to a per-shard batch\n  size when calling the `input_fn` and `model_fn`. Users should specify\n  global batch size in constructor, and then get the batch size for each shard\n  in `input_fn` and `model_fn` by `params['batch_size']`.\n\n  - For training, `model_fn` gets per-core batch size; `input_fn` may get\n    per-core or per-host batch size depending on `per_host_input_for_training`\n    in `TPUConfig` (See docstring for TPUConfig for details).\n\n  - For evaluation and prediction, `model_fn` gets per-core batch size and\n    `input_fn` get per-host batch size.\n\n  Evaluation\n  ==========\n\n  `model_fn` should return `TPUEstimatorSpec`, which expects the `eval_metrics`\n  for TPU evaluation. However, if eval_on_tpu is False, `model_fn` must return\n  `EstimatorSpec` and the evaluation will execute on CPU or GPU; in this case\n  the following discussion on TPU evaluation does not apply.\n\n  `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`, where\n  `tensors` could be a list of any nested structure of `Tensor`s (See\n  `TPUEstimatorSpec` for details).  `metric_fn` takes the `tensors` and returns\n  a dict from metric string name to the result of calling a metric function,\n  namely a `(metric_tensor, update_op)` tuple.\n\n  One can set `use_tpu` to `False` for testing. All training, evaluation, and\n  predict will be executed on CPU. `input_fn` and `model_fn` will receive\n  `train_batch_size` or `eval_batch_size` unmodified as `params['batch_size']`.\n\n  Current limitations:\n  --------------------\n\n  1. TPU evaluation only works on a single host (one TPU worker) except\n     BROADCAST mode.\n\n  2. `input_fn` for evaluation should **NOT** raise an end-of-input exception\n     (`OutOfRangeError` or `StopIteration`). And all evaluation steps and all\n     batches should have the same size.\n\n  Example (MNIST):\n  ----------------\n\n  ```\n  # The metric Fn which runs on CPU.\n  def metric_fn(labels, logits):\n    predictions = tf.argmax(logits, 1)\n    return {\n      'accuracy': tf.metrics.precision(\n          labels=labels, predictions=predictions),\n    }\n\n  # Your model Fn which runs on TPU (eval_metrics is list in this example)\n  def model_fn(features, labels, mode, config, params):\n    ...\n    logits = ...\n\n    if mode = tf.estimator.ModeKeys.EVAL:\n      return tpu_estimator.TPUEstimatorSpec(\n          mode=mode,\n          loss=loss,\n          eval_metrics=(metric_fn, [labels, logits]))\n\n  # or specify the eval_metrics tensors as dict.\n  def model_fn(features, labels, mode, config, params):\n    ...\n    final_layer_output = ...\n\n    if mode = tf.estimator.ModeKeys.EVAL:\n      return tpu_estimator.TPUEstimatorSpec(\n          mode=mode,\n          loss=loss,\n          eval_metrics=(metric_fn, {\n              'labels': labels,\n              'logits': final_layer_output,\n          }))\n  ```\n\n  Prediction\n  ==========\n\n  Prediction on TPU is an experimental feature to support large batch inference.\n  It is not designed for latency-critical system. In addition, due to some\n  usability issues, for prediction with small dataset, CPU `.predict`, i.e.,\n  creating a new `TPUEstimator` instance with `use_tpu=False`, might be more\n  convenient.\n\n  Note: In contrast to TPU training/evaluation, the `input_fn` for prediction\n  *should* raise an end-of-input exception (`OutOfRangeError` or\n  `StopIteration`), which serves as the stopping signal to `TPUEstimator`. To be\n  precise, the ops created by `input_fn` produce one batch of the data.\n  The `predict()` API processes one batch at a time. When reaching the end of\n  the data source, an end-of-input exception should be raised by one of these\n  operations. The user usually does not need to do this manually. As long as the\n  dataset is not repeated forever, the `tf.data` API will raise an end-of-input\n  exception automatically after the last batch has been produced.\n\n  Note: Estimator.predict returns a Python generator. Please consume all the\n  data from the generator so that TPUEstimator can shutdown the TPU system\n  properly for user.\n\n  Current limitations:\n  --------------------\n  1. TPU prediction only works on a single host (one TPU worker).\n\n  2. `input_fn` must return a `Dataset` instance rather than `features`. In\n  fact, .train() and .evaluate() also support Dataset as return value.\n\n  Example (MNIST):\n  ----------------\n  ```\n  height = 32\n  width = 32\n  total_examples = 100\n\n  def predict_input_fn(params):\n    batch_size = params['batch_size']\n\n    images = tf.random_uniform(\n        [total_examples, height, width, 3], minval=-1, maxval=1)\n\n    dataset = tf.data.Dataset.from_tensor_slices(images)\n    dataset = dataset.map(lambda images: {'image': images})\n\n    dataset = dataset.batch(batch_size)\n    return dataset\n\n  def model_fn(features, labels, params, mode):\n     # Generate predictions, called 'output', from features['image']\n\n    if mode == tf.estimator.ModeKeys.PREDICT:\n      return tf.contrib.tpu.TPUEstimatorSpec(\n          mode=mode,\n          predictions={\n              'predictions': output,\n              'is_padding': features['is_padding']\n          })\n\n  tpu_est = TPUEstimator(\n      model_fn=model_fn,\n      ...,\n      predict_batch_size=16)\n\n  # Fully consume the generator so that TPUEstimator can shutdown the TPU\n  # system.\n  for item in tpu_est.predict(input_fn=input_fn):\n    # Filter out item if the `is_padding` is 1.\n    # Process the 'predictions'\n  ```\n\n  Exporting\n  =========\n\n  `export_savedmodel` exports 2 metagraphs, one with `tag_constants.SERVING`,\n  and another with `tag_constants.SERVING` and `tag_constants.TPU`.\n  At serving time, these tags are used to select metagraph to load.\n\n  Before running the graph on TPU, TPU system needs to be initialized. If\n  TensorFlow Serving model-server is used, this is done automatically. If\n  not, please call `session.run(tpu.initialize_system())`.\n\n  `tpu.outside_compilation` can be used to wrap TPU incompatible ops in\n  `model_fn`.\n\n  Example:\n  ----------------\n\n  ```\n  def model_fn(features, labels, mode, config, params):\n    ...\n    logits = ...\n    export_outputs = {\n      'logits': export_output_lib.PredictOutput(\n        {'logits': logits})\n    }\n\n    def host_call(logits):\n      class_ids = math_ops.argmax(logits)\n      classes = string_ops.as_string(class_ids)\n      export_outputs['classes'] =\n        export_output_lib.ClassificationOutput(classes=classes)\n\n    tpu.outside_compilation(host_call, logits)\n\n    ...\n  ```\n\n  "

    def __init__(self, model_fn=None, train_cache_fn=None, eval_cache_fn=None, model_dir=None, config=None, params=None, use_tpu=True, train_batch_size=None, eval_batch_size=None, predict_batch_size=None, batch_axis=None, eval_on_tpu=True, export_to_tpu=True, warm_start_from=None):
        "Constructs an `TPUEstimator` instance.\n\n    Args:\n      model_fn: Model function as required by `Estimator` which returns\n        EstimatorSpec or TPUEstimatorSpec. `training_hooks`, 'evaluation_hooks',\n        and `prediction_hooks` must not capure any TPU Tensor inside the\n        model_fn.\n      model_dir: Directory to save model parameters, graph and etc. This can\n        also be used to load checkpoints from the directory into a estimator to\n        continue training a previously saved model. If `None`, the model_dir in\n        `config` will be used if set. If both are set, they must be same. If\n        both are `None`, a temporary directory will be used.\n      config: An `tpu_config.RunConfig` configuration object. Cannot be `None`.\n      params: An optional `dict` of hyper parameters that will be passed into\n        `input_fn` and `model_fn`.  Keys are names of parameters, values are\n        basic python types. There are reserved keys for `TPUEstimator`,\n        including 'batch_size'.\n      use_tpu: A bool indicating whether TPU support is enabled. Currently, -\n        TPU training and evaluation respect this bit, but eval_on_tpu can\n        override execution of eval. See below. - Predict still happens on CPU.\n      train_batch_size: An int representing the global training batch size.\n        TPUEstimator transforms this global batch size to a per-shard batch\n        size, as params['batch_size'], when calling `input_fn` and `model_fn`.\n        Cannot be `None` if `use_tpu` is `True`. Must be divisible by total\n        number of replicas.\n      eval_batch_size: An int representing evaluation batch size. Must be\n        divisible by total number of replicas.\n      predict_batch_size: An int representing the prediction batch size. Must be\n        divisible by total number of replicas.\n      batch_axis: A python tuple of int values describing how each tensor\n        produced by the Estimator `input_fn` should be split across the TPU\n        compute shards. For example, if your input_fn produced (images, labels)\n        where the images tensor is in `HWCN` format, your shard dimensions would\n        be [3, 0], where 3 corresponds to the `N` dimension of your images\n        Tensor, and 0 corresponds to the dimension along which to split the\n        labels to match up with the corresponding images. If None is supplied,\n        and per_host_input_for_training is True, batches will be sharded based\n        on the major dimension. If tpu_config.per_host_input_for_training is\n        False or `PER_HOST_V2`, batch_axis is ignored.\n      eval_on_tpu: If False, evaluation runs on CPU or GPU. In this case, the\n        model_fn must return `EstimatorSpec` when called with `mode` as `EVAL`.\n      export_to_tpu: If True, `export_savedmodel()` exports a metagraph for\n        serving on TPU besides the one on CPU.\n      warm_start_from: Optional string filepath to a checkpoint or SavedModel to\n        warm-start from, or a `tf.estimator.WarmStartSettings` object to fully\n        configure warm-starting.  If the string filepath is provided instead of\n        a `WarmStartSettings`, then all variables are warm-started, and it is\n        assumed that vocabularies and Tensor names are unchanged.\n\n    Raises:\n      ValueError: `params` has reserved keys already.\n    "
        if ((config is None) or (not isinstance(config, tpu_config.RunConfig))):
            raise ValueError('`config` must be provided with type `tpu_config.RunConfig`')
        if ((params is not None) and any(((k in params) for k in _RESERVED_PARAMS_KEYS))):
            raise ValueError('{} are reserved keys but existed in params {}.'.format(_RESERVED_PARAMS_KEYS, params))
        if use_tpu:
            if (train_batch_size is None):
                raise ValueError('`train_batch_size` cannot be `None`')
            util_lib.check_positive_integer(train_batch_size, 'train_batch_size')
            if ((config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.PER_SHARD_V1) and config.tpu_config.num_cores_per_replica):
                raise ValueError('Model parallelism only supports per host input for training. Please adjust TPURunconfig.per_host_input_for_training.')
            if (eval_batch_size is not None):
                util_lib.check_positive_integer(eval_batch_size, 'eval_batch_size')
            if (predict_batch_size is not None):
                util_lib.check_positive_integer(predict_batch_size, 'predict_batch_size')
        estimator_lib._verify_model_fn_args(model_fn, params)
        model_function = self._augment_model_fn(model_fn, train_cache_fn, eval_cache_fn, batch_axis)
        self._log_every_n_steps = config.log_step_count_steps
        config = config.replace(log_step_count_steps=None)
        params = (params or {})
        super(TPUEstimator, self).__init__(model_fn=model_function, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)
        self._iterations_per_training_loop = self._config.tpu_config.iterations_per_loop
        self._ctx = tpu_context._get_tpu_context(self._config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu, eval_on_tpu)
        self._export_to_tpu = export_to_tpu
        self._is_input_fn_invoked = None
        self._rendezvous = {}

    def _add_meta_graph_for_mode(self, builder, input_receiver_fn_map, checkpoint_path, save_variables=True, mode=model_fn_lib.ModeKeys.PREDICT, export_tags=None, check_variables=True):
        if (self._export_to_tpu and (mode != model_fn_lib.ModeKeys.PREDICT)):
            raise NotImplementedError('TPUEstimator only handles mode PREDICT for exporting when `export_to_tpu` is `True`; got {}.'.format(mode))
        super(TPUEstimator, self)._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables, mode=mode, export_tags=export_tags, check_variables=check_variables)
        if self._export_to_tpu:
            input_receiver_fn_map = {_REWRITE_FOR_INFERENCE_MODE: input_receiver_fn_map[mode]}
            export_tags = [tag_constants.SERVING, tag_constants.TPU]
            mode = _REWRITE_FOR_INFERENCE_MODE
            super(TPUEstimator, self)._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables=False, mode=mode, export_tags=export_tags, check_variables=False)

    def _call_model_fn(self, features, labels, mode, config):
        if (mode == _REWRITE_FOR_INFERENCE_MODE):
            return self._call_model_fn_for_inference(features, labels, mode, config)
        else:
            return super(TPUEstimator, self)._call_model_fn(features, labels, mode, config)

    def _call_model_fn_for_inference(self, features, labels, mode, config):
        'Wraps `_call_model_fn` for `export_savedmodel`.'
        if (mode != _REWRITE_FOR_INFERENCE_MODE):
            raise ValueError('mode must be {}; got {}.'.format(_REWRITE_FOR_INFERENCE_MODE, mode))
        capture = _CapturedObject()

        def computation():
            'Compute tpu tensors used in export_outputs.\n\n      Passed to rewrite_for_inference so that model_fn will be called under\n      the rewriting contexts. Only tpu tensors are returned, but export_outputs\n      and scaffold are captured.\n\n      Returns:\n         A list of Tensors used in export_outputs and not marked for\n         outside_compilation.\n      '
            mode = model_fn_lib.ModeKeys.PREDICT
            estimator_spec = self._call_model_fn(features, labels, mode, config)
            tensors_dict = collections.OrderedDict(((k, _export_output_to_tensors(v)) for (k, v) in six.iteritems(estimator_spec.export_outputs)))
            tensors = nest.flatten(tensors_dict)
            tpu_tensors = [t for t in tensors if (t is not None)]
            capture.capture((estimator_spec, tensors_dict, tensors))
            return tpu_tensors
        tpu_tensors_on_cpu = tpu.rewrite_for_inference(computation)
        (estimator_spec, tensors_dict, tensors) = capture.get()
        new_tensors = []
        for t in tensors:
            if (t is None):
                new_tensors.append(None)
            else:
                new_tensors.append(tpu_tensors_on_cpu.pop(0))
        new_tensors_dict = nest.pack_sequence_as(tensors_dict, new_tensors)
        export_outputs = estimator_spec.export_outputs
        new_export_outputs = collections.OrderedDict(((k, _clone_export_output_with_tensors(export_outputs[k], v)) for (k, v) in six.iteritems(new_tensors_dict)))
        return estimator_spec._replace(export_outputs=new_export_outputs)

    def _create_global_step(self, graph):
        'Creates a global step suitable for TPUs.\n\n    Args:\n      graph: The graph in which to create the global step.\n\n    Returns:\n      A global step `Tensor`.\n\n    Raises:\n      ValueError: if the global step tensor is already defined.\n    '
        return _create_global_step(graph)

    def _convert_train_steps_to_hooks(self, steps, max_steps):
        with self._ctx.with_mode(model_fn_lib.ModeKeys.TRAIN) as ctx:
            if ctx.is_running_on_cpu():
                return super(TPUEstimator, self)._convert_train_steps_to_hooks(steps, max_steps)
        if ((steps is None) and (max_steps is None)):
            raise ValueError('For TPU training, one of `steps` or `max_steps` must be set. Cannot be both `None`.')
        if (steps is not None):
            util_lib.check_positive_integer(steps, 'Train steps')
        if (max_steps is not None):
            util_lib.check_positive_integer(max_steps, 'Train max_steps')
        return [_TPUStopAtStepHook(self._iterations_per_training_loop, steps, max_steps)]

    def _convert_eval_steps_to_hooks(self, steps):
        with self._ctx.with_mode(model_fn_lib.ModeKeys.EVAL) as ctx:
            if ctx.is_running_on_cpu():
                return super(TPUEstimator, self)._convert_eval_steps_to_hooks(steps)
        if (steps is None):
            raise ValueError('Evaluate `steps` must be set on TPU. Cannot be `None`.')
        util_lib.check_positive_integer(steps, 'Eval steps')
        return [evaluation._StopAfterNEvalsHook(num_evals=steps), _SetEvalIterationsHook(steps)]

    def _call_input_fn(self, input_fn, mode):
        'Calls the input function.\n\n    Args:\n      input_fn: The input function.\n      mode: ModeKeys\n\n    Returns:\n      In TPU mode, returns an input_fn to be called later in model_fn.\n      Otherwise, calls the input_fn and returns either fatures or\n        (features, labels).\n\n    Raises:\n      ValueError: if input_fn takes invalid arguments or does not have `params`.\n    '
        input_fn_args = function_utils.fn_args(input_fn)
        config = self.config
        kwargs = {}
        if ('params' in input_fn_args):
            kwargs['params'] = self.params
        else:
            raise ValueError('input_fn ({}) does not include params argument, required by TPUEstimator to pass batch size as params["batch_size"]'.format(input_fn))
        if ('config' in input_fn_args):
            kwargs['config'] = config
        if ('mode' in input_fn_args):
            kwargs['mode'] = mode
        self._is_input_fn_invoked = True
        with self._ctx.with_mode(mode) as ctx:
            batch_size_for_input_fn = ctx.batch_size_for_input_fn
            if (batch_size_for_input_fn is not None):
                _add_item_to_params(kwargs['params'], _BATCH_SIZE_KEY, batch_size_for_input_fn)
            if ctx.is_running_on_cpu(is_export_mode=False):
                with ops.device('/device:CPU:0'):
                    return input_fn(**kwargs)

            def _input_fn(ctx):
                _add_item_to_params(kwargs['params'], _CTX_KEY, ctx)
                return input_fn(**kwargs)
            return _input_fn

    def _validate_features_in_predict_input(self, result):
        "Skip the validation.\n\n    For TPUEstimator, we do not need to check the result type. `_InputPipeline`\n    has stronger check. Parent class's check generates confusing warning msg.\n\n    Args:\n      result: `features` returned by input_fn.\n    "
        pass

    def train(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.TRAIN] = rendezvous
        try:
            return super(TPUEstimator, self).train(input_fn=input_fn, hooks=npu_hooks_append(hooks_list=hooks), steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
        except Exception:
            rendezvous.record_error('training_loop', sys.exc_info())
        finally:
            rendezvous.record_done('training_loop')
            rendezvous.raise_errors()

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.EVAL] = rendezvous
        try:
            return super(TPUEstimator, self).evaluate(input_fn, steps=steps, hooks=hooks, checkpoint_path=checkpoint_path, name=name)
        except Exception:
            rendezvous.record_error('evaluation_loop', sys.exc_info())
        finally:
            rendezvous.record_done('evaluation_loop')
            rendezvous.raise_errors()

    def predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.PREDICT] = rendezvous
        try:
            for result in super(TPUEstimator, self).predict(input_fn=input_fn, predict_keys=predict_keys, hooks=hooks, checkpoint_path=checkpoint_path, yield_single_examples=yield_single_examples):
                (yield result)
        except Exception:
            rendezvous.record_error('prediction_loop', sys.exc_info())
        finally:
            rendezvous.record_done('prediction_loop')
            rendezvous.raise_errors()
        rendezvous.record_done('prediction_loop')
        rendezvous.raise_errors()

    def _augment_model_fn(self, model_fn, train_cache_fn, eval_cache_fn, batch_axis):
        'Returns a new model_fn, which wraps the TPU support.'

        def _model_fn(features, labels, mode, config, params):
            'A Estimator `model_fn` for TPUEstimator.'
            with self._ctx.with_mode(mode) as ctx:
                model_fn_wrapper = _ModelFnWrapper(model_fn, train_cache_fn, eval_cache_fn, config, params, ctx)
                if self._is_input_fn_invoked:
                    is_export_mode = False
                else:
                    is_export_mode = True
                self._is_input_fn_invoked = None
                if (self._log_every_n_steps is not None):
                    examples_hook = ExamplesPerSecondHook(ctx.global_batch_size, output_dir=self.model_dir, every_n_steps=self._log_every_n_steps)
                if ctx.is_running_on_cpu(is_export_mode=is_export_mode):
                    logging.info('Running %s on CPU', mode)
                    estimator_spec = model_fn_wrapper.call_without_tpu(features, labels, is_export_mode=is_export_mode)
                    if (self._log_every_n_steps is not None):
                        estimator_spec = estimator_spec._replace(training_hooks=(estimator_spec.training_hooks + (examples_hook,)))
                    return estimator_spec
                assert (labels is None), '`labels` passed to `model_fn` must be `None`.'
                assert callable(features), '`input_fn` is not callable.'
                input_fn = features
                input_holders = _InputPipeline(input_fn, batch_axis, ctx)
                (enqueue_ops, dequeue_fn, input_hooks, run_infeed_loop_on_coordinator) = input_holders.generate_infeed_enqueue_ops_and_dequeue_fn()
                graph = ops.get_default_graph()
                for enqueue_op in enqueue_ops:
                    if isinstance(enqueue_op, list):
                        graph.get_collection_ref(_TPU_ENQUEUE_OPS).extend(enqueue_op)
                    else:
                        graph.add_to_collection(_TPU_ENQUEUE_OPS, enqueue_op)
                if (mode == model_fn_lib.ModeKeys.TRAIN):
                    (compile_op, loss, host_call, scaffold, training_hooks) = _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                    host_ops = host_call.create_tpu_hostcall()
                    if (host_ops is None):
                        host_ops = []
                    shutdown_hooks = []
                    shutdown_mode = os.environ.get('TF_TPU_GRACEFUL_SHUTDOWN_MODE', 'shutdown_worker')
                    if shutdown_mode:
                        if (shutdown_mode == 'shutdown_worker'):
                            finalizer_hooks = [session_support.ShutdownLameWorkers(timeout_ms=(60 * 1000))]
                        elif (shutdown_mode == 'shutdown_computation'):
                            finalizer_hooks = [session_support.RestartComputation(timeout_ms=(60 * 1000))]
                        else:
                            raise ValueError(('Unknown TF_TPU_GRACEFUL_SHUTDOWN_MODE "%s"' % shutdown_mode))
                        shutdown_hooks.append(session_support.GracefulShutdownHook(checkpoint_prefix=(self.model_dir + '/model.ckpt'), on_shutdown_hooks=finalizer_hooks))
                    with ops.control_dependencies([loss]):
                        global_step = array_ops.identity(training.get_global_step())
                    hooks = (input_hooks + shutdown_hooks)
                    hooks.extend([TPUInfeedOutfeedSessionHook(ctx, enqueue_ops, host_ops, tpu_compile_op=compile_op, run_infeed_loop_on_coordinator=run_infeed_loop_on_coordinator, rendezvous=self._rendezvous[mode], master=self._config.master, session_config=self._session_config), InstallSignalHandlerHook()])
                    if (self._log_every_n_steps is not None):
                        logging_hook_frequency = (((self._log_every_n_steps + self._config.tpu_config.iterations_per_loop) - 1) // self._config.tpu_config.iterations_per_loop)
                        hooks.append(training.LoggingTensorHook({'loss': array_ops.identity(loss), 'step': global_step}, every_n_iter=logging_hook_frequency))
                        examples_hook._set_steps_per_run(self._config.tpu_config.iterations_per_loop)
                        hooks.append(examples_hook)
                    if training_hooks:
                        hooks.extend(training_hooks)
                    chief_hooks = []
                    if (self._config.save_checkpoints_secs or self._config.save_checkpoints_steps):
                        checkpoint_hook = training.CheckpointSaverHook(self.model_dir, save_secs=self._config.save_checkpoints_secs, save_steps=self._config.save_checkpoints_steps, scaffold=scaffold)
                        checkpoint_hook._set_steps_per_run(self._config.tpu_config.iterations_per_loop)
                        chief_hooks.append(checkpoint_hook)
                    summary.scalar(model_fn_lib.LOSS_METRIC_KEY, loss)
                    with ops.control_dependencies([loss]):
                        update_ops = _sync_variables_ops(ctx)
                    _validate_tpu_training_graph()
                    train_op = control_flow_ops.group(*update_ops)
                    graph.add_to_collection(_TPU_TRAIN_OP, train_op)
                    return model_fn_lib.EstimatorSpec(mode, loss=loss, training_chief_hooks=chief_hooks, training_hooks=hooks, train_op=train_op, scaffold=scaffold)
                if (mode == model_fn_lib.ModeKeys.EVAL):
                    (compile_op, total_loss, host_calls, scaffold, eval_hooks) = _eval_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                    iterations_per_loop_var = _create_or_get_iterations_per_loop()
                    mean_loss = math_ops.div(total_loss, math_ops.cast(iterations_per_loop_var, dtype=total_loss.dtype))
                    with ops.control_dependencies([mean_loss]):
                        internal_ops_to_run = _sync_variables_ops(ctx)
                        internal_ops_to_run.append(_increase_eval_step_op(iterations_per_loop_var))
                    host_call_ret = host_calls.create_tpu_hostcall()
                    eval_metric_ops = {}
                    eval_update_ops = []
                    eval_metrics = host_call_ret.get('eval_metrics', {})
                    if eval_metrics:
                        with ops.control_dependencies(internal_ops_to_run):
                            dummy_update_op = control_flow_ops.no_op()
                        for (k, v) in eval_metrics.items():
                            eval_metric_ops[k] = (v[0], dummy_update_op)
                            eval_update_ops.append(v[1])
                    else:
                        with ops.control_dependencies(internal_ops_to_run):
                            mean_loss = array_ops.identity(mean_loss)
                    if ('host_call' not in host_call_ret):
                        host_ops = []
                    else:
                        host_ops = host_call_ret['host_call']
                    hooks = ([TPUInfeedOutfeedSessionHook(ctx, enqueue_ops, (eval_update_ops + host_ops), tpu_compile_op=compile_op, run_infeed_loop_on_coordinator=run_infeed_loop_on_coordinator, rendezvous=self._rendezvous[mode], master=self._config.evaluation_master, session_config=self._session_config)] + input_hooks)
                    if eval_hooks:
                        hooks.extend(eval_hooks)
                    return model_fn_lib.EstimatorSpec(mode, loss=mean_loss, evaluation_hooks=hooks, eval_metric_ops=eval_metric_ops, scaffold=scaffold)
                assert (mode == model_fn_lib.ModeKeys.PREDICT)
                (compile_op, dummy_predict_op, host_calls, scaffold, prediction_hooks) = _predict_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                with ops.control_dependencies([dummy_predict_op]):
                    internal_ops_to_run = _sync_variables_ops(ctx)
                    with ops.control_dependencies(internal_ops_to_run):
                        dummy_predict_op = control_flow_ops.no_op()
                enqueue_ops.append(dummy_predict_op)
                host_call_ret = host_calls.create_tpu_hostcall()
                if ('host_call' not in host_call_ret):
                    host_ops = []
                else:
                    host_ops = host_call_ret['host_call']
                predictions = host_call_ret['predictions']
                _verify_cross_hosts_transfer_size(predictions, message='The estimated size for TPUEstimatorSpec.predictions is too large.')
                signals = host_call_ret['signals']
                with ops.control_dependencies(host_ops):
                    host_ops = []
                    scalar_stopping_signal = _StopSignals.as_scalar_stopping_signal(signals)
                    predictions = _PaddingSignals.slice_tensor_or_dict(predictions, signals)
                hooks = ([_StoppingPredictHook(scalar_stopping_signal), TPUInfeedOutfeedSessionHookForPrediction(ctx, enqueue_ops, host_ops, rendezvous=self._rendezvous[mode], tpu_compile_op=compile_op, master=self._config.master, session_config=self._session_config)] + input_hooks)
                if prediction_hooks:
                    hooks.extend(prediction_hooks)
                return model_fn_lib.EstimatorSpec(mode, prediction_hooks=hooks, predictions=predictions, scaffold=scaffold)
        return _model_fn

def _export_output_to_tensors(export_output):
    'Get a list of `Tensors` used in `export_output`.\n\n  Args:\n    export_output: an `ExportOutput` object such as `ClassificationOutput`,\n      `RegressionOutput`, or `PredictOutput`.\n\n  Returns:\n    a list of tensors used in export_output.\n\n  Raises:\n    ValueError: if `export_output` is not one of `ClassificationOutput`,\n        `RegressionOutput`, or `PredictOutput`.\n  '
    if isinstance(export_output, export_output_lib.ClassificationOutput):
        return [export_output.scores, export_output.classes]
    elif isinstance(export_output, export_output_lib.RegressionOutput):
        return [export_output.value]
    elif isinstance(export_output, export_output_lib.PredictOutput):
        return list(export_output.outputs.values())
    else:
        raise ValueError('`export_output` must be have type `ClassificationOutput`, `RegressionOutput`, or `PredictOutput`; got {}.'.format(export_output))

def _clone_export_output_with_tensors(export_output, tensors):
    'Clones `export_output` but with new `tensors`.\n\n  Args:\n    export_output: an `ExportOutput` object such as `ClassificationOutput`,\n      `RegressionOutput`, or `PredictOutput`.\n    tensors: a list of `Tensors` used to construct a new `export_output`.\n\n  Returns:\n    A dict similar to `export_output` but with `tensors`.\n\n  Raises:\n    ValueError: if `export_output` is not one of `ClassificationOutput`,\n        `RegressionOutput`, or `PredictOutput`.\n  '
    if isinstance(export_output, export_output_lib.ClassificationOutput):
        if (len(tensors) != 2):
            raise ValueError('tensors must be of length 2; got {}.'.format(len(tensors)))
        return export_output_lib.ClassificationOutput(*tensors)
    elif isinstance(export_output, export_output_lib.RegressionOutput):
        if (len(tensors) != 1):
            raise ValueError('tensors must be of length 1; got {}'.format(len(tensors)))
        return export_output_lib.RegressionOutput(*tensors)
    elif isinstance(export_output, export_output_lib.PredictOutput):
        return export_output_lib.PredictOutput(dict(zip(export_output.outputs.keys(), tensors)))
    else:
        raise ValueError('`export_output` must be have type `ClassificationOutput`, `RegressionOutput`, or `PredictOutput`; got {}.'.format(export_output))

def _eval_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
    'Executes `model_fn_wrapper` multiple times on all TPU shards.'
    iterations_per_loop_var = _create_or_get_iterations_per_loop()
    (single_tpu_eval_step, host_calls, captured_scaffold_fn, captured_eval_hooks) = model_fn_wrapper.convert_to_single_tpu_eval_step(dequeue_fn)

    def multi_tpu_eval_steps_on_single_shard():
        loop_vars = [_ZERO_LOSS]
        if (model_fn_wrapper._eval_cache_fn is not None):
            batch_size = ctx.global_batch_size
            num_shards = ctx._config._tpu_config.num_shards
            loop_vars += model_fn_wrapper._eval_cache_fn((batch_size // num_shards))
        return training_loop.repeat(iterations_per_loop_var, single_tpu_eval_step, loop_vars)
    (compile_op, ret) = tpu.split_compile_and_shard(multi_tpu_eval_steps_on_single_shard, inputs=[], num_shards=ctx.num_replicas, outputs_from_all_shards=False, device_assignment=ctx.device_assignment)
    loss = ret[0]
    scaffold = _get_scaffold(captured_scaffold_fn)
    return (compile_op, loss, host_calls, scaffold, captured_eval_hooks.get())

def _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
    'Executes `model_fn_wrapper` multiple times on all TPU shards.'
    iterations_per_loop_var = _create_or_get_iterations_per_loop()
    (single_tpu_train_step, host_call, captured_scaffold_fn, captured_training_hooks) = model_fn_wrapper.convert_to_single_tpu_train_step(dequeue_fn)

    def multi_tpu_train_steps_on_single_shard():
        loop_vars = [_INITIAL_LOSS]
        if (model_fn_wrapper._train_cache_fn is not None):
            batch_size = ctx.global_batch_size
            num_shards = ctx._config._tpu_config.num_shards
            loop_vars += model_fn_wrapper._train_cache_fn((batch_size // num_shards))
        return training_loop.repeat(iterations_per_loop_var, single_tpu_train_step, loop_vars)
    (compile_op, ret) = tpu.split_compile_and_shard(multi_tpu_train_steps_on_single_shard, inputs=[], num_shards=ctx.num_replicas, outputs_from_all_shards=False, device_assignment=ctx.device_assignment)
    loss = ret[0]
    scaffold = _get_scaffold(captured_scaffold_fn)
    return (compile_op, loss, host_call, scaffold, captured_training_hooks.get())

def _predict_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
    'Executes `model_fn_wrapper` multiple times on all TPU shards.'
    (single_tpu_predict_step, host_calls, captured_scaffold_fn, captured_predict_hooks) = model_fn_wrapper.convert_to_single_tpu_predict_step(dequeue_fn)

    def multi_tpu_predict_steps_on_single_shard():

        def cond(scalar_stopping_signal):
            return math_ops.logical_not(_StopSignals.should_stop(scalar_stopping_signal))
        inputs = [_StopSignals.NON_STOPPING_SIGNAL]
        outputs = training_loop.while_loop(cond, single_tpu_predict_step, inputs=inputs, name=b'loop')
        return outputs
    (compile_op, dummy_predict_op) = tpu.split_compile_and_shard(multi_tpu_predict_steps_on_single_shard, inputs=[], num_shards=ctx.num_replicas, outputs_from_all_shards=False, device_assignment=ctx.device_assignment)
    dummy_predict_op = dummy_predict_op[0]
    scaffold = _get_scaffold(captured_scaffold_fn)
    return (compile_op, dummy_predict_op, host_calls, scaffold, captured_predict_hooks.get())

def _wrap_computation_in_while_loop(device, op_fn):
    'Wraps the ops generated by `op_fn` in tf.while_loop.'

    def computation(i):
        with ops.control_dependencies(op_fn()):
            return (i + 1)
    iterations_per_loop_var = _create_or_get_iterations_per_loop()
    with ops.device(device):
        iterations = array_ops.identity(iterations_per_loop_var)
        return control_flow_ops.while_loop((lambda i: (i < iterations)), computation, [constant_op.constant(0)], parallel_iterations=1)

def _wrap_computation_in_while_loop_with_stopping_signals(device, op_fn):
    'Wraps the ops generated by `op_fn` in tf.while_loop.'

    def cond(scalar_stopping_signal):
        return math_ops.logical_not(_StopSignals.should_stop(scalar_stopping_signal))

    def computation(unused_scalar_stopping_signal):
        return_value = op_fn()
        execute_ops = return_value['ops']
        signals = return_value['signals']
        with ops.control_dependencies(execute_ops):
            return _StopSignals.as_scalar_stopping_signal(signals)
    with ops.device(device):
        return control_flow_ops.while_loop(cond, computation, [_StopSignals.NON_STOPPING_SIGNAL], parallel_iterations=1)

def _validate_tpu_training_graph():
    'Validate graph before running distributed training.\n\n  Raises:\n    ValueError: If the graph seems invalid for running on device\n  '
    operations = ops.get_default_graph().get_operations()
    cross_replica_sum_ops = [o for o in operations if (o.type == _CROSS_REPLICA_SUM_OP)]
    if (not cross_replica_sum_ops):
        raise ValueError('CrossShardOptimizer must be used for model training on TPUs.')

class _CapturedObject(object):
    'A placeholder to capture an object.\n\n  This is useful when we need to capture a Python object in the Tensorflow\n  control flow body function and use it outside the control flow.\n  '

    def __init__(self):
        self._object = None
        self._captured = False

    def capture(self, o):
        if self._captured:
            raise RuntimeError('InternalError: Object can capture only once. Please file bug.')
        self._captured = True
        self._object = o

    def get(self):
        if (not self._captured):
            raise RuntimeError('InternalError: Object is not captured properly before `get`. Please file bug.')
        return self._object

def _get_scaffold(captured_scaffold_fn):
    'Retrieves the Scaffold from `captured_scaffold_fn`.'
    with _CapturingContext(message='Inside scaffold_fn'):
        scaffold_fn = captured_scaffold_fn.get()
        if scaffold_fn:
            scaffold = scaffold_fn()
            if (scaffold is None):
                raise ValueError('TPUEstimatorSpec.scaffold_fn returns None, which is not allowed')
        else:
            scaffold = None
    if scaffold:
        wrapped_finalize = scaffold.finalize

        def _finalize():
            with _CapturingContext('Inside Scaffold.finalize'):
                wrapped_finalize()
        scaffold.finalize = _finalize
    return scaffold

class _CapturingContext(control_flow_ops.ControlFlowContext):
    'Tracks references to Tensors defined in TPU replication.'

    def __init__(self, message):
        control_flow_ops.ControlFlowContext.__init__(self)
        self._message = message

    def to_control_flow_context_def(self, context_def, export_scope=None):
        super(_CapturingContext, self).to_control_flow_context_def(context_def, export_scope)

    def AddOp(self, op):
        for c in op.inputs:
            if (tpu._TPU_REPLICATE_ATTR in c.op.node_def.attr):
                raise ValueError('{}: Op {} depends on TPU computation {}, which is not allowed.'.format(self._message, op, c))

    def __enter__(self):
        self._g = ops.get_default_graph()
        self._old = self._g._get_control_flow_context()
        self._g._set_control_flow_context(self)

    def __exit__(self, _, __, ___):
        self._g._set_control_flow_context(self._old)

class _Inputs(object):
    'A data structure representing the input_fn returned values.\n\n  This also supports the returned value from input_fn as `Dataset`.\n  '

    def __init__(self, features=None, labels=None, dataset=None, signals=None):
        if ((dataset is not None) and ((features is not None) or (labels is not None) or (signals is not None))):
            raise RuntimeError('Internal Error: Either (features and labels) or dataset should be provided, not both. Please file bug')
        self._features = features
        self._labels = labels
        self._signals = signals
        self._dataset = dataset
        self._iterator = None

    @staticmethod
    def from_input_fn(return_values):
        'Returns an `_Inputs` instance according to `input_fn` return value.'
        if isinstance(return_values, dataset_ops.DatasetV2):
            dataset = return_values
            return _Inputs(dataset=dataset)
        (features, labels) = _Inputs._parse_inputs(return_values)
        return _Inputs(features, labels)

    @staticmethod
    def _parse_inputs(return_values):
        if isinstance(return_values, tuple):
            (features, labels) = return_values
        else:
            (features, labels) = (return_values, None)
        return (features, labels)

    @property
    def is_dataset(self):
        'Returns True if the return value from input_fn is Dataset.'
        return (self._dataset is not None)

    def dataset_initializer(self):
        "Returns the dataset's initializer.\n\n    The initializer must be run before calling `features_and_labels`.\n    "
        self._iterator = dataset_ops.make_initializable_iterator(self._dataset)
        return self._iterator.initializer

    def features_and_labels(self):
        'Gets `features` and `labels`.'
        if self.is_dataset:
            if (self._iterator is None):
                raise RuntimeError('Internal error: Must run dataset_initializer before calling features_and_labels(). Please file a bug!')
            return _Inputs._parse_inputs(self._iterator.get_next())
        return (self._features, self._labels)

    def signals(self):
        return self._signals

    @property
    def dataset(self):
        return self._dataset

class _InputsWithStoppingSignals(_Inputs):
    'Inputs with `_StopSignals` inserted into the dataset.'

    def __init__(self, dataset, batch_size, add_padding=False, num_invocations_per_step=1):
        assert (dataset is not None)
        user_provided_dataset = dataset.map(_InputsWithStoppingSignals.insert_stopping_signal(stop=False, batch_size=batch_size, add_padding=add_padding))
        if (num_invocations_per_step == 1):
            final_batch_dataset = dataset.take(1).map(_InputsWithStoppingSignals.insert_stopping_signal(stop=True, batch_size=batch_size, add_padding=add_padding))
        else:
            final_batch_dataset = dataset.take(1).map(_InputsWithStoppingSignals.insert_stopping_signal(stop=True, batch_size=batch_size, add_padding=add_padding))
            final_batch_dataset = final_batch_dataset.repeat(((2 * num_invocations_per_step) - 1))

            def _set_mask(data_dict):
                signals = data_dict['signals']
                signals['padding_mask'] = array_ops.ones_like(signals['padding_mask'])
                data_dict['signals'] = signals
                return data_dict
            final_batch_dataset = final_batch_dataset.map(_set_mask)
        dataset = user_provided_dataset.concatenate(final_batch_dataset).prefetch(2)
        super(_InputsWithStoppingSignals, self).__init__(dataset=dataset)
        self._current_inputs = None

    def features_and_labels(self):
        if (self._current_inputs is not None):
            raise RuntimeError('Internal Error: The previous inputs have not been properly consumed. First call features_and_labels, then call signals.')
        inputs_with_signals = self._iterator.get_next()
        features = inputs_with_signals['features']
        labels = inputs_with_signals.get('labels')
        self._current_inputs = inputs_with_signals
        return (features, labels)

    def signals(self):
        'Returns the `Signals` from `_Inputs`.'
        if (self._current_inputs is None):
            raise RuntimeError('Internal Error: The current inputs have not been properly generated. First call features_and_labels, then call signals.')
        signals = self._current_inputs['signals']
        self._current_inputs = None
        return signals

    @staticmethod
    def insert_stopping_signal(stop, batch_size, add_padding=False):
        'Inserts stopping_signal into dataset via _map_fn.\n\n    Here we change the data structure in the dataset, such that the return value\n    is a dictionary now and `features`, `labels`, and `signals` are three\n    distinguished keys in that dict. This provides a better structure, which\n    eases the process to decompose the inputs (see `features_and_labels`).\n\n    Args:\n      stop: bool, state of current stopping signals.\n      batch_size: int, batch size.\n      add_padding: bool, whether to pad the tensor to full batch size.\n\n    Returns:\n      A map_fn passed to dataset.map API.\n    '

        def _map_fn(*args):
            'The map fn to insert signals.'
            if (len(args) == 1):
                args = args[0]
            (features, labels) = _Inputs._parse_inputs(args)
            new_input_dict = {}
            if add_padding:
                (padding_mask, features, labels) = _PaddingSignals.pad_features_and_labels(features, labels, batch_size)
                new_input_dict['features'] = features
                if (labels is not None):
                    new_input_dict['labels'] = labels
            else:
                new_input_dict['features'] = features
                if (labels is not None):
                    new_input_dict['labels'] = labels
                padding_mask = None
            new_input_dict['signals'] = _StopSignals(stop=stop, batch_size=batch_size, padding_mask=padding_mask).as_dict()
            return new_input_dict
        return _map_fn

class _StopSignals(object):
    'Signals class holding all logic to handle TPU stopping condition.'
    NON_STOPPING_SIGNAL = False
    STOPPING_SIGNAL = True

    def __init__(self, stop, batch_size, padding_mask=None):
        self._stop = stop
        self._batch_size = batch_size
        self._padding_mask = padding_mask

    def as_dict(self):
        'Returns the signals as Python dict.'
        shape = [self._batch_size, 1]
        dtype = dtypes.bool
        if self._stop:
            stopping = array_ops.ones(shape=shape, dtype=dtype)
        else:
            stopping = array_ops.zeros(shape=shape, dtype=dtype)
        signals = {'stopping': stopping}
        if (self._padding_mask is not None):
            signals['padding_mask'] = self._padding_mask
        return signals

    @staticmethod
    def as_scalar_stopping_signal(signals):
        return array_ops.identity(signals['stopping'][0][0])

    @staticmethod
    def should_stop(scalar_stopping_signal):
        'Detects whether scalar_stopping_signal indicates stopping.'
        if isinstance(scalar_stopping_signal, ops.Tensor):
            return math_ops.logical_and(scalar_stopping_signal, _StopSignals.STOPPING_SIGNAL)
        else:
            return bool(scalar_stopping_signal)

class _PaddingSignals(object):
    'Signals class holding all logic to handle padding.'

    @staticmethod
    def pad_features_and_labels(features, labels, batch_size):
        'Pads out the batch dimension of features and labels.'
        real_batch_size = array_ops.shape(_PaddingSignals._find_any_tensor(features))[0]
        batch_size_tensor = constant_op.constant(batch_size, dtypes.int32)
        check_greater = check_ops.assert_greater_equal(batch_size_tensor, real_batch_size, data=(batch_size_tensor, real_batch_size), message='The real batch size should not be greater than batch_size.')
        with ops.control_dependencies([check_greater]):
            missing_count = (batch_size_tensor - real_batch_size)

        def pad_single_tensor(tensor):
            'Pads out the batch dimension of a tensor to the complete batch_size.'
            rank = len(tensor.shape)
            assert (rank > 0)
            padding = array_ops.stack(([[0, missing_count]] + ([[0, 0]] * (rank - 1))))
            padded_shape = ((batch_size,) + tuple(tensor.shape[1:]))
            padded_tensor = array_ops.pad(tensor, padding)
            padded_tensor.set_shape(padded_shape)
            return padded_tensor

        def nest_pad(tensor_or_dict):
            return nest.map_structure(pad_single_tensor, tensor_or_dict)
        features = nest_pad(features)
        if (labels is not None):
            labels = nest_pad(labels)
        padding_mask = _PaddingSignals._padding_mask(real_batch_size, missing_count, batch_size)
        return (padding_mask, features, labels)

    @staticmethod
    def slice_tensor_or_dict(tensor_or_dict, signals):
        'Slice the real Tensors according to padding mask in signals.'
        padding_mask = signals['padding_mask']
        batch_size = array_ops.shape(padding_mask)[0]

        def verify_batch_size(tensor):
            check_batch_size = math_ops.equal(batch_size, tensor.shape[0])
            with ops.control_dependencies([check_batch_size]):
                return array_ops.identity(tensor)

        def slice_single_tensor(tensor):
            rank = len(tensor.shape)
            assert (rank > 0)
            real_batch_size = (batch_size - math_ops.reduce_sum(padding_mask))
            return verify_batch_size(tensor)[0:real_batch_size]
        sliced_padding_mask = slice_single_tensor(padding_mask)
        assert_padding_mask = math_ops.equal(math_ops.reduce_sum(sliced_padding_mask), 0)
        with ops.control_dependencies([assert_padding_mask]):
            should_stop = _StopSignals.should_stop(_StopSignals.as_scalar_stopping_signal(signals))
        is_full_batch = math_ops.equal(math_ops.reduce_sum(padding_mask), 0)

        def slice_fn(tensor):
            return control_flow_ops.cond(math_ops.logical_or(should_stop, is_full_batch), (lambda : verify_batch_size(tensor)), (lambda : slice_single_tensor(tensor)))
        return nest.map_structure(slice_fn, tensor_or_dict)

    @staticmethod
    def _find_any_tensor(batch_features):
        tensors = [x for x in nest.flatten(batch_features) if isinstance(x, ops.Tensor)]
        if (not tensors):
            raise ValueError('Cannot find any Tensor in features dict.')
        return tensors[0]

    @staticmethod
    def _padding_mask(real_batch_size, missing_count, batch_size):
        padding_mask = array_ops.concat([array_ops.zeros((real_batch_size,), dtype=dtypes.int32), array_ops.ones((missing_count,), dtype=dtypes.int32)], axis=0)
        padding_mask.set_shape((batch_size,))
        return padding_mask

def _verify_cross_hosts_transfer_size(tensor_dict, message):
    total_size = 0
    tensor_structure = {}
    for (key, tensor) in tensor_dict.items():
        shape = tensor.shape
        size = (np.product(shape) * tensor.dtype.size)
        tensor_structure[key] = shape
        total_size += size
    if (total_size >= _ONE_GIGABYTE):
        raise ValueError('{} The transfer size is larger than the protobuf limit. Please consider to use Tensors with smaller shapes or reduce batch size. Given:\n{}'.format(message, '\n'.join([' -- Key: {}, Shape: {}'.format(k, v) for (k, v) in tensor_structure.items()])))

def _add_item_to_params(params, key, value):
    'Adds a new item into `params`.'
    if isinstance(params, hparam.HParams):
        if (key in params):
            params.set_hparam(key, value)
        else:
            params.add_hparam(key, value)
    else:
        params[key] = value

def export_estimator_savedmodel(estimator, export_dir_base, serving_input_receiver_fn, assets_extra=None, as_text=False, checkpoint_path=None, strip_default_attrs=False):
    'Export `Estimator` trained model for TPU inference.\n\n  Args:\n    estimator: `Estimator` with which model has been trained.\n    export_dir_base: A string containing a directory in which to create\n      timestamped subdirectories containing exported SavedModels.\n    serving_input_receiver_fn: A function that takes no argument and returns a\n      `ServingInputReceiver` or `TensorServingInputReceiver`.\n    assets_extra: A dict specifying how to populate the assets.extra directory\n      within the exported SavedModel, or `None` if no extra assets are needed.\n    as_text: whether to write the SavedModel proto in text format.\n    checkpoint_path: The checkpoint path to export.  If `None` (the default),\n      the most recent checkpoint found within the model directory is chosen.\n    strip_default_attrs: Boolean. If `True`, default-valued attributes will be\n      removed from the NodeDefs.\n\n  Returns:\n    The string path to the exported directory.\n  '
    config = tpu_config.RunConfig(model_dir=estimator.model_dir)
    est = TPUEstimator(estimator._model_fn, config=npu_run_config_init(run_config=config), params=estimator.params, use_tpu=True, train_batch_size=2048, eval_batch_size=2048)
    return est.export_savedmodel(export_dir_base, serving_input_receiver_fn, assets_extra, as_text, checkpoint_path, strip_default_attrs)
