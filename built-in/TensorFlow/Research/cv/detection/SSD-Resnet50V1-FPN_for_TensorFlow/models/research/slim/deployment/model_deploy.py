
'Deploy Slim models across multiple clones and replicas.\n\n# TODO(sguada) docstring paragraph by (a) motivating the need for the file and\n# (b) defining clones.\n\n# TODO(sguada) describe the high-level components of model deployment.\n# E.g. "each model deployment is composed of several parts: a DeploymentConfig,\n# which captures A, B and C, an input_fn which loads data.. etc\n\nTo easily train a model on multiple GPUs or across multiple machines this\nmodule provides a set of helper functions: `create_clones`,\n`optimize_clones` and `deploy`.\n\nUsage:\n\n  g = tf.Graph()\n\n  # Set up DeploymentConfig\n  config = model_deploy.DeploymentConfig(num_clones=2, clone_on_cpu=True)\n\n  # Create the global step on the device storing the variables.\n  with tf.device(config.variables_device()):\n    global_step = slim.create_global_step()\n\n  # Define the inputs\n  with tf.device(config.inputs_device()):\n    images, labels = LoadData(...)\n    inputs_queue = slim.data.prefetch_queue((images, labels))\n\n  # Define the optimizer.\n  with tf.device(config.optimizer_device()):\n    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)\n\n  # Define the model including the loss.\n  def model_fn(inputs_queue):\n    images, labels = inputs_queue.dequeue()\n    predictions = CreateNetwork(images)\n    slim.losses.log_loss(predictions, labels)\n\n  model_dp = model_deploy.deploy(config, model_fn, [inputs_queue],\n                                 optimizer=optimizer)\n\n  # Run training.\n  slim.learning.train(model_dp.train_op, my_log_dir,\n                      summary_op=model_dp.summary_op)\n\nThe Clone namedtuple holds together the values associated with each call to\nmodel_fn:\n  * outputs: The return values of the calls to `model_fn()`.\n  * scope: The scope used to create the clone.\n  * device: The device used to create the clone.\n\nDeployedModel namedtuple, holds together the values needed to train multiple\nclones:\n  * train_op: An operation that run the optimizer training op and include\n    all the update ops created by `model_fn`. Present only if an optimizer\n    was specified.\n  * summary_op: An operation that run the summaries created by `model_fn`\n    and process_gradients.\n  * total_loss: A `Tensor` that contains the sum of all losses created by\n    `model_fn` plus the regularization losses.\n  * clones: List of `Clone` tuples returned by `create_clones()`.\n\nDeploymentConfig parameters:\n  * num_clones: Number of model clones to deploy in each replica.\n  * clone_on_cpu: True if clones should be placed on CPU.\n  * replica_id: Integer.  Index of the replica for which the model is\n      deployed.  Usually 0 for the chief replica.\n  * num_replicas: Number of replicas to use.\n  * num_ps_tasks: Number of tasks for the `ps` job. 0 to not use replicas.\n  * worker_job_name: A name for the worker job.\n  * ps_job_name: A name for the parameter server job.\n\nTODO(sguada):\n  - describe side effect to the graph.\n  - what happens to summaries and update_ops.\n  - which graph collections are altered.\n  - write a tutorial on how to use this.\n  - analyze the possibility of calling deploy more than once.\n\n\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import collections
import tensorflow as tf
slim = tf.contrib.slim
__all__ = ['create_clones', 'deploy', 'optimize_clones', 'DeployedModel', 'DeploymentConfig', 'Clone']
Clone = collections.namedtuple('Clone', ['outputs', 'scope', 'device'])
DeployedModel = collections.namedtuple('DeployedModel', ['train_op', 'summary_op', 'total_loss', 'clones'])
_deployment_params = {'num_clones': 1, 'clone_on_cpu': False, 'replica_id': 0, 'num_replicas': 1, 'num_ps_tasks': 0, 'worker_job_name': 'worker', 'ps_job_name': 'ps'}

def create_clones(config, model_fn, args=None, kwargs=None):
    'Creates multiple clones according to config using a `model_fn`.\n\n  The returned values of `model_fn(*args, **kwargs)` are collected along with\n  the scope and device used to created it in a namedtuple\n  `Clone(outputs, scope, device)`\n\n  Note: it is assumed that any loss created by `model_fn` is collected at\n  the tf.GraphKeys.LOSSES collection.\n\n  To recover the losses, summaries or update_ops created by the clone use:\n  ```python\n    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)\n    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, clone.scope)\n    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone.scope)\n  ```\n\n  The deployment options are specified by the config object and support\n  deploying one or several clones on different GPUs and one or several replicas\n  of such clones.\n\n  The argument `model_fn` is called `config.num_clones` times to create the\n  model clones as `model_fn(*args, **kwargs)`.\n\n  If `config` specifies deployment on multiple replicas then the default\n  tensorflow device is set appropriatly for each call to `model_fn` and for the\n  slim variable creation functions: model and global variables will be created\n  on the `ps` device, the clone operations will be on the `worker` device.\n\n  Args:\n    config: A DeploymentConfig object.\n    model_fn: A callable. Called as `model_fn(*args, **kwargs)`\n    args: Optional list of arguments to pass to `model_fn`.\n    kwargs: Optional list of keyword arguments to pass to `model_fn`.\n\n  Returns:\n    A list of namedtuples `Clone`.\n  '
    clones = []
    args = (args or [])
    kwargs = (kwargs or {})
    with slim.arg_scope([slim.model_variable, slim.variable], device=config.variables_device()):
        for i in range(0, config.num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device('/cpu:0'):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(True if (i > 0) else None)):
                        outputs = model_fn(*args, **kwargs)
                    clones.append(Clone(outputs, clone_scope, clone_device))
    return clones

def _gather_clone_loss(clone, num_clones, regularization_losses):
    'Gather the loss for a single clone.\n\n  Args:\n    clone: A Clone namedtuple.\n    num_clones: The number of clones being deployed.\n    regularization_losses: Possibly empty list of regularization_losses\n      to add to the clone losses.\n\n  Returns:\n    A tensor for the total loss for the clone.  Can be None.\n  '
    sum_loss = None
    clone_loss = None
    regularization_loss = None
    with tf.device('/cpu:0'):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if (num_clones > 1):
                clone_loss = tf.div(clone_loss, (1.0 * num_clones), name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    if (clone_loss is not None):
        tf.summary.scalar('/'.join(filter(None, ['Losses', clone.scope, 'clone_loss'])), clone_loss)
    if (regularization_loss is not None):
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
    return sum_loss

def _optimize_clone(optimizer, clone, num_clones, regularization_losses, **kwargs):
    'Compute losses and gradients for a single clone.\n\n  Args:\n    optimizer: A tf.Optimizer  object.\n    clone: A Clone namedtuple.\n    num_clones: The number of clones being deployed.\n    regularization_losses: Possibly empty list of regularization_losses\n      to add to the clone losses.\n    **kwargs: Dict of kwarg to pass to compute_gradients().\n\n  Returns:\n    A tuple (clone_loss, clone_grads_and_vars).\n      - clone_loss: A tensor for the total loss for the clone.  Can be None.\n      - clone_grads_and_vars: List of (gradient, variable) for the clone.\n        Can be empty.\n  '
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if (sum_loss is not None):
        with tf.device('/cpu:0'):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return (sum_loss, clone_grad)

def optimize_clones(clones, optimizer, regularization_losses=None, **kwargs):
    'Compute clone losses and gradients for the given list of `Clones`.\n\n  Note: The regularization_losses are added to the first clone losses.\n\n  Args:\n   clones: List of `Clones` created by `create_clones()`.\n   optimizer: An `Optimizer` object.\n   regularization_losses: Optional list of regularization losses. If None it\n     will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to\n     exclude them.\n   **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.\n\n  Returns:\n   A tuple (total_loss, grads_and_vars).\n     - total_loss: A Tensor containing the average of the clone losses including\n       the regularization loss.\n     - grads_and_vars: A List of tuples (gradient, variable) containing the sum\n       of the gradients for each variable.\n\n  '
    grads_and_vars = []
    clones_losses = []
    num_clones = len(clones)
    if (regularization_losses is None):
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        with tf.name_scope(clone.scope):
            (clone_loss, clone_grad) = _optimize_clone(optimizer, clone, num_clones, regularization_losses, **kwargs)
            if (clone_loss is not None):
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)
            regularization_losses = None
    total_loss = tf.add_n(clones_losses, name='total_loss')
    grads_and_vars = _sum_clones_gradients(grads_and_vars)
    return (total_loss, grads_and_vars)

def deploy(config, model_fn, args=None, kwargs=None, optimizer=None, summarize_gradients=False):
    'Deploys a Slim-constructed model across multiple clones.\n\n  The deployment options are specified by the config object and support\n  deploying one or several clones on different GPUs and one or several replicas\n  of such clones.\n\n  The argument `model_fn` is called `config.num_clones` times to create the\n  model clones as `model_fn(*args, **kwargs)`.\n\n  The optional argument `optimizer` is an `Optimizer` object.  If not `None`,\n  the deployed model is configured for training with that optimizer.\n\n  If `config` specifies deployment on multiple replicas then the default\n  tensorflow device is set appropriatly for each call to `model_fn` and for the\n  slim variable creation functions: model and global variables will be created\n  on the `ps` device, the clone operations will be on the `worker` device.\n\n  Args:\n    config: A `DeploymentConfig` object.\n    model_fn: A callable. Called as `model_fn(*args, **kwargs)`\n    args: Optional list of arguments to pass to `model_fn`.\n    kwargs: Optional list of keyword arguments to pass to `model_fn`.\n    optimizer: Optional `Optimizer` object.  If passed the model is deployed\n      for training with that optimizer.\n    summarize_gradients: Whether or not add summaries to the gradients.\n\n  Returns:\n    A `DeployedModel` namedtuple.\n\n  '
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    clones = create_clones(config, model_fn, args, kwargs)
    first_clone = clones[0]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone.scope)
    train_op = None
    total_loss = None
    with tf.device('/cpu:0'):
        if optimizer:
            with tf.device('/cpu:0'):
                global_step = slim.get_or_create_global_step()
            (total_loss, clones_gradients) = optimize_clones(clones, optimizer)
            if clones_gradients:
                if summarize_gradients:
                    summaries |= set(_add_gradients_summaries(clones_gradients))
                grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_op = tf.identity(total_loss, name='train_op')
        else:
            clones_losses = []
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            for clone in clones:
                with tf.name_scope(clone.scope):
                    clone_loss = _gather_clone_loss(clone, len(clones), regularization_losses)
                    if (clone_loss is not None):
                        clones_losses.append(clone_loss)
                    regularization_losses = None
            if clones_losses:
                total_loss = tf.add_n(clones_losses, name='total_loss')
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone.scope))
        if (total_loss is not None):
            summaries.add(tf.summary.scalar('total_loss', total_loss))
        if summaries:
            summary_op = tf.summary.merge(list(summaries), name='summary_op')
        else:
            summary_op = None
    return DeployedModel(train_op, summary_op, total_loss, clones)

def _sum_clones_gradients(clone_grads):
    'Calculate the sum gradient for each shared variable across all clones.\n\n  This function assumes that the clone_grads has been scaled appropriately by\n  1 / num_clones.\n\n  Args:\n    clone_grads: A List of List of tuples (gradient, variable), one list per\n    `Clone`.\n\n  Returns:\n     List of tuples of (gradient, variable) where the gradient has been summed\n     across all clones.\n  '
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        for (g, v) in grad_and_vars:
            assert (v == var)
            if (g is not None):
                grads.append(g)
        if grads:
            if (len(grads) > 1):
                sum_grad = tf.add_n(grads, name=(var.op.name + '/sum_grads'))
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads

def _add_gradients_summaries(grads_and_vars):
    'Add histogram summaries to gradients.\n\n  Note: The summaries are also added to the SUMMARIES collection.\n\n  Args:\n    grads_and_vars: A list of gradient to variable pairs (tuples).\n\n  Returns:\n    The _list_ of the added summaries for grads_and_vars.\n  '
    summaries = []
    for (grad, var) in grads_and_vars:
        if (grad is not None):
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad
            summaries.append(tf.summary.histogram((var.op.name + ':gradient'), grad_values))
            summaries.append(tf.summary.histogram((var.op.name + ':gradient_norm'), tf.global_norm([grad_values])))
        else:
            tf.logging.info('Var %s has no gradient', var.op.name)
    return summaries

class DeploymentConfig(object):
    'Configuration for deploying a model with `deploy()`.\n\n  You can pass an instance of this class to `deploy()` to specify exactly\n  how to deploy the model to build.  If you do not pass one, an instance built\n  from the default deployment_hparams will be used.\n  '

    def __init__(self, num_clones=1, clone_on_cpu=False, replica_id=0, num_replicas=1, num_ps_tasks=0, worker_job_name='worker', ps_job_name='ps'):
        'Create a DeploymentConfig.\n\n    The config describes how to deploy a model across multiple clones and\n    replicas.  The model will be replicated `num_clones` times in each replica.\n    If `clone_on_cpu` is True, each clone will placed on CPU.\n\n    If `num_replicas` is 1, the model is deployed via a single process.  In that\n    case `worker_device`, `num_ps_tasks`, and `ps_device` are ignored.\n\n    If `num_replicas` is greater than 1, then `worker_device` and `ps_device`\n    must specify TensorFlow devices for the `worker` and `ps` jobs and\n    `num_ps_tasks` must be positive.\n\n    Args:\n      num_clones: Number of model clones to deploy in each replica.\n      clone_on_cpu: If True clones would be placed on CPU.\n      replica_id: Integer.  Index of the replica for which the model is\n        deployed.  Usually 0 for the chief replica.\n      num_replicas: Number of replicas to use.\n      num_ps_tasks: Number of tasks for the `ps` job. 0 to not use replicas.\n      worker_job_name: A name for the worker job.\n      ps_job_name: A name for the parameter server job.\n\n    Raises:\n      ValueError: If the arguments are invalid.\n    '
        if (num_replicas > 1):
            if (num_ps_tasks < 1):
                raise ValueError('When using replicas num_ps_tasks must be positive')
        if ((num_replicas > 1) or (num_ps_tasks > 0)):
            if (not worker_job_name):
                raise ValueError('Must specify worker_job_name when using replicas')
            if (not ps_job_name):
                raise ValueError('Must specify ps_job_name when using parameter server')
        if (replica_id >= num_replicas):
            raise ValueError('replica_id must be less than num_replicas')
        self._num_clones = num_clones
        self._clone_on_cpu = clone_on_cpu
        self._replica_id = replica_id
        self._num_replicas = num_replicas
        self._num_ps_tasks = num_ps_tasks
        self._ps_device = (('/job:' + ps_job_name) if (num_ps_tasks > 0) else '')
        self._worker_device = (('/job:' + worker_job_name) if (num_ps_tasks > 0) else '')

    @property
    def num_clones(self):
        return self._num_clones

    @property
    def clone_on_cpu(self):
        return self._clone_on_cpu

    @property
    def replica_id(self):
        return self._replica_id

    @property
    def num_replicas(self):
        return self._num_replicas

    @property
    def num_ps_tasks(self):
        return self._num_ps_tasks

    @property
    def ps_device(self):
        return self._ps_device

    @property
    def worker_device(self):
        return self._worker_device

    def caching_device(self):
        'Returns the device to use for caching variables.\n\n    Variables are cached on the worker CPU when using replicas.\n\n    Returns:\n      A device string or None if the variables do not need to be cached.\n    '
        if (self._num_ps_tasks > 0):
            return (lambda op: op.device)
        else:
            return None

    def clone_device(self, clone_index):
        'Device used to create the clone and all the ops inside the clone.\n\n    Args:\n      clone_index: Int, representing the clone_index.\n\n    Returns:\n      A value suitable for `tf.device()`.\n\n    Raises:\n      ValueError: if `clone_index` is greater or equal to the number of clones".\n    '
        if (clone_index >= self._num_clones):
            raise ValueError('clone_index must be less than num_clones')
        device = ''
        if (self._num_ps_tasks > 0):
            device += self._worker_device
        if self._clone_on_cpu:
            device += '/device:CPU:0'
        else:
            device += ('/device:GPU:%d' % clone_index)
        return device

    def clone_scope(self, clone_index):
        'Name scope to create the clone.\n\n    Args:\n      clone_index: Int, representing the clone_index.\n\n    Returns:\n      A name_scope suitable for `tf.name_scope()`.\n\n    Raises:\n      ValueError: if `clone_index` is greater or equal to the number of clones".\n    '
        if (clone_index >= self._num_clones):
            raise ValueError('clone_index must be less than num_clones')
        scope = ''
        if (self._num_clones > 1):
            scope = ('clone_%d' % clone_index)
        return scope

    def optimizer_device(self):
        'Device to use with the optimizer.\n\n    Returns:\n      A value suitable for `tf.device()`.\n    '
        if ((self._num_ps_tasks > 0) or (self._num_clones > 0)):
            return (self._worker_device + '/device:CPU:0')
        else:
            return ''

    def inputs_device(self):
        'Device to use to build the inputs.\n\n    Returns:\n      A value suitable for `tf.device()`.\n    '
        device = ''
        if (self._num_ps_tasks > 0):
            device += self._worker_device
        device += '/device:CPU:0'
        return device

    def variables_device(self):
        'Returns the device to use for variables created inside the clone.\n\n    Returns:\n      A value suitable for `tf.device()`.\n    '
        device = ''
        if (self._num_ps_tasks > 0):
            device += self._ps_device
        device += '/device:CPU:0'

        class _PSDeviceChooser(object):
            'Slim device chooser for variables when using PS.'

            def __init__(self, device, tasks):
                self._device = device
                self._tasks = tasks
                self._task = 0

            def choose(self, op):
                if op.device:
                    return op.device
                node_def = (op if isinstance(op, tf.NodeDef) else op.node_def)
                if node_def.op.startswith('Variable'):
                    t = self._task
                    self._task = ((self._task + 1) % self._tasks)
                    d = ('%s/task:%d' % (self._device, t))
                    return d
                else:
                    return op.device
        if (not self._num_ps_tasks):
            return device
        else:
            chooser = _PSDeviceChooser(device, self._num_ps_tasks)
            return chooser.choose
