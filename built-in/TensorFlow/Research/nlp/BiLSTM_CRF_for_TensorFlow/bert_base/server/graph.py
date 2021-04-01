
from npu_bridge.npu_init import *
import contextlib
import json
import os
from enum import Enum
from termcolor import colored
from .helper import import_tf, set_logger
import sys
sys.path.append('..')
from bert_base.bert import modeling
__all__ = ['PoolingStrategy', 'optimize_bert_graph', 'optimize_ner_model', 'optimize_class_model']

class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4
    LAST_TOKEN = 5
    CLS_TOKEN = 4
    SEP_TOKEN = 5

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()

def optimize_bert_graph(args, logger=None):
    if (not logger):
        logger = set_logger(colored('GRAPHOPT', 'cyan'), args.verbose)
    try:
        if (not os.path.exists(args.model_pb_dir)):
            os.mkdir(args.model_pb_dir)
        pb_file = os.path.join(args.model_pb_dir, 'bert_model.pb')
        if os.path.exists(pb_file):
            return pb_file
        tf = import_tf(verbose=args.verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
        config = npu_config_proto(config_proto=tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True))
        config_fp = os.path.join(args.model_dir, args.config_name)
        init_checkpoint = os.path.join((args.tuned_model_dir or args.bert_model_dir), args.ckpt_name)
        if args.fp16:
            logger.warning('fp16 is turned on! Note that not all CPU GPU support fast fp16 instructions, worst case you will have degraded performance!')
        logger.info(('model config: %s' % config_fp))
        logger.info(('checkpoint%s: %s' % ((' (override by the fine-tuned model)' if args.tuned_model_dir else ''), init_checkpoint)))
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))
        logger.info('build graph...')
        input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_type_ids')
        jit_scope = (tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress)
        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]
            model = modeling.BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, token_type_ids=input_type_ids, use_one_hot_embeddings=False)
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            minus_mask = (lambda x, m: (x - (tf.expand_dims((1.0 - m), axis=(- 1)) * 1e+30)))
            mul_mask = (lambda x, m: (x * tf.expand_dims(m, axis=(- 1))))
            masked_reduce_max = (lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1))
            masked_reduce_mean = (lambda x, m: (tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)))
            with tf.variable_scope('pooling'):
                if (len(args.pooling_layer) == 1):
                    encoder_layer = model.all_encoder_layers[args.pooling_layer[0]]
                else:
                    all_layers = [model.all_encoder_layers[l] for l in args.pooling_layer]
                    encoder_layer = tf.concat(all_layers, (- 1))
                input_mask = tf.cast(input_mask, tf.float32)
                if (args.pooling_strategy == PoolingStrategy.REDUCE_MEAN):
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif (args.pooling_strategy == PoolingStrategy.REDUCE_MAX):
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif (args.pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX):
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask), masked_reduce_max(encoder_layer, input_mask)], axis=1)
                elif ((args.pooling_strategy == PoolingStrategy.FIRST_TOKEN) or (args.pooling_strategy == PoolingStrategy.CLS_TOKEN)):
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                elif ((args.pooling_strategy == PoolingStrategy.LAST_TOKEN) or (args.pooling_strategy == PoolingStrategy.SEP_TOKEN)):
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, (seq_len - 1)], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes)
                elif (args.pooling_strategy == PoolingStrategy.NONE):
                    pooled = mul_mask(encoder_layer, input_mask)
                else:
                    raise NotImplementedError()
            if args.fp16:
                pooled = tf.cast(pooled, tf.float16)
            pooled = tf.identity(pooled, 'final_encodes')
            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()
        with tf.Session(config=npu_session_config_init(session_config=config)) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(tmp_g, [n.name[:(- 2)] for n in input_tensors], [n.name[:(- 2)] for n in output_tensors], [dtype.as_datatype_enum for dtype in dtypes], False)
            logger.info('freeze...')
            tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:(- 2)] for n in output_tensors], use_fp16=args.fp16)
        logger.info(('write graph to a tmp file: %s' % args.model_pb_dir))
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
    except Exception:
        logger.error('fail to optimize the graph!', exc_info=True)

def convert_variables_to_constants(sess, input_graph_def, output_node_names, variable_names_whitelist=None, variable_names_blacklist=None, use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if (use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT)):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))
    inference_graph = extract_sub_graph(input_graph_def, output_node_names)
    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if (node.op in ['Variable', 'VariableV2', 'VarHandleOp']):
            variable_name = node.name
            if (((variable_names_whitelist is not None) and (variable_name not in variable_names_whitelist)) or ((variable_names_blacklist is not None) and (variable_name in variable_names_blacklist))):
                continue
            variable_dict_names.append(variable_name)
            if (node.op == 'VarHandleOp'):
                variable_names.append((variable_name + '/Read/ReadVariableOp:0'))
            else:
                variable_names.append((variable_name + ':0'))
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))
    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if (input_node.name in found_variables):
            output_node.op = 'Const'
            output_node.name = input_node.name
            dtype = input_node.attr['dtype']
            data = found_variables[input_node.name]
            if (use_fp16 and (dtype.type == types_pb2.DT_FLOAT)):
                output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(data.astype('float16'), dtype=types_pb2.DT_HALF, shape=data.shape)))
            else:
                output_node.attr['dtype'].CopyFrom(dtype)
                output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type, shape=data.shape)))
            how_many_converted += 1
        elif ((input_node.op == 'ReadVariableOp') and (input_node.input[0] in found_variables)):
            output_node.op = 'Identity'
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr['T'].CopyFrom(input_node.attr['dtype'])
            if ('_class' in input_node.attr):
                output_node.attr['_class'].CopyFrom(input_node.attr['_class'])
        else:
            output_node.CopyFrom(input_node)
        patch_dtype(input_node, 'dtype', output_node)
        patch_dtype(input_node, 'T', output_node)
        patch_dtype(input_node, 'DstT', output_node)
        patch_dtype(input_node, 'SrcT', output_node)
        patch_dtype(input_node, 'Tparams', output_node)
        if (use_fp16 and ('value' in output_node.attr) and (output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT)):
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(output_node.attr['value'].tensor.float_val[0], dtype=types_pb2.DT_HALF)))
        output_graph_def.node.extend([output_node])
    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def

def optimize_ner_model(args, num_labels, logger=None):
    '\n    加载中文NER模型\n    :param args:\n    :param num_labels:\n    :param logger:\n    :return:\n    '
    if (not logger):
        logger = set_logger(colored('NER_MODEL, Lodding...', 'cyan'), args.verbose)
    try:
        if (args.model_pb_dir is None):
            tmp_file = os.path.join(os.getcwd(), 'predict_optimizer')
            if (not os.path.exists(tmp_file)):
                os.mkdir(tmp_file)
        else:
            tmp_file = args.model_pb_dir
        pb_file = os.path.join(tmp_file, 'ner_model.pb')
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file
        import tensorflow as tf
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=npu_session_config_init()) as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))
                from bert_base.train.models import create_model
                (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=None, labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0, lstm_size=args.lstm_size)
                pred_ids = tf.identity(pred_ids, 'pred_ids')
                saver = tf.train.Saver()
            with tf.Session(config=npu_session_config_init()) as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_ids'])
                logger.info('model cut finished !!!')
        logger.info(('write graph to a tmp file: %s' % pb_file))
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error(('fail to optimize the graph! %s' % e), exc_info=True)

def optimize_class_model(args, num_labels, logger=None):
    '\n    加载中文分类模型\n    :param args:\n    :param num_labels:\n    :param logger:\n    :return:\n    '
    if (not logger):
        logger = set_logger(colored('CLASSIFICATION_MODEL, Lodding...', 'cyan'), args.verbose)
    try:
        if (args.model_pb_dir is None):
            tmp_file = os.path.join(os.getcwd(), 'predict_optimizer')
            if (not os.path.exists(tmp_file)):
                os.mkdir(tmp_file)
        else:
            tmp_file = args.model_pb_dir
        pb_file = os.path.join(tmp_file, 'classification_model.pb')
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file
        import tensorflow as tf
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=npu_session_config_init()) as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))
                from bert_base.train.models import create_classification_model
                segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')
                (loss, per_example_loss, logits, probabilities) = create_classification_model(bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, labels=None, num_labels=num_labels)
                probabilities = tf.identity(probabilities, 'pred_prob')
                saver = tf.train.Saver()
            with tf.Session(config=npu_session_config_init()) as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_prob'])
                logger.info('predict cut finished !!!')
        logger.info(('write graph to a tmp file: %s' % pb_file))
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error(('fail to optimize the graph! %s' % e), exc_info=True)
