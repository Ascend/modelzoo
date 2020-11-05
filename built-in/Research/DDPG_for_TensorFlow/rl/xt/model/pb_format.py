"""
save model in pb file format.
"""

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from xt.model.tf_compat import K


def pb_model(h5_model, file_name, out_prefix="output_"):
    """
    Describe: output model in pb file
    :param h5_model:
    :param file_name:
    :param out_prefix:
    """
    output_dir = os.path.dirname(file_name) + "/" + "pb_model"
    model_name = os.path.basename(file_name) + ".pb"
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
