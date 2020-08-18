import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import re
import os
from operator import itemgetter

class PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i + 1])

def stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op = stage_area.put(tensors)
    get_tensors = stage_area.get()
    tf.add_to_collection('STAGING_AREA_PUTS', put_op)
    return put_op, get_tensors


def sort_and_load_ckpts(log_dir):
    ckpts = []
    for f in os.listdir(log_dir):
        m = re.match(r'model.ckpt-([0-9]+).index', f)
        if m is None:
            continue
        fullpath = os.path.join(log_dir, f)
        ckpts.append({'step': int(m.group(1)),
                      'path': os.path.splitext(fullpath)[0],
                      'mtime': os.stat(fullpath).st_mtime,
                      })
    ckpts.sort(key=itemgetter('step'))
    return ckpts


