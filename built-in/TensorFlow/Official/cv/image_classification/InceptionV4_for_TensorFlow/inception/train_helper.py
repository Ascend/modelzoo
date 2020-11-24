import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import re
import os
from operator import itemgetter

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

