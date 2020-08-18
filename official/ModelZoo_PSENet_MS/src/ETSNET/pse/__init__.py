import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# if subprocess.call(['make', '-C', BASE_DIR]) != 0:
#     raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

from .adaptor import pse as cpse

def pse(polys, min_area):
    ret = np.array(cpse(polys, min_area), dtype='int32')
    return ret


