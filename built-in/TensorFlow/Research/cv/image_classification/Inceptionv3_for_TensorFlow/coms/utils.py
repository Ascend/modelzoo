import platform

import tensorflow as tf
from  tensorflow.python import pywrap_tensorflow as pyw
from tensorflow.python.client import device_lib as _device_lib

def isLinuxSys():
    if platform.system() == "Linux":
        return True
    return False

def isWinSys():
    if platform.system() == "Windows":
        return True
    return False

def isHasCpu():
    info = _device_lib.list_local_devices()
    for dev in info:
        if 'CPU' in dev.name:
            return True
    return False

def isHasGpu():
    info = _device_lib.list_local_devices()

    for dev in info:
        # print(dev.name)
        if 'GPU' in dev.name:
            return True
    return False

if __name__ == '__main__':
    # path = ''
    # model_name = 'YOLO_small.ckpt'
    # if isLinuxSys():
    #     path = '/home/zhuhao/DataSets/YOLO/v1model/' + model_name
    #
    # # saver = tf.train.Saver(max_to_keep=1)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # model_file = tf.train.latest_checkpoint(path)
    #     reader = pyw.NewCheckpointReader(path)
    #     var_to_shape_map = reader.get_variable_to_shape_map()
    #     from  pprint import pprint
    #     pprint(var_to_shape_map)
    print(isHasGpu())


