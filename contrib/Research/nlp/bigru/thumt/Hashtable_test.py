from thumt import npu_config

using_npu = False
config = npu_config() if using_npu else None

keys_tensor = tf.constant(['a', 'b', 'c'])
vals_tensor = tf.constant([7, 8, 9])
input_tensor = tf.constant(['a', 'c'])
init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
table = tf.lookup.StaticHashTable(init, default_value=-1)
init_ops = init.initialize(table)
a, b = table.export() # key, value
c = table.lookup(input_tensor)
with tf.Session(config=npu_config()) as sess:
    sess.run(init_ops) # is equal to # sess.run(table._initialize())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
 



###  test2
## https://github.com/tensorflow/tensorflow/blob/b0784e587b62eec6967196b745bba4db3a90ab0c/tensorflow/python/ops/lookup_ops.py#L37
from tensorflow.python.ops import control_flow_ops
import tensorflow.python.ops as ops
initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
tables_init = control_flow_ops.group(*initializers, name="init_all_tables")
print('initializers =\n', initializers)
print('Table initializers are ============\n', a)
a, b = src_table.export() # key, value
with tf.Session(config=config) as sess:
    # sess.run(tables_init)
    # sess.run(tf.tables_initializer()) # or sess.run(tables_init)
    # print(a.eval())
    print(src_table._default_value)
    print(features["source"].eval())



### test3
a = tf.lookup.KeyValueTensorInitializer(tf.constant(params.vocabulary["source"]),
        tf.constant([i for i in range(len(params.vocabulary["source"]))], dtype=tf.int64))
b = tf.lookup.KeyValueTensorInitializer(tf.constant(params.vocabulary["target"]),
        tf.constant([i for i in range(len(params.vocabulary["target"]))], dtype=tf.int64))

tgt_table = tgt_table._initialize()
src_table = tf.lookup.StaticHashTable(
      a,
      params.mapping["source"][params.unk])
tgt_table = tf.lookup.StaticHashTable(
      b,
      params.mapping["target"][params.unk])