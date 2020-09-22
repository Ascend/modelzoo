import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pdb

new_model = tf.keras.models.load_model('./checkpoints/yolov4-416')

full_model = tf.function(lambda x: new_model(x))
full_model = full_model.get_concrete_function(x=tf.TensorSpec((None,416,416,3),'float32'))

forzen_func = convert_variables_to_constants_v2(full_model)
forzen_func.graph.as_graph_def()

layers = [op.name for op in forzen_func.graph.get_operations()]
print("-"*50)
print("Frozen model layers:")
for layer in layers:
    print(layer)

print("*"*50)
print("Frozen model input:")
print(forzen_func.inputs)
print("Frozen model output:")
print(forzen_func.outputs)

tf.io.write_graph(
    graph_or_graph_def=forzen_func.graph,
    logdir="./",
    name="yolov4_frozen.pb",
    as_text=False
)