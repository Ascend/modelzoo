import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pdb
import numpy as np

model = tf.keras.models.load_model('./Wide_ResNet/')

imgs = np.fromfile("./input/000000.bin",dtype="float32").reshape(1,32,32,3)
print(model.predict(imgs))
