# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import ssd_architecture,ssd_model
import argparse
import ssd_constants
import numpy as np
import math
import itertools as it
from object_detection import box_coder
import dataloader
from object_detection import box_list
from object_detection import faster_rcnn_box_coder

NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default='./ckpt/model.ckpt-108000',
                        help="""set checkpoint path""")
    parser.add_argument("--resnet_checkpoint", type=str, default='./resnet34_pretrain/model.ckpt-28152',
                        help="The path of the resnet34 checkpoint")
    parser.add_argument("--val_json_file", type=str, default='./coco_official_2017/annotations/instances_val2017.json',
                        help="The path of the val_json_file.")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="batch size")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def concat_outputs(cls_outputs, box_outputs):
  """Concatenate predictions into a single tensor.

  This function takes the dicts of class and box prediction tensors and
  concatenates them into a single tensor for comparison with the ground truth
  boxes and class labels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width,
      num_anchors * num_classses].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
  Returns:
    concatenanted cls_outputs and box_outputs.
  """
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  print(keys)
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []

  for i, k in enumerate(keys):
    # TODO(taylorrobie): confirm that this reshape, transpose,
    # reshape is correct.
    scale = ssd_constants.FEATURE_SIZES[i] # 不同特征尺度, 38,19,10,5,3,1
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES) # （4,81）、（6,81）...
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32,38,38)+ (4,81)=(32,38,38,4,81)
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 38^2 * 4, 81)
    flat_cls.append(tf.reshape(tf.reshape(
        cls_outputs[k], intermediate_shape), final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4) # (4,4), (6,4)...
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32, 19,19) + (6,4) 为避免歧义，以第二个default box为例
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 19^2 * 6, 4)
    flat_box.append(tf.reshape(tf.reshape(
        box_outputs[k], intermediate_shape), final_shape))

  return tf.concat(flat_cls, axis=1), tf.concat(flat_box, axis=1)

'''
def concat_outputs(cls_outputs, box_outputs):
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []

  for i, k in enumerate(keys):
    # TODO(taylorrobie): confirm that this reshape, transpose,
    # reshape is correct.
    scale = ssd_constants.FEATURE_SIZES[i] # 不同特征尺度, 38,19,10,5,3,1
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES) # （4,81）、（6,81）...
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32,38,38)+ (4,81)=(32,38,38,4,81)
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 38^2 * 4, 81)
    flat_cls.append(tf.reshape(tf.reshape(
        cls_outputs[k], intermediate_shape), final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4) # (4,4), (6,4)...
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape # (32, 19,19) + (6,4) 为避免歧义，以第二个default box为例
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1]) # (32, 19^2 * 6, 4)
    flat_box.append(tf.reshape(tf.reshape(
        box_outputs[k], intermediate_shape), final_shape))

  return tf.concat(flat_cls, axis=1), tf.concat(flat_box, axis=1)

def DefaultBoxes():
    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    fk = ssd_constants.IMAGE_SIZE / np.array(ssd_constants.STEPS)
    default_boxes = []
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
        sk1 = ssd_constants.SCALES[idx] / ssd_constants.IMAGE_SIZE
        sk2 = ssd_constants.SCALES[idx + 1] / ssd_constants.IMAGE_SIZE
        sk3 = math.sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]
        for alpha in ssd_constants.ASPECT_RATIOS[idx]:
            w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))
        assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]
        for i, j in it.product(range(feature_size), repeat=2):
            for w, h in all_sizes:
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                box = tuple(np.clip(k, 0, 1) for k in (cy, cx, h, w))
                default_boxes.append(box)
    default_boxes_ltrb = tuple(to_ltrb(*i) for i in default_boxes)
    return default_boxes_ltrb

def batch_decode(encoded_boxes, box_coder, anchors):

  def get_center_coordinates_and_sizes(anchors):
      ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(anchors))
      width = xmax - xmin
      height = ymax - ymin
      ycenter = ymin + height / 2.
      xcenter = xmin + width / 2.
      return [ycenter, xcenter, height, width]

  def decode(box_coder, rel_codes, anchors):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if box_coder:
      ty /= box_coder[0]
      tx /= box_coder[1]
      th /= box_coder[2]
      tw /= box_coder[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

  encoded_boxes.get_shape().assert_has_rank(3)

  decoded_boxes = tf.stack([
      decode(box_coder, boxes, tf.convert_to_tensor(anchors))
      for boxes in tf.unstack(encoded_boxes)
  ])
  return decoded_boxes
'''

def main():
    args = parse_args()
    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[args.batchsize, 300, 300, 3], name="input")
    inputs -= tf.constant(NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=inputs.dtype)
    inputs /= tf.constant(NORMALIZATION_STD, shape=[1, 1, 3], dtype=inputs.dtype)
    
    hparams = ssd_model.default_hparams()
    params = dict(
        hparams.values(),
        num_examples_per_epoch=120000,
        resnet_checkpoint=args.resnet_checkpoint,
        val_json_file=args.val_json_file,
        mode=eval,
        model_dir='result_npu',
        eval_samples=5000,
    )
    eval_params = dict(params)
    eval_params['batch_size'] = 32

    cls_outputs, box_outputs = ssd_architecture.ssd(
        inputs, params=eval_params, is_training_bn=False)

    flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)

    ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)

    anchors = box_list.BoxList(
        tf.convert_to_tensor(dataloader.DefaultBoxes()('ltrb')))

    decoded_boxes = box_coder.batch_decode(
        encoded_boxes=flattened_box, box_coder=ssd_box_coder, anchors=anchors)

    pred_scores = tf.nn.softmax(flattened_cls, axis=2)

    #output1 = tf.argmax(decoded_boxes, axis=1, output_type=tf.int32, name='output1')
    #output2 = tf.argmax(pred_scores, axis=1, output_type=tf.int32, name='output2')


    #freeze graph
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './', 'model.pb')  # save unfrozen graph
        freeze_graph.freeze_graph(
            input_graph='./model.pb',  # unfrozen graph
            input_saver='',
            input_binary=False,
            input_checkpoint=args.ckpt_path,
            #output_node_names="Softmax,stack_1",  # graph outputs node
            output_node_names="Softmax,stack",  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./ssd-resnet34_' + str(args.batchsize) + "batch.pb",  # output pb name
            clear_devices=False,
            initializer_nodes='')
    print("done")

    '''
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './', 'model.pb')    #save unfrozen graph
        freeze_graph.freeze_graph(
		        input_graph='./model.pb',   # unfrozen graph
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=args.ckpt_path,
		        output_node_names="ssd/class_net/class-3/BiasAdd,ssd/box_net/box-3/BiasAdd,\
                                   ssd/class_net/class-4/BiasAdd,ssd/box_net/box-4/BiasAdd,\
                                   ssd/class_net/class-5/BiasAdd,ssd/box_net/box-5/BiasAdd,\
                                   ssd/class_net/class-6/BiasAdd,ssd/box_net/box-6/BiasAdd,\
                                   ssd/class_net/class-7/BiasAdd,ssd/box_net/box-7/BiasAdd,\
                                   ssd/class_net/class-8/BiasAdd,ssd/box_net/box-8/BiasAdd",  # graph outputs node
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./SSD-Resnet34_tf.pb',   # output pb name
		        clear_devices=False,
		        initializer_nodes='')
    print("done")
    '''

if __name__ == '__main__': 
    main()

