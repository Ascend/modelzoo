#! -*- coding:utf-8 -*-
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
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
import json
import os,time
import collections
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RAW_SHAPE = "raw_shape"
IS_PADDED = "is_padded"
SOURCE_ID = "source_id"
MIN_SCORE = 0.05
DUMMY_SCORE = -1e5
MAX_NUM_EVAL_BOXES = 200
OVERLAP_CRITERIA = 0.5
CLASS_INV_MAP = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=1,
                        help="""batchsize""")
    parser.add_argument('--model_path', default='ssd-resnet34_1batch.pb',
                        help="""pb path""")
    parser.add_argument('--data_path', default = 'coco2017/test',
                        help = """the data path""")
    parser.add_argument('--val_json_file', default='coco_official_2017/annotations/instances_val2017.json',
                        help="""the val json file path""")
    parser.add_argument('--input_tensor_name', default='input:0',
                        help="""the output1 tensor name""")
    parser.add_argument('--output_tensor_name1', default='Softmax:0',
                        help="""the output1 tensor name""")
    parser.add_argument('--output_tensor_name2', default='stack:0',
                        help="""the output2 tensor name""")


    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def load_model(model_file):
    with tf.gfile.GFile(model_file, "rb") as gf:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(gf.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def top_k(input, k=1, sorted=True):
    """Top k max pooling
    Args:
        input(ndarray): convolutional feature in heigh x width x channel format
        k(int): if k==1, it is equal to normal max pooling
        sorted(bool): whether to return the array sorted by channel value
    Returns:
        ndarray: k x (height x width)
        ndarray: k
    """
    ind = np.argpartition(input, -k)[..., -k:]
    def get_entries(input, ind, sorted):
        if len(ind.shape) == 1:
            if sorted:
                ind = ind[np.argsort(-input[ind])]
            return input[ind], ind
        output, ind = zip(*[get_entries(inp, id, sorted) for inp, id in zip(input, ind)])
        return np.array(output), np.array(ind)
    return get_entries(input, ind, sorted)

def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
    '''
    scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
    top_k_scores, top_k_indices = tf.nn.top_k(
        scores_trans, k=pre_nms_num_detections, sorted=True)
    return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
        top_k_indices, [0, 2, 1])
    '''
    scores_trans = np.transpose(scores_in, (0, 2, 1))
    top_k_scores, top_k_indices = top_k(scores_trans, k = pre_nms_num_detections)
    return np.transpose(top_k_scores, (0, 2, 1)), np.transpose(top_k_indices, (0, 2, 1))


def _load_images_info(images_info_file):
  """Loads object annotation JSON file."""
  f = open(images_info_file, encoding='utf-8')
  info_dict = json.load(f)

  img_to_obj_annotation = collections.defaultdict(list)
  for annotation in info_dict['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)
  return info_dict['images'],img_to_obj_annotation

def get_image_obj(images_info_file, input_images):
    f = open(images_info_file, encoding='utf-8')
    info_dict = json.load(f)
    img_obj = collections.defaultdict(list)
    img_info_list = []
    image_list_new = []
    for image in info_dict['images']:
        img_info = {}
        image_name = image['file_name']
        if image_name not in input_images:
            continue
        img_info['source_id'] = image['id']
        img_info['raw_shape'] = [image['height'], image['width'], 3]
        img_info_list.append(img_info)
        image_list_new.append(image_name)

    return img_info_list, image_list_new


def _read_inputImage(filename):
    image_list = []
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            file = file.split('.')[0] + ".jpg"
            image_list.append(file)
    return image_list



def image_process(image_path, images_name):
    ###image process
    imagelist = []
    images_count = 0
    for image_name in images_name:
        with tf.Session().as_default():
        #with tf.Session() as sess:
            image_file = os.path.join(image_path, image_name)
            image = tf.gfile.FastGFile(image_file, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)
            '''
            #把宽和高变成奇数
            filename = tf.placeholder(tf.string, [], name='filename')
            img = sess.run(image, feed_dict={filename:image_file})
            height = img.shape[0]
            width = img.shape[1]
            height = height - 1 if height % 2 == 0 else height
            width = width - 1 if width % 2 == 0 else width
            image = tf.image.crop_to_bounding_box(image, 0, 0, height, width)
            '''
            image = tf.image.resize_images(image, size=(300, 300))
            image /= 255.
            images_count = images_count + 1
            if tf.shape(image)[2].eval() == 1:
                image = tf.image.grayscale_to_rgb(image)
            image = image.eval()
            imagelist.append(image)
        tf.reset_default_graph()
    return np.array(imagelist), images_count

class Classifier(object):
    # set batch_size
    args = parse_args()
    batch_size = int(args.batch_size)

    def __init__(self):
        # --------------------------------------------------------------------------------
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 1）run on Ascend NPU
        custom_op.parameter_map["use_off_line"].b = True

        # 2）recommended use fp16 datatype to obtain better performance
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        # 3）disable remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        # 4）set graph_run_mode=0，obtain better performance
        custom_op.parameter_map["graph_run_mode"].i = 0

        '''
        # 是否开启dump功能
        custom_op.parameter_map["enable_dump"].b = True
        # dump数据存放路径
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/autotest/xwx5322041/dump")
        # dump模式，默认仅dump算子输出数据，还可以dump算子输入数据，取值：input/output/all
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        '''
        # --------------------------------------------------------------------------------

        # load model， set graph input nodes and output nodes
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor1 = self.graph.get_tensor_by_name(args.output_tensor_name1)
        self.output_tensor2 = self.graph.get_tensor_by_name(args.output_tensor_name2)

        # create session
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load fronzen graph
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def infer(self, batch_size, batch_data, labels_list, images_count):
        dataOutput = []
        total_time = 0
        count = 0
        for data in batch_data:
            t = time.time()
            classes, boxes = self.sess.run([self.output_tensor1, self.output_tensor2], feed_dict={self.input_tensor: data.reshape(int(batch_size),300,300,3)})
            total_time = total_time + time.time() - t
            pred_scores, indices = select_top_k_scores(classes,200)
            output_index = 0
            while output_index < int(batch_size) and count < images_count:
                dataOutput.append({"pred_box": boxes[output_index],
                            "source_id": labels_list[count]['source_id'],
                            "indices": indices[output_index],
                            "pred_scores": pred_scores[output_index],
                            "raw_shape": labels_list[count]['raw_shape']})
                output_index = output_index + 1
                count = count + 1

        return dataOutput, total_time

    def batch_process(self, image_data):
        """
        images preprocess
        :return:
        """
        # Get the batch information of the current input data, and automatically adjust the data to the fixed batch
        n_dim = image_data.shape[0]
        batch_size = self.batch_size

        # if data is not enough for the whole batch, you need to complete the data
        m = n_dim % batch_size
        if m < batch_size and m > 0:
            # The insufficient part shall be filled with 0 according to n dimension
            pad = np.zeros((batch_size - m, 300, 300, 3)).astype(np.float32)
            image_data = np.concatenate((image_data, pad), axis=0)

        # Define the Minis that can be divided into several batches
        mini_batch = []
        i = 0
        while i < n_dim:
            # Define the Minis that can be divided into several batches
            mini_batch.append(image_data[i: i + batch_size, :, :, :])
            i += batch_size

        return mini_batch

def decode_single(bboxes_in,
                  scores_in,
                  indices,
                  criteria,
                  max_output,
                  max_num=200):
  """Implement Non-maximum suppression.

    Reference to https://github.com/amdegroot/ssd.pytorch

  Args:
    bboxes_in: a Tensor with shape [N, 4], which stacks box regression outputs
      on all feature levels. The N is the number of total anchors on all levels.
    scores_in: a Tensor with shape [ssd_constants.MAX_NUM_EVAL_BOXES,
      num_classes]. The top ssd_constants.MAX_NUM_EVAL_BOXES box scores for each
      class.
    indices: a Tensor with shape [ssd_constants.MAX_NUM_EVAL_BOXES,
      num_classes]. The indices for these top boxes for each class.
    criteria: a float number to specify the threshold of NMS.
    max_output: maximum output length.
    max_num: maximum number of boxes before NMS.

  Returns:
    boxes, labels and scores after NMS.
  """

  bboxes_out = []
  scores_out = []
  labels_out = []

  for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
    class_indices = indices[:, i]
    bboxes = bboxes_in[class_indices, :]
    score = np.squeeze(score, 1)

    # skip background
    if i == 0:
      continue

    mask = score > MIN_SCORE
    if not np.any(mask):
      continue

    bboxes, score = bboxes[mask, :], score[mask]

    # remain_list = []
    # for r in range(bboxes.shape[0]):
    #   if bboxes[r, 0] < 0 or bboxes[r, 1] < 0 or bboxes[r, 2] < 0 or bboxes[r, 3] < 0 or bboxes[r, 0] >= bboxes[r, 2] or \
    #           bboxes[r, 1] >= bboxes[r, 3]:
    #     continue
    #   remain_list.append(r)
    # bboxes = bboxes[remain_list, :]
    # score = score[remain_list]

    remain_list = []
    for r in range(bboxes.shape[0]):
      for j in range(4):
        if bboxes[r, j] < 0:
          bboxes[r, j] = 0.00001
      if bboxes[r, 0] >= bboxes[r, 2]:
        bboxes[r, 2] = bboxes[r, 0] + 0.00001
      if bboxes[r, 1] >= bboxes[r, 3]:
        bboxes[r, 3] = bboxes[r, 1] + 0.00001
      remain_list.append(r)
    bboxes = bboxes[remain_list, :]
    score = score[remain_list]


    score_idx_sorted = np.argsort(score)
    score_sorted = score[score_idx_sorted]

    score_idx_sorted = score_idx_sorted[-max_num:]
    candidates = []

    # perform non-maximum suppression
    while len(score_idx_sorted):
      idx = score_idx_sorted[-1]
      bboxes_sorted = bboxes[score_idx_sorted, :]
      bboxes_idx = bboxes[idx, :]
      iou = calc_iou(bboxes_idx, bboxes_sorted)

      score_idx_sorted = score_idx_sorted[iou < criteria]
      candidates.append(idx)

    bboxes_out.append(bboxes[candidates, :])
    scores_out.append(score[candidates])
    labels_out.extend([i]*len(candidates))

  if len(scores_out) == 0:
    tf.logging.info("No objects detected. Returning dummy values.")
    return (
        np.zeros(shape=(1, 4), dtype=np.float32),
        np.zeros(shape=(1,), dtype=np.int32),
        np.ones(shape=(1,), dtype=np.float32) * DUMMY_SCORE,
    )

  bboxes_out = np.concatenate(bboxes_out, axis=0)
  scores_out = np.concatenate(scores_out, axis=0)
  labels_out = np.array(labels_out)

  max_ids = np.argsort(scores_out)[-max_output:]

  return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

def calc_iou(target, candidates):
  target_tiled = np.tile(target[np.newaxis, :], (candidates.shape[0], 1))
  # Left Top & Right Bottom
  lt = np.maximum(target_tiled[:,:2], candidates[:,:2])

  rb = np.minimum(target_tiled[:,2:], candidates[:,2:])

  delta = np.maximum(rb - lt, 0)

  intersect = delta[:,0] * delta[:,1]

  delta1 = target_tiled[:, 2:] - target_tiled[:, :2]
  area1 = delta1[:,0] * delta1[:,1]
  delta2 = candidates[:, 2:] - candidates[:, :2]
  area2 = delta2[:,0] * delta2[:,1]

  iou = intersect/(area1 + area2 - intersect)
  return iou

def compute_map(labels_and_predictions,
                coco_gt,
                use_cpp_extension=True,
                nms_on_tpu=True):
  """Use model predictions to compute mAP.

  The evaluation code is largely copied from the MLPerf reference
  implementation. While it is possible to write the evaluation as a tensor
  metric and use Estimator.evaluate(), this approach was selected for simplicity
  and ease of duck testing.

  Args:
    labels_and_predictions: A map from TPU predict method.
    coco_gt: ground truch COCO object.
    use_cpp_extension: use cocoeval C++ library.
    nms_on_tpu: do NMS on TPU.
  Returns:
    Evaluation result.
  """
  predictions = []
  tic = time.time()

  if nms_on_tpu:
    p = []
    for i in labels_and_predictions:
      for j in i:
        p.append(np.array(j, dtype=np.float32))
    predictions = np.concatenate(list(p)).reshape((-1, 7))
  else:
    k = 0
    for example in labels_and_predictions:
      if IS_PADDED in example and example[
          IS_PADDED]:
        continue
      #print(k)
      k += 1
      htot, wtot, _ = example[RAW_SHAPE]
      pred_box = example['pred_box']
      pred_scores = example['pred_scores']
      indices = example['indices']
      loc, label, prob = decode_single(
          pred_box, pred_scores, indices, OVERLAP_CRITERIA,
          MAX_NUM_EVAL_BOXES, MAX_NUM_EVAL_BOXES)

      for loc_, label_, prob_ in zip(loc, label, prob):
        # Ordering convention differs, hence [1], [0] rather than [0], [1]
        predictions.append([
            int(example[SOURCE_ID]),
            loc_[1] * wtot, loc_[0] * htot, (loc_[3] - loc_[1]) * wtot,
            (loc_[2] - loc_[0]) * htot, prob_,
            CLASS_INV_MAP[label_]
        ])

  toc = time.time()
  tf.logging.info('Prepare predictions DONE (t={:0.2f}s).'.format(toc - tic))

  if coco_gt is None:
    coco_gt = create_coco(
        FLAGS.val_json_file, use_cpp_extension=use_cpp_extension)

  if use_cpp_extension:
    coco_dt = coco_gt.LoadRes(np.array(predictions, dtype=np.float32))
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type='bbox')
    coco_eval.Evaluate()
    coco_eval.Accumulate()
    coco_eval.Summarize()
    stats = coco_eval.GetStats()

  else:
    coco_dt = coco_gt.loadRes(np.array(predictions))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats

  print('Current AP: {:.5f}'.format(stats[0]))
  metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                  'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
  coco_time = time.time()
  tf.logging.info('COCO eval DONE (t={:0.2f}s).'.format(coco_time - toc))

  # Prefix with "COCO" to group in TensorBoard.
  return {'COCO/' + key: value for key, value in zip(metric_names, stats)}


def create_coco(val_json_file, use_cpp_extension=True):
  """Creates Microsoft COCO helper class object and return it."""
  if val_json_file.startswith('gs://'):
    _, local_val_json = tempfile.mkstemp(suffix='.json')
    tf.gfile.Remove(local_val_json)

    tf.gfile.Copy(val_json_file, local_val_json)
    atexit.register(tf.gfile.Remove, local_val_json)
  else:
    local_val_json = val_json_file

  if use_cpp_extension:
    coco_gt = coco.COCO(local_val_json, False)
  else:
    coco_gt = COCO(local_val_json)
  return coco_gt

def main():
    args = parse_args()
    tf.reset_default_graph()

    image_list = _read_inputImage(args.data_path)
    image_obj, image_list = get_image_obj(args.val_json_file, image_list)

    print("########NOW Start Preprocess!!!#########")
    images, images_count = image_process(args.data_path, image_list)

    ###batch
    print("########NOW Start Batch!!!#########")
    classifier = Classifier()
    batch_images = classifier.batch_process(images)

    ###do inference
    print("########NOW Start inference!!!#########")
    dataOutput, total_time = classifier.infer(args.batch_size, batch_images, image_obj, images_count)

    coco_gt = create_coco(
        args.val_json_file, use_cpp_extension=False)
    compute_map(
        dataOutput,
        coco_gt,
        use_cpp_extension=False,
        nms_on_tpu=False)

    print('+-------------------------------------------------+')
    print('images number = ', images_count)
    print('images/sec = ', images_count / total_time)
    print('+-------------------------------------------------+')

if __name__ == '__main__':
    main()
