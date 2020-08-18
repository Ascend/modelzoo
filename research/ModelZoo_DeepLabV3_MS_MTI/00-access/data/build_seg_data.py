import sys
import os
from mindspore.mindrecord import FileWriter
import gflags
import numpy as np

FLAGS = gflags.FLAGS

gflags.DEFINE_string('data_root', '', 
    'root path of training data')

gflags.DEFINE_string('data_lst', '', 
    'list of training data')

gflags.DEFINE_string('dst_path', '',
    'where records are saved')

gflags.DEFINE_integer('num_shards', 8, 'number of shards')

gflags.DEFINE_boolean('shuffle', True, 'shuffle or not')

seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"},
    "data": {"type": "bytes"}}


if __name__ == '__main__':

    FLAGS(sys.argv)
    datas = []
    with open(FLAGS.data_lst) as f:
        lines = f.readlines()
    if FLAGS.shuffle:
        np.random.shuffle(lines)
    
    dst_dir = '/'.join(FLAGS.dst_path.split('/')[:-1])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    print('number of samples:', len(lines))
    writer = FileWriter(file_name=FLAGS.dst_path, shard_num=FLAGS.num_shards)
    writer.add_schema(seg_schema, "seg_schema")
    cnt = 0
    for l in lines:
        img_path, label_path = l.strip().split(' ')
        sample_ = {"file_name": img_path.split('/')[-1]}
        with open(os.path.join(FLAGS.data_root, img_path), 'rb') as f:
            sample_['data'] = f.read()
        with open(os.path.join(FLAGS.data_root, label_path), 'rb') as f:
            sample_['label'] = f.read()
        datas.append(sample_)
        cnt += 1
        if cnt % 1000 == 0:
            writer.write_raw_data(datas)
            print('number of samples written:', cnt)
            datas = []

    if len(datas) > 0:
        writer.write_raw_data(datas)
    writer.commit()
    print('number of samples written:', cnt)

    



