import os
import numpy as np
from lib.load_data import load_info
from lib.label_trans import *
import argparse
_DATA_ROOT = {
    'ucf101': {
        'rgb': '../data/jpegs_256',
        'flow': '../data/tvl1_flow//{:s}'
    }
}

def get_videosize(dataset, mode, spilt):
    _, test_info_rgb, class_num, _ = load_info(dataset, root=_DATA_ROOT[dataset], mode='rgb', split=spilt)
    print('test_info_rgb:', test_info_rgb)
    video_size = len(test_info_rgb)
    print('video_size:', video_size)
    return video_size

def main(dataset, mode, split):
    video_size = get_videosize(dataset, mode, split)
    OUTPUT = './data/rgb/output/'
    files = os.listdir(OUTPUT)
    check_num = 0
    label_map = get_label_map(os.path.join(
        './data', dataset, 'label_map.txt'))
    print('label_map', label_map)
    for file in files:
        if file.endswith(".bin"):
            top_1 = np.fromfile(OUTPUT+'/'+file, dtype='float32')
            inf_label = int(np.argmax(top_1))
            print('inf_label:', inf_label)
            # true_count += inf_label
            try:
                pic_name = str(file.split("_")[1]) #+".JPEG"
                print('pic_name:', pic_name)
                print(label_map.index(pic_name))
                print("video_name:%s, inference label:%d, gt_label: %d" % (pic_name, inf_label, label_map.index(pic_name)))
                if inf_label == label_map.index(pic_name):
                    check_num += 1
            except:
                print("Can't find %s in the label file: %s" % (pic_name))
    accuracy = check_num / video_size
    print('test accuracy: %.4f' % (accuracy))

if __name__ == '__main__':
    description = 'Test Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    # main(dataset='ucf101', mode='rgb', spilt=1)
    main(**vars(p.parse_args()))
