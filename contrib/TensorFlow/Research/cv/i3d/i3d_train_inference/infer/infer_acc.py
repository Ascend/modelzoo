import os
import numpy as np
import argparse
import glob


def get_label_map(file):
    label_map = []
    with open(file) as f:
        for line in f.readlines():
            label_map.append(line.strip())
    return label_map


def get_videosize():
    path_file_number = glob.glob('./out_rgb/*.bin')
    video_size = len(path_file_number)
    print('video_size:', video_size)
    return video_size

def main(dataset, mode, split):
    video_size = get_videosize()
    OUTPUT = './out_rgb/'
    files = os.listdir(OUTPUT)
    check_num = 0
    label_map = get_label_map(os.path.join('../data', dataset, 'label_map.txt'))
    for file in files:
        if file.endswith(".bin"):
            np_list = np.fromfile(OUTPUT+'/'+file, dtype='float32')
            inf_label = int(np.argmax(np_list))
            try:
                pic_name = str(file.split("_")[1])
                gt_label = label_map.index(pic_name)
                print("video_name:%s, inference label:%d, gt_label: %d" % (pic_name, inf_label, label_map.index(pic_name)))
                if inf_label == gt_label:
                    check_num += 1
            except:
                print("Can't find %s in the label file: %s" % (pic_name))
    print('check_num:', check_num, video_size)
    accuracy = check_num/video_size
    print('test accuracy: %.4f' % (accuracy))

if __name__ == '__main__':
    description = 'Test Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    main(**vars(p.parse_args()))
