import sys
import os
import cv2
import numpy as np

def yolov4_onnx(src_info, output_path):
    in_files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(src_info, 'r') as file:
        contents = file.read().split('\n')
    for i in contents[:-1]:
        in_files.append(i.split()[1])

    i = 0
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        img0 = cv2.imread(file)
        resized = cv2.resize(img0, (608, 608), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        print("shape:", img_in.shape)

        # save img_tensor as binary file for om inference input
        temp_name = file[file.rfind('/') + 1:]
        img_in.tofile(os.path.join(output_path, temp_name.split('.')[0] + ".bin"))


if __name__ == "__main__":
    src_info = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    yolov4_onnx(src_info, bin_path)
