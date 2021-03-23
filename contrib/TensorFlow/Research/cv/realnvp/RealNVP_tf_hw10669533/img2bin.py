# Copyright 2019 Huawei Technologies Co., Ltd
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
from __future__ import print_function
import argparse
import os
import sys
import json
# import getpass
try:
    import cv2 as cv 
except:
    '''
    output = getpass.getuser()
    if output != 'root':
        print('[ERROR] opencv-python not installed.')
        print("[ERROR] Please use the root user to execute this script again")
        exit(0)
    '''
    if sys.version_info.major == 2:
        confirm = raw_input("[info] Begin to install opencv-python, input[Y/N]:")
    else:
        confirm = input("[info] Begin to install opencv-python, input[Y/N]:")
    
    if confirm == 'Y' or confirm == 'y':
        print('[info] Starting to install opencv-python...') 
        if sys.version_info.major == 2:
            import commands
            ret, output = commands.getstatusoutput("sudo yum install -y opencv-python")
            # ret =  os.popen('sudo yum install -y opencv-python')
            if ret != 0:
                # ret = os.popen('sudo apt-get install -y python-opencv')
                ret, output = commands.getstatusoutput("sudo apt-get install -y python-opencv")
                if ret != 0:
                    print('[ERROR] install opencv-python failed,please check env.')
                    exit(0)
        else:
            ret = os.system('sudo python3 -m pip install opencv-python')
            if ret != 0:
                print('[ERROR] install opencv-python failed,please check env.')
                exit(0)
        
        import cv2 as cv
    else:
        print("[info] The installation has been cancled.")
        exit(0)
    
try:
    import numpy as np  
except:
    if sys.version_info.major == 2:
        os.system('pip2 install numpy')
    else:
        os.system('python3 -m pip install numpy')
    import numpy as np 

def get_args():
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='''eg1: python2 img2bin.py
         -i ./images -w 416 -h 416 -f BGR -a NHWC -t uint8 -m [104,117,123] -c [1,1,1] -o ./out 
         eg2: python2 img2bin.py -i ./test.txt -t uint8 -o ./out''')
    parser.add_argument('-i', '--input', required=True, type=str, \
        help='folder of input image or file of other input.')
    parser.add_argument('-w', '--width', type=int, \
        help='resized image width before inference.')
    parser.add_argument('-h', '--height', type=int, \
        help='resized image height before inference.')
    parser.add_argument('-f', '--output_image_format', default='BGR', type=str, \
        help='output image format in (BGR/RGB/YUV/GRAY).')
    parser.add_argument('-a', '--output_format', default='NCHW', type=str, \
        help='output format in (NCHW/NHWC).')
    parser.add_argument('-t', '--output_type', required=True, type=str, \
        help='output type in (float32/uint8/int32/uint32).')
    parser.add_argument('-m', '--mean', default='[0, 0, 0]', \
        help='reduce mean for each channel.')   
    parser.add_argument('-c', '--coefficient', default='[1, 1, 1]', \
        help='multiplying coefficients for each channel.')
    parser.add_argument('-o', '--output', default='./', \
        help='output path.')
    return parser.parse_args()


def eprint(*args, **kwargs):
    """print error message to stderr
    """
    print(*args, file=sys.stderr, **kwargs)


def check_args(args):
    """check console parameters according to restrictions.
    :return: True or False
    """
    check_flag = True
    is_dir = True  
    if os.path.isdir(args.input):
        if not os.listdir(args.input):
            eprint('[ERROR] input image path=%r is empty.' % args.input)
            check_flag = False
    elif os.path.isfile(args.input):
        is_dir = False
    else:
        eprint('[ERROR] input path=%r does not exist.' % args.input)
        check_flag = False
    if args.output_image_format not in ('BGR','RGB', 'YUV', 'GRAY'):
        eprint("ERROR:Convert to %d is not support"%(args.output_image_format))
        check_flag = False
    return check_flag, is_dir
   


def convert_img(args, input_img):
    if args.output_image_format == 'BGR':
        converted_input_img = input_img
    elif args.output_image_format == 'RGB':
        converted_input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
    elif args.output_image_format == 'YUV':
        if input_img.shape[0] % 2 == 1:
            if input_img.shape[1] % 2 == 1:
                input_img = cv.resize(input_img, ((input_img.shape[0] + 1), (input_img.shape[1] + 1)))
            else:
                input_img = cv.resize(input_img, ((input_img.shape[0] + 1), input_img.shape[1]))
        elif input_img.shape[1] % 2 == 1:
            input_img = cv.resize(input_img, (input_img.shape[0], input_img.shape[1] + 1))
        converted_input_img = cv.cvtColor(input_img, cv.COLOR_BGR2YUV_I420)
    elif args.output_image_format == 'GRAY':
        converted_input_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    return converted_input_img
        

def resize_img(args, input_img):
    old_size = input_img.shape[0:2]
    target_size = [args.height, args.width]
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img_new = cv.resize(input_img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    resized_img = cv.copyMakeBorder(img_new, top, bottom, left, right, cv.BORDER_CONSTANT, None,(0, 0, 0))
    #resized_img = cv.resize(input_img, (args.width, args.height))
    return resized_img


def change_type(args, input_img):
    if args.output_type == 'float32':
        change_type_img = input_img.astype(np.float32)
    elif args.output_type == 'int32':
        change_type_img = input_img.astype(np.int32)
    elif args.output_type == 'uint32':
        change_type_img = input_img.astype(np.uint32)
    else:
        change_type_img = input_img.astype(np.uint8)
    return change_type_img


def mean(args, input_img):
    if isinstance (args.mean, str):
        args.mean = json.loads(args.mean)
    input_img = input_img.astype(np.float32)
    if args.output_image_format == 'GRAY':
        input_img[:, :] -= args.mean[0]
    elif args.output_image_format in ('BGR', 'RGB'):
        input_img[:, :, 0] -= args.mean[0]
        input_img[:, :, 1] -= args.mean[1]
        input_img[:, :, 2] -= args.mean[2]
    else:
        input_img[: int(args.width / 1.5), :] -= args.mean[0]
        input_img[int(args.width / 1.5) :, :: 2] -= args.mean[1]
        input_img[int(args.width / 1.5) :, 1: 2] -= args.mean[2]
    return input_img


def coefficient(args, input_img):
    if isinstance (args.coefficient, str):
        args.coefficient = json.loads(args.coefficient)
    input_img = input_img.astype(np.float32)
    if args.output_image_format == 'GRAY':
        input_img[:, :] *= args.coefficient[0]
    elif args.output_image_format in ('BGR', 'RGB'):
        input_img[:, :, 0] *= args.coefficient[0]
        input_img[:, :, 1] *= args.coefficient[1]
        input_img[:, :, 2] *= args.coefficient[2]
    else:
        input_img[: int(args.width / 1.5), :] *= args.coefficient[0]
        input_img[int(args.width / 1.5) :, :: 2] *= args.coefficient[1]
        input_img[int(args.width / 1.5) :, 1: 2] *= args.coefficient[2]
    return input_img


def change_format(args, input_img):
    if args.output_format == 'NCHW':
        if args.output_image_format in ('RGB', 'BGR'):
            change_format_img = input_img.transpose(2,0,1).copy()        
            return change_format_img
    return input_img


def mkdir_output(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return


def process(args, file_path):
    if file_path.endswith(".txt"):
        if sys.version_info.major == 2:
            import ConfigParser as configparser
        else:
            import configparser

        config = configparser.ConfigParser()
        config.read(file_path)
        if sys.version_info.major == 2:
            input_node = json.loads(config.get('baseconf', 'input_node'))
            shape = json.loads(config.get('baseconf', 'shape'))
        else:
            input_node = json.loads(config['baseconf']['input_node'])
            shape = json.loads(config['baseconf']['shape'])
        input_node_np = np.array(input_node)
        change_type_img_info = change_type(args, input_node_np)
        img_info = np.reshape(change_type_img_info, shape)
        out_path = os.path.join(args.output, os.path.splitext(os.path.split(file_path)[1])[0] + ".bin")
        mkdir_output(args)
        img_info.tofile(out_path) 
    else:
        input_img = cv.imread(file_path) 
        if args.output_image_format == 'YUV':
            resized_img1 = resize_img(args, input_img) 
            converted_img = convert_img(args, resized_img1) 
            mean_img = mean(args, converted_img)
        else: 
            converted_img = convert_img(args, input_img)    
            resized_img = resize_img(args, converted_img)  
            mean_img = mean(args, resized_img)
        coefficient_img = coefficient(args, mean_img)
        change_type_img = change_type(args, coefficient_img)
        change_format_img = change_format(args, change_type_img)
        out_path = os.path.join(args.output, os.path.splitext(os.path.split(file_path)[1])[0] + ".bin")
        mkdir_output(args)
        change_format_img.tofile(out_path)
        

def main():
    """main function to receive params them change data to bin.
    """
    args = get_args()
    ret,is_dir = check_args(args)
    if ret:
        if is_dir:
            files_name = os.listdir(args.input)
            for file_name in files_name:
                file_path = os.path.join(args.input, file_name)
                process(args, file_path)  
        else:
            process(args, args.input)    
        print("[info] bin file generated successfully.")

if __name__ == '__main__':
    main()



