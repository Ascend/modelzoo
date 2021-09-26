# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from constants import verbosity, save_dir, overlap, \
    model_name, tests_path, input_width, input_height, scale_fact
from utils import float_im

'''
   om推理+推理结果解析保存(可少量图执行，且提前先准备好对应的om)  
   python3 predict_directory_npu.py --img_dir='输入图' --output_dir="结果图"
   
'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--img_dir', type=str,default='DIV2K_test_100/DIV2K_train_LR_bicubic_801_900_X2/',
                    help='directory of images')
parser.add_argument('--output_dir', '-d', type=str, default='./DIV2K_test_predicted',
                    help='the name of the "output" folder')

args = parser.parse_args()



def predict(image_path):
    """
    Super-resolution on the input image using the model.

    :param args:
    :return:
        'predictions' contains an array of every single cropped sub-image once enhanced (the outputs of the model).
        'image' is the original image, untouched.
        'crops' is the array of every single cropped sub-image that will be used as input to the model.
    """
    image = skimage.io.imread(image_path)[:, :, :3]  # removing possible extra channels (Alpha)
    print("Image shape:", image.shape)

    predictions = []
    images = []

    # Padding and cropping the image
    overlap_pad = (overlap, overlap)  # padding tuple
    pad_width = (overlap_pad, overlap_pad, (0, 0))  # assumes color channel as last
    padded_image = np.pad(image, pad_width, 'constant')  # padding the border
    crops = seq_crop(padded_image)  # crops into multiple sub-parts the image based on 'input_' constants

    # Arranging the divided image into a single-dimension array of sub-images
    for i in range(len(crops)):         # amount of vertical crops
        for j in range(len(crops[0])):  # amount of horizontal crops
            current_image = crops[i][j]
            images.append(current_image)

    print("Moving on to predictions. Amount:", len(images))
    upscaled_overlap = overlap * 2
    for p in range(len(images)):
        if p % 3 == 0 and verbosity == 2:
            print("--prediction #", p)

        # Hack due to some GPUs that can only handle one image at a time
        input_img = (np.expand_dims(images[p], 0))  # Add the image to a batch where it's the only member
        input_img.astype('float32').tofile(os.path.join("./input_bins","test.bin"))
        os.system("../Benchmark/out/benchmark --om ResCNN_{}_{}_1batch.om --dataDir ./input_bins --modelType ResCNN --outDir results --batchSize 1 --imgType bin --useDvpp 0".format(input_img.shape[1],input_img.shape[2]))
        pred = np.fromfile("results/ResCNN/davinci_test_output0.bin",dtype='float32').reshape(input_img.shape[1]*4,input_img.shape[2]*4,3)
        #pred = model.predict(input_img)[0]          # returns a list of lists, one for each image in the batch

        # Cropping the useless parts of the overlapped predictions (to prevent the repeated erroneous edge-prediction)
        pred = pred[upscaled_overlap:pred.shape[0]-upscaled_overlap, upscaled_overlap:pred.shape[1]-upscaled_overlap]

        predictions.append(pred)
    return predictions, image, crops


def show_pred_output(input, pred):
    plt.figure(figsize=(20, 20))
    plt.suptitle("Results")

    plt.subplot(1, 2, 1)
    plt.title("Input : " + str(input.shape[1]) + "x" + str(input.shape[0]))
    plt.imshow(input, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 2, 2)
    plt.title("Output : " + str(pred.shape[1]) + "x" + str(pred.shape[0]))
    plt.imshow(pred, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()


# adapted from  https://stackoverflow.com/a/52463034/9768291
def seq_crop(img):
    """
    To crop the whole image in a list of sub-images of the same size.
    Size comes from "input_" variables in the 'constants' (Evaluation).
    Padding with 0 the Bottom and Right image.

    :param img: input image
    :return: list of sub-images with defined size (as per 'constants')
    """
    sub_images = []  # will contain all the cropped sub-parts of the image
    j, shifted_height = 0, 0
    while shifted_height < (img.shape[0] - input_height):
        horizontal = []
        shifted_height = j * (input_height - overlap)
        i, shifted_width = 0, 0
        while shifted_width < (img.shape[1] - input_width):
            shifted_width = i * (input_width - overlap)
            horizontal.append(crop_precise(img,
                                           shifted_width,
                                           shifted_height,
                                           input_width,
                                           input_height))
            i += 1
        sub_images.append(horizontal)
        j += 1

    return sub_images


def crop_precise(img, coord_x, coord_y, width_length, height_length):
    """
    To crop a precise portion of an image.
    When trying to crop outside of the boundaries, the input to padded with zeros.

    :param img: image to crop
    :param coord_x: width coordinate (top left point)
    :param coord_y: height coordinate (top left point)
    :param width_length: width of the cropped portion starting from coord_x (toward right)
    :param height_length: height of the cropped portion starting from coord_y (toward bottom)
    :return: the cropped part of the image
    """
    tmp_img = img[coord_y:coord_y + height_length, coord_x:coord_x + width_length]
    return float_im(tmp_img)  # From [0,255] to [0.,1.]


# adapted from  https://stackoverflow.com/a/52733370/9768291
def reconstruct(predictions, crops):
    """
    Used to reconstruct a whole image from an array of mini-predictions.
    The image had to be split in sub-images because the GPU's memory
    couldn't handle the prediction on a whole image.

    :param predictions: an array of upsampled images, from left to right, top to bottom.
    :param crops: 2D array of the cropped images
    :return: the reconstructed image as a whole
    """

    # unflatten predictions
    def nest(data, template):
        data = iter(data)
        return [[next(data) for _ in row] for row in template]

    if len(crops) != 0:
        predictions = nest(predictions, crops)

    # At this point "predictions" is a 3D image of the individual outputs
    H = np.cumsum([x[0].shape[0] for x in predictions])
    W = np.cumsum([x.shape[1] for x in predictions[0]])
    D = predictions[0][0]
    recon = np.empty((H[-1], W[-1], D.shape[2]), D.dtype)
    for rd, rs in zip(np.split(recon, H[:-1], 0), predictions):
        for d, s in zip(np.split(rd, W[:-1], 1), rs):
            d[...] = s

    # Removing the pad from the reconstruction
    tmp_overlap = overlap * (scale_fact - 1)  # using "-2" leaves the outer edge-prediction error
    return recon[tmp_overlap:recon.shape[0]-tmp_overlap, tmp_overlap:recon.shape[1]-tmp_overlap]


if __name__ == '__main__':
    print("   -  ", args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    imgs = os.listdir(args.img_dir)
    imgs.sort()
    for single_img in imgs:
        preds, original, crops = predict(os.path.join(args.img_dir,single_img))  # returns the predictions along with the original
        enhanced = reconstruct(preds, crops)    # reconstructs the enhanced image from predictions

        # Save and display the result
        enhanced = np.clip(enhanced, 0, 1)
        plt.imsave(os.path.join(args.output_dir,single_img), enhanced, cmap=plt.cm.gray)
        show_pred_output(original, enhanced)

