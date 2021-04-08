from __future__ import print_function
from __future__ import division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random, copy
######modify for npu start######
#from scipy.misc import imread
from imageio import imread
######modify for npu end######
import ast
sys.path.append("models")

# from newUnet import build_unet
from deepUnet import build_deepUnet
from deep import build_deep
from AUnet import build_AUnet

#from unetmod import build_unetmod
from Encoder_Decoder import build_encoder_decoder


from tensorflow.contrib.layers import flatten
#from cnn_gru import build_cnn_rnn

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score


# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

smooth = 1.0
def smoothL1(logits, masks):
    logits = tf.nn.tanh(logits)
    x = tf.abs(masks-logits)
    var = 0.5
    t_loss = tf.where(x < var, 0.5 * x **2, var * ( x - 0.5 * var))
    return tf.reduce_sum(t_loss, axis=0)



def dice_loss(y_pred, y_true):
    y_pred_f = flatten(y_pred)
    y_true_f = flatten(y_true)
    product = (y_pred_f * y_true_f)
    intersection = tf.reduce_sum(product)
    #union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)
    coefficient = (2.0 * intersection + smooth)/(tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + smooth)
    dice_loss = (-coefficient)
    return dice_loss

def focal_loss(logits, labels):
    gamma = 2
    alpha = 0.25
    # y_pred=tf.nn.sigmoid(logits)
    # labels=tf.to_float(labels)
    # L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-(1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    #pt_1 = tf.where(tf.equal(masks, 1), pred, tf.ones_like(pred))
    #pt_0 = tf.where(tf.equal(masks, 0), pred, tf.zeros_like(pred))

    #loss = -tf.reduce_sum(alpha*tf.pow(1.0 - pt_1, gamma)*tf.log(pt_1))-tf.reduce_sum((1-alpha)*tf.pow(pt_0, gamma)*tf.log(1.0 - pt_0))
    return L


def generalized_dice_loss(y_pred, y_true):
    y_pred_f = flatten(y_pred)
    y_true_f = flatten(y_true)
    product = tf.multiply(y_pred_f, y_true_f)
    intersection = tf.reduce_sum(product)

# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    #if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        #raise Exception('Image and label must have the same dimensions!')

    if (crop_width < image.shape[1]) and (crop_height < image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)

        if len(label.shape) == 3:
            image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
        image = np.resize(image, (256, 256, 3))
        label = np.resize(label, (256, 256, 3))
    #else:
        #raise Exception('Crop shape exceeds image dimensions!')
    return image,label
# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def IoU(pred, valid, cl):
    tp = np.count_nonzero(np.logical_and(pred == cl, valid == cl))
    fn = np.count_nonzero(np.logical_and(pred != cl, valid == cl))
    fp = np.count_nonzero(np.logical_and(pred == cl, valid != cl))
    tn = np.count_nonzero(np.logical_and(pred != cl, valid != cl))
    return tp, fn, fp, tn

def divided_IoU(tp, fn, fp):
    try:
        return float(sum(tp)) / (sum(tp) + sum(fn) + sum(fp))
    except ZeroDivisionError:
        return 0


def divided_PixelAcc(tp, fn):
    try:
        return float(sum(tp)) / (sum(tp) + sum(fn))
    except ZeroDivisionError:
        return 0



def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou

def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out
eps = 25
def region_growing(img, start_point, seed):
    #print("processing...")
    list = []
    img = img[:, :, 1]
    outimg = np.zeros_like(img)
    #print(outimg.shape)
    max = -2000
    for row in range(len(outimg)):
        for col in range(len(outimg[0])):
            outimg[row, col] = img[row, col]
            if img[row,col] > max:
                max = img[row,col]

    list.append((start_point[0], start_point[1]))
    #processed = []
    factor = eps
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = max
        for coord in get8n(pix[0], pix[1], img.shape):
            if outimg[coord[0], coord[1]] != max and img[coord[0], coord[1]] > seed - factor and img[coord[0], coord[1]] < seed + factor:
                outimg[coord[0], coord[1]] = max
                list.append(coord)
        list.pop(0)

    #print("Region growing: done. ")
    #print("Starting post processing...")
    outimg2 = copy.deepcopy(outimg)
    for row in range(len(outimg)):
        for col in range(len(outimg[0])):
            counter = 0
            for coord in get8n(row, col, img.shape):
                if(outimg[coord[0], coord[1]] == max):
                    counter = counter + 1
            if counter > 2:
                outimg2[row, col] = max
    return outimg2

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou


def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes)

    total_pixels = 0.0

    for n in range(len(image_files)):
        #image = cv2.cvtColor(cv2.imread(image_files[n],-1), cv2.COLOR_BGR2RGB)#imread(image_files[n], mode="RGB")
        image = imread(image_files[n], mode="RGB")

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)


        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)

def get_model(model_name, input, num_classes, keep_prob, gpu=0, drop=False):
    #gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    #print('MODEL', model_name)
    if model_name == "encoder" or model_name=="encoder-decoder":
        network = build_encoder_decoder_skip(input, num_classes)
    elif model_name == "deepUnet" or model_name == "deepunet" or model_name == "deepUNet":
        network = build_deepUnet(input, num_classes)
    elif model_name == "UNet" or model_name == "unet" or model_name == "Unet":
        network = build_unet2(input, num_classes=num_classes)
    elif model_name == "aunet" or model_name == "Aunet" or model_name == "attentionNet":
        ######Modify for NPU#######
        # network = build_AUnet(input, num_classes=num_classes)
        network = build_AUnet(input, n_classes=num_classes)
        ######Modify for NPU#######
    elif model_name == "deep":
        network = build_deep(input, num_classes, gpu)
    else:
        raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help", model_name)
    return network
