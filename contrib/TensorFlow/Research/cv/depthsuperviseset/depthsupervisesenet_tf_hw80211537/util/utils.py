
import os, glob, copy, random, cv2, scipy, time
import tensorflow as tf
from six import iteritems
from sklearn.metrics import roc_curve
import numpy as np
from scipy import interp
interval_save = 500
suffix = '.jpg'

def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)

def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)

def write_arguments_to_file(args, filename):
    """
    :param args:
    :param filename:
    :return: write args parameter to file
    """
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

############## Data Load ########################
class Oulu:
    def __init__(self, protocol, mode):
        ### protocol ###
        protocol_dict = {}
        protocol_dict['oulu_protocal_1'] = {'train': {'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6],
                                                      'users': list(range(1, 21)), 'PAI': [1, 2, 3, 4, 5]},
                                            'dev': {'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6],
                                                    'users': list(range(21, 36)), 'PAI': [1, 2, 3, 4, 5]},
                                            'test': {'session': [3], 'phones': [1, 2, 3, 4, 5, 6],
                                                     'users': list(range(36, 56)), 'PAI': [1, 2, 3, 4, 5]}
                                            }

        protocol_dict['oulu_protocal_2'] = {'train': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6],
                                                      'users': list(range(1, 21)), 'PAI': [1, 2, 4]},
                                            'dev': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6],
                                                    'users': list(range(21, 36)), 'PAI': [1, 2, 4]},
                                            'test': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6],
                                                     'users': list(range(36, 56)), 'PAI': [1, 3, 5]}
                                            }

        protocol_dict['oulu_protocal_3'] = {'train': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5],
                                                      'users': list(range(1, 21)), 'PAI': [1, 2, 3, 4, 5]},
                                            'dev': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5],
                                                    'users': list(range(21, 36)), 'PAI': [1, 2, 3, 4, 5]},
                                            'test': {'session': [1, 2, 3], 'phones': [6],
                                                     'users': list(range(36, 56)), 'PAI': [1, 2, 3, 4, 5]}
                                            }
        for i in range(6):
            protocol_dict['oulu_protocal_3@%d'%(i+1)] = copy.deepcopy(protocol_dict['oulu_protocal_3'])
            protocol_dict['oulu_protocal_3@%d'%(i+1)]['train']['phones'] = []
            protocol_dict['oulu_protocal_3@%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['oulu_protocal_3@%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['oulu_protocal_3@%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['oulu_protocal_3@%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['oulu_protocal_3@%d'%(i+1)]['dev']['phones'].append(j+1)


        protocol_dict['oulu_protocal_4'] = {'train': {'session': [1, 2], 'phones': [1, 2, 3, 4, 5],
                                                      'users': list(range(1, 21)), 'PAI': [1, 2, 4]},
                                            'dev': {'session': [1, 2], 'phones': [1, 2, 3, 4, 5],
                                                    'users': list(range(21, 36)), 'PAI': [1, 2, 4]},
                                            'test': {'session': [3], 'phones': [6],
                                                     'users': list(range(36, 56)), 'PAI': [1, 3, 5]}
                                            }
        for i in range(6):
            protocol_dict['oulu_protocal_4@%d'%(i+1)] = copy.deepcopy(protocol_dict['oulu_protocal_4'])
            protocol_dict['oulu_protocal_4@%d'%(i+1)]['train']['phones'] = []
            protocol_dict['oulu_protocal_4@%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['oulu_protocal_4@%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['oulu_protocal_4@%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['oulu_protocal_4@%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['oulu_protocal_4@%d'%(i+1)]['dev']['phones'].append(j+1)


        protocol_dict['oulu_protocal_all'] = {'train': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6],
                                                        'users': list(range(1, 21)), 'PAI': [1, 2, 3, 4, 5]},
                                              'dev': {'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6],
                                                      'users': list(range(21, 36)), 'PAI': [1, 2, 3, 4, 5]},
                                              'test': {'session': [3], 'phones': [1, 2, 3, 4, 5, 6],
                                                       'users': list(range(36, 56)), 'PAI': [1, 2, 3, 4, 5]}
                                              }

        self.protocol_dict = protocol_dict
        if not (protocol in self.protocol_dict.keys()):
            print('error: Protocal should be ', list(self.protocol_dict.keys()))
            exit(1)
        self.protocol = protocol
        self.mode = mode
        self.protocol_info = protocol_dict[protocol][mode]

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_') #1_1_01_1.avi - ['1', '1', '01', '1.avi']
        if not len(name_split) == 4:
            return False
        [phones_, session_, users_, PAI_] = [int(x) for x in name_split]
        if (phones_ in self.protocol_info['phones']) and (session_ in self.protocol_info['session']) \
                and (users_ in self.protocol_info['users']) and (PAI_ in self.protocol_info['PAI']):
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('********** Dataset Info **********')
        print('Data:Oulu, protocol:{}, Mode:{}'.format(self.protocol, self.mode))
        print('All counts={} vs Protocal counts={}'.format(len(file_list),len(res_list)))
        print('**********************************')
        return res_list

class ImageClass():
    """
    Stores the paths of images for a given video
    input: video_name, image_paths
    output: class(include three functions)
    """
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
    def __len__(self):
        return len(self.image_paths)

def load_oulu_npu(data_path, protocol, mode, modal='ccrop', is_sort=True):
    """
    Function: load Oulu-NPU
    Protocol: oulu_protocal_1, ...
    """
    if mode == 'train':
        data_path = os.path.join(data_path, 'Train_images')
    elif mode == 'dev':
        data_path = os.path.join(data_path, 'Dev_images')
    elif mode == 'test':
        data_path = os.path.join(data_path, 'Test_images')
    dataset = []
    FILES_LIST = glob.glob(os.path.join(data_path, '*'))
    if len(FILES_LIST) == 0:
        print('{} empty, please check it !'.format(data_path))
        exit(1)
    data_object = Oulu(protocol, mode)
    FILES_LIST = data_object.dataset_process(FILES_LIST)
    for i in range(len(FILES_LIST)):
        video_name = FILES_LIST[i]
        facedir = os.path.join(video_name, modal)
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
        if is_sort:image_paths.sort()  ### Guaranteed continuous frames
        else:random.shuffle(image_paths)  ### Shuffle continuous frames
        label_p = video_name.split('_')[-1]
        video_name = '_'.join(['oulu', label_p])
        dataset.append(ImageClass(video_name, image_paths))  ### 1_1_01_1, 151 images
    return dataset

### First(training): real(label=0), attack(label=1) ###
def video_2_label(video_name):
    label = int(video_name.split('_')[-1])
    label = 0 if label == 1 else 1
    data_name = video_name.split('_')[0]
    return label, data_name
def get_sframe_paths_labels(dataset, phase, num='one', ratio=1):
    image_paths_flat = []
    labels_flat = []
    data_name_flat = []
    for i in range(len(dataset)):
        label, data_name = video_2_label(dataset[i].name)
        if phase == 'train':
            if label == 0:ratio_ = 1 ### real
            else:ratio_ = ratio      ### fake
            sample_image_paths = \
                [dataset[i].image_paths[sam_idx] for sam_idx in range(0, len(dataset[i].image_paths), ratio_)]
            image_paths_flat += sample_image_paths
            labels_flat += [label] * len(sample_image_paths)
            data_name_flat += [data_name] * len(sample_image_paths)
        elif (phase == 'dev') or (phase == 'test'):
            if num == 'one':load_num = 1
            elif num == 'all':load_num = len(dataset[i].image_paths)
            ### image_paths_flat += random.sample(dataset[i].image_paths, batch_size_val)
            image_paths_flat += dataset[i].image_paths[1:1 + load_num] ### In order to get stable results
            labels_flat += [label] * load_num
            data_name_flat += [data_name] * load_num
    assert len(image_paths_flat) == len(labels_flat) == len((data_name_flat))
    return image_paths_flat, labels_flat, data_name_flat

def distort_color(image, color_ordering, alpha=8, beta=0.2, gamma=0.05):
    if (color_ordering ==1) or (color_ordering ==0):
        image = image
    elif color_ordering ==2:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
    elif color_ordering ==3:
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==4:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==5:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==6:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==7:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==8:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
    elif color_ordering ==9:
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
        image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
    elif color_ordering ==10:
        image = tf.image.random_saturation(image, lower=1.0 - beta, upper=1.0 + beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
    else:
        print('color_ordering is error!', color_ordering)
        exit(0)
    return image
def get_aug_flag(flag, set):
    return tf.equal(set, flag)
def align_imagee_py(image_decoded, target_image_size):
    image_decoded = image_decoded.decode(encoding='UTF-8')
    image_size = cv2.imread(image_decoded).shape
    size_h, size_w = image_size[0], image_size[1]
    resize_flag = False
    if (size_h < target_image_size[0]) or (size_w < target_image_size[1]):
        resize_flag = True
        size_h, size_w = target_image_size[0], target_image_size[1]
        size_h = np.array(size_h).astype('int64')
        size_w = np.array(size_w).astype('int64')
    return size_h, size_w, resize_flag
def resize_py_image(image_decoded, image_size):
    image_size = tuple(image_size)
    image_resized = cv2.resize(image_decoded, image_size)
    return image_resized
def random_rotate_image(image, max_angle, domain):
    domain = domain.decode(encoding='UTF-8')
    tan = tf.contrib.image.rotate(image, 50)
    if domain == 'oulu':
        modal = 'ccrop'
    else:modal = domain
    if ('color' == modal) or ('profile' == modal) or ('ccrop' == modal):
        angle = np.random.uniform(low=-max_angle, high=max_angle)
    else:angle = max_angle
    return scipy.ndimage.interpolation.rotate(image, angle), angle

def replace_py(image_name_1, domain):
    image_name_1 = image_name_1.decode(encoding='UTF-8')
    domain = domain.decode(encoding='UTF-8')
    if domain == 'oulu':
        modal_1, modal_2 = 'ccrop', 'prnet'
        image_name_2 = image_name_1.replace(os.path.basename(image_name_1), 'prn_depth' + suffix)
    image_name_2 = image_name_2.replace(modal_1, modal_2)
    if not check_if_exist(image_name_2):
        return image_name_1
    return image_name_2

def is_preprocess_imagenet(image, flag):
    image = tf.cast(image, tf.float32)
    return image
def depth_image_label(image, label):
    label = tf.to_float(label, name='to_float')
    return tf.multiply(image, tf.subtract(1.0, label))

def create_pipeline(threads, input_queue, seed, color_size, depth_size, color_mean_div, depth_mean_div,
                    batch_size_p, batch_size,data_augment=[0, 0, 0, 0, 0], disorder_para=[8, 0.2, 0.05]):
    ### Training
    max_angle = data_augment[0]
    RANDOM_FLIP = data_augment[1]
    RANDOM_CROP = data_augment[2]
    RANDOM_COLOR = data_augment[3]
    is_std = data_augment[4]
    c_alpha = int(disorder_para[0])
    c_beta = disorder_para[1]
    c_gamma = disorder_para[2]
    ### testing
    crop_flag = RANDOM_CROP
    color_flag = 0
    images_and_labels_list = []
    for _ in range(threads):
        head_filenames, label, _ = input_queue.dequeue()
        head_color_images = []
        head_depth_images = []
        head_depth_labels = []
        for head_filename in tf.unstack(head_filenames):
        ####### Head file for temporal network
        ####### color ########
            image_color_modal = tf.image.decode_image(tf.io.read_file(head_filename), channels=3)
            ### @0: get info of rgb_size
            rgb_size_h, rgb_size_w, resize_flag = \
                tuple(tf.py_func(align_imagee_py, [head_filename, color_size], [tf.int64, tf.int64, tf.bool]))
            ### @1: resize image_color_modal
            image_color_modal = tf.cond(resize_flag,
                            lambda: tf.py_func(resize_py_image,[image_color_modal, (rgb_size_w, rgb_size_h)], tf.uint8),
                            lambda: image_color_modal)
            ### @2: RANDOM_ROTATE
            image_color_modal, angle = \
                tuple(tf.py_func(random_rotate_image, [image_color_modal, max_angle, 'oulu'], [tf.uint8, tf.double]))

            ### @3: Distort_color
            if RANDOM_COLOR == 1: color_flag += 1
            image_color_modal = tf.cond(get_aug_flag(RANDOM_COLOR, 1),
                                    lambda: distort_color(image_color_modal, color_flag, c_alpha, c_beta, c_gamma),
                                    lambda: tf.identity(image_color_modal))
            ### @4: Random flip
            image_color_modal = tf.cond(get_aug_flag(RANDOM_FLIP, 1),
                                        lambda: tf.image.random_flip_left_right(image_color_modal, seed=seed),
                                        lambda: tf.identity(image_color_modal))
            ### @5: Crop_Resize
            if (RANDOM_CROP == 1): crop_flag = int(not crop_flag)
            image_color_modal = tf.cond(get_aug_flag(crop_flag, 1),
                                        lambda: tf.image.random_crop(image_color_modal, color_size + (3,), seed=seed),
                                        lambda: tf.py_func(resize_py_image, [image_color_modal, color_size], tf.uint8))
            ### FIXED_STANDARDIZATION
            image_color_modal = (tf.cast(image_color_modal, tf.float32) - color_mean_div[0]) / color_mean_div[1]
            image_color_modal = is_preprocess_imagenet(image_color_modal, is_std)
            image_color_modal.set_shape(color_size + (3,))
            head_color_images.append(image_color_modal)

        ####### depth ########
            ### @1: resize image_depth_modal
            image_depth_modal = tf.image.decode_image(
                tf.io.read_file(tf.py_func(replace_py, [head_filename, 'oulu'], tf.string)), channels=3)
            image_depth_modal = tf.py_func(resize_py_image, [image_depth_modal, (rgb_size_w, rgb_size_h)], tf.uint8)
            ### @2: RANDOM_ROTATE
            image_depth_modal, _ = \
                tuple(tf.py_func(random_rotate_image, [image_depth_modal, angle, 'depth'], [tf.uint8, tf.double]))
            ### @3: Distort_color
            image_depth_modal = tf.cond(get_aug_flag(RANDOM_COLOR, 1),
                                        lambda: distort_color(image_depth_modal, color_flag, c_alpha, c_beta, c_gamma),
                                        lambda: tf.identity(image_depth_modal))
            ### @4: Random flip
            image_depth_modal = tf.cond(get_aug_flag(RANDOM_FLIP, 1),
                            lambda: tf.image.random_flip_left_right(image_depth_modal, seed=seed),
                            lambda: tf.identity(image_depth_modal))
            ### @5: Crop_Resize
            if (RANDOM_CROP == 1): crop_flag = crop_flag
            image_depth_modal = tf.cond(get_aug_flag(crop_flag, 1),
                            lambda: tf.image.random_crop(image_depth_modal, color_size + (3,), seed=seed),
                            lambda: tf.py_func(resize_py_image, [image_depth_modal, color_size], tf.uint8))

            image_depth_modal = tf.py_func(resize_py_image, [image_depth_modal, depth_size], tf.uint8)
            ### FIXED_STANDARDIZATION
            image_depth_modal = (tf.cast(image_depth_modal, tf.float32) - depth_mean_div[0]) / depth_mean_div[1]
            image_depth_modal = is_preprocess_imagenet(image_depth_modal, is_std)
            image_depth_modal.set_shape(depth_size + (3,))
            head_depth_images.append(image_depth_modal)
            ###
            depth_label = depth_image_label(image_depth_modal, label[0])
            depth_label.set_shape(depth_size + (3,))
            head_depth_labels.append(depth_label)
        ########## end images #######
        images_and_labels_list.append(
            [head_color_images, head_depth_images, head_depth_labels, label, ['oulu'], head_filenames])
    color_batch, depth_batch, depth_label_batch, label_batch, domain_batch, filename_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_p,shapes=[color_size + (3,),depth_size + (3,), depth_size + (3,),
        (), (), ()], enqueue_many=True, capacity=4 * threads * batch_size, allow_smaller_final_batch=True)
    return color_batch, depth_batch, depth_label_batch, label_batch, domain_batch, filename_batch

def get_saver_tf(max_to_keep=1024):
    def show_vars(trainable_list):
        for i in range(len(trainable_list)):
            var_name = trainable_list[i].name
            print('{} {}'.format(i, var_name))
    ### save global_variables ###
    global_list = tf.compat.v1.global_variables()
    print('***** Network parameter Info *******')
    print('*** global={}'.format(len(global_list)))
    ### Only save trainable ###
    trainable_list = tf.compat.v1.trainable_variables()
    print('*** trainable={}'.format(len(trainable_list)))
    #### save trainable + bn ###
    bn_moving_vars = [g for g in global_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]
    trainable_list += bn_moving_vars
    print('*** trainable@bn_moving_vars={}'.format(len(trainable_list)))
    saver = tf.compat.v1.train.Saver(trainable_list, max_to_keep=max_to_keep)
    ### move out some variable ###
    restore_trainable_list = [t for t in trainable_list if 'logits' not in t.name]
    ###
    # show_vars(trainable_list)
    # show_vars(restore_trainable_list)
    saver_restore = tf.compat.v1.train.Saver(restore_trainable_list)
    return saver, saver_restore, trainable_list, restore_trainable_list

def get_train_op(total_loss, trainable_list, global_step, optimizer, learning_rate):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    elif optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimization algorithm')
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        grads = opt.compute_gradients(total_loss, var_list=trainable_list)
        train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op

def get_lr(initial_lr, lr_decay_epochs, epoch_idx, lr_decay=0.1):
    lr = initial_lr
    for s in lr_decay_epochs:
        if epoch_idx >= s:
            lr *= lr_decay
    return lr

def write_ScoreImages(save_path, color_batch, depth_batch, depth_label_batch, label_batch, domain_batch,
                      filename_batch, color_mean_div, depth_mean_div, batch_it, logits, depth_map, accuracy, fid,
                      is_save_images=True):
    ### parameters ###
    logit_acc_mean = 0.0
    depth_acc_mean = 0.0
    accuracy_mean = 0.0
    def realProb(logits):
        x = np.array(logits)
        if np.isinf(np.sum(np.exp(x))):
            return 0
        y = np.exp(x[0]) / np.sum(np.exp(x))
        return y
    for frame_ind in range(len(color_batch)):
        frame_all = batch_it * len(color_batch) + frame_ind
        sample_label = label_batch[frame_ind]
        domain = domain_batch[frame_ind].decode(encoding='UTF-8')
        P_list = filename_batch[frame_ind].decode(encoding='UTF-8').split('/')
        P = [domain, 'G({})'.format(str(sample_label)), P_list[-1]]
        sample_name = '_'.join(P)
        assert depth_map[frame_ind, :, :, :].shape[-1] == 1
        depth_map_image = np.squeeze(depth_map[frame_ind, :, :, :], axis=-1)
        depth_map_image = (depth_map_image * depth_mean_div[1]) + depth_mean_div[0]
        if (is_save_images) and (frame_ind == 0):
        ### sptial_image ###
            ### @1: color
            color_image = color_batch[frame_ind, :, :, 0:3]
            color_image = (color_image * color_mean_div[1]) + color_mean_div[0]
            cv2.imwrite(
                os.path.join(save_path, sample_name + '@' + str(batch_it) + '@' + str(frame_ind) + '@1_Color.jpg'),
                cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            ### @2: Depth
            depth_image = depth_batch[frame_ind, :, :, :]
            depth_image = (depth_image * depth_mean_div[1]) + depth_mean_div[0]
            cv2.imwrite(os.path.join(save_path, sample_name + '@' + str(batch_it) + '@' + str(frame_ind) + '@2_Depth.jpg'),
                        cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR))
            ###
            depth_image = depth_batch[frame_ind, :, :, :]
            depth_binary = np.array(depth_image > depth_image.min(), np.float32) * 255.0
            cv2.imwrite(os.path.join(save_path, sample_name + '@' + str(batch_it) + '@' + str(frame_ind) + '@3_Binary.jpg'),
                    cv2.cvtColor(depth_binary, cv2.COLOR_RGB2BGR))
            ### @3: Depth_Label
            depth_label = depth_label_batch[frame_ind, :, :, :]
            depth_label = (depth_label * depth_mean_div[1]) + depth_mean_div[0]
            cv2.imwrite(os.path.join(save_path, sample_name + '@' + str(batch_it) + '@' + str(frame_ind) + '@3_Label.jpg'),
                    cv2.cvtColor(depth_label, cv2.COLOR_RGB2BGR))
            ### @4: Depth_Map
            cv2.imwrite(os.path.join(save_path, sample_name + '@' + str(batch_it) + '@' + str(frame_ind) + '@4_Map.jpg'),
                    cv2.cvtColor(depth_map_image, cv2.COLOR_RGB2BGR))
        ### testing ###
        if (save_path.find('dev') == -1) and (save_path.find('test') == -1):
            continue
        ### compute logical score ###
        out = np.argmax(np.array(logits[frame_ind]))
        logit_acc = int(out == sample_label)
        logit_acc_mean += float(logit_acc)
        accuracy_mean += accuracy
        logit_score = realProb(logits[frame_ind])
        ### compute depth score ###
        depth_map_image_norm = depth_map_image / 255.0  ### 0~1
        depth_binary = np.array(depth_batch[frame_ind, :, :, :] > depth_batch[frame_ind, :, :, :].min(), np.float32)
        depth_mean_score = np.sum(depth_map_image_norm) / np.sum(depth_binary[..., 0])
        # assert np.min(depth_batch[frame_ind, :, :, :]) == 0
        # print(np.sum(depth_map_image_norm), np.sum(depth_binary[..., 0]), depth_mean_score)
        # assert depth_mean_score <= 20.0
        depth_score = [depth_mean_score, 1.0 - depth_mean_score]
        out = np.argmax(np.array(depth_score))
        depth_acc = int(out == sample_label)
        depth_acc_mean += float(depth_acc)
        map_score = realProb(depth_score)
        ### write score ###
        fid.write(domain + '@GT_' + str(sample_label) + ',' + str(logit_score) + ',' + str(map_score) + '\n')
    if (save_path.find('dev') == -1) and (save_path.find('test') == -1): return True
    else:print('* batch_it={}, sample_name={}, batch_size={}, all_samples={}, depth_acc={} logit_acc/accuracy={}/{}'
        .format(batch_it + 1, sample_name, (frame_ind + 1), frame_all + 1, str(depth_acc_mean / (frame_ind + 1)),
        str(logit_acc_mean / (frame_ind + 1)), str(accuracy_mean / (frame_ind + 1))))

def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    ### Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)


def eval_acc(threshold, diff):
    """d[0],d[1],d[2] = videoname,score,label"""
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[1]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[2]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * (np.count_nonzero(y_true == y_predict) / len(y_true))
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def metric_casia_race(predicts, scores_dev, labels_dev, scores_test, labels_test):
    def get_4_index(predicts, thresh):
        """d[0],d[1],d[2] = videoname, score, label"""
        FN = 0
        FN_images = []
        FP = 0
        FP_images = []
        TP = 0
        TN = 0
        for d in predicts:
            if (int(d[2]) == 1) and (float(d[1]) >= thresh):
                TP += 1
            elif (int(d[2]) == 1) and (float(d[1]) < thresh):
                FN += 1
                FN_images.append([d[0], d[1]])
            elif (int(d[2]) == 0) and (float(d[1]) < thresh):
                TN += 1
            elif (int(d[2]) == 0) and (float(d[1]) >= thresh):
                FP += 1
                FP_images.append([d[0], d[1]])
        return FP, FN, TP, TN, FN_images, FP_images

    def get_eer_threhold(fpr, tpr, threshold):
        differ_tpr_fpr_1 = tpr + fpr - 1.0
        right_index = np.argmin(np.abs(differ_tpr_fpr_1))
        best_th = threshold[right_index]
        eer = fpr[right_index]
        return eer, best_th

    dev_fpr, dev_tpr, dev_thre = roc_curve(copy.deepcopy(labels_dev), copy.deepcopy(scores_dev), pos_label=1)
    dev_eer, dev_best_thre = get_eer_threhold(dev_fpr, dev_tpr, dev_thre)

    FP, FN, TP, TN, FN_images, FP_images = get_4_index(predicts, dev_best_thre)
    FAR = FP / (FP + TP + 1e-10)    ### False Acceptance Rate
    FRR = FN / (TN + FN + 1e-10)    ### False Rejection Rate
    HTER = (FAR + FRR) / 2          ### Half Total Error Rate
    APCER = FP / (TN + FP + 1e-10)  ### Attack Presentation Classification Error Rate == FPR
    TNR = 1 - APCER                 ### True Negative Rate
    NPCER = FN / (FN + TP + 1e-10)  ### Normal Presentation Classification Error Rate
    TPR = 1 - NPCER                 ### True Positive Rate(TPR)
    ACER = (APCER + NPCER) / 2      ### Average Classification Error Rate
    ACC = (TP + TN) / (1e-10 + TP + FP + FN + TN)  ### classification accuracy

    test_fpr, test_tpr, test_threshold = roc_curve(copy.deepcopy(labels_test), copy.deepcopy(scores_test), pos_label=1)
    test_EER, test_best_thre = get_eer_threhold(test_fpr, test_tpr, test_threshold)
    Recall2 = interp(0.007, test_fpr, test_tpr)
    # Recall2 = interp(0.01, test_fpr, test_tpr)
    Recall3 = interp(0.001, test_fpr, test_tpr)
    Recall4 = interp(0.0001, test_fpr, test_tpr)

    return dev_best_thre, FP, FN, TP, TN, FN_images, FP_images, HTER, APCER, NPCER, ACER, \
           test_EER, Recall2, Recall3, Recall4

def metric_zcx(scores_dev, labels_dev, scores_test, labels_test):
    def get_eer_threhold(fpr, tpr, threshold):
        differ_tpr_fpr_1 = tpr + fpr - 1.0
        right_index = np.argmin(np.abs(differ_tpr_fpr_1))
        best_th = threshold[right_index]
        eer = fpr[right_index]
        return eer, best_th
    scores_dev = np.array(scores_dev)
    labels_dev = np.array(labels_dev)
    scores_test = np.array(scores_test)
    labels_test = np.array(labels_test)

    dev_fpr, dev_tpr, dev_thre = roc_curve(labels_dev, scores_dev, pos_label=1)
    dev_eer, dev_best_thre = get_eer_threhold(dev_fpr, dev_tpr, dev_thre)

    real_scores = scores_test[labels_test == 1]
    attack_scores = scores_test[labels_test == 0]
    APCER = np.mean(np.array(attack_scores >= dev_best_thre, np.float32))
    BPCER = np.mean(np.array(real_scores < dev_best_thre, np.float32))

    test_fpr, test_tpr, test_threshold = roc_curve(labels_test, scores_test, pos_label=1)
    test_EER, test_best_thre = get_eer_threhold(test_fpr, test_tpr, test_threshold)
    results = [test_EER, APCER, BPCER, (APCER + BPCER) / 2.0]
    return dev_best_thre, results

def metric_LAJ(scores_dev, labels_dev, scores_test, labels_test):
    def get_4_index(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
        fn = np.sum(np.logical_and(predict_issame, actual_issame))
        tn = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        fp = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        return tp, fn, tn, fp

    def metric(threshold, dist, actual_issame):
        tp, fn, tn, fp = get_4_index(threshold, dist, actual_issame)
        apcer = fp / (tn * 1.0 + fp * 1.0)
        npcer = fn / (fn * 1.0 + tp * 1.0)
        acer = (apcer + npcer) / 2.0
        return tp, fn, tn, fp, apcer, npcer, acer

    def find_threshold(dist, actual_issame):
        min_acer = 100
        best_threshold = 0
        thresholds = np.arange(0.0, 1.0, 0.01)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, _, _, _, acer = metric(threshold, dist, actual_issame)
            if acer < min_acer:
                best_threshold = threshold
                min_acer = acer
        return best_threshold, acer

    def get_err_threhold(fpr, tpr, threshold):
        differ_tpr_fpr_1 = tpr + fpr - 1.0
        right_index = np.argmin(np.abs(differ_tpr_fpr_1))
        best_th = threshold[right_index]
        eer = fpr[right_index]
        return eer, best_th

    dev_best_thre, min_acer = find_threshold(scores_dev, labels_dev)
    tp, fn, tn, fp, apcer, npcer, acer = metric(dev_best_thre, scores_test, labels_test)

    test_fpr, test_tpr, test_threshold = roc_curve(labels_test, scores_test, pos_label=1)
    test_EER, test_best_thre = get_err_threhold(test_fpr, test_tpr, test_threshold)

    ### EER ###
    # fpr, tpr, dev_thre = roc_curve(labels_dev, scores_dev, pos_label=1)
    # plt.figure()
    # plt.plot(1 - tpr, thresholds, marker='*', label='far')
    # plt.plot(fpr, thresholds, marker='o', label='fpr')
    # plt.legend()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.xlabel('thresh')
    # plt.ylabel('far/fpr')
    # plt.title(' find eer')
    # plt.show()

    return dev_best_thre, tp, fn, tn, fp, test_EER, apcer, npcer, acer

### <real:label==1, fake:label==0> (metric)
def performance(phases, scores_dir, score_ind=1):
    '''
    :param phases: ['train', 'dev', 'test']
    :param score_ind: 1:prob_score 2:exp_score
    '''
    predicts_test = []
    predicts_dev = []
    scores_test = []
    labels_test = []
    scores_dev = []
    labels_dev = []
    for phase in phases[1:]:
        locals()['predicts_' + phase] = []
        score_fid = open(os.path.join(scores_dir, phase.capitalize() + '_scores.txt'), 'r')
        lines = score_fid.readlines()
        score_fid.close()
        for line in lines:
            label = 1 - int(line.split(',')[0].split('_')[-1])
            locals()['predicts_' + phase].append([line.split(',')[0], float(line.split(',')[score_ind]), label])
            locals()['scores_' + phase].append(float(line.split(',')[score_ind]))  ### '1_275_1_1_1_G(0)_0001.jpg@GT_0'
            locals()['labels_' + phase].append(int(label))

    ### method_1: LFW type
    thresholds = np.arange(0, 1, 0.0010)
    dev_best_thre = find_best_threshold(thresholds, predicts_dev)
    accurace_dev = eval_acc(dev_best_thre, predicts_dev)
    accurace_test = eval_acc(dev_best_thre, predicts_test)
    print('@_LFW: Dev_best_thre={} Accurace_dev={} Accurace_test={}'.format(dev_best_thre, accurace_dev, accurace_test))
    ### method_2: CASIA-Race
    Dev_best_thre, FP, FN, TP, TN, FN_images, FP_images, HTER, APCER, NPCER, ACER, \
    test_EER, Recall2, Recall3, Recall4=metric_casia_race(predicts_test, scores_dev, labels_dev, scores_test, labels_test)
    # print("@_ZSF: Dev_best_thre={} TP={} FN={} FP={} TN={} EER={} APCER={} NPCER={} ACER={}".format(
    #     Dev_best_thre, TP, FN, FP, TN, test_EER, APCER,  NPCER, ACER))
    ### method_3: zcx
    Dev_best_thre, results = metric_zcx(scores_dev, labels_dev, scores_test, labels_test)
    # print('@_ZCX: Dev_best_thre={} EER={} APCER={} BPCER={} ACER={}'.
    #       format(Dev_best_thre, results[0], results[1], results[2], results[3]))
    ### method_4: LAJ
    dev_best_thre, tp, fn, tn, fp, test_EER, apcer, npcer, acer = \
        metric_LAJ(scores_dev, labels_dev, scores_test, labels_test)
    if acer < ACER:
        Dev_best_thre, TP, FN, FP, TN, test_EER, APCER, NPCER, ACER = \
        dev_best_thre, tp, fn, fp, tn, test_EER, apcer, npcer, acer
    print("@_Best: Dev_best_thre={} TP={} FN={} FP={} TN={} EER={} APCER={} NPCER={} ACER={}".format(
        Dev_best_thre, TP, FN, FP, TN, test_EER, APCER, NPCER, ACER))
    return Dev_best_thre, TP, FN, FP, TN, test_EER, APCER, NPCER, ACER, Recall2, Recall3, Recall4
