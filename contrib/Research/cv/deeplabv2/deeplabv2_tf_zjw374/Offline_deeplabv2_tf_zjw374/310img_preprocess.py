import tensorflow as tf
import numpy as np
import os

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

def read_image(img_path, label_path):

    img_contents = tf.read_file(img_path)
    label_contents = tf.read_file(label_path)

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    label = tf.image.decode_png(label_contents, channels=1)
    new_size = [321, 321]
    img = tf.image.resize_images(img,  new_size, method=RESIZE_METHOD)
    label = tf.image.resize_images(label, new_size, method=RESIZE_METHOD)
    
    return img, label

def main():

    f = open('./data/val.txt', 'r')
    val_list = f.readlines()
    for line in val_list:
        tf.reset_default_graph()
        with tf.Session() as sess:
            img_path = './dataset' + line.split(' ')[0]
            label_path = './dataset' + line.split(' ')[1].split('\n')[0]
            #print("Process %s" %(img_path))
            img, label = read_image(img_path, label_path)
            img_np, label_np = sess.run([img, label])
            print(img_np.dtype,label_np.dtype)
            print(img_np.shape, label_np.shape)
            img_bin_path = './bin_dataset' + line.split(' ')[0]+'.bin'
            label_bin_path ='./bin_dataset' + line.split(' ')[1].split('\n')[0]+'.bin'
            print(img_bin_path)
            img_np.tofile(img_bin_path)
            label_np.tofile(label_bin_path)
        tf.get_default_graph().finalize()
        #tf.get_default_graph().finalize()

if __name__ == "__main__":
    main()



