import config as config
import tensorflow as tf
import os


def input_fn_tfrecord(data_dir, tag, batch_size=16,
                      num_epochs=1, num_parallel=16, perform_shuffle=False, line_per_sample=1000):

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.FixedLenFeature(shape=(line_per_sample, ), dtype=tf.float32),
            'feat_ids': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.int64),
            'feat_vals': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.float32),
        }
        sample = tf.parse_single_example(data_record, features)
        return sample

    all_files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in all_files if f.startswith(tag)]
    #dataset = tf.data.TFRecordDataset(files).map(extract_fn, num_parallel_calls=num_parallel).batch(int(batch_size)).repeat(num_epochs)
    dataset = tf.data.TFRecordDataset(files).map(extract_fn, num_parallel_calls=num_parallel).batch(int(batch_size), drop_remainder=True).repeat(num_epochs)
    # Randomizes input using a window of batch_size elements (read into memory)
    #if perform_shuffle:
    #    dataset = dataset.shuffle(config.batch_size * 10)

    # epochs from blending together.
    return dataset
