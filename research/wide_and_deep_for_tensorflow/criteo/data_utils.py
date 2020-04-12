import config as config
import tensorflow as tf
import os


def input_fn_tfrecord(tag, batch_size=16,
                      num_epochs=1, num_parallel=16, perform_shuffle=False, line_per_sample=1000):

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.FixedLenFeature(shape=(line_per_sample, ), dtype=tf.float32),
            'feat_ids': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.int64),
            'feat_vals': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.float32),
        }
        sample = tf.parse_single_example(data_record, features)
        sample['feat_ids'] = tf.cast(sample['feat_ids'], dtype=tf.int32)
        sample['feat_vals'] = sample['feat_vals']*10
        return sample

    path = config.record_path
    all_files = os.listdir(path)
    files = [os.path.join(path,f) for f in all_files if f.startswith(tag)]
    dataset = tf.data.TFRecordDataset(files).map(extract_fn, num_parallel_calls=num_parallel).batch(int(batch_size), drop_remainder=True).repeat(num_epochs)
    # Randomizes input using a window of batch_size elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(int(config.batch_size * 1))

    # epochs from blending together.
    return dataset
