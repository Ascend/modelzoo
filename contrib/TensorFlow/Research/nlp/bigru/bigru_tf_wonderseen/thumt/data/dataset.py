# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator
import os
import tensorflow as tf
from thumt.data.vocab_utils import *
from tensorflow.python.data.ops import dataset_ops
import random
import numpy as np
from thumt.multiprocess.multiprocess import Pool
from thumt.parameter_config import data_dir


def re_path(path):
    if path.startswith("s3:"):
        return path
    else:
        return os.path.join(data_dir, path)


def read_txt(filename):
    with tf.gfile.GFile(filename, "r") as reader:
        return [l.strip() for l in reader.readlines()]


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True, src_length=60, tgt_length=61):
    """ Batch examples
    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """
    constant = True
    max_length = max_length or batch_size
    min_length = 8
    mantissa_bits = mantissa_bits

    # Compute boundaries
    x = min_length
    boundaries = []

    while x < max_length:
        boundaries.append(x)
        x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

    # Whether the batch size is constant
    if not constant:
        batch_sizes = [max(1, batch_size // length)
                       for length in boundaries + [max_length]]
        batch_sizes = [b * shard_multiplier for b in batch_sizes]
        bucket_capacities = [2 * b for b in batch_sizes]
    else:
        batch_sizes = batch_size * shard_multiplier
        bucket_capacities = [2 * n for n in boundaries + [max_length]]
        

    max_length *= length_multiplier
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    max_length = max_length if drop_long_sequences else 10 ** 9

    # The queue to bucket on will be chosen based on maximum length
    if using_dynamic:
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)
    else:
        max_example_length = max(src_length, tgt_length)

    with tf.name_scope("batch_examples"):
        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            keep_input=(max_example_length <= max_length),
            dynamic_pad=using_dynamic,
            shapes=[[max_example_length], [], [tgt_length], []] if not using_dynamic else None,
        )
    return outputs


def multiprocess_dataset(dataset):
    global params
    SEQ_LENGTH = params.train_encode_length
    DECODE_LENGTH = params.train_decode_length
    eos = params.eos
    bos = params.bos
    src_vocab = params.vocabulary["source"]
    src_unk_id = params.mapping["source"][params.unk]
    tgt_vocab = params.vocabulary["target"]
    tgt_unk_id = params.mapping["target"][params.unk]
    pad_id = params.mapping["source"][params.pad]
    src_max_len, tgt_max_len = SEQ_LENGTH, DECODE_LENGTH
    src_vocab_table = CPUVocab(src_vocab, default_value=src_unk_id)
    tgt_vocab_table = CPUVocab(tgt_vocab, default_value=tgt_unk_id)


    ## Add start or end symbols
    dataset = list(map(
        lambda src, tgt: (
            src.split() + [params.eos],
            [params.bos] + tgt.split() + [params.eos]
        ),
        [i[0] for i in dataset],
        [i[1] for i in dataset],
    ))

    ## Filter the sentences
    dataset = list(filter(
       lambda x:(
           len(x[0]) <= src_max_len and len(x[1]) <= tgt_max_len
       ),
       dataset
       )
    )
    # dataset = list(map(
    #     lambda src, tgt:(
    #     src[:SEQ_LENGTH],
    #     tgt[:DECODE_LENGTH],
    #     ),
    #     [i[0] for i in dataset],
    #     [i[1] for i in dataset]
    #     )
    # )

    ## Convert to dictionary
    dataset = list(map(
            lambda src, tgt:(
                src_vocab_table(src),
                tgt_vocab_table(tgt)
            ),
            [i[0] for i in dataset],
            [i[1] for i in dataset],
        )
    )

    ## padding
    features = list(map(
        lambda src, tgt: {
            "source_length": len(src),
            "target_length": len(tgt),
            "source": src + [pad_id] * (src_max_len - len(src)) if len(src) < src_max_len else src,
            "target": tgt + [pad_id] * (tgt_max_len - len(tgt)) if len(tgt) < tgt_max_len else tgt,

        },
        [d[0] for d in dataset],
        [d[1] for d in dataset],
    ))
    return features


def get_training_input(filenames, mparams, with_bucket=False):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    global params
    params = mparams # used in process_dataset multiprocess, but notice that maybe unsafe here.
    SEQ_LENGTH = params.train_encode_length
    DECODE_LENGTH = params.train_decode_length


    with tf.device("/cpu:0"):
        filenames = [re_path(f) for f in filenames]

        src_dataset = read_txt(filenames[0])
        tgt_dataset = read_txt(filenames[1])
        dataset = list(zip(src_dataset, tgt_dataset))

        ## single thread process
        def process_dataset(dt):
            # Split string
            dt = list(map(
                    lambda src, tgt: (
                    src.split() + [params.eos],
                    [params.bos] + tgt.split() + [params.eos]
                ),
                [i[0] for i in dt],
                [i[1] for i in dt],
            ))

            # dataset = list(map(
            #         lambda src, tgt:(
            #             src[:SEQ_LENGTH],
            #             tgt[:DECODE_LENGTH],
            #         ),
            #         [i[0] for i in dt],
            #         [i[1] for i in dt],
            #     )
            # )

            dt = list(
                filter(
                    lambda x:(
                        len(x[0]) <= SEQ_LENGTH and len(x[1]) <= DECODE_LENGTH
                    ),
                    dt
                )
            )

            src_vocab_table = CPUVocab(params.vocabulary["source"], default_value=params.mapping["source"][params.unk])
            tgt_vocab_table = CPUVocab(params.vocabulary["target"], default_value=params.mapping["target"][params.unk])
            dt = list(map(
                    lambda src, tgt:(
                        src_vocab_table(src),
                        tgt_vocab_table(tgt)
                    ),
                    [i[0] for i in dt],
                    [i[1] for i in dt],
                )
            )

            ## padding to static shape
            pad_id = params.mapping["source"][params.pad]
            features = list(map(
                lambda src, tgt: {
                    "source_length": len(src),
                    "target_length": len(tgt),
                 "source": src + [pad_id] * (SEQ_LENGTH - len(src)) if len(src) < SEQ_LENGTH else src,
                 "target": tgt + [pad_id] * (DECODE_LENGTH - len(tgt)) if len(tgt) < DECODE_LENGTH else tgt,
                },
                [d[0] for d in dt],
                [d[1] for d in dt],
            ))
            return features


        if_multi_thread_processing = False
        if if_multi_thread_processing:
            ## multi thread process     
            num_thread = 4
            p = Pool(num_thread)
            slice_num = len(dataset) // (num_thread*100) + 1        
            dataset = [dataset[i * slice_num : (i + 1) * slice_num] for i in range(num_thread * 100)]
            features = []
            ## for dt in p.map_async(multiprocess_dataset, dataset).get(): features.extend(dt)
            for dt in p.imap_unordered(multiprocess_dataset, dataset): features.extend(dt)
        else:
            ## single thread process
            features = process_dataset(dataset)


        ## global shuffle at first
        import random
        random.shuffle(features)

        from_tensor_slice = False # True: from_tensor_slice or False: from_generator
        if from_tensor_slice: ## from_slices (limited by the graph size)
            ## colloction
            sources = []
            targets = []
            source_lengths = []
            target_lengths = []
            for feature in features:
                sources.append(feature["source"])
                targets.append(feature["target"])
                source_lengths.append(feature["source_length"])
                target_lengths.append(feature["target_length"])

            def input_fn(params):
                """The actual input function."""
                batch_size = params.batch_size
                num_examples = len(features)
                d = tf.data.Dataset.from_tensor_slices({
                    "source":
                        tf.constant(
                            sources,
                            shape=[num_examples, SEQ_LENGTH],
                            dtype=tf.int32),
                    "target":
                        tf.constant(
                            targets,
                            shape=[num_examples, DECODE_LENGTH],
                            dtype=tf.int32),
                    "source_length":
                        tf.constant(
                            source_lengths,
                            shape=[num_examples],
                            dtype=tf.int32),
                    "target_length":
                        tf.constant(
                            target_lengths,
                            shape=[num_examples],
                            dtype=tf.int32),
                })

                ## locally shuffled iterator
                d = d.shuffle(params.buffer_size).repeat()

                ## if without bucket, directly into a batch-version, else comment out this line
                d = d.batch(batch_size=batch_size, drop_remainder=True)

                iterator = dataset_ops.make_one_shot_iterator(d)
                itera = iterator.get_next()
                return itera
            return input_fn(features, params)

        else:
        ## from_generator (bucket plus)
            def input_fn(params):
                batch_size = params.batch_size
                def nmt_generator():
                    for feature in features:
                        yield (
                            feature["source"],
                            feature["target"],
                            feature["source_length"],
                            feature["target_length"]
                        )
                d = tf.data.Dataset.from_generator(
                    nmt_generator,
                    (
                        tf.int32, # "source"
                        tf.int32, # "target"
                        tf.int32, # "source_length"
                        tf.int32, # "target_length"
                    ),
                    (
                        tf.TensorShape([SEQ_LENGTH]),
                        tf.TensorShape([DECODE_LENGTH]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                    ) 
                )

                ## locally shuffled iterator
                d = d.shuffle(params.buffer_size).repeat()

                ## if without bucket, directly into a batch-version, else comment out this line
                if not with_bucket:
                    d = d.batch(batch_size=batch_size, drop_remainder=True)

                iterator = dataset_ops.make_one_shot_iterator(d)
                itera = iterator.get_next()
                return itera

            source_tensor, target_tensor, source_length_tensor, target_length_tensor = input_fn(params)
            feature_tensor_dict = {
                "source": source_tensor,
                "target": target_tensor,
                "source_length": source_length_tensor,
                "target_length": target_length_tensor,
            }
            if not with_bucket:
                return feature_tensor_dict

            ## bucket
            features_bucket = batch_examples(feature_tensor_dict, params.batch_size,
                                      params.max_length, params.mantissa_bits,
                                      shard_multiplier=len(params.device_list),
                                      length_multiplier=params.length_multiplier,
                                      constant=params.constant_batch_size,
                                      num_threads=params.num_threads,
                                      src_length=SEQ_LENGTH, tgt_length=DECODE_LENGTH)
        
            return features_bucket 


def get_training_input_from_generator_without_bucket(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    SEQ_LENGTH = params.train_encode_length
    DECODE_LENGTH = params.train_decode_length
    with tf.device("/cpu:0"):
        filenames = [re_path(f) for f in filenames]

        src_dataset = read_txt(filenames[0])
        tgt_dataset = read_txt(filenames[1])
        dataset = list(zip(src_dataset, tgt_dataset))

        # dataset = sort_and_zip_files(filenames)
        import random
        random.shuffle(dataset)
        def process_dataset(dataset):
            ## Split string
            dataset = list(map(
                lambda src, tgt: (
                    src.split() + [params.eos],
                    [params.bos] + tgt.split() + [params.eos]
                ),
                [i[0] for i in dataset],
                [i[1] for i in dataset],
            ))

            ## Convert to dictionary
            # dataset = list(map(
            #         lambda src, tgt:(
            #             src[:SEQ_LENGTH],
            #             tgt[:DECODE_LENGTH],
            #         ),
            #         [i[0] for i in dataset],
            #         [i[1] for i in dataset],
            #     )
            # )
            dataset = list(filter(
                    lambda x:(
                        len(x[0]) <= SEQ_LENGTH and len(x[1]) <= DECODE_LENGTH
                    ),
                    dataset
                )
            )

            src_vocab_table = CPUVocab(params.vocabulary["source"], default_value=params.mapping["source"][params.unk])
            tgt_vocab_table = CPUVocab(params.vocabulary["target"], default_value=params.mapping["target"][params.unk])
            
            dataset = list(map(
                    lambda src, tgt:(
                        src_vocab_table(src),
                        tgt_vocab_table(tgt)
                    ),
                    [i[0] for i in dataset],
                    [i[1] for i in dataset],
                )
            )

            ## padding
            pad_id = params.mapping["source"][params.pad]
            features = list(map(
                lambda src, tgt: {
                    "source_length": np.array(len(src)),
                    "target_length": np.array(len(tgt)),
                    "source": np.array(src + [pad_id] * (SEQ_LENGTH - len(src)) if len(src) < SEQ_LENGTH else src),
                    "target": np.array(tgt + [pad_id] * (DECODE_LENGTH - len(tgt)) if len(tgt) < DECODE_LENGTH else tgt),

                },
                [d[0] for d in dataset],
                [d[1] for d in dataset],
            ))
            return features


        features = process_dataset(dataset)
        

        def input_fn(params):
            """The actual input function."""
            batch_size = params.batch_size
            
            ## from_generator
            def nmt_generator():
                random.shuffle(features)
                for feature in features:
                    yield (
                        feature["source"],
                        feature["target"],
                        feature["source_length"],
                        feature["target_length"]
                    )
            d = tf.data.Dataset.from_generator(
                nmt_generator,
                (
                    tf.int32, # "source"
                    tf.int32, # "target"
                    tf.int32, # "source_length"
                    tf.int32, # "target_length"
                ),
                (
                    tf.TensorShape([SEQ_LENGTH]),
                    tf.TensorShape([DECODE_LENGTH]),
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                ) 
            )

            ## something else massive
            d = d.shuffle(params.buffer_size).repeat()

            ## if without bucket, directly into a batch-version
            d = d.batch.batch(batch_size=batch_size, drop_remainder=True)
            iterator = dataset_ops.make_one_shot_iterator(d)
            itera = iterator.get_next()
            return itera

        source_tensor, target_tensor, source_length_tensor, target_length_tensor = input_fn(params)
        feature_tensor_dict = {
            "source": source_tensor,
            "target": target_tensor,
            "source_length": source_length_tensor,
            "target_length": target_length_tensor,
        }

        return feature_tensor_dict


def get_training_input_from_tensor_slice_without_bucket(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    SEQ_LENGTH = params.train_encode_length
    DECODE_LENGTH = params.train_decode_length
    with tf.device("/cpu:0"):
        filenames = [re_path(f) for f in filenames]

        src_dataset = read_txt(filenames[0])
        tgt_dataset = read_txt(filenames[1])
        dataset = list(zip(src_dataset, tgt_dataset))

        # dataset = sort_and_zip_files(filenames)

        def process_dataset(dataset): 
            ## Split string
            dataset = list(map(
                lambda src, tgt: (
                    src.split() + [params.eos],
                    [params.bos] + tgt.split() + [params.eos]
                ),
                [i[0] for i in dataset],
                [i[1] for i in dataset],
            ))

            ## Convert to dictionary
            # dataset = list(map(
            #         lambda src, tgt:(
            #             src[:SEQ_LENGTH],
            #             tgt[:DECODE_LENGTH],
            #         ),
            #         [i[0] for i in dataset],
            #         [i[1] for i in dataset],
            #     )
            # )
            dataset = list(filter(
                    lambda x:(
                        len(x[0]) <= SEQ_LENGTH and len(x[1]) <= DECODE_LENGTH
                    ),
                    dataset
                )
            )


            src_vocab_table = CPUVocab(params.vocabulary["source"], default_value=params.mapping["source"][params.unk])
            tgt_vocab_table = CPUVocab(params.vocabulary["target"], default_value=params.mapping["target"][params.unk])
            
            dataset = list(map(
                    lambda src, tgt:(
                        src_vocab_table(src),
                        tgt_vocab_table(tgt)
                    ),
                    [i[0] for i in dataset],
                    [i[1] for i in dataset],
                )
            )

            ## padding
            pad_id = params.mapping["source"][params.pad]
            features = list(map(
                lambda src, tgt: {
                    "source_length": np.array(len(src)),
                    "target_length": np.array(len(tgt)),
                    "source": np.array(src + [pad_id] * (SEQ_LENGTH - len(src)) if len(src) < SEQ_LENGTH else src),
                    "target": np.array(tgt + [pad_id] * (DECODE_LENGTH - len(tgt)) if len(tgt) < DECODE_LENGTH else tgt),

                },
                [d[0] for d in dataset],
                [d[1] for d in dataset],
            ))
            return features

        features = process_dataset(dataset)
        
        ## colloction
        ## Note that: from_slices is limited by the graph space
        sources = []
        targets = []
        source_lengths = []
        target_lengths = []
        for feature in features:
            sources.append(feature["source"])
            targets.append(feature["target"])
            source_lengths.append(feature["source_length"])
            target_lengths.append(feature["target_length"])


        def input_fn(params):
            """The actual input function."""
            batch_size = params.batch_size

            ## from_slices limit the GPU space
            num_examples = len(features)
            d = tf.data.Dataset.from_tensor_slices({
                "source":
                    tf.constant(
                        sources,
                        shape=[num_examples, SEQ_LENGTH],
                        dtype=tf.int32),
                "target":
                    tf.constant(
                        targets,
                        shape=[num_examples, DECODE_LENGTH],
                        dtype=tf.int32),
                "source_length":
                    tf.constant(
                        source_lengths,
                        shape=[num_examples],
                        dtype=tf.int32),
                "target_length":
                    tf.constant(
                        target_lengths,
                        shape=[num_examples],
                        dtype=tf.int32),
            })

            ## something else massive
            d = d.shuffle(params.buffer_size).repeat()

            ## if without bucket, directly into a batch-version
            d = d.batch.batch(batch_size=batch_size, drop_remainder=True)
            iterator = dataset_ops.make_one_shot_iterator(d)
            itera = iterator.get_next()
            return itera

        ## for from_tensor_slice
        return input_fn(params)


def sort_input_file(filename, reverse=True):
    # Read file  
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    names = [re_path(name) for name in names]
    
    files = [tf.gfile.GFile(name) for name in names]
    return [read_txt(names[0]), read_txt(names[1])]

    '''
    ## original version
    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1), reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
    return [list(x) for x in zip(*sorted_inputs)]
    '''


def get_evaluation_input(inputs, params):
    SEQ_LENGTH = params.eval_encode_length
    DECODE_LENGTH = params.eval_decode_length

    with tf.device("/cpu:0"):
        # Create datasets
        src_dataset = inputs[0]
        tgt_dataset = inputs[1]

        dataset = list(zip(src_dataset, tgt_dataset))
        
        ## Split string
        dataset = list(map(
            lambda src, tgt: (
                src.split() + [params.eos],
                tgt.split() + [params.eos]
            ),
            [i[0] for i in dataset],
            [i[1] for i in dataset],
        ))

        ## Convert to dictionary
        dataset = list(map(
                lambda src, ref:(
                    src[:SEQ_LENGTH],
                    ref[:DECODE_LENGTH],
                ),
                [i[0] for i in dataset],
                [i[1] for i in dataset],
            )
        )

        src_vocab_table = CPUVocab(params.vocabulary["source"], default_value=params.mapping["source"][params.unk])
        
        dataset = list(map(
                lambda src, ref:(
                    src_vocab_table(src),
                    ref
                ),
                [i[0] for i in dataset],
                [i[1] for i in dataset],
            )
        )


        ## padding
        pad_id = params.mapping["source"][params.pad]
        features = list(map(
            lambda src, ref: {
                "source": src + [pad_id] * (SEQ_LENGTH - len(src)) if len(src) < SEQ_LENGTH else src,
                "references": [ref + [params.pad] * (DECODE_LENGTH - len(ref)) if len(ref) < DECODE_LENGTH else ref],
                "source_length": len(src),
            },
            [d[0] for d in dataset],
            [d[1] for d in dataset],
        ))


        ## colloction
        sources = []
        referencess = []
        source_lengths = []
        for feature in features:
            sources.append(feature["source"])
            source_lengths.append(feature["source_length"])
            referencess.append(feature["references"])

        ## colloction
        sources = []
        source_lengths = []
        referencess = []
        for feature in features:
            sources.append(feature["source"])
            source_lengths.append(feature["source_length"])
            referencess.append(feature["references"])


        def input_fn(params):
            """The actual input function."""
            batch_size = params.eval_batch_size * len(params.device_list)
            num_examples = len(features)
            d = tf.data.Dataset.from_tensor_slices({
                "source":
                    tf.constant(sources, shape=[num_examples, SEQ_LENGTH], dtype=tf.int32),
                "references":
                    tf.constant(referencess, shape=[num_examples, params.reference_num, DECODE_LENGTH], dtype=tf.string),
                "source_length":
                    tf.constant(
                        source_lengths,
                        shape=[num_examples],
                        dtype=tf.int32),
                }           
            )
            d = d.batch(batch_size=batch_size, drop_remainder=True)#False) # result in a unfixed batch_size, not suitable for NPU
            iterator = dataset_ops.make_one_shot_iterator(d).get_next()
            return iterator
    features = input_fn(params)
    return features
