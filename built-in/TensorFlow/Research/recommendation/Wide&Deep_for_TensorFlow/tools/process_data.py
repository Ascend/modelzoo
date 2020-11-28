# coding:utf-8

import os
import pickle
import collections
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd

TRAIN_LINE_COUNT = 45840617
TEST_LINE_COUNT = 6042135

class CriteoStatsDict():
    def __init__(self):
        self.field_size = 39 # value_1-13;  cat_1-26;
        self.val_cols = [ "val_{}".format(i+1) for i in range(13)]
        self.cat_cols = [ "cat_{}".format(i+1) for i in range(26)]
        #
        self.val_min_dict = { col: 0 for col in self.val_cols }
        self.val_max_dict = { col: 0 for col in self.val_cols }
        self.cat_count_dict = { col: collections.defaultdict(int)  for col in self.cat_cols }
        #
        self.oov_prefix = "OOV_"

        self.cat2id_dict = {}
        self.cat2id_dict.update( {col : i for i,col in enumerate( self.val_cols ) })
        self.cat2id_dict.update( {self.oov_prefix + col  :  i + len(self.val_cols) for i,col in enumerate(self.cat_cols) })
        # { "val_1": , ..., "val_13": ,  "OOV_cat_1": , ..., "OOV_cat_26": }
    #
    def stats_vals(self, val_list):
        assert len(val_list) == len(self.val_cols)
        def map_max_min(i,val):
            key = self.val_cols[i]
            if val != "":
                if float(val) > self.val_max_dict[ key ]:
                    self.val_max_dict[ key ] = float(val)
                if float(val) < self.val_min_dict[ key ]:
                    self.val_min_dict[ key ] = float(val)
            #
        for i,val in enumerate(val_list):
            map_max_min(i,val)
    #
    def stats_cats(self, cat_list):
        assert len(cat_list) == len(self.cat_cols)
        def map_cat_count(i,cat):
            key = self.cat_cols[i]
            self.cat_count_dict[ key ][ cat ] += 1
        #
        for i,cat in enumerate(cat_list):
            map_cat_count(i,cat)
    #
    def save_dict(self, output_path, prefix=""):
        with open(os.path.join(output_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(output_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(output_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)
    #
    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print( "val_max_dict.items()[:50]: {}".format( list(self.val_max_dict.items()) ) )
        print( "val_min_dict.items()[:50]: {}".format( list(self.val_min_dict.items()) ) )
        #
    #
    def get_cat2id(self, threshold=100):
        before_all_count = 0
        after_all_count = 0
        for key,cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1]> threshold, cat_count_d.items() ))
            for cat_str,count in new_cat_count_d.items():
                self.cat2id_dict[ key +"_"+ cat_str ] = len(self.cat2id_dict)
        # print("before_all_count: {}".format( before_all_count )) # before_all_count: 33762577
        # print("after_all_count: {}".format( after_all_count )) # after_all_count: 184926
        print( "cat2id_dict.size: {}".format( len(self.cat2id_dict) ) )
        print( "cat2id_dict.items()[:50]: {}".format( list(self.cat2id_dict.items())[:50] ) )
    #
    def map_cat2id(self, values, cats):
        def minmax_scale_value(i,val):
            # min_v = float(self.val_min_dict[ "val_{}".format(i+1) ])
            max_v = float(self.val_max_dict[ "val_{}".format(i+1) ])
            # return ( float(val) - min_v ) * 1.0 / (max_v - min_v)
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i,val in enumerate(values):
            if val == "":
                id_list.append( i )
                weight_list.append( 0 )
            else:
                key = "val_{}".format(i+1)
                id_list.append( self.cat2id_dict[key] )
                weight_list.append( minmax_scale_value(i, float(val) ) )
        #
        for i,cat_str in enumerate(cats):
            key = "cat_{}".format(i+1) + "_" + cat_str
            if key in self.cat2id_dict:
                id_list.append( self.cat2id_dict[key] )
            else:
                id_list.append( self.cat2id_dict[ self.oov_prefix + "cat_{}".format(i+1) ] )
            weight_list.append( 1.0 )
        return id_list, weight_list
#



def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
#

def statsdata(data_file_path, output_path, criteo_stats):
    with open(data_file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for line in file_in:
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                errorline_list.append(count)
                print("line: {}".format(line) )
                continue
            if count % 1000000 == 0:
                print( "Have handle {}w lines.".format(count//10000) )
            # if count % 5000000 == 0:
            #     criteo_stats.save_dict(output_path, prefix="{}w_".format(count//10000))
            label = items[0]
            values = items[1:14]
            cats = items[14:]
            assert len(values) == 13, "values.size： {}".format( len(values) )
            assert len(cats) == 26, "cats.size： {}".format( len(cats) )
            criteo_stats.stats_vals( values )
            criteo_stats.stats_cats( cats )
    criteo_stats.save_dict(output_path)
#


def add_write(file_path, wrt_str):
    with open(file_path, 'a', encoding="utf-8") as file_out:
        file_out.write( wrt_str + "\n" )
#


def random_split_trans2h5(in_file_path, output_path, criteo_stats, part_rows=2000000, test_size=0.1, seed=2020):
    test_size = int(TRAIN_LINE_COUNT * test_size)
    train_size = TRAIN_LINE_COUNT - test_size
    all_indices = [i for i in range(TRAIN_LINE_COUNT)]
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    print( "all_indices.size: {}".format( len(all_indices) ) )
    lines_count_dict = collections.defaultdict(int)
    test_indices_set = set(all_indices[ : test_size])
    print( "test_indices_set.size: {}".format( len(test_indices_set) ) )
    print( "----------" * 10 + "\n" * 2 )

    train_feature_file_name = os.path.join(output_path, "train_input_part_{}.h5")
    train_label_file_name = os.path.join(output_path, "train_output_part_{}.h5")
    test_feature_file_name = os.path.join(output_path, "test_input_part_{}.h5")
    test_label_file_name = os.path.join(output_path, "test_output_part_{}.h5")
    train_feature_list = []
    train_label_list = []
    test_feature_list = []
    test_label_list = []
    with open(in_file_path, encoding="utf-8") as file_in:
        count = 0
        train_part_number = 0
        test_part_number = 0
        for i,line in enumerate(file_in):
            count += 1
            if count % 1000000 == 0:
                print( "Have handle {}w lines.".format(count//10000) )
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                continue
            label = float(items[0])
            values = items[1:14]
            cats = items[14:]
            assert len(values) == 13, "values.size： {}".format( len(values) )
            assert len(cats) == 26, "cats.size： {}".format( len(cats) )
            ids, wts = criteo_stats.map_cat2id(values, cats)
            if i not in test_indices_set:
                train_feature_list.append( ids + wts )
                train_label_list.append( label )
            else:
                test_feature_list.append( ids + wts )
                test_label_list.append( label )
            if (len(train_label_list) > 0) and (len(train_label_list) % part_rows == 0):
                pd.DataFrame( np.asarray(train_feature_list) ).to_hdf( train_feature_file_name.format(train_part_number), key="fixed")
                pd.DataFrame( np.asarray(train_label_list) ).to_hdf( train_label_file_name.format(train_part_number), key="fixed")
                train_feature_list = []
                train_label_list = []
                train_part_number += 1
            if (len(test_label_list) > 0) and (len(test_label_list) % part_rows == 0):
                pd.DataFrame( np.asarray(test_feature_list) ).to_hdf( test_feature_file_name.format(test_part_number), key="fixed")
                pd.DataFrame( np.asarray(test_label_list) ).to_hdf( test_label_file_name.format(test_part_number), key="fixed")
                test_feature_list = []
                test_label_list = []
                test_part_number += 1
        #
        if len(train_label_list) > 0:
            pd.DataFrame( np.asarray(train_feature_list) ).to_hdf( train_feature_file_name.format(train_part_number), key="fixed")
            pd.DataFrame( np.asarray(train_label_list) ).to_hdf( train_label_file_name.format(train_part_number), key="fixed")
            train_part_number += 1
        if len(test_label_list) > 0:
            pd.DataFrame( np.asarray(test_feature_list) ).to_hdf( test_feature_file_name.format(test_part_number), key="fixed")
            pd.DataFrame( np.asarray(test_label_list) ).to_hdf( test_label_file_name.format(test_part_number), key="fixed")
            test_part_number += 1

    return train_part_number,test_part_number
#
def convert_tfrecords(hdf_in, hdf_out, output_filename):
    # label id1,id2,...,idn   val1,val2,...,valn
    #rf = open(input_filename, "r")

    X = pd.read_hdf(hdf_in).values
    y = pd.read_hdf(hdf_out).values

    line_num = 0
    samples_per_line = 1000
    num_inputs = 39 # features num of each sample
    writer = tf.python_io.TFRecordWriter(output_filename)
    print("Starting to convert {} to {}...".format(hdf_in, output_filename))

    ids = []
    values = []
    labels = []
    new_line_num = 1
    number_of_batches = np.ceil(1. * X.shape[0] / samples_per_line)
    counter = 0
    sample_index = np.arange(X.shape[0])
    np.random.shuffle(sample_index)
    while True:
        #line = rf.readline()

        batch_index = sample_index[samples_per_line * counter:samples_per_line * (counter + 1)]
        counter+=1
        X_batch = X[batch_index]
        id_list = X_batch[:,0:num_inputs]
        val_list = X_batch[:,num_inputs:]
        label = y[batch_index]
        if len(label) < 1000:
            break
        #print(id_list[0],val_list[0],label[0])
        #print(len(id_list[0]),len(val_list[0]),len(label[0]))
        #data, id_str, val_str = line.split(" ")
        #label = float(data[0])
        #id_list = map(int, id_str.split(','))
        #val_list = map(float, val_str.split(','))
        if len(id_list[0]) != num_inputs or len(val_list[0]) != num_inputs:
            continue
        for i in range(len(label)):
            labels.append(label[i,:])
            ids.extend(id_list[i,:].astype(int))
            values.extend(val_list[i,:])
        #line_num += 1
        # Write samples one by one
        #if line_num % samples_per_line == 0:
            #print("new line num is %d" % new_line_num)
        assert(len(ids) == num_inputs * samples_per_line)
        new_line_num += 1
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
            "feat_ids":
                tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
            "feat_vals":
                tf.train.Feature(float_list=tf.train.FloatList(value=values))
        }))
        writer.write(example.SerializeToString())
        ids = []
        values = []
        labels = []
    writer.close()
    # drop data not satisfy samples_per_line
    print("Starting to convert {} to {} done...".format(hdf_in, output_filename))

def trans_h5_to_tfrecord(train_part_number, test_part_number, output_h5_path, output_tfrecord_path):
    train_feature_file_name = os.path.join(output_h5_path, "train_input_part_{}.h5")
    train_label_file_name = os.path.join(output_h5_path, "train_output_part_{}.h5")
    test_feature_file_name = os.path.join(output_h5_path, "test_input_part_{}.h5")
    test_label_file_name = os.path.join(output_h5_path, "test_output_part_{}.h5")


    train_tf_record_file_name = os.path.join(output_tfrecord_path, "train_part_{}.tfrecord")
    test_tf_record_file_name = os.path.join(output_tfrecord_path, "test_part_{}.tfrecord")

    for train_num in range(train_part_number):
        input1 = train_feature_file_name.format(train_num)
        input2 = train_label_file_name.format(train_num)
        output = train_tf_record_file_name.format(train_num)
        convert_tfrecords(input1,input2,output)

    for test_num in range(test_part_number):
        input1 = test_feature_file_name.format(test_num)
        input2 = test_label_file_name.format(test_num)
        output = test_tf_record_file_name.format(test_num)
        convert_tfrecords(input1,input2,output)






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get and Process datasets')
    parser.add_argument('--base_path', default="/opt/npu/surong/data/", help='The path to save dataset')
    parser.add_argument('--output_h5_path', default="/opt/npu/surong/data/h5_data/", help='The path to save dataset')
    parser.add_argument('--output_tfrecord_path', default="/opt/npu/surong/data/tf_record/", help='The path to save dataset')
    args, _ = parser.parse_known_args()
   
    # base_path = "/opt/npu/data/origin_criteo_data/dataset/"
    base_path = args.base_path
    data_path = base_path + "origin_data/"
    mkdir_path(data_path)
    # step 1, get data;
    os.system( "wget -P {} -c https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz --no-check-certificate".format( base_path ) )
    os.system( "tar -zxvf {}dac.tar.gz".format( data_path ) )
    criteo_stats = CriteoStatsDict()
    # step 2, stats the vocab and normalize value
    data_file_path = base_path + "origin_data/train.txt"
    stats_output_path = base_path + "stats_dict/"
    mkdir_path(stats_output_path)
    statsdata(data_file_path, stats_output_path, criteo_stats)
    print( "----------" * 10 )
    criteo_stats.load_dict(dict_path=stats_output_path, prefix="")
    criteo_stats.get_cat2id(threshold=100)
    # step 3, transform data trans2h5; version 2: np.random.shuffle
    #in_file_path = base_path + "origin_data/train.txt"
    in_file_path = data_file_path
    mkdir_path(args.output_h5_path)
    train_part_number,test_part_number = random_split_trans2h5(in_file_path, args.output_h5_path, criteo_stats, part_rows=2000000, test_size=0.1, seed=2020)
    
    mkdir_path(args.output_tfrecord_path)
    trans_h5_to_tfrecord(train_part_number, test_part_number, args.output_h5_path, args.output_tfrecord_path)




