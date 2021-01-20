BASE_DIR = '/home/guowei/mpiwork/general_id_cut_wei'
graph_path = './'

num_gpu = 1
num_inputs = 39
num_features = 200000
#num_features = 6779244
batch_size = 10000
multi_hot_flags=[False]
multi_hot_len = 1
iterations_per_loop=10
n_epoches =1 
n_step = 50000

line_per_sample = 1000
# n_step_update = 10

dim_uid=1e5
#cut_dim=30
#cut_dim = 37
#common_feat_num = 13
common_feat_num = 21340
#common_feat_num=6779244
#common_feat_num = 200000
general_feat_num = 163622 # [21340, 21340+163622-1]
record_path = "/autotest/deepFM/data/general_split"
#record_path = "/home/daihongtao/wideDeep_multi/new_e2e_test/data/tf_record_qiefen_Large_6780000/tf_record_20200716_threshold_1"
#general_feat_num =6780000
#13个连续特征加17个低维的离散特征是common
common_dim = 30
#剩下的9个高维的离散特征是general
general_dim = 9
#common_dim = 13
#general_dim = 26
embed_size = 80
#test_record = "/home/guohuifeng/sjtu-1023/test.svm.100w.tfrecord.1000perline"
#train_record = "/home/guohuifeng/sjtu-1023/train.svm.1000w.tfrecord.1000perline"
#test_record = "/home/guohuifeng/sjtu-multi-card/test.svm.100w.tfrecord.1000perline"
#train_record = "/home/guohuifeng/sjtu-multi-card/train.svm.1000w.tfrecord.1000perline"
#record_path = "/home/dataset/guohuifeng/criteo_tfrecord_a"
#record_path = "/home/daihongtao/wideDeep_multi/new_e2e_test/data/tf_record"
#record_path = "general_split"
#record_path = "/home/daihongtao/wideDeep_multi/general_id_cut_ingraph-new/data/general_split"
#record_path = "/home/daihongtao/wideDeep_multi/new_e2e_test/data/tf_record_qiefen_Large_6780000/tf_record_20200716_threshold_1"
#record_path = "/home/daihongtao/wideDeep_multi/new_e2e_test/data/tf_record_qiefen/tf_record_20200819_threshold_1750000"
train_tag = 'train'
test_tag = 'test'

#record_path = "/home/guohuifeng/sjtu-multi-card"
#train_tag = 'train.svm' 
#test_tag = 'test.svm'


train_size = 41257636
test_size = 4582981
