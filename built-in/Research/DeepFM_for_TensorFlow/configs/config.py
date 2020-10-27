BASE_DIR = './'

num_gpu = 1 
num_inputs = 39
num_features = 200000
batch_size = 16000
multi_hot_flags = [False]
multi_hot_len = 1
n_epoches = 20
iteration_per_loop = 10
#one_step = 60/iteration_per_loop # for one step debug
one_step = 0 
line_per_sample = 1000
graph_path = "./"


# n_step_update = 10

#test_record = "/home/guohuifeng/sjtu-1023/test.svm.100w.tfrecord.1000perline"
#train_record = "/home/guohuifeng/sjtu-1023/train.svm.1000w.tfrecord.1000perline"
#test_record = "/home/guohuifeng/sjtu-multi-card/test.svm.100w.tfrecord.1000perline"
#train_record = "/home/guohuifeng/sjtu-multi-card/train.svm.1000w.tfrecord.1000perline"
#record_path = "./tf_record"
record_path = "/autotest/CI_daily/ModelZoo_DeepFM_TF/data/deepfm"
train_tag = 'train_part'
test_tag = 'test_part'

#record_path = "/home/guohuifeng/sjtu-multi-card"
#train_tag = 'train.svm' 
#test_tag = 'test.svm'


train_size = 41257636
test_size = 4582981
