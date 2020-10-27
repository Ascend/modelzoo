BASE_DIR = './'

num_gpu = 8
num_inputs = 39
num_features = 200000


batch_size = 16000
multi_hot_flags = [False]
multi_hot_len = 1
###
#n_epoches =50
#iterations_per_loop = 10
n_epoches = 1
iterations_per_loop = 1
#one_step = 50/iterations_per_loop # for one step debug
one_step = 0
line_per_sample = 1000

#record_path = '/data/tf_record'
record_path = '/autotest/CI_daily/ModelZoo_WideDeep_TF/data/tf_record'
train_tag = 'train_part'
test_tag = 'test_part'
writer_path = './model/'
graph_path = './model/'

train_size = 41257636
test_size = 4582981


