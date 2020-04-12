BASE_DIR = './'

num_gpu = 1 
num_inputs = 39 
num_features = 200000
batch_size = 10000
multi_hot_flags = [False]
multi_hot_len = 1
n_epoches = 20 
one_step = 0 # for one step debug

line_per_sample = 1000

# n_step_update = 10

record_path = "../tf_record/"
train_tag = 'train_part'
test_tag = 'test_part'
writer_path = './model/'
graph_path = './model/'

train_size = 41257636
test_size = 4582981

