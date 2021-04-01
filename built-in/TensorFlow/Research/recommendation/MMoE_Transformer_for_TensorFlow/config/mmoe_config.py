base_dir = 'result/'
num_workers = 16
shuffle_seed_idx = False
shuffle_dataset = False

batch_size = 256
deep = [100, 100, 100]
cross = 6
embed_size = 64
l2 = 1e-4
l1 = 1e-7
lr = 1e-4
decay_step = 1
decay_rate = 0.6
num_experts = 8
num_expert_units = 16
keep_prob = 0.9

data_para = {"train_path": "dataset/train/tfrecord_data_padLen300_threshold10/",
             "train_size": 1000000,
             "test_path": "dataset/test/tfrecord_data_padLen300_threshold10/",
             "input_dim": 1038446,
             "fields_num": 15}

train_para = {
    "pos_weight": 1.0,
    "n_epoch": 1,  # 10
    "early_stop_steps": 300}

transformer_params = {
    'd_model': embed_size*8,
    'd_ff':  embed_size*8,
    'num_blocks': 2,
    'num_heads': 8,
    'maxlen1': 300,
    'maxlen2': 300,
    'keep_prob': 0.7}