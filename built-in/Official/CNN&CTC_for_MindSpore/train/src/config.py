class Config_CNNCTC():
    # model config
    CHARACTER = '0123456789abcdefghijklmnopqrstuvwxyz'
    NUM_CLASS = len(CHARACTER) + 1
    HIDDEN_SIZE = 512
    FINAL_FEATURE_WIDTH = 26

    # dataset config
    IMG_H = 32
    IMG_W = 100
    TRAIN_DATASET_PATH = '/home/workspace/mindspore_dataset/CNNCTC_Data/ST_MJ/'
    TRAIN_DATASET_INDEX_PATH = '/home/workspace/mindspore_dataset/CNNCTC_Data/st_mj_fixed_length_index_list.pkl'
    TRAIN_BATCH_SIZE = 192
    TRAIN_DATASET_SIZE = 19200
    TEST_DATASET_PATH = '/home/workspace/mindspore_dataset/CNNCTC_Data/IIIT5k_3000'
    TEST_BATCH_SIZE = 256
    TEST_DATASET_SIZE = 2976
    TRAIN_EPOCHS = 3

    # training config
    CKPT_PATH = ''
    SAVE_PATH = './'
    LR = 1e-4
    LR_PARA = 5e-4
    MOMENTUM = 0.8
    LOSS_SCALE = 8096
    SAVE_CKPT_PER_N_STEP = 2000
    KEEP_CKPT_MAX_NUM = 5
