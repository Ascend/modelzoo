#!/bin/bash
#set -x
#start offline inference
python3 predict_directory_npu.py --img_dir DIV2K_test_100/DIV2K_train_LR_bicubic_801_900_X2/ --output_dir ./DIV2K_test_predicted

#post process
python3 evaluation.py --HR_data_dir /DIV2K_test_100/DIV2K_train_HR_801_900  --inference_result ./DIV2K_test_predicted
