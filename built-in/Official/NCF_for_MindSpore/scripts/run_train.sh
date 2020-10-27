# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python ./train.py --data_path '/home/x00352035/ncf_data' --dataset 'ml-20m'  --train_epochs 5 --batch_size 1048576 --output_path './output/' --loss_file_name 'loss.log' --checkpoint_path './checkpoint/' 
python ./train.py --data_path '/home/x00352035/ncf_data' --dataset 'ml-1m'  --train_epochs 20 --batch_size 256 --output_path './output/' --loss_file_name 'loss.log' --checkpoint_path './checkpoint/' 
