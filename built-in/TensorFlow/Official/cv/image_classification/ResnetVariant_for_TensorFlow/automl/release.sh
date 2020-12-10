# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
# 1. benchmark

# 1.1 remove algorithms config files
#       auto_lane, faster_rcnn, sm-nas
rm -r ./benchmark/algs/faster_rcnn
rm -r ./benchmark/algs/nas/jdd_ea.yml
rm -r ./benchmark/algs/nas/sm_nas.yml

# 1.2 remove env config
rm -r ./benchmark/env
rm ./benchmark/run_benchmark_roma.py

rm -r contrib

# 2. deploy

# 2.1 remove not used 3rd packages

# 3. docs

# 3.1 algorithms, 
#   a. remove algorithms not released
#   b. remove author from all algorithm files manually
rm ./docs/cn/algorithms/combo_nas.md
rm ./docs/cn/algorithms/images/combo_*
rm ./docs/cn/algorithms/jdd_ea.md
rm ./docs/cn/algorithms/images/jdd*
rm ./docs/cn/algorithms/sm-nas.md
rm ./docs/cn/algorithms/images/sm_nas*

rm ./docs/en/algorithms/sm-nas.md
rm ./docs/en/algorithms/images/sm_nas*

# 3.2 benchmark, do nothing

# 3.3 examples, remove the algorithms not released manually

# 3.4 model zoo, do nothing

# 3.5 tasks, remove wsd file, and remove the algorithms not released manually
rm ./docs/cn/tasks/images/*.wsd
rm ./docs/en/tasks/images/*.wsd

# 3.6 user, remove dloop
rm ./docs/cn/user/dloop.md
rm ./docs/cn/user/images/dloop*
# rm ./docs/en/user/dloop.md
rm ./docs/en/user/images/dloop*
# rm ./docs/cn/user/evaluate_service.md
rm ./docs/cn/user/roma_user_guide.md
# modify configuration.md, remove network not released manually

# 3.7 developer, do nothing

# 3.8 modify README.md, remove algorithms not released manually

# 4. evaluate_servics, remove it
# rm -r ./evaluate_service

# 5. examples, remove algorithms
rm -r ./examples/run_example_roma.py

rm -r ./examples/faster_rcnn
rm -r ./examples/nas/jdd_ea
rm -r ./examples/nas/sm_nas
rm -r ./examples/fine_grained_space

# 6. roma, remove it
rm -r ./roma

# 7. test, remove it
rm -r ./test

# 8. vega

# 8.1 algorithms
rm -r ./vega/algorithms/nas/jdd_ea
rm -r ./vega/algorithms/nas/sm_nas
# remove algorithms from __init__.py file
sed -i '/jdd_ea/d'  ./vega/algorithms/nas/__init__.py
sed -i '/sm_nas/d'  ./vega/algorithms/nas/__init__.py

# 8.3 core

# 8.3.1 evluator
rm ./vega/core/evaluator/hava_d_evaluator.py
sed -i '/hava_d_evaluator/d'  ./vega/core/backend_register.py

# 8.3.2 metrics
rm ./vega/core/metrics/pytorch/detection_metric.py
rm ./vega/core/metrics/pytorch/jdd_psnr_metric.py
rm ./vega/core/metrics/pytorch/recall_eval.py
sed -i '/detection_metric/d'  ./vega/core/metrics/pytorch/__init__.py
sed -i '/jdd_psnr_metric/d'  ./vega/core/metrics/pytorch/__init__.py
sed -i '/recall_eval/d'  ./vega/core/metrics/pytorch/__init__.py

# 8.3.2 pipeline
rm ./vega/core/pipeline/horovod/run_horovod_train.sh

# 8.4 datasets
rm ./vega/datasets/conf/jdd.py
rm ./vega/datasets/conf/mdc.py
rm ./vega/datasets/conf/coco.py

rm ./vega/datasets/pytorch/common/mdc_util.py
rm ./vega/datasets/pytorch/jdd_data.py
rm ./vega/datasets/pytorch/coco.py
sed -i '/jdd_data/d'  ./vega/datasets/pytorch/__init__.py
sed -i '/coco/d'  ./vega/datasets/pytorch/__init__.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/cifar10.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/cifar100.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/cityscapes.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/div2k.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/imagenet.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/fmnist.py
sed -i '/download_dataset/d'  ./vega/datasets/pytorch/mnist.py

# 8.5 search_space

# fine grained
rm -r ./vega/search_space/fine_grained_space
rm -r ./vega/search_space/networks/pytorch/operator
sed -i '/fine_grained_space/d'  ./vega/search_space/__init__.py

# faster-rcnn
rm -r ./vega/search_space/networks/pytorch/backbones/resnet_det.py
sed -i '/resnet_det/d'  ./vega/search_space/networks/pytorch/backbones/__init__.py
rm -r ./vega/search_space/networks/pytorch/blocks/resnet_block_det.py
sed -i '/resnet_block_det/d'  ./vega/search_space/networks/pytorch/blocks/__init__.py
rm -r ./vega/search_space/networks/pytorch/detectors/faster_rcnn.py
sed -i '/faster_rcnn/d'  ./vega/search_space/networks/pytorch/detectors/__init__.py
rm -r ./vega/search_space/networks/pytorch/heads/bbox_head.py
rm -r ./vega/search_space/networks/pytorch/heads/rpn_head.py
sed -i '/bbox_head/d'  ./vega/search_space/networks/pytorch/heads/__init__.py
sed -i '/rpn_head/d'  ./vega/search_space/networks/pytorch/heads/__init__.py
rm -r ./vega/search_space/networks/pytorch/roi_extractors/
sed -i '/roi_extractors/d'  ./vega/search_space/networks/pytorch/__init__.py
rm -r ./vega/search_space/networks/pytorch/shared_heads/
sed -i '/shared_heads/d'  ./vega/search_space/networks/pytorch/__init__.py
rm -r ./vega/search_space/networks/pytorch/utils
sed -i '/utils/d'  ./vega/search_space/networks/pytorch/__init__.py

# jdd
rm -r ./vega/search_space/networks/pytorch/jddbodys
sed -i '/jddbodys/d'  ./vega/search_space/networks/pytorch/__init__.py

# remove file at root path
rm mr_checklist.md
