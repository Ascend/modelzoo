
# Contents
- [Contents](#contents)
- [Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)
Jasper is a family of end-to-end ASR models that replace acoustic and pronunciation models with a convolutional neural network.


[Paper](https://arxiv.org/abs/1904.03288): Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gaddecc. [Jasper: An End-to-End Convolutional Neural Acoustic Model.]((https://arxiv.org/abs/1904.03288)).

# [Model Architecture](#contents)
Jasper uses mel-filterbank features calculated from 20ms
windows with a 10ms overlap, and outputs a probability distribution over characters per frame2
. Jasper has a block architecture: a Jasper BxR model has B blocks, each with R subblocks. Each sub-block applies the following operations: a 1Dconvolution, batch norm, ReLU, and dropout. All sub-blocks in
a block have the same number of output channels.

# [Dataset](#contents)
- LibriSpeech is a ~1,000 hours of 16kHz read English speech corpus. The data is obtained from audiobooks read from the LibriVox project, and has been segmented and aligned.

# [Environment Requirements](#contents)
- Hardware（Ascend）
  - Prepare hardware environment with Ascend processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get access to the resources.
- Framework
  - [Tensorflow 1.15.0](https://www.tensorflow.org/versions/)
- Requirement module
```bash
# run pip to install reqirement
pip install -r requirement.txt

```
# [Quick Start](#contents)
After installing Tensorflow via the official website, you can start training and evaluation as follows:
```bash
# run standalone training example
bash scripts/run_1p.sh

# run distributed training example
bash scripts/run_8p.sh

# run evaluation example
bash scripts/run_eval.sh
```

For distributed training, an hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell

│   README.md
│   run.py
│   requirements.txt
├ ─ configs
│   └ ─ speech2text
│           jasper10x5_LibriSpeech_nvgrad.py
│           jasper10x5_LibriSpeech_nvgrad_masks.py
│           jasper5x3_LibriSpeech_nvgrad_masks_8p.py       #standalone training configuration
│           jasper5x3_LibriSpeech_nvgrad_masks_1p.py       #distributed trainining configuration
├ ─ open_seq2seq
└ ─ scripts
        8p.json                                            #hccl configuration file with json format
        eval_1p.sh
        run_1p.sh                                          #run standalone training
        run_8p.sh                                          #run distributed training
        run_eval.sh                                        #run evaluation
        train_1p.sh
        train_8p.sh
```

## [Checkpoint to pb](#contents)

```
cd generate_pb
python ckpt2pb.py
```



## [Online Inference](#contents)

```
python3.7.5 jasper_online_inference.py

args:
--model_path	 #original path of pb model,default value: ../model/jasper_infer_float32.pb
--data_dir		 #parents dir of dev.json file，default value：../datasets
--output_dir	 #the output dir of preprocess of jasper bin files.default value:../datasets
--pre_process  	 #weather execute preprocess.option value:True/False，default value is: False
--post_process   #weather execute postprocess.option value:True/False，default value is: True
--batchSize		 #batch size of inference.default value is 1
```



# ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

