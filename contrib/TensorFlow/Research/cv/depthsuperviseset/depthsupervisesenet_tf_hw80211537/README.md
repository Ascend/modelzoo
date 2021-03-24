## Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision

## Environment
- HUAWEI CLOUD
- Python 3.7
- Tensorflow-1.15

## Datasets
- OULU-NPU

## Installation
- Clone depthsuperviseset repository. We'll call the directory that you cloned depthsupervisesenet_tf_hw80211537 as $ROOT_PATH.
    ```Shell
  git clone --recursive https://gitee.com/liuajian/modelzoo/tree/master/contrib/TensorFlow/Research/cv/depthsuperviseset/depthsupervisesenet_tf_hw80211537
    ```
    
## Requirements
- Account: hw80211537
- Obs: ajian3

## Usage
- data_url: obs://ajian3/Oulu-Train/Oulu-Train
- train_url: Jobs
- Start training: python train_depth_yun.py
- Start testing: python test_depth_yun.py
If you do not train the model by yourself, you can download our [trained model](https://pan.baidu.com/s/13B7N51in8hMmAATveen4TQ). code: gec4

## Results
- The testing results of these methods based on multi-shot setting are as follows(%): 
   ```Shell
   ---------------------------------------------
   |  Method   | APCER(%) | BPCER(%) | ACER(%) |
   |Aux(Depth) |   2.7    |   2.7    |   2.7   |
   |   Ours    |   3.5    |   1.4    |   2.4   |
   ---------------------------------------------
   Note that the metric of ACER is the final indicator, and the smaller value means better performance.
  ```
## Citation
  ```Shell
Please cite the following papers in your publications if it helps your research:
@inproceedings{Liu2018Learning,
  title={Learning deep models for face anti-spoofing: Binary or auxiliary supervision},
  author={Liu, Yaojie and Jourabloo, Amin and Liu, Xiaoming},
  booktitle={CVPR},
  year={2018}
}
  ```
## Questions
 
Please contact 'ajianliu92@gmail.com'











