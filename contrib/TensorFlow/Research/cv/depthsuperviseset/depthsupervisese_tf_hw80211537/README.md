## Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision

## Environment
- HUAWEI CLOUD
- Python 3.7
- Tensorflow-1.15

## Datasets
- [OULU-NPU](https://sites.google.com/site/oulunpudatabase/) 

## Installation
- Clone FAS_ModelZoo_v4 repository. We'll call the directory that you cloned ReId_Eigen as $ROOT_PATH.
    ```Shell
  git clone --recursive https://github.com/liuajian/FAS_ModelZoo_v4.git
    ```
    
## Requirements
- Account: hw80211537
- Obs: ajian3

## Usage
- data_url: Oulu-Train
- train_url: Jobs
- Start training: python train_depth_yun.py
- Start testing: python test_depth_yun.py

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











