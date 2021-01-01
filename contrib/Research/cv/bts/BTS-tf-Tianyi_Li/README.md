
# Note
This repository is a fully TensorFlow implementation of BTS model. It is forked from [original repository](https://github.com/saeid-h/bts-fully-tf). The code is modified to run with TensorFlow 1.6 without extra C++ code.

# BTS
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Video Demo 1
[![Screenshot](https://img.youtube.com/vi/2fPdZYzx9Cg/maxresdefault.jpg)](https://www.youtube.com/watch?v=2fPdZYzx9Cg)
## Video Demo 2
[![Screenshot](https://img.youtube.com/vi/1J-GSb0fROw/maxresdefault.jpg)](https://www.youtube.com/watch?v=1J-GSb0fROw)

## Testing with [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
```shell
$ cd ~/workspace/bts/utils
# Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
$ cd ..
$ mkdir models
# Get BTS model trained with NYU Depth V2
$ python utils/download_from_gdrive.py 1KgLFgZa9U4X7Kq1_U908idFo_GBfu04d models/bts_nyu.zip
$ cd models
$ unzip bts_nyu.zip
```
Once the preparation steps completed, you can test BTS using following commands.
```
$ cd ~/workspace/bts
$ python bts_test.py arguments_test_nyu.txt
```
This will save results to ./result_bts_nyu. With a single RTX 2080 Ti it takes about 34 seconds for processing 654 testing images. 

## Evaluation
Following command will evaluate the prediction results for NYU Depvh V2.
```
$ python eval_with_pngs.py --pred_path ./result_bts_nyu/raw/ --gt_path ../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
```

You should see outputs like this:
```
Raw png files reading done
Evaluating 654 files
GT files reading done
0 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.880,   0.978,   0.996,   0.113,   0.060,   0.356,   0.142,  11.333,   0.048
Done.
```

## Preparation for Training
### NYU Depvh V2
First, you need to download DenseNet-161 model pretrained with ImageNet.
```
# Get DenseNet-161 model pretrained with ImageNet
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1rn7xBF5eSISFKL2bIa8o3d8dNnsrlWfJ models/densenet161_imagenet.zip
$ cd models && unzip densenet161_imagenet.zip
```
Then, download the dataset we used in this work.
```
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ../dataset/nyu_depth_v2/sync.zip
$ cd ../dataset/nyu_depth_v2
$ unzip sync.zip
```

Or, you can prepare the dataset by yourself using original files from official site [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
There are two options for downloading original files: Single file downloading and Segmented-files downloading.

Single file downloading:
```
$ cd ~/workspace/dataset/nyu_depth_v2
$ mkdir raw && cd raw
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip
$ unzip nyu_depth_v2_raw.zip
```
Segmented-files downloading:
```
$ cd ~/workspace/dataset/nyu_depth_v2
$ mkdir raw && cd raw
$ aria2c -x 16 -i ../../../bts/utils/nyudepthv2_archives_to_download.txt
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1xBwO6qU8UCS69POJJ0-9luaG_1pS1khW ../dataset/nyu_depth_v2/raw/bathroom_0039.zip
$ python utils/download_from_gdrive.py 1IFoci9kns6vOV833S7osV6c5HmGxZsBp ../dataset/nyu_depth_v2/raw/bedroom_0076a.zip
$ python utils/download_from_gdrive.py 1ysSeyiOiOI1EKr1yhmKy4jcYiXdgLP4f ../dataset/nyu_depth_v2/raw/living_room_0018.zip
$ python utils/download_from_gdrive.py 1QkHkK46VuKBPszB-mb6ysFp7VO92UgfB ../dataset/nyu_depth_v2/raw/living_room_0019.zip
$ python utils/download_from_gdrive.py 1g1Xc3urlI_nIcgWk8I-UaFXJHiKGzK6w ../dataset/nyu_depth_v2/raw/living_room_0020.zip
$ parallel unzip ::: *.zip
```
Get official toolbox for rgb and depth synchronization.
```
$ cd ~/workspace/bts/utils
$ wget http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip
$ unzip toolbox_nyu_depth_v2.zip
$ cd toolbox_nyu_depth_v2
$ mv ../sync_project_frames_multi_threads.m .
$ mv ../train_scenes.txt .
```
Run script "sync_project_frames_multi_threads.m" using MATLAB to get synchronized RGB and depth images.
This will save rgb-depth pairs in "~/workspace/dataset/nyu_depth_v2/sync".
Once the dataset is ready, you can train the network using following command.
```
$ cd ~/workspace/bts
$ python bts_main.py arguments_train_nyu.txt
```
You can check the training using tensorboard:
```
$ tensorboard --logdir ./models/bts_nyu_test/
```
Open localhost:6006 with your favorite browser to see the progress of training.

### KITTI
You can also train BTS with KITTI dataset by following procedures.
First, download the ground truth depthmaps from [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
Then, download and unzip the raw dataset using following commands.
```
$ cd ~/workspace/dataset
$ mkdir kitti_dataset && cd kitti_dataset
$ mv ~/Downloads/data_depth_annotated.zip .
$ unzip data_depth_annotated.zip
$ aria2c -x 16 -i ../../bts/utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip
```
Finally, we can train our network with
```
$ cd ~/workspace/bts
$ python bts_main.py arguments_train_eigen.txt
```

## Testing and Evaluation with [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
Once you have KITTI dataset and official ground truth depthmaps, you can test and evaluate our model with following commands.
```
# Get KITTI model trained with KITTI Eigen split
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1w4WbSQxui8GTDEsjX5xb4m7_-5yCznhQ models/bts_eigen.zip
$ cd models && unzip bts_eigen.zip
```
Test and save results.
```
$ cd ~/workspace/bts
$ python bts_test.py arguments_test_eigen.txt
```
This will save results to ./result_bts_eigen.
Finally, we can evaluate the prediction results with
```
$ python eval_with_pngs.py --pred_path ./result_bts_eigen/raw/ --gt_path ../dataset/kitti_dataset/data_depth_annotated/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
You should see outputs like this:
```
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.951,   0.993,   0.998,   0.064,   0.256,   2.796,   0.100,   9.175,   0.028
Done.
```

## License
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
This Software is licensed under GPL-3.0-or-later.
