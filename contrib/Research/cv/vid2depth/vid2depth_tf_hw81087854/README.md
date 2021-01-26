![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# vid2depth

**Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints**

Reza Mahjourian, Martin Wicke, Anelia Angelova

CVPR 2018

Project website: [https://sites.google.com/view/vid2depth](https://sites.google.com/view/vid2depth)

ArXiv: [https://arxiv.org/pdf/1802.05522.pdf](https://arxiv.org/pdf/1802.05522.pdf)

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/sample_video_small.gif'></a>
</p>

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/approach.png' width=400></a>
</p>

## 1. Installation

### Requirements

#### Python Packages

```shell
mkvirtualenv venv  # Optionally create a virtual environment.
pip install absl-py
pip install matplotlib
pip install numpy
pip install scipy
pip install tensorflow
```

#### For building the ICP op (work in progress)

* Bazel: https://bazel.build/

### Download vid2depth

```shell
git clone --depth 1 https://github.com/tensorflow/models.git
```

## 2. Datasets

### Download KITTI dataset (174GB)

```shell
mkdir -p ~/vid2depth/kitti-raw-uncompressed
cd ~/vid2depth/kitti-raw-uncompressed
wget https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt
unzip "*.zip"
```

### Download Cityscapes dataset (110GB) (optional)

You will need to register in order to download the data.  Download the following files:

* leftImg8bit_sequence_trainvaltest.zip
* camera_trainvaltest.zip

### Download Bike dataset (17GB) (optional)

```shell
mkdir -p ~/vid2depth/bike-uncompressed
cd ~/vid2depth/bike-uncompressed
wget https://storage.googleapis.com/brain-robotics-data/bike/BikeVideoDataset.tar
tar xvf BikeVideoDataset.tar
```

## 3. Inference

### Download trained model

```shell
mkdir -p ~/vid2depth/trained-model
cd ~/vid2depth/trained-model
wget https://storage.cloud.google.com/vid2depth/model/model-119496.zip
unzip model-119496.zip
```

### Run inference

```shell
cd tensorflow/models/research/vid2depth
python inference.py \
  --kitti_dir ~/vid2depth/kitti-raw-uncompressed \
  --output_dir ~/vid2depth/inference \
  --kitti_video 2011_09_26/2011_09_26_drive_0009_sync \
  --model_ckpt ~/vid2depth/trained-model/model-119496
```

## 4. Training

### Prepare KITTI training sequences

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name kitti_raw_eigen \
  --dataset_dir ~/vid2depth/kitti-raw-uncompressed \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3
```

### Prepare Cityscapes training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name cityscapes \
  --dataset_dir ~/vid2depth/cityscapes-uncompressed \
  --data_dir ~/vid2depth/data/cityscapes \
  --seq_length 3
```

### Prepare Bike training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name bike \
  --dataset_dir ~/vid2depth/bike-uncompressed \
  --data_dir ~/vid2depth/data/bike \
  --seq_length 3
```

### Compile the ICP op (work in progress)

The ICP op depends on multiple software packages (TensorFlow, Point Cloud
Library, FLANN, Boost, HDF5).  The Bazel build system requires individual BUILD
files for each of these packages.  We have included a partial implementation of
these BUILD files inside the third_party directory.  But they are not ready for
compiling the op.  If you manage to build the op, please let us know so we can
include your contribution.

```shell
cd tensorflow/models/research/vid2depth
bazel build ops:pcl_demo  # Build test program using PCL only.
bazel build ops:icp_op.so
```

For the time being, it is possible to run inference on the pre-trained model and
run training without the icp loss.

### Run training

```shell
# Train
cd tensorflow/models/research/vid2depth
python train.py \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3 \
  --reconstr_weight 0.85 \
  --smooth_weight 0.05 \
  --ssim_weight 0.15 \
  --icp_weight 0 \
  --checkpoint_dir ~/vid2depth/checkpoints
```

### Run training on NPU

```shell
# Train
python3.7 train.py \
  --data_dir ./data/kitti_raw_eigen \
  --seq_length 3 \
  --reconstr_weight 0.85 \
  --smooth_weight 0.05 \
  --ssim_weight 0.15 \
  --icp_weight 0 \
  --checkpoint_dir ./checkpoints
```
or simplely run the shell script:
```
bash train_npu.sh
```
Train log of NPU:
```
2021-01-05 23:02:19.923832: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:114, total_bytes:4, shape:, tensor_ptr:281459378523712, output281459345860592
2021-01-05 23:02:19.923878: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:115, total_bytes:4, shape:, tensor_ptr:281459380395072, output281459345830976
2021-01-05 23:02:19.923893: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:116, total_bytes:4, shape:, tensor_ptr:281459379966976, output281459345862128
2021-01-05 23:02:19.923905: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:117, total_bytes:4, shape:, tensor_ptr:281459379664064, output281459345829232
2021-01-05 23:02:19.923918: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:118, total_bytes:4, shape:, tensor_ptr:281459379549120, output281459345745168
2021-01-05 23:02:19.923947: I tf_adapter/kernels/geop_npu.cc:573] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp15_0[ 134782831us]
I0105 23:02:20.097224 281473551822864 train.py:169] Epoch: [ 1] [    1/   46] time: 249.73s (249s total) loss: 2.643
2021-01-05 23:02:21.111948: I tf_adapter/optimizers/get_attr_optimize_pass.cc:64] NpuAttrs job is localhost
2021-01-05 23:02:21.112550: I tf_adapter/optimizers/get_attr_optimize_pass.cc:128] GetAttrOptimizePass_9 success. [0 ms]
2021-01-05 23:02:21.112582: I tf_adapter/optimizers/mark_start_node_pass.cc:82] job is localhost Skip the optimizer : MarkStartNodePass.
2021-01-05 23:02:21.112617: I tf_adapter/optimizers/mark_noneed_optimize_pass.cc:102] mix_compile_mode is True
2021-01-05 23:02:21.112631: I tf_adapter/optimizers/mark_noneed_optimize_pass.cc:103] iterations_per_loop is 1
2021-01-05 23:02:21.112690: I tf_adapter/optimizers/om_partition_subgraphs_pass.cc:1763] OMPartition subgraph_17 begin.
2021-01-05 23:02:21.112702: I tf_adapter/optimizers/om_partition_subgraphs_pass.cc:1764] mix_compile_mode is True
2021-01-05 23:02:21.112711: I tf_adapter/optimizers/om_partition_subgraphs_pass.cc:1765] iterations_per_loop is 1
2021-01-05 23:02:21.112750: I tf_adapter/optimizers/om_partition_subgraphs_pass.cc:354] FindNpuSupportCandidates enableDP:0, mix_compile_mode: 1, hasMakeIteratorOp:0, hasIteratorOp:0
```

## Reference
If you find our work useful in your research please consider citing our paper:

```
@inproceedings{mahjourian2018unsupervised,
  title={Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints},
    author={Mahjourian, Reza and Wicke, Martin and Angelova, Anelia},
    booktitle = {CVPR},
    year={2018}
}
```

## Contact

To ask questions or report issues please open an issue on the tensorflow/models
[issues tracker](https://github.com/tensorflow/models/issues). Please assign
issues to [@rezama](https://github.com/rezama).

## Credits

This implementation is derived from [SfMLearner](https://github.com/tinghuiz/SfMLearner) by [Tinghui Zhou](https://github.com/tinghuiz).
