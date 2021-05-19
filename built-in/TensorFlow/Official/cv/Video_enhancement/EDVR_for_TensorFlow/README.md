# Video Restoration with Enhanced Deformable Convolution Networks
## Introduction
This is an EDVR implementation based on tensorflow, which supports running on NPU, GPU and CPU. For EDVR itself, please refer to the original [paper](https://arxiv.org/abs/1905.02716).

## Requirements
- tensorflow==1.15
- cv2
- yacs
- python3.7

## Structure

```sh
edvr
|-- ascendcv
|   |-- dataloader
|   |-- layers
|   |-- runner
|   `-- utils
|-- ascendvsr
|   |-- config
|   |-- layers
|   `-- models
|-- configs
|   `-- edvr.yaml
|-- data
|   `-- reds
|       |-- images
|       |   `-- 001
|       |       |-- blur4
|       |       `-- truth
|       `-- sets
|-- scripts
|   |-- 2p.json
|   |-- 8p.json
|   |-- download_REDS.py
|   |-- make_reds_dataset.py
|   |-- prepare_8p.sh
|   |-- prepare_reds_dataset.sh
|   |-- regroup_reds_dataset.py
|   |-- run_1p_train
|   |-- run_1p_train.sh
|   |-- run_8p_train.sh
|   |-- run_evaluation.sh
|   |-- run_freeze.sh
|   `-- run_inference.sh
`-- tools
    `-- main.py
```

- ascendcv: some basic layers and runner
- ascendvsr: basic vsr model and edvr moels
- configs: specific configuration yaml files
- data: dataset folder
- scripts: top shell scripts
- tools: top python entrance script

## Prepare dataset

We take the REDS4 dataset for example.

1. Download the datasets splits

```bash
# pwd: path/to/edvr
datadir=./data/reds

mkdir -p ${datadir}
python3 scripts/download_REDS.py --root_dir ${datadir} --train_sharp --train_sharp_bicubic --val_sharp --val_sharp_bicubic
```
This step will download the above splits and save them to ``data/reds``. Unzip these files in ``data/reds``.

2. Merge and regroup the REDS4 dataset splits as in the paper

```bash
if [ ! -d ${datadir}/images ]; then
    mkdir -p ${datadir}/images
fi

python3 scripts/regroup_reds_dataset.py ${datadir}
```
This step will merge the val splits into the train splits, and concatenate them together. The val splits will be renamed to index [240 - 269]. 

3. Prepare dataset metadata. Of all the 270 clips, the 000, 011, 015, 020 will be used as the actual validation sets as is conducted by the paper. 

```sh
if [ ! -d ${datadir}/sets ]; then
    mkdir -p ${datadir}/sets
fi

python3 scripts/make_reds_dataset.py ${datadir}
```

**We've integrated these steps in a single shell script**. One can run the script directly for the REDS4 dataset：

```sh
bash scripts/prepare_reds_dataset.sh
```

Note that the download may take much time, since the dataset is quite large.

### Dataset folder structure

```sh
data/reds
|-- images
|   |-- 000
|   |   |-- blur4
|   |   |   |-- 00000000.png
|   |   |   |-- 00000001.png
|   |   |   |-- 00000002.png
|   |   |   |-- 00000003.png
|   |   |   |-- ...
|   |   |   `-- 00000099.png
|   |   `-- truth
|   |       |-- 00000000.png
|   |       |-- 00000001.png
|   |       |-- 00000002.png
|   |       |-- 00000003.png
|   |       |-- ...
|   |       `-- 00000099.png
|   |-- 001
|   |   |-- blur4
|   |   `-- truth
|   |-- 002
|   |   |-- blur4
|   |   `-- truth
|   |-- 003
|   |-- ...
|   |-- ...
|   `-- 269
`-- sets
    |-- train.json
    `-- val.json
```

### Prepare your own dataset

We suggest users to follow our dataset folder structure and protocal, especially when validation and inference.

## Training

1. Training EDVR with device 0: 

    ```sh
   # Set ascend environment
   source scripts/env.sh
   
    bash scripts/run_1p_train.sh 0 1
    ```

    This first argument 0 indicates the device id, while the second is the number of device used (a.k.a, device rank).

    One can configure the training and dataset details by defining a new ``yaml`` file. Please refer to ``ascendvsr/config/defaults.py`` for all the default configure items, and ``configs/edvr.yaml`` for example.

    The training status, e.g. loss value, step time, fps, will be printed on the screen every certain steps (``cfg.solver.print_interval``), and the checkpoint will be saved every ``cfg.solver.checkpoint_interval`` steps in the ``cfg.output_dir`` (default to ``outputs/edvr``). 

    Training process with batchsize=4:

    > 2020-12-25 08:34:46 Step:20, lr:0.00040000, loss:9476.21899544, time:209.62ms, fps:19.08 <br>
    > 2020-12-25 08:34:51 Step:40, lr:0.00040000, loss:9483.95068774, time:210.05ms, fps:19.04 <br>
    > 2020-12-25 08:34:55 Step:60, lr:0.00040000, loss:9374.16324098, time:212.36ms, fps:18.84 <br>
    > 2020-12-25 08:34:59 Step:80, lr:0.00040000, loss:9282.14592199, time:215.24ms, fps:18.58 <br>
    > 2020-12-25 08:35:03 Step:100, lr:0.00040000, loss:9225.38266487, time:214.05ms, fps:18.69 <br>
    > 2020-12-25 08:35:08 Step:120, lr:0.00040000, loss:9147.80390486, time:216.20ms, fps:18.50 <br>
    > 2020-12-25 08:35:12 Step:140, lr:0.00040000, loss:9152.80633528, time:210.89ms, fps:18.97 <br>
    > 2020-12-25 08:35:16 Step:160, lr:0.00040000, loss:9033.11667964, time:221.91ms, fps:18.03 <br>
    > 2020-12-25 08:35:21 Step:180, lr:0.00040000, loss:8864.82630793, time:214.61ms, fps:18.64 <br>
    > 2020-12-25 08:35:25 Step:200, lr:0.00040000, loss:8713.98834194, time:213.67ms, fps:18.72 <br>
    > 2020-12-25 08:35:29 Step:220, lr:0.00040000, loss:8513.21058127, time:211.83ms, fps:18.88 <br>
    > 2020-12-25 08:35:33 Step:240, lr:0.00040000, loss:8290.43024285, time:212.71ms, fps:18.81 <br>
    > 2020-12-25 08:35:38 Step:260, lr:0.00040000, loss:8128.87668816, time:212.40ms, fps:18.83 <br>
    > 2020-12-25 08:35:42 Step:280, lr:0.00040000, loss:7993.71964797, time:215.15ms, fps:18.59 <br>
    > 2020-12-25 08:35:46 Step:300, lr:0.00040000, loss:7808.95153805, time:214.33ms, fps:18.66 <br>
    > 2020-12-25 08:35:50 Step:320, lr:0.00040000, loss:7670.07353983, time:215.52ms, fps:18.56 <br>
    > 2020-12-25 08:35:55 Step:340, lr:0.00039999, loss:7567.70286385, time:213.47ms, fps:18.74 <br>
    > 2020-12-25 08:35:59 Step:360, lr:0.00039999, loss:7432.47448890, time:209.44ms, fps:19.10 <br>
    > 2020-12-25 08:36:03 Step:380, lr:0.00039999, loss:7331.30056232, time:214.45ms, fps:18.65 <br>
    > 2020-12-25 08:36:07 Step:400, lr:0.00039999, loss:7172.49776584, time:212.36ms, fps:18.84 <br>
    > 2020-12-25 08:36:12 Step:420, lr:0.00039999, loss:7051.55225264, time:209.99ms, fps:19.05 <br>
    > 2020-12-25 08:36:16 Step:440, lr:0.00039999, loss:6938.33076964, time:214.65ms, fps:18.64 <br>
    > 2020-12-25 08:36:20 Step:460, lr:0.00039999, loss:6866.46246160, time:212.75ms, fps:18.80 <br>
    > 2020-12-25 08:36:24 Step:480, lr:0.00039999, loss:6809.04133592, time:212.99ms, fps:18.78 <br>
    > 2020-12-25 08:36:29 Step:500, lr:0.00039999, loss:6736.72430321, time:210.67ms, fps:18.99 <br>

2. Training EDVR with 8 devices:

    ```sh
   # Set ascend environment
   source scripts/env.sh
   
    bash scripts/prepare_8p.sh
    bash scripts/run_8p_train.sh
    ```

    Training process with batchsize=4 per device:

    > 2021-01-05 14:05:28 Step:20, lr:0.00040000, loss:10128.23235736, time:225.60ms, fps:141.85 <br>
    > 2021-01-05 14:05:32 Step:40, lr:0.00040000, loss:9911.20655139, time:230.63ms, fps:138.75 <br>
    > 2021-01-05 14:05:37 Step:60, lr:0.00040000, loss:9746.84064846, time:225.71ms, fps:141.78 <br>
    > 2021-01-05 14:05:41 Step:80, lr:0.00040000, loss:9610.47396385, time:225.90ms, fps:141.66 <br>
    > 2021-01-05 14:05:46 Step:100, lr:0.00040000, loss:9442.95130887, time:223.77ms, fps:143.00 <br>
    > 2021-01-05 14:05:50 Step:120, lr:0.00040000, loss:9300.51361234, time:228.85ms, fps:139.83 <br>
    > 2021-01-05 14:05:55 Step:140, lr:0.00040000, loss:9029.22531498, time:227.36ms, fps:140.75 <br>
    > 2021-01-05 14:05:59 Step:160, lr:0.00040000, loss:8850.09716859, time:225.94ms, fps:141.63 <br>
    > 2021-01-05 14:06:04 Step:180, lr:0.00040000, loss:8620.74724067, time:228.48ms, fps:140.05 <br>
    > 2021-01-05 14:06:09 Step:200, lr:0.00040000, loss:8384.87165179, time:227.64ms, fps:140.57 <br>
    > 2021-01-05 14:06:13 Step:220, lr:0.00040000, loss:8193.46840854, time:230.90ms, fps:138.59 <br>
    > 2021-01-05 14:06:18 Step:240, lr:0.00040000, loss:8052.94862988, time:229.94ms, fps:139.17 <br>
    > 2021-01-05 14:06:22 Step:260, lr:0.00040000, loss:7884.65315172, time:226.39ms, fps:141.35 <br>
    > 2021-01-05 14:06:27 Step:280, lr:0.00040000, loss:7719.58562702, time:225.88ms, fps:141.67 <br>
    > 2021-01-05 14:06:31 Step:300, lr:0.00040000, loss:7546.77749729, time:229.57ms, fps:139.39 <br>
    > 2021-01-05 14:06:36 Step:320, lr:0.00040000, loss:7387.72234128, time:228.84ms, fps:139.83 <br>
    > 2021-01-05 14:06:40 Step:340, lr:0.00039999, loss:7278.32803386, time:229.31ms, fps:139.55 <br>
    > 2021-01-05 14:06:45 Step:360, lr:0.00039999, loss:7133.84139243, time:226.94ms, fps:141.01 <br>
    > 2021-01-05 14:06:50 Step:380, lr:0.00039999, loss:7018.06870953, time:227.69ms, fps:140.54 <br>
    > 2021-01-05 14:06:54 Step:400, lr:0.00039999, loss:6941.95861095, time:228.10ms, fps:140.29 <br>
    > 2021-01-05 14:06:59 Step:420, lr:0.00039999, loss:6833.76219723, time:225.16ms, fps:142.12 <br>
    > 2021-01-05 14:07:03 Step:440, lr:0.00039999, loss:6742.74311336, time:226.48ms, fps:141.29 <br>
    > 2021-01-05 14:07:08 Step:460, lr:0.00039999, loss:6676.80909872, time:227.10ms, fps:140.91 <br>
    > 2021-01-05 14:07:12 Step:480, lr:0.00039999, loss:6571.46454924, time:226.12ms, fps:141.52 <br>
    > 2021-01-05 14:07:17 Step:500, lr:0.00039999, loss:6452.07785241, time:223.62ms, fps:143.10 <br>

**Note**: The given scripts ``edvr.yaml``, ``run_1p_train.sh`` are only for reference.
    In ``run_1p_train.sh``, the training proceeds with ``batchsize=4`` and runs for only 1000 steps, 
which cannot reach the reproduced precision (PSNR 31.24dB on REDS4). If one wants to fully 
reproduce the given PSNR, please run ``run_1p_train_precision_overwatch.sh`` (for 1 device) or 
``run_8p_train_precision_overwatch.sh`` (for 8 devices). The former requires several days to complete 
the training, while the latter also takes one day at least.

## Evaluation

It's easy to evaluate the checkpoint or any other model file.

```sh
bash scripts/run_evaluation.sh 0 outputs/edvr/EDVR-600000
```

The first argument represents the device_id, and the second the certain checkpoint. The output will be like:

```sh
Evaluate 000
Video 000 PSNR = 27.58365249633789
        Inference time: 102.73
Evaluate 011
Video 011 PSNR = 30.992197036743164
        Inference time: 102.67
Evaluate 015
Video 015 PSNR = 33.03461837768555
        Inference time: 102.22
Evaluate 020
Video 020 PSNR = 29.36363983154297
        Inference time: 102.83
PSNR = 30.243528366088867
```

## Inference

Similar to evalution: 

```sh
bash scripts/run_inference.sh 0 outputs/edvr/EDVR-600000
```

The inference results (images) will be saved in ``${cfg.output_dir}/test`` folder. Since the writing time is the 
bottleneck of the total time, the project uses a standalone queue and mutliple threads to write the super-resolution
 image to hard disks. One may find the hint after the inference bar:

> Writing images to files. This may take some time. Please DO NOT manually interrupt !!!

and the prompt does not show. This is normal and please do not interrupt, because the threads are still writing the 
rest images.

## Freeze Graph

```sh
bash scripts/run_freeze.sh outputs/edvr/EDVR-600000
```

The frozen pb file will be saved at ``${cfg.output_dir}/EDVR.pb``. 

If one wants to use 4D ``[N*D, H, W, C]`` input instead of 5D ``[N, D, H, W, C]`` where ``D`` represents the number of 
consecutive frames of EDVR, please 
set ``model.input_format_dimension=4`` when freezing; additionally, if one wants to use unknown batchsize in pb model, please set the 
``data.eval_batch_size=-1`` when freezing. The output node of the model is by default of ``fp32`` data type. One can set 
``model.convert_output_to_uint8=True`` when freezing so that the scripts will add a cast op before output converting to 
``tf.uint8``.

## Precision & Performance

Evaluated on REDS4 dataset

|      | Training Input Size (per device) | Training Time (ms/step) | Inference Input Size (single device) | Inference Time (ms/step) | PSNR (dB)   |
| ---- | -------------------------------- | ----------------------- | ------------------------------------ | ------------------------ | ----------- |
| 1p   | [32, 5, 64, 64, 3]               | 875                     | [1, 5, 180, 320, 3]                  | 102                      | 30.24352837 |
| 8p   | [4, 5, 64, 64, 3]                | 230                     | [1, 5, 180, 320, 3]                  | 102                      | 30.24139595 |

## NOTES

1. To run on GPU, one can simply set the ``cfg.device`` to ``GPU``, and make sure to set the implementation of 
   deformable convolution to the composed by ``cfg.edvr.impl='tf'``. The rest remains the same. However, the project only support for a single GPU device.
2. (**IMPORTANT**) The standalone *deformable convolution* operator is available after version C76B220 (included). Otherwise one can only use the composed tensorflow operator for deformable convolution, but the performance is rather poor compared to the former one. Setting ``cfg.edvr.impl`` to ``tf`` so that the deformable convolution will be running on GPU/CPU, while with ``cfg.edvr.impl='npu'``  the NPU deformable operator is used.

