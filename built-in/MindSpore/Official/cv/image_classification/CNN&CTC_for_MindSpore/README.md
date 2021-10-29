# Contents

- [Contents](#contents)
- [CNNCTC Description](#cnnctc-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
- [ModelZoo Homepage](#modelzoo-homepage)


# [CNNCTC Description](#contents)
This paper proposes three major contributions to addresses scene text recognition (STR).
First, we examine the inconsistencies of training and evaluation datasets, and the performance gap results from inconsistencies.
Second, we introduce a unified four-stage STR framework that most existing STR models fit into.
Using this framework allows for the extensive evaluation of previously proposed STR modules and the discovery of previously
unexplored module combinations. Third, we analyze the module-wise contributions to performance in terms of accuracy, speed,
and memory demand, under one consistent set of training and evaluation datasets. Such analyses clean up the hindrance on the current
comparisons to understand the performance gain of the existing modules.
[Paper](https://arxiv.org/abs/1904.01906): J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, “What is wrong with scene text recognition model comparisons? dataset and model analysis,” ArXiv, vol. abs/1904.01906, 2019.

# [Model Architecture](#contents)
This is an example of training CNN+CTC model for text recognition on MJSynth and SynthText dataset with MindSpore.



# [Dataset](#contents)

The [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](https://github.com/ankush-me/SynthText) dataset are used for model training. The [The IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) dataset is used for evaluation.

- step 1:
All the datasets have been preprocessed and stored in .lmdb format and can be downloaded following this [introduction](https://github.com/ankush-me/SynthText#pre-generated-dataset)

- step 2:
Uncompress the downloaded file, rename the MJSynth dataset as MJ, the SynthText dataset as ST and the IIIT dataset as IIIT.

- step 3:
Move above mentioned three datasets into `cnnctc_data` folder, and the structure should be as below:
```
|--- CNNCTC/
    |--- cnnctc_data/
        |--- ST/
            data.mdb
            lock.mdb
        |--- MJ/
            data.mdb
            lock.mdb
        |--- IIIT/
            data.mdb
            lock.mdb
    
    ......
```

- step 4:
Preprocess the dataset by running:
```
python src/preprocess_dataset.py
```

This takes around 75 minutes.

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. 
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.



# [Environment Requirements](#contents)

- Hardware（Ascend）
  
  - Prepare hardware environment with Ascend processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 

  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)

  
  

# [Quick Start](#contents)

- Install dependencies:
```
pip install lmdb
pip install Pillow
pip install tqdm
pip install six
```

- Standalone Training:
```
bash scripts/run_standalone_train_ascend.sh $PRETRAINED_CKPT
```

- Distributed Training:
```
bash scripts/run_distribute_train_ascend.sh $RANK_TABLE_FILE $PRETRAINED_CKPT
```

- Evaluation:
```
bash scripts/run_eval_ascend.sh $TRAINED_CKPT
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)
The entire code structure is as following:
```
|--- CNNCTC/
    |---README.md    // descriptions about cnnctc
    |---train.py    // train scripts
    |---eval.py    // eval scripts
    |---scripts
        |---run_standalone_train_ascend.sh    // shell script for standalone on ascend
        |---run_distribute_train_ascend.sh    // shell script for distributed on ascend
        |---run_eval_ascend.sh    // shell script for eval on ascend
    |---src
        |---__init__.py    // init file
        |---cnn_ctc.py    // cnn_ctc network
        |---config.py    // total config
        |---callback.py    // loss callback file
        |---dataset.py    // process dataset
        |---util.py    // routine operation
        |---generate_hccn_file.py    // generate distribute json file
        |---preprocess_dataset.py    // preprocess dataset

```


## [Script Parameters](#contents)
Parameters for both training and evaluation can be set in `config.py`.

Arguments:
  * `--CHARACTER`: Character labels.
  * `--NUM_CLASS`: The number of classes including all character labels and the <blank> label for CTCLoss.
  * `--HIDDEN_SIZE`: Model hidden size.
  * `--FINAL_FEATURE_WIDTH`: The number of features.
  * `--IMG_H`： The height of input image.
  * `--IMG_W`： The width of input image.
  * `--TRAIN_DATASET_PATH`： The path to training dataset.
  * `--TRAIN_DATASET_INDEX_PATH`： The path to training dataset index file which determines the order .
  * `--TRAIN_BATCH_SIZE`： Training batch size. The batch size and index file must ensure input data is in fixed shape.
  * `--TRAIN_DATASET_SIZE`： Training dataset size.
  * `--TEST_DATASET_PATH`： The path to test dataset.
  * `--TEST_BATCH_SIZE`： Test batch size.
  * `--TEST_DATASET_SIZE`：Test dataset size.
  * `--TRAIN_EPOCHS`：Total training epochs.
  * `--CKPT_PATH`：The path to model checkpoint file, can be used to resume training and evaluation.
  * `--SAVE_PATH`：The path to save model checkpoint file.
  * `--LR`：Learning rate for standalone training.
  * `--LR_PARA`：Learning rate for distributed training.
  * `--MOMENTUM`：Momentum.
  * `--LOSS_SCALE`：Loss scale to prevent gradient underflow.
  * `--SAVE_CKPT_PER_N_STEP`：Save model checkpoint file per N steps.
  * `--KEEP_CKPT_MAX_NUM`：The maximum number of saved model checkpoint file.

## [Training Process](#contents)

### Training

- Standalone Training:
```
bash scripts/run_standalone_train_ascend.sh $PRETRAINED_CKPT
```

Results and checkpoints are written to `./train` folder. Log can be found in `./train/log` and loss values are recorded in `./train/loss.log`.

`$PRETRAINED_CKPT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

- Distributed Training:
```
bash scripts/run_distribute_train_ascend.sh $RANK_TABLE_FILE $PRETRAINED_CKPT
```

Results and checkpoints are written to `./train_parallel_{i}` folder for device `i` respectively.
 Log can be found in `./train_parallel_{i}/log_{i}.log` and loss values are recorded in `./train_parallel_{i}/loss.log`.

`$RANK_TABLE_FILE` is needed when you are running a distribute task on ascend. 
`$PATH_TO_CHECKPOINT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

### Training Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in loss.log.

 
```
# distribute training result(8p)
epoch: 1 step: 1 , loss is 76.25, average time per step is 0.335177839748392712
epoch: 1 step: 2 , loss is 73.46875, average time per step is 0.36798572540283203
epoch: 1 step: 3 , loss is 69.46875, average time per step is 0.3429678678512573
epoch: 1 step: 4 , loss is 64.3125, average time per step is 0.33512671788533527
epoch: 1 step: 5 , loss is 58.375, average time per step is 0.33149147033691406
epoch: 1 step: 6 , loss is 52.7265625, average time per step is 0.3292975425720215
...
epoch: 1 step: 8689 , loss is 9.706798802612482, average time per step is 0.3184656601312549
epoch: 1 step: 8690 , loss is 9.70612545289855, average time per step is 0.3184725407765116
epoch: 1 step: 8691 , loss is 9.70695776049204, average time per step is 0.31847309686135555
epoch: 1 step: 8692 , loss is 9.707279624277456, average time per step is 0.31847339290613375
epoch: 1 step: 8693 , loss is 9.70763437950938, average time per step is 0.3184720295013031
epoch: 1 step: 8694 , loss is 9.707695425072046, average time per step is 0.31847410284595573
epoch: 1 step: 8695 , loss is 9.708408273381295, average time per step is 0.31847338271072345
epoch: 1 step: 8696 , loss is 9.708703753591953, average time per step is 0.3184726025560777
epoch: 1 step: 8697 , loss is 9.709536406025824, average time per step is 0.31847212061114694
epoch: 1 step: 8698 , loss is 9.708542263610315, average time per step is 0.3184715309307257
```

## [Evaluation Process](#contents)

### Evaluation
- Evaluation:
```
bash scripts/run_eval_ascend.sh $TRAINED_CKPT
```

The model will be evaluated on the IIIT dataset, sample results and overall accuracy will be printed.


# [Model Description](#contents)
## [Performance](#contents)

### Training Performance 

| Parameters                 | FasterRcnn                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 09/28/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                      |
| Dataset                    |  MJSynth,SynthText                                                  |
| Training Parameters        | epoch=3,  batch_size=192          |
| Optimizer                  | RMSProp                                                         |
| Loss Function              | CTCLoss                                      |
| Speed                      | 1pc: 300 ms/step;  8pcs: 310 ms/step                          |
| Total time                 | 1pc: 18 hours;  8pcs: 2.3 hours                          |
| Parameters (M)             | 177                                                         |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc |


### Evaluation Performance

| Parameters          | FasterRcnn                   |
| ------------------- | --------------------------- |
| Model Version       | V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/28/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | IIIT5K    |
| batch_size          | 192                         |
| outputs             | Accuracy                 |
| Accuracy            |  85%  |
| Model for inference | 675M (.ckpt file)         |

## [How to use](#contents)
### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/network_migration.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target=cfg.device_target)
  context.set_context(device_id=cfg.device_id)

  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### Continue Training on the Pretrained Model

- running on Ascend

  ```
  # Load dataset
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = CNNCTC(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = P.CTCLoss(preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False,                   loss_scale_manager=None)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```


# [ModelZoo Homepage](#contents)
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
