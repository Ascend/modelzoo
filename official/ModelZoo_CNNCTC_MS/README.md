## Overview
This is an example of training CNN+CTC model for text recognition on MJSynth and SynthText dataset with MindSpore.

## Dataset

<!---
[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](https://github.com/ankush-me/SynthText) are used for model training.
 
[The IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) are used for evaluation.
-->

The [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](https://github.com/ankush-me/SynthText) dataset are used for model training. The [The IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) dataset is used for evaluation.

- step 1:
All the datasets have been preprocessed and stored in .lmdb format and can be downloaded [**HERE**](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt).

- step 2:
Uncompress the downloaded file, rename the MJSynth dataset as MJ, the SynthText dataset as ST and the IIIT dataset as IIIT.

- step 3:
Move above mentioned three datasets into `./data` folder, and the structure should be as below:
```
|--- CNNCTC/
    |--- data/
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
python scripts/preprocess_dataset.py
```

This takes around 75 minutes.


### Code Structure
The entire code structure is as following:
```
|--- CNNCTC/
    README.md
    requirements.txt
    train.py
    train_para.py
    eval.py

    |--- data
        |--- ST/
        |--- MJ/
        |--- IIIT/
        |--- ST_MJ/
        st_mj_fixed_length_index_list.pkl

    |--- scripts
        run_standalone_train.sh
        run_distribute_train.sh
        generate_hccn_file.py
        preprocess_dataset.py
    
    |--- src
        |--- CNNCTC/
            model.py

        config.py
        callback.py 
        dataset.py
        util.py

```

## Running Code

- Install dependencies:
```
pip install -r requirements.txt
```

- Training:
```
bash scripts/run_standalone_train.sh $PATH_TO_CHECKPOINT
```

Results and checkpoints are written to `./train` folder. Log can be found in `./train/log` and loss values are recorded in `./train/loss.log`.

`$PATH_TO_CHECKPOINT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

- Training using 8P:
```
bash scripts/run_distribute_train.sh $PATH_TO_CHECKPOINT
```

Results and checkpoints are written to `./train_parallel_device_{i}` folder for device `i` respectively.
 Log can be found in `./train_parallel_device_{i}/log_device_{i}.log` and loss values are recorded in `./train_parallel_device_{i}/loss.log`.

`$PATH_TO_CHECKPOINT` is the path to model checkpoint and it is **optional**. If none is given the model will be trained from scratch.

- Evaluation:
```
python eval.py --ckpt_path [PATH_TO_CHECKPOINT]
```

The model will be evaluated on the IIIT dataset, sample results and overall accuracy will be printed.


### Parameter configuration
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

## Performance

### Result

#### Test accuracy results

| Dataset |   Accuracy   |
| :--------: | :-----------: |
|     IIIT     | 84.3% |

#### Training performance results

| **NPUs** | time per step |
| :------: | :---------------: |
|    1     |   260 ms   |
|    8     |   260 ms   |

