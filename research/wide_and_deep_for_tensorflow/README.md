## Overview
### Get The Dataset
===============

Criteo released the data set after the end of the competition. However, the
format of the released data set is different from the one used in the
competition; the format used in the comptition is in csv format, while the
released one is in text format.

Our scripts require csv format. If you already have the dataset used in the
competition, please skip to the next section. Otherwise, we provide a script to
convert the files in text format to csv format. Please follow these steps:

1. Download the data set from Criteo.

    http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

2. Decompress and make sure the files are correct.

    $ md5sum dac.tar.gz
    df9b1b3766d9ff91d5ca3eb3d23bed27  dac.tar.gz

    $ tar -xzf dac.tar.gz

    $ md5sum train.txt test.txt
    4dcfe6c4b7783585d4ae3c714994f26a  train.txt
    94ccf2787a67fd3d6e78a62129af0ed9  test.txt


3. Convert the original data to tf record data.

Now you can follow the steps in `Step-by-step` section to run our experiments.

---

The code sample in this directory uses the low level `tf.Session` API.

## Running the code
### Training

You can run the code locally as follows:

```
python3.7 widedeep_main_record_multigpu_fp16_huifeng.py 10000
```

The model is saved to `model` by default, which can be changed using the `--model_dir` flag.

To run the *wide* or *deep*-only models, set the `--model_type` flag to `wide` or `deep`.

The final accuracy should be over 80.8% with any of the three model types.


### Parameters
you can set the parameter in config.py, some of the key parameters are list as:
```
--batch_size: samples in one step
--num_gpu: number of training 1980
--n_epoch: total training epoches
--width: the width of wide network
--depth: the depth of deep network
--record_path: data path
--one_step: debug by step
```