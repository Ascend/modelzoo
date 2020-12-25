# 乐府

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Quick start guide](#quick-start-guide)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)   
    * [Evaluation](#Evaluation)


## Description

Yuefu is proposed by Huawei’s Noah’s Ark. It proposes a method for generating ancient Chinese poetry based on the GPT model. It uses large-scale Chinese news corpus training to obtain a Chinese GPT model, and collects a large number of ancient Chinese poems, words, and couplets in a specific format. After sorting, and then inputting the sorted data into the GPT model for fine-tune, a high-quality ancient poetry generation system is obtained.

Reference paper: Yi Liao, Yasheng Wang, Qun Liu, Xin Jiang: “GPT-based Generation for Classical Chinese Poetry”, 2019

## Requirements

- Prepare python environment and yuefu checkopint.

## Quick start guide

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


### Running the example

#### Evaluation

Set the checkpoint path, the default is ModelZoo_Yuefu_TF/

Set `main_1p.sh`, configuration parameters for non-distributed training of Transformer model.

```
max_decode_len: the maximum generation length, the default setting is 80

Title: Enter the subject of the poetry you want to generate

Type: Enter the subject of the poetry you want to generate, including five-character quatrain poetry, five-character quatrain, seven-character quatrain, seven-character quatrain

export JOB_ID=10086
export DEVICE_ID=3
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=${dir}/device_table_1p.json
export USE_NPU=True
export POETRY_TYPE=-1
export max_decode_len=80
python3 poetry_v2.py --title="Mid-Autumn Festival" --type="Seven Words Quatrains"
```

- Set options in main_1p.sh, including max_decode_len and other hyperparameters,Run the script: 

    ``` bash
    sh main_1p.sh
    ```

