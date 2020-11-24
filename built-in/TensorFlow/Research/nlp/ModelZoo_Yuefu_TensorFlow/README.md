# 乐府
## Description
This example implements training and evaluation of Transformer Model, which is introduced in the following paper:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## Requirements
- Prepare python environment and yuefu checkopint.

## Example structure

```shell
.
└─YueFu
  ├─README.md
  ├─poetry_v2.py
  ├─tokenization.py
  ├─white_list.txt
  ├─main_1p.sh
  └─black_list.txt
```

## Running the example

### Inferring
- if environment is Atlas Data Center Solution V100R020C10, please use the following settings：

    ```
    export install_path=/usr/local/Ascend/nnae/latest# driver包依赖export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH#fwkacllib 包依赖export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATHexport PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATHexportPATH=${install_path}/fwkacllib/ccec_compiler/bin:{install_path}/fwkacllib/bin:$PATH#tfplugin 包依赖export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH# opp包依赖export ASCEND_OPP_PATH=${install_path}/opp
    ```

    set the checkpoint path, the default is ModelZoo_Yuefu_TF/

    Run `main_1p.sh` for non-distributed training of Transformer model.

    ```bash
    title is the parameter denote the name of the poetry;
    type is the type of poetry, including 五言律诗, 五言绝句, 七言绝句, 七言律诗;
    ```

- Set options in main_1p.sh, including max_decode_len and other hyperparameters

    ``` bash
    sh main_1p.sh
    ```

