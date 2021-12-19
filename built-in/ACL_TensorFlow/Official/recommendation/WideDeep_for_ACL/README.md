

# widedeep Inference for Tensorflow 

This repository provides a script and recipe to Inference the widedeep model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL/Research/recommendation/WideDeep_for_ACL
```

### 2. Download weights and test dataset
[Weights files,access code:**ascend**](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=ps4oG2NRMXJdKgQrPTQLN2xbRMW0y1ENptC+xgiaJBjyVS8X1vzxx3oBtbozGX5utCjbEdzYpBgLXj6IBanV++xpRfqeZY/WOqR7eGcKMbLsbPD/QvsqFYTIgPbYIFgtJnhmpp2t3V4akctZr6rbOc5bjzGaZiq/dU6r+GwXaemxgLzTBoUMrTNJdKsvD+QZBppUjiF23f6xOHwqUquOXr7HPlFtx+K0ImJOXbDdZlYoSJAb1wGZ6RgNgmWNX691n4hWhjGQ4qkqckVqOV+UZrRaca66qT7i+GMsd5TNb/iAQ6b8R9wpGIkqgS/y17gawgeqGlL3Hy0aEToOCYMUESrnw30waqxA7E5/ahP6GCO3brhvmkefNqA/8yweYnNB78Ii6Mc4cgM7fX7kWfJbsp7HqfTF39ywkQe2/ecCqJG8aQDG7yolKrYQOLUiP8+oRUYUSI3dERHDcXPDf4atAbZN9Y/1XXAhmhc+E3xy4HbIa+uZy0Oik3Jhkvl1i5zB2Gb83QdyIQuCKUclTJXo/OQo46BJl6HcWELY+UfqZhTXDy5FZWYrnjfzyzOdq+GY0n0/fofy2/8LrvUldHgmp2jojS6jroyEQvT03vkzOzixRDZrWtFUYQmTev7+YBprV+GfmWm60TU0Olznc65Gqrjqs1xtTHwglKqYn1/22z03wtreRdhgfLcrf3pN8RdybScmRWMp28Ro9gR+0lITa5ct22Lh+B4WhDRSc6SYckwsa58Kzg/D4ctRAcrHjsaQAz73nfVj1DX7qaIWVRPizi6uaSR5CZbAmjClX+FFlibpnLmvFlBJ4NSmKAOoxKx+VCw8szniylSmp2OFRQ70dQ==)

and put them to **'acl/bin'**

[test dataset](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/WideDeep_ID0028_for_ACL/acl/data/adult.test?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656054988&Signature=UXJ9XFtUdgDeM2PDVesztNBzQXs%3D)

put test dataset to **'acl/data/'**


### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export ASCEND_OPP_PATH=/usr/local/Ascend/opp
  export PATH=/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/atc/lib64:/usr/local/Ascend/toolkit/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
  export PYTHONPATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
  ```

- convert pb to om

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/widedeep_tf.pb)

  ```
  atc --model=acl/model/widedeep_tf.pb --framework=3 --output=acl/model/widedeep_framework_tensorflow_aipp_0_batch_1_input_fp32_output_fp32 --soc_version=Ascend310 --input_shape="Input:1,51"
  ```

- Build the program

  ```
  cd benchmark-master
  bash run.sh
  ```

- Run the program:

  ```
  cd ../
  bash benchmark_tf.sh --batchSize=1 --modelPath=acl/model/widedeep_framework_tensorflow_aipp_0_batch_1_input_fp32_output_fp32.om --dataPath=acl/data/adult.test
  ```


## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Top1    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 16281     | 85.44 % |

