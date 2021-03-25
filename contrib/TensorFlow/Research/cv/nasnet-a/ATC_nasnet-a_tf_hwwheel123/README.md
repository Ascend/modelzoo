## 1.Install dnmetis_backend

As README.md:
backend_C++/dnmetis_backend/README.md

## 2.Download dataset and model(.om)

1.download  Imagenet-val dataset

2.download  NASNET-A model(.om) URL: obs://hwwheel23/frozen_nasnet_large.om 

3.process the original Imagenet-val dataset as list:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/234302_a572d632_5418572.jpeg "无标题.jpg")



## 3.Start execute the inference:

sh run_nasnet.sh

or 

python3.7 main.py --model=/data/hwwheel123/model/frozen_nasnet_large.om --image_size='331,331,3' --inputs='input:0' --outputs='final_layer/predictions:0' --precision=fp16


## 4.ATC offline model generate (optional):

1.download  NASNET-A model(.pb) URL: obs://hwwheel23/frozen_nasnet_large.pb 

2.atc --model=$MODEL_DIR/frozen_nasnet_large.pb --framework=3 --input_shape='images:1,331,331,3' --output=$MODEL_DIR/frozen_nasnet_large --mode=0 --out_nodes='final_layer/predictions:0' --soc_version=Ascend310  --input_fp16_nodes=input --output_type=FP16

## 5.Imagenet2012-val Top1 Accuracy:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/235610_5fbc1f27_5418572.png "屏幕截图.png")
