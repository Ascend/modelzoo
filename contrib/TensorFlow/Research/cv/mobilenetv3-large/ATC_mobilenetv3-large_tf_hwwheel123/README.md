## 1.Install dnmetis_backend

As README.md:
backend_C++/dnmetis_backend/README.md

## 2.Download dataset and model(.om)

1.download  Imagenet-val dataset

2.download  mobilenet-v3-large model(.om) URL: obs://hwwheel23/mobilenet-v3-large.om 

3.process the original Imagenet-val dataset as list:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/234302_a572d632_5418572.jpeg "无标题.jpg")



## 3.Start execute the inference:

sh run_mobilenet-v3-large.sh


## 4.ATC offline model generate (optional):

1.download  mobilenet-v3-large model(.pb) URL: obs://hwwheel23/mobilenet-v3-large.pb 

2.atc --model=$MODEL_DIR/mobilenet-v3-large.pb --framework=3 --input_shape='input:1,224,224,3' --output=$MODEL_DIR/mobilenet-v3-large --mode=0 --out_nodes='MobilenetV3/Predictions/Softmax:0' --soc_version=Ascend310  --log=info --input_fp16_nodes="input" --output_type=FP16

## 5.Imagenet2012-val Top1 Accuracy:

--------------------------------------------------------------------------------------------------------------------

img_orig: ILSVRC2012_val_00049997.JPEG label: 27 predictions: 27 

img_orig: ILSVRC2012_val_00049998.JPEG label: 233 predictions: 233 

img_orig: ILSVRC2012_val_00049999.JPEG label: 983 predictions: 983 

img_orig: ILSVRC2012_val_00050000.JPEG label: 356 predictions: 356 

[Accuracy] Predict total jpeg: 50000  Accuracy:  0.75508

[INFO] AclBackend unload OK

--------------------------------------------------------------------------------------------------------------------

## 6.Imagenet2012-val Perf:

--------------------------------------------------------------------------------------------------------------------

python3.7 main.py --model=/data/hwwheel123/model/mobilenet-v3-large.om --image_size='224,224,3' --inputs='input:0' --outputs='MobilenetV3/Predictions/Softmax:0' --precision=fp16    --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=1000

[INFO] AclBackend init OK

[INFO] AclBackend load OK

[INFO] start warmup AclBackend predict

[INFO] end warmup AclBackend predict

[Perf] Predict total jpeg: 1000  Cost all time(s):  3.8576412200927734

[INFO] AclBackend unload OK


--------------------------------------------------------------------------------------------------------------------

