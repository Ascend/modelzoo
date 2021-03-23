## 1.Install dnmetis_backend

As README.md:
backend_C++/dnmetis_backend/README.md

## 2.Download dataset and model(.om)

1.download  Imagenet-val dataset

2.download  xception model(.om) URL: obs://hwwheel23/xception_tf.om 

3.process the original Imagenet-val dataset as list:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/234302_a572d632_5418572.jpeg "无标题.jpg")



## 3.Start execute the inference:

sh run_xception.sh


## 4.ATC offline model generate (optional):

1.download  xception model(.pb) URL: obs://hwwheel23/xception_tf.pb 

2.atc --model=$MODEL_DIR/xception_tf.pb --framework=3 --input_shape='input_1:1,299,299,3' --output=$MODEL_DIR/xception_tf --mode=0 --out_nodes='predictions/Softmax:0' --soc_version=Ascend310  --log=info --input_fp16_nodes="input_1" --output_type=FP16

## 5.Imagenet2012-val Top1 Accuracy:

--------------------------------------------------------------------------------------------------------------------
img_orig: ILSVRC2012_val_00049998.JPEG label: 232 predictions: 230 

img_orig: ILSVRC2012_val_00049999.JPEG label: 982 predictions: 982 

img_orig: ILSVRC2012_val_00050000.JPEG label: 355 predictions: 355 

[Accuracy] Predict total jpeg: 50000  Accuracy:  0.78436
[INFO] AclBackend unload OK

--------------------------------------------------------------------------------------------------------------------

python3.7 main.py --model=/data/hwwheel123/model/xception_tf.om --image_size='299,299,3' --inputs='input_1:0' --outputs='dense_1/Softmax:0' --precision=fp16

[INFO] AclBackend init OK

[INFO] AclBackend load OK

[INFO] start warmup AclBackend predict

[INFO] end warmup AclBackend predict

[Perf] Predict total jpeg: 10  Cost all time(s):  0.1485152244567871

[INFO] AclBackend unload OK

--------------------------------------------------------------------------------------------------------------------

![输入图片说明](https://images.gitee.com/uploads/images/2020/0919/210429_36b8fdd0_5418572.png "屏幕截图.png")
