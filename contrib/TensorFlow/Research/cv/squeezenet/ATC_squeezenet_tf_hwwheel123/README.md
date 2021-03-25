## 1.Install dnmetis_backend

As README.md:
backend_C++/dnmetis_backend/README.md

## 2.Download dataset and model(.om)

1.download  Imagenet-val dataset

2.download  squeezenet model(.om) URL: obs://hwwheel23/squeezenet.om 

3.process the original Imagenet-val dataset as list:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/234302_a572d632_5418572.jpeg "无标题.jpg")



## 3.Start execute the inference:

sh run_squeezenet.sh


## 4.ATC offline model generate (optional):

1.download  squeezenet model(.pb) URL: obs://hwwheel23/squeezenet.pb 

2.atc --model=$MODEL_DIR/squeezenet.pb --framework=3 --input_shape='ImageTensor:1,227,227,3' --output=$MODEL_DIR/squeezenet --mode=0 --out_nodes='Softmax:0' --soc_version=Ascend310  --log=info --input_fp16_nodes="ImageTensor" --output_type=FP16

## 5.Imagenet2012-val Top1 Accuracy:

--------------------------------------------------------------------------------------------------------------------

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=1024

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=2048

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=3096

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=5000

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=10240

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=20480

python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --accuracy     --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=50000

[Accuracy] Predict total jpeg: 1024  Accuracy:  0.5703125

[Accuracy] Predict total jpeg: 2048  Accuracy:  0.55712890625

[Accuracy] Predict total jpeg: 3096  Accuracy:  0.5545865633074936

[Accuracy] Predict total jpeg: 5000  Accuracy:  0.5584

[Accuracy] Predict total jpeg: 10240  Accuracy:  0.55849609375

[Accuracy] Predict total jpeg: 20480  Accuracy:  0.553125

[Accuracy] Predict total jpeg: 50000  Accuracy:  0.55022


--------------------------------------------------------------------------------------------------------------------

## 6.Imagenet2012-val Perf:

--------------------------------------------------------------------------------------------------------------------

root@ecs-38b9:/data/hwwheel123/modelzoo/contrib/ATC_squeezenet_tf_c00506053# python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=1000

[INFO] AclBackend init OK

[INFO] AclBackend load OK

[INFO] start warmup AclBackend predict

[INFO] end warmup AclBackend predict

[Perf] Predict total jpeg: 1000  Cost all time(s):  1.5852699279785156

[INFO] AclBackend unload OK

root@ecs-38b9:/data/hwwheel123/modelzoo/contrib/ATC_squeezenet_tf_c00506053# python3.7 main.py --model=/data/hwwheel123/model/squeeze_tf.om --image_size='227,227,3' --inputs='ImageTensor:0' --outputs='Softmax:0' --precision=fp16 --dataset_path=/data/hwwheel123/ILSVRC2012_img_val --count=1000

[INFO] AclBackend init OK

[INFO] AclBackend load OK

[INFO] start warmup AclBackend predict

[INFO] end warmup AclBackend predict

[Perf] Predict total jpeg: 1000  Cost all time(s):  1.5843868255615234

[INFO] AclBackend unload OK



--------------------------------------------------------------------------------------------------------------------

