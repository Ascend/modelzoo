## 1.Install dnmetis_backend

As README.md:
backend_C++/dnmetis_backend/README.md

## 2.Download dataset and model(.om)

1.download  Imagenet-val dataset URL: http://www.image-net.org/download-imageurls

2.download  NoisyStudent model(.om) URL: obs://hwwheel23/noisy_student_efficientnet-l2.om 

3.process the original Imagenet-val dataset as list:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0918/234302_a572d632_5418572.jpeg "无标题.jpg")



## 3.Start execute the inference:

sh run_efficientnet-b8.sh

or 

python3.7 main.py --model=/data/hwwheel123/model/noisy_student_efficientnet-l2.om --image_size='800,800,3' --inputs='images:0' --outputs='Softmax:0' --precision=fp16


## 4.ATC offline model generate (optional):

1.download  NoisyStudent model(.pb) URL: obs://hwwheel23/noisy_student_efficientnet-l2.pb 

2.atc --model=$MODEL_DIR/noisy_student_efficientnet-l2.pb --framework=3 --input_shape='images:1,800,800,3' --output=$MODEL_DIR/noisy_student_efficientnet-l2 --mode=0 --out_nodes='Softmax:0' --soc_version=Ascend310  --input_fp16_nodes=images --output_type=FP16

## 5.Imagenet2012-val Top1 Accuracy:

![输入图片说明](https://images.gitee.com/uploads/images/2020/0919/005135_096ca9da_5418572.png "屏幕截图.png")
