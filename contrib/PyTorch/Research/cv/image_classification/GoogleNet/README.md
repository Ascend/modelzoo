# GoogleNet 

This implements training of googlenet on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## GoogleNet Detail

Details, see ./googlenet.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py`or `main-8p.py` with the desired model architecture and the path to the ImageNet dataset:



# 1p training 1p
bash script/run_1p.sh

# 8p training 8p
bash script/run_8p.sh

# eval default 8p， should support 1p
bash script/eval.sh

# online inference demo 
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

## GoogleNet training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 515       | 1        | 150      | O2       |
| 69.807   | 4653      | 8        | 150      | O2       |
