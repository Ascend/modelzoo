# VGG-16

This implements training of vgg19 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## VGG-16 Detail

As of the current date, Ascend-Pytorch is still have some bug in nn.Dropout(). For details, see ./vgg.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
cp ./scripts/*.sh ./

# O2 training 1p
bash train_1p.sh

# O2 training 8p
bash train_8p.sh
```

## VGG-229 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 360       | 1        | 240      | O2       |
| 72.933   | 2500      | 8        | 240      | O2       |
