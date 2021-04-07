# ResNet18 

This implements training of ResNet18 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).



## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# O2 training
bash scripts/train_1p.sh

# O2 training 8p
bash scripts/train_8p.sh
```

## ResNet18 training result
| Acc@1 | FPS  | Npu_nums | Epochs | AMP_Type |
| :---: | :--: | :------: | :----: | :------: |
|   -   | 4121 |    1     |  120   |    O2    |
| 69.91 | 6400 |    8     |  120   |    O2    |

