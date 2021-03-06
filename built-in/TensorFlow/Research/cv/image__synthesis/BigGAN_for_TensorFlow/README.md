# BigGAN-Tensorflow
Simple Tensorflow implementation of ["Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)](https://arxiv.org/abs/1809.11096)

![main](./assets/main.png)

## Issue
* **The paper** used `orthogonal initialization`, but `I used random normal initialization.` The reason is, when using the orthogonal initialization, it did not train properly.
* I have applied a hierarchical latent space, but **not** a class embeddedding.

## Usage
### dataset
* `mnist` and `cifar10` are used inside keras
* For `your dataset`, put images like this:

```
├── dataset
   └── train
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
```
### train
单P训练
cd scripts;bash run_npu_1p.sh

8P训练
cd scripts;bash run_npu_8p.sh

### test

验证


cd scripts;bash eval.sh 



### PB generation
We added BigGAN/frozen_graph.py for the pb file generation.
To run this code, backpack your codes first, then
####  change the last line of scripts/npu_run.sh to 'python3.7 BigGAN/frozen_graph.py'

```bash
$ bash scripts/run_npu_supervised.sh
```** 