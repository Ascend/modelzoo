# Examples Reference

## Example Description

New algorithm dataset description:

| Dataset | Default Path | Data Source |
| :--- | :--- | :--: |
| Cifar10 | /cache/datasets/cifar10/ | [Download](https://www.cs.toronto.edu/~kriz/cifar.html) |

New algorithm pre-trained model description:

| Algorithm | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| Adelaide-EA | (model_name).pth | /cache/models/(model_name).pth | |

## Examples' Input and Output

Example of Prune-EA:

| Stage | Option | Content |
| :--: | :--: | :-- |
| nas | Input | Config File: compression/prune-ea/prune.yml <br> Pre-Trained Model: /cache/models/resnet20.pth <br> Dataset: /cache/datasets/cifar10 |
| nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
| nas | approximate running time | (random_models + num_generation * num_individual) * epochs / Number of GPUs * Training time per epoch |
| fully train | Input | Config File: compression/prune-ea/prune.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
| fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
| fully train | approximate running time | epochs * Training time per epoch |
