# GAT
Graph Attention Networks (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora (files can be accessed from [https://dataset-cora.obs.dualstack.cn-north-4.myhuaweicloud.com](https://dataset-cora.obs.dualstack.cn-north-4.myhuaweicloud.com));
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` contains a pre-trained Cora model;
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);
    * preprocessing utilities for the PPI benchmark (`process_ppi.py`).

Finally, `execute_cora.py` puts all of the above together and may be used to execute a full training run on Cora. The reported result from original paper is 83.0&plusmn;0.7%. This implemetation achieves around 82.9%.

## Dependencies

The script has been tested running under Python 3.7 Ascend 910 environment, with the following packages installed (along with their dependencies):

- `numpy`
- `scipy`
- `networkx`
- `tensorflow`

## Usage

```
python3 execute_cora.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--batch_size 1 \
	--nb_epochs 200 \
	--lr 0.005 \
	--l2_coef 0.0005
```


## License
MIT
