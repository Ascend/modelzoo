Autorec : Autoencoder meets Collaborative Filtering
(Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th International Conference on World Wide Web (pp. 111-112). ACM

TensorFlow Implementation for I-AutoRec on Ascend 910 environment

Dataset ml-1m can be accessed from [https://dataset-ml-1m.obs.dualstack.cn-north-4.myhuaweicloud.com](https://dataset-ml-1m.obs.dualstack.cn-north-4.myhuaweicloud.com)

Reported ml-1m RMSE from the original paper : 0.831; from this implementation : 0.809.

```
python3 main.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--hidden_neuron 1024 \
	--train_epoch 200 \
	--batch_size 256
```
