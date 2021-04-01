# 概要

网络名/用途：GRU4Rec 

论文链接/名: http : //arxiv.org/abs/1511.06939

源码链接：https://github.com/Songweiping/GRU4Rec_TensorFlow

## 环境

- Python 3.7.5 
- TensorFlow 
- numpy >= 1.12.1
- Pandas 

### 训练

运行 `sh run_npu_1p.sh`，可以开始训练。

详细参数修改请修改shell脚本中的python命令；
Other optional parameters include:   
 --layer: Number of GRU layers. Default is 1.  
 --size: Number of hidden units in GRU model. Default is 100.   
 --epoch: Runing epochs. Default is 3.   
 --lr : Initial learning rate. Default is 0.001.   
 --train: Specify whether training(1) or evaluating(0). Default is 1.   
 --hidden_act: Activation function used in GRU units. Default is tanh.   
 --final_act: Final activation function. Default is softmax.    
 --loss: Loss functions, cross-entropy, bpr or top1 loss. Default is cross-entropy.      
 --dropout: Dropout rate. Default is 0.5.


###精度和性能
GPU：Tesla V100-SXM2-32GB
Epoch 0 Step 1  lr: 0.001000    loss: 3.911899
Each Step time is:0.585581
Epoch 0 Step 1000       lr: 0.001000    loss: 3.435232
Each Step time is:0.004362
Epoch 0 Step 2000       lr: 0.000960    loss: 3.101549
Each Step time is:0.003441
Epoch 0 Step 3000       lr: 0.000922    loss: 2.896644
Each Step time is:0.003598
Epoch 1 Step 4000       lr: 0.000885    loss: 2.270550
Each Step time is:0.004265
Epoch 1 Step 5000       lr: 0.000849    loss: 2.173362
Each Step time is:0.004291
Epoch 1 Step 6000       lr: 0.000815    loss: 2.104624
Each Step time is:0.003798
Epoch 2 Step 7000       lr: 0.000783    loss: 1.878428
Each Step time is:0.004253
Epoch 2 Step 8000       lr: 0.000751    loss: 1.821544
Each Step time is:0.004219
Epoch 2 Step 9000       lr: 0.000721    loss: 1.781683
Each Step time is:0.004222
Epoch 3 Step 10000      lr: 0.000693    loss: 1.652992
Each Step time is:0.003286
Epoch 3 Step 11000      lr: 0.000665    loss: 1.618463
Each Step time is:0.004682
Epoch 3 Step 12000      lr: 0.000638    loss: 1.592724
Each Step time is:0.003945
Epoch 4 Step 13000      lr: 0.000613    loss: 1.517120
Each Step time is:0.004413
Epoch 4 Step 14000      lr: 0.000588    loss: 1.491452
Each Step time is:0.003047
Epoch 4 Step 15000      lr: 0.000565    loss: 1.470285
Each Step time is:0.004823
Epoch 5 Step 16000      lr: 0.000542    loss: 1.411028
Each Step time is:0.004945
Epoch 5 Step 17000      lr: 0.000520    loss: 1.393256
Each Step time is:0.004386
Epoch 5 Step 18000      lr: 0.000500    loss: 1.379420
Each Step time is:0.005300
Epoch 6 Step 19000      lr: 0.000480    loss: 1.339496
Each Step time is:0.005045
Epoch 6 Step 20000      lr: 0.000460    loss: 1.328781
Each Step time is:0.003675
Epoch 6 Step 21000      lr: 0.000442    loss: 1.315069
Each Step time is:0.004629
Epoch 7 Step 22000      lr: 0.000424    loss: 1.285930
Each Step time is:0.003749
Epoch 7 Step 23000      lr: 0.000407    loss: 1.278718
Each Step time is:0.004132
Epoch 7 Step 24000      lr: 0.000391    loss: 1.261955
Each Step time is:0.004080
Epoch 8 Step 25000      lr: 0.000375    loss: 1.241525
Each Step time is:0.004286
Epoch 8 Step 26000      lr: 0.000360    loss: 1.236466
Each Step time is:0.004063
Epoch 8 Step 27000      lr: 0.000346    loss: 1.222925
Each Step time is:0.005332
Epoch 9 Step 28000      lr: 0.000332    loss: 1.201749
Each Step time is:0.004180
Epoch 9 Step 29000      lr: 0.000319    loss: 1.201249
Each Step time is:0.004601
Epoch 9 Step 30000      lr: 0.000306    loss: 1.189128
Each Step time is:0.004267

NPU：
Epoch 0 Step 1  lr: 0.001000    loss: 3.911966
Each Step time is:4.246922
Epoch 0 Step 1000       lr: 0.001000    loss: 3.419568
Each Step time is:0.023352
Epoch 0 Step 2000       lr: 0.000960    loss: 3.079019
Each Step time is:0.030417
Epoch 0 Step 3000       lr: 0.000922    loss: 2.873609
Each Step time is:0.017977
Epoch 1 Step 4000       lr: 0.000885    loss: 2.234102
Each Step time is:0.020465
Epoch 1 Step 5000       lr: 0.000849    loss: 2.147734
Each Step time is:0.019830
Epoch 1 Step 6000       lr: 0.000815    loss: 2.085524
Each Step time is:0.019779
Epoch 2 Step 7000       lr: 0.000783    loss: 1.858841
Each Step time is:0.031110
Epoch 2 Step 8000       lr: 0.000751    loss: 1.812654
Each Step time is:0.021023
Epoch 2 Step 9000       lr: 0.000721    loss: 1.778383
Each Step time is:0.019872
Epoch 3 Step 10000      lr: 0.000693    loss: 1.648445
Each Step time is:0.020836
Epoch 3 Step 11000      lr: 0.000665    loss: 1.618356
Each Step time is:0.020097
Epoch 3 Step 12000      lr: 0.000638    loss: 1.596578
Each Step time is:0.022056
Epoch 4 Step 13000      lr: 0.000613    loss: 1.518200
Each Step time is:0.019850
Epoch 4 Step 14000      lr: 0.000588    loss: 1.496975
Each Step time is:0.019566
Epoch 4 Step 15000      lr: 0.000565    loss: 1.478350
Each Step time is:0.021537
Epoch 5 Step 16000      lr: 0.000542    loss: 1.421001
Each Step time is:0.020090
Epoch 5 Step 17000      lr: 0.000520    loss: 1.408438
Each Step time is:0.019964
Epoch 5 Step 18000      lr: 0.000500    loss: 1.393376
Each Step time is:0.022634
Epoch 6 Step 19000      lr: 0.000480    loss: 1.346111
Each Step time is:0.020254
Epoch 6 Step 20000      lr: 0.000460    loss: 1.339697
Each Step time is:0.025045
Epoch 6 Step 21000      lr: 0.000442    loss: 1.327170
Each Step time is:0.018506
Epoch 7 Step 22000      lr: 0.000424    loss: 1.302010
Each Step time is:0.018757
Epoch 7 Step 23000      lr: 0.000407    loss: 1.293618
Each Step time is:0.018338
Epoch 7 Step 24000      lr: 0.000391    loss: 1.278249
Each Step time is:0.018477
Epoch 8 Step 25000      lr: 0.000375    loss: 1.251435
Each Step time is:0.019069
Epoch 8 Step 26000      lr: 0.000360    loss: 1.248180
Each Step time is:0.019417
Epoch 8 Step 27000      lr: 0.000346    loss: 1.235918
Each Step time is:0.018264
Epoch 9 Step 28000      lr: 0.000332    loss: 1.218329
Each Step time is:0.019223
Epoch 9 Step 29000      lr: 0.000319    loss: 1.218511
Each Step time is:0.019388
Epoch 9 Step 30000      lr: 0.000306    loss: 1.206509
Each Step time is:0.018461
