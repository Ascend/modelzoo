python3 cifar10.py --dataset $CIFAR10 > train.log 2>&1
#����жϣ����ܼ�����ckpt/��־�ؼ��֡����ȼ��lossֵ/accucy�ؼ��֡����ܼ���ʱ���/ThroughOutput�ȹؼ���
key1="\[GEOP\]"  #���ܼ����


if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "Loss" "train.log"` -ne '0' ] && [ `grep -c "Extracting Finished" "train.log"` -ne '0' ] ;then   #���Ը�����Ҫ��������߼�
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi