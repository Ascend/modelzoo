set -x
max_steps=20

pip install -r requirements.txt
ln -s ${dbresnet_dataset} ./

start_time=`date +%s`

python3.7 train.py -TRAIN_MAX_STEPS $max_steps \
2>&1 | tee train_log.log

end_time=`date +%s`
key1=">>>>loss_val"

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.