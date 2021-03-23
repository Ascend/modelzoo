set -x
max_steps=20
save_steps=100
output="logs"

pip install -r requirements.txt
sudo apt-get install libgeos-dev -y
ln -s ${dbresnet_dataset} ./
rm -rf $output/*

start_time=`date +%s`
python3.7 train.py -m $max_steps \
    -s $save_steps \
    -p NPU \
    2>&1 | tee train.log

end_time=`date +%s`
key1="total loss"

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.