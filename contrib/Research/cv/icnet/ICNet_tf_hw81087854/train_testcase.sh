python3.7 train.py \
--train-beta-gamma \
--random-scale \
--random-mirror  \
--dataset ade20k \
--filter-scale 2 >train.log 2>&1

key1="[GEOP]"
key2="xxx"
key3="xxx"

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi
