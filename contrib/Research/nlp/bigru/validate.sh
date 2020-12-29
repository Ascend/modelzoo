## For NPU debug
# export PRINT_MODEL=1
# export DUMP_GRAPH=2
# export ENABLE_NETWORK_ANALYSIS_DEBUG=1
# export SLOG_PRINT_TO_STDOUT=1
# export RANK_ID=3
# export RANK_SIZE=1

## Clear something massive
rm -rf *.pbtxt
rm -rf ge*.txt
rm train/*.json
# rm -rf train/*

## Export NPU setting
echo ${PWD}
device_phy_id=4
# export JOB_ID=11086
export DEVICE_ID=$device_phy_id
export DEVICE_INDEX=$device_phy_id

## Validate
#script -f log.txt 
python3.7 thumt/bin/validate.py \
--input thumt/data/newstest2015.tc.32k.de thumt/data/newstest2015.tc.en \
--vocabulary vocab.32k.de.txt vocab.32k.en.txt \
--model rnnsearch \
--validation thumt/data/newstest2015.tc.32k.de \
--references thumt/data/newstest2015.tc.en \
--parameters=batch_size=128,device_list=[0],train_steps=800000 
#> nohup-bigru.log 2>&1 &
