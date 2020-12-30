cd ..

## Clear something massive
rm train/*.json

python3.7 thumt/bin/ckpt2pb.py \
--input thumt/data/newstest2015.tc.32k.de thumt/data/newstest2015.tc.en \
--vocabulary vocab.32k.de.txt vocab.32k.en.txt \
--model rnnsearch \
--validation thumt/data/newstest2015.tc.32k.de \
--references thumt/data/newstest2015.tc.en \
--parameters=batch_size=128,device_list=[0],train_steps=800000 

cd msame
