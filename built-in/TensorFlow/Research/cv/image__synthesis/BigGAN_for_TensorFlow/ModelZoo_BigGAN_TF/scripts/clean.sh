 ps -ef | grep TdtMain | awk '{print $2}' | xargs kill -9
curpath=`pwd`
echo "$curpath"

rm -rf $curpath/D0/*.pbtxt
rm -rf $curpath/D1/*.pbtxt
rm -rf $curpath/D2/*.pbtxt
rm -rf $curpath/D3/*.pbtxt
rm -rf $curpath/D4/*.pbtxt
rm -rf $curpath/D5/*.pbtxt
rm -rf $curpath/D6/*.pbtxt
rm -rf $curpath/D7/*.pbtxt

rm $curpath/D0/*.txt
rm $curpath/D1/*.txt
rm $curpath/D2/*.txt
rm $curpath/D3/*.txt
rm $curpath/D4/*.txt
rm $curpath/D5/*.txt
rm $curpath/D6/*.txt
rm $curpath/D7/*.txt

rm -rf /var/log/npu/slog/*.log
rm ckpt* -rf
find ./ -name "*.pyc" | xargs rm -rf
find ./ -name __pycache__ | xargs rm -rf
rm /var/log/npu/dataset/* -rf
