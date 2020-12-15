rm -rf /var/log/npu/slog/*
curpath=`pwd`
echo "$curpath"
rm -rf $curpath/D0/checkpoint/*
rm -rf $curpath/D1/checkpoint/*
rm -rf $curpath/D2/checkpoint/*
rm -rf $curpath/D3/checkpoint/*
rm -rf $curpath/D4/checkpoint/*
rm -rf $curpath/D5/checkpoint/*
rm -rf $curpath/D6/checkpoint/*
rm -rf $curpath/D7/checkpoint/*

rm -rf $curpath/D0/kernel_meta/*
rm -rf $curpath/D1/kernel_meta/*
rm -rf $curpath/D2/kernel_meta/*
rm -rf $curpath/D3/kernel_meta/*
rm -rf $curpath/D4/kernel_meta/*
rm -rf $curpath/D5/kernel_meta/*
rm -rf $curpath/D6/kernel_meta/*
rm -rf $curpath/D7/kernel_meta/*

rm -rf $curpath/D0/logs/*
rm -rf $curpath/D1/logs/*
rm -rf $curpath/D2/logs/*
rm -rf $curpath/D3/logs/*
rm -rf $curpath/D4/logs/*
rm -rf $curpath/D5/logs/*
rm -rf $curpath/D6/logs/*
rm -rf $curpath/D7/logs/*

rm -rf $curpath/D0/samples/*
rm -rf $curpath/D1/samples/
rm -rf $curpath/D2/samples/
rm -rf $curpath/D3/samples/
rm -rf $curpath/D4/samples/
rm -rf $curpath/D5/samples/
rm -rf $curpath/D6/samples/
rm -rf $curpath/D7/samples/

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

cd $curpath/D0
bash $curpath/D0/scripts/npu_run_8p.sh&
cd $curpath/D1
bash $curpath/D1/scripts/npu_run_8p.sh&
cd $curpath/D2
bash $curpath/D2/scripts/npu_run_8p.sh&
cd $curpath/D3
bash $curpath/D3/scripts/npu_run_8p.sh&
cd $curpath/D4
bash $curpath/D4/scripts/npu_run_8p.sh&
cd $curpath/D5
bash $curpath/D5/scripts/npu_run_8p.sh&
cd $curpath/D6
bash $curpath/D6/scripts/npu_run_8p.sh&
cd $curpath/D7
bash $curpath/D7/scripts/npu_run_8p.sh
