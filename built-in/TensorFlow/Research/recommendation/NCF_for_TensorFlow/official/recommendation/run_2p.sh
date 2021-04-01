#run ncf 2p
rm -rf /var/log/npu/slog/*
rm -f slog/*
curpath=`pwd`
echo "$curpath"
cd $curpath/D0
$curpath/D0/run_2p.sh&
cd $curpath/D1
$curpath/D1/run_2p.sh&
