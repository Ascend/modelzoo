#run ncf 2p
rm -rf /var/log/npu/slog/*
rm -f slog/*
curpath=`pwd`
echo "$curpath"
cd $curpath/../official/recommendation/D0
$curpath/../official/recommendation/D0/run_2p.sh&
cd $curpath/../official/recommendation/D1
$curpath/../official/recommendation/D1/run_2p.sh&
