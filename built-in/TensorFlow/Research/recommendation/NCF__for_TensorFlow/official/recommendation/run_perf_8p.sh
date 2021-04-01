#run ncf 8p
rm -rf /var/log/npu/slog/*
curpath=`pwd`
echo "$curpath"
cd $curpath/D0
$curpath/D0/train_ncf_geop_wlx.sh&
cd $curpath/D1
$curpath/D1/run_8p.sh&
cd $curpath/D2
$curpath/D2/run_8p.sh&
cd $curpath/D3
$curpath/D3/run_8p.sh&
cd $curpath/D4
$curpath/D4/run_8p.sh&
cd $curpath/D5
$curpath/D5/run_8p.sh&
cd $curpath/D6
$curpath/D6/run_8p.sh&
cd $curpath/D7
$curpath/D7/run_8p.sh&
