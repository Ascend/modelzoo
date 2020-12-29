
# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"

# DataDump
export FLAG_ENABLE_DUMP=False
export DUMP_PATH=/var/log/npu/dump
export DUMP_STEP="0|2"
export DUMP_MODE="all"
mkdir -p $DUMP_PATH
chown -R HwHiAiUser:HwHiAiUser $DUMP_PATH

cd transformer_p1
sh -x transformer_main_p1.sh >log 2>&1 &
cd ..

cd transformer_p2
sh -x transformer_main_p2.sh >log 2>&1 &
cd ..

cd transformer_p3
sh -x transformer_main_p3.sh >log 2>&1 &
cd ..

cd transformer_p4
sh -x transformer_main_p4.sh >log 2>&1 &
cd ..

cd transformer_p5
sh -x transformer_main_p5.sh >log 2>&1 &
cd ..

cd transformer_p6
sh -x transformer_main_p6.sh >log 2>&1 &
cd ..

cd transformer_p7
sh -x transformer_main_p7.sh >log 2>&1 &
cd ..

cd transformer_p8
sh -x transformer_main_p8.sh >log 2>&1 &
cd ..




