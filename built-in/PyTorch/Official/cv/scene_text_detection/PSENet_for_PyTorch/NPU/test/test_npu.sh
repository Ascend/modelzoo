su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\""
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 0"
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 4"

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1

python test_npu.py \
	--long_size 2240 \
	--npu 1\
	--resume "/home/z00524916/deploy/PSENet/8p/best/npu8pbatch64lr4_0.3401_0.9416_0.8407_0.9017_521.pth"\
	--data_dir '/home/z00524916/data/ICDAR/Challenge/' \
	--output_file 'npu8p64r4521'