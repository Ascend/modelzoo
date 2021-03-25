## Here is the script for BiGRU translating sentences in the MSAME mode

source_length_path=input/source_length/
source_path=input/source/
ulimit -c 0
#./../tools/msame/out/msame \
./msame \
	--model ./om_model/bigru.om \
	--input ${source_path},${source_length_path} \
	--output ./output_offline \
	--outfmt TXT
