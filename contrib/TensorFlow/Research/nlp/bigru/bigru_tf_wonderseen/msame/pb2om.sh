export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}


#/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc -h
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./pb_model/bigru.pb \
	--framework=3 \
	--output=./om_model/bigru \
	--soc_version=Ascend910 \
	--input_shape="source:1,100;source_length:1" \
	--log=info \
	--out_nodes="translation:0"
        #--out_nodes="translation:0;rnnsearch/decoder/logit_sequence:0"

