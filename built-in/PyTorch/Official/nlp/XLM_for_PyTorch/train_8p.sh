source ./test/env.sh
OUTPATH=data/processed/XLM_en_zh/50k
export NNPU=8
export WORLD_SIZE=$NNPU
export MASTER_ADDR=$(hostname -I |awk '{print $1}')
export MASTER_PORT=29500
for((RANK_ID=0;RANK_ID<NNPU;RANK_ID++))
do
    export RANK=$RANK_ID
    if [ $(uname -m) = "aarch64" ]
    then
        let a=0+RANK_ID*24
        let b=23+RANK_ID*24
        taskset -c $a-$b python3.7 train.py --exp_name xlm_en_zh \
            --dump_path ./dumped        \
            --data_path ./$OUTPATH      \
            --lgs 'en-zh'          \
            --clm_steps ''          \
            --mlm_steps 'en,zh'          \
            --emb_dim 1024               \
            --n_layers 12                \
            --n_heads 16                 \
            --dropout 0.1                \
            --attention_dropout 0.1      \
            --gelu_activation true       \
            --batch_size 16              \
            --bptt 256                   \
            --optimizer npu_fused_adam_v2,lr=0.00005     \
            --epoch_size 300000               \
            --max_epoch 100000                \
            --validation_metrics _valid_mlm_ppl          \
            --stopping_criterion _valid_mlm_ppl,15       \
            --fp16 true     \
            --amp 1 \
            --seed 1 \
            --local_rank $RANK_ID &
    else
        python3.7 train.py --exp_name xlm_en_zh \
            --dump_path ./dumped        \
            --data_path ./$OUTPATH      \
            --lgs 'en-zh'          \
            --clm_steps ''          \
            --mlm_steps 'en,zh'          \
            --emb_dim 1024               \
            --n_layers 12                \
            --n_heads 16                 \
            --dropout 0.1                \
            --attention_dropout 0.1      \
            --gelu_activation true       \
            --batch_size 16              \
            --bptt 256                   \
            --optimizer npu_fused_adam_v2,lr=0.00005     \
            --epoch_size 300000               \
            --max_epoch 100000                \
            --validation_metrics _valid_mlm_ppl          \
            --stopping_criterion _valid_mlm_ppl,15       \
            --fp16 true     \
            --amp 1 \
            --seed 1 \
            --local_rank $RANK_ID &
    fi
done
