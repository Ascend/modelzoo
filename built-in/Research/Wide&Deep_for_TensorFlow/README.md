Wide&Deep

1P:
/scripts/run_npu_1p.sh

8P:
/scripts/run_npu_8p.sh


The set path needs to be changed based on the actual path.
configs/config.py: record_path

result file will be saved in result file.

1P result:
    #dropout=1 eval auc 0.80768 dropout=0.8  eval auc 0.808996(50epoch)

    model = WideDeep(graph, [input_dim, num_inputs, config.multi_hot_flags,
                       config.multi_hot_len],
                     [80, [1024, 512, 256, 128], 'relu'],
                     ['uniform', -0.01, 0.01, seeds[4:14], None],
                     [['adam', 3e-4, 9e-8, 0.8, 5], ['ftrl', 3.5e-2, 1, 3e-8, 1e-6]],
                     [1.0, 9e-6],
                     _input_d
                     )
8P result:
   #dropout=1 eval auc 0.8049
