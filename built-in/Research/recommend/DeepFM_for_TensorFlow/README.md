DeepFM

1P:
/scripts/run_npu_1p.sh

8P:
/scripts/run_npu_8p.sh


The set path needs to be changed based on the actual path.
configs/config.py: record_path

The result file will be saved in result file.

1P result:
    # dropout=0.8  eval auc 0.810(10epoch)

    model = FMNN_v2([input_dim, num_inputs, config.multi_hot_flags,
                   config.multi_hot_len],
                  [80, [1024, 512, 256, 128], 'relu'],
                  ['uniform', -0.01, 0.01, seeds[4:14], None],
                  ['adam', 5e-4, 5e-8, 0.95, 625],
                  [0.8, 8e-5],
                  _input_d
                  )
8P result:
   #dropout=0.8  eval auc 0.80788(15epoch)