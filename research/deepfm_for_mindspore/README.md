# DeepFM Model for CTR

The entire code structure is divided into three parts:
```
├── scripts
│   ├── download.sh    "Used for downloading the raw datasets"
│   ├── run_eval.sh    "The shell script for evaluation"
│   └── run_train.sh   "The shell script for training"
├── src
│   ├── callback.py    "Callback class: EvaluateCallback, LossCallback"
│   ├── config.py      "Configure args for data, model, train"
│   ├── datasets.py    "Dataset loader class"
│   ├── deepfm.py      "Model structure code, include: DenseLayer, DeepFMModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid, ModelBuilder"
│   ├── metric.py      "Metric class: AUCMetric"
│   └── utils.py       "Some tools function: init functions, write function"
├── test.py            "The main script for predict, load checkpoint file"
└── train.py           "The main script for train and eval, init by the config.py in deepfm_model_zoo/src/config.py"
```

Usage:
```
bash run_train.sh

bash run_eval.sh
```

**Notice:** You can configure the corresponding parameters in the `deepfm_model_zoo/src/config.py`.

