# GraphSage: Representation Learning on Large Graphs
An NPU adapted implementation of GraphSAGE. 
The current code supports NPU (1980), GPU and CPU. 

This repository is based on the official implementation of GraphSAGE available [here](https://github.com/williamleif/GraphSAGE). 

## Paper
[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) 

William L. Hamilton, Rex Ying

NIPS, 2017.

## Usage

### 1. Setup data
Please download data and place the data in the `data` folder as the structure as follows.
The full Reddit and PPI datasets (described in the paper) are available on the [project website](http://snap.stanford.edu/graphsage/).
If your folder structure is different, you may need to change the corresponding paths in the code.

```
GraphSAGE
├── graphsage
├── eval_scripts
├── data
│   ├── toy-ppi
│   │   ├── toy-ppi-class_map.json
│   │   ├── toy-ppi-feats.npy
│   │   ├── toy-ppi-G.json
│   │   ├── toy-ppi-id_map.json
│   │   ├── toy-ppi-walks.txt
│   ├── ppi
│   │   ├── ppi-class_map.json
│   │   ├── ppi-feats.npy
│   │   ├── ppi-G.json
│   │   ├── ppi-id_map.json
│   │   ├── ppi-walks.txt
│   ├── reddit
│   │   ├── reddit-class_map.json
│   │   ├── reddit-feats.npy
│   │   ├── reddit-G.json
│   │   ├── reddit-id_map.json
│   │   ├── reddit-walks.txt

```

As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

To run the model on a new dataset, you need to make data files in the format described above.
To run random walks for the unsupervised model and to generate the <prefix>-walks.txt file)
you can use the `run_walks` function in `graphsage.utils`.

### 2. Run NPU (1980)

#### 2.1. Unsupervised learning
Train and eval a unsupervised model

1pi
```bash
$ bash scripts/run_npu_unsupervised.sh
```

#### 2.2. Supervised learning
Train and eval a unsupervised model

1p
```bash
$ bash scripts/run_npu_supervised.sh
```

8p
```bash
$ bash scripts/run_npu_supervised_8p.sh
```

### 3. Run GPU

#### 3.1. Unsupervised learning
Train and eval a unsupervised model
```bash
$ bash scripts/run_gpu_unsupervised.sh
```

#### 3.2. Supervised learning
Train and eval a unsupervised model
```bash
$ bash scripts/run_gpu_supervised.sh
```

### 4. Run CPU

#### 4.1. Unsupervised learning
Train a unsupervised model
```bash
$ bash scripts/run_cpu_unsupervised.sh
```

#### 4.2. Supervised learning
Train and eval a unsupervised model
```bash
$ bash scripts/run_cpu_supervised.sh
```

## Experimental results
| Model        | Aggregator | Device   | Dataset | Batch size | lr      | epoch | F1 (micro) | F1 (macro) | Speed (ms/iter) | 
|:------------:|:----------:|:--------:|:-------:|:----------:|:-------:|:-----:|:----------:|:----------:|:---------------:|
| Unsupervised | Meanpool   | NPU (1P) | PPI     | 512        | 0.00001 | 20    | 0.505      | 0.312      | 15.23           |
| Unsupervised | Maxpool    | NPU (1P) | PPI     | 512        | 0.00001 | 20    | 0.494      | 0.299      | 16.72           |
| Unsupervised | Mean       | NPU (1P) | PPI     | 512        | 0.00001 | 20    | 0.500      | 0.277      | 7.46            |
| Unsupervised | GCN        | NPU (1P) | PPI     | 512        | 0.00001 | 20    | 0.469      | 0.247      | 12.16           |
| Supervised   | Meanpool   | NPU (1P) | PPI     | 512        | 0.001   | 500   | 0.742      | 0.682      | 11.80           |
| Supervised   | Maxpool    | NPU (1P) | PPI     | 512        | 0.001   | 500   | 0.771      | 0.716      | 13.18           |
| Supervised   | Mean       | NPU (1P) | PPI     | 512        | 0.001   | 500   | 0.654      | 0.549      | 5.84            |
| Supervised   | GCN        | NPU (1P) | PPI     | 512        | 0.001   | 500   | 0.534      | 0.390      | 8.34            |
