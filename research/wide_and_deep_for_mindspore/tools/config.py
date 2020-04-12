import argparse


def argparse_init():
    parser = argparse.ArgumentParser(description='NCF')

    parser.add_argument("--data_path", type=str, default="../test_raw_data/")  # The location of the input data.
    parser.add_argument("--epochs", type=int, default=15)  # The number of epochs used to train.
    parser.add_argument("--batch_size", type=int, default=10000)  # Batch size for training and evaluation
    parser.add_argument("--eval_batch_size", type=int, default=10000)  # The batch size used for evaluation.
    parser.add_argument("--field_size", type=int, default=39)  # field size for training and evaluation
    parser.add_argument("--vocab_size", type=int, default=184965)  # vocab size for training and evaluation
    parser.add_argument("--emb_dim", type=int, default=80)  # emb dim for training and evaluation
    parser.add_argument("--deep_layers_dim", type=int, nargs='+', default=[1024, 512, 256, 128])  # The sizes of hidden layers for MLP
    parser.add_argument("--deep_layers_act", type=str, default='relu')  # The act of hidden layers for MLP
    parser.add_argument("--keep_prob", type=float, default=1.0)  # The Embedding size of MF model.

    parser.add_argument("--output_path", type=str, default="../output/")  # The location of the output file.
    parser.add_argument("--ckpt_path", type=str, default="../checkpoints/")  # The location of the checkpoints file.
    parser.add_argument("--eval_file_name", type=str, default="eval.log")  # Eval output file.
    parser.add_argument("--loss_file_name", type=str, default="loss.log")  # Loss output file.
    return parser


class Config_WideDeep():
    def __init__(self):
        self.data_path = './test_raw_data/'
        self.epochs = 15
        self.batch_size = 10000
        self.eval_batch_size = 10000
        self.field_size = 39
        self.vocab_size = 184965
        self.emb_dim = 80
        self.deep_layers_dim = [1024, 512, 256, 128]
        self.deep_layers_act = 'relu'
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.keep_prob = 1.0
        self.l2_coef = 8e-5

        self.output_path = "../output/"
        self.eval_file_name = "eval.log"
        self.loss_file_name = "loss.log"
        self.ckpt_path = "../checkpoints/"

    def argparse_init(self):
        parser = argparse_init()
        args, _ = parser.parse_known_args()
        self.data_path = args.data_path
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.field_size = args.field_size
        self.vocab_size = args.vocab_size
        self.emb_dim = args.emb_dim
        self.deep_layers_dim = args.deep_layers_dim
        self.deep_layers_act = args.deep_layers_act
        self.keep_prob = args.keep_prob
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.l2_coef = 8e-5

        self.output_path = args.output_path
        self.eval_file_name = args.eval_file_name
        self.loss_file_name = args.loss_file_name
        self.ckpt_path = args.ckpt_path


