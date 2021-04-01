#! -*- coding:utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import tensorflow as tf
import argparse

# from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator
# from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set


def main():

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    seed = 42

    args = parse_args()

    train_news_file = os.path.join(args.data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(args.data_path, 'train', r'behaviors.tsv')
    valid_news_file = os.path.join(args.data_path, 'valid', r'news.tsv')
    valid_behaviors_file = os.path.join(args.data_path, 'valid', r'behaviors.tsv')
    wordEmb_file = os.path.join(args.data_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(args.data_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(args.data_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(args.data_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(args.data_path, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(args.data_path, "utils", r'naml.yaml')

    # mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(args.MIND_type)
    #
    # if not os.path.exists(train_news_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'train'), mind_train_dataset)
    #
    # if not os.path.exists(valid_news_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'valid'), mind_dev_dataset)
    # if not os.path.exists(yaml_file):
    #     download_deeprec_resources(mind_url, os.path.join(args.data_path, 'utils'), mind_utils)

    hparams = prepare_hparams(yaml_file,
                              wordEmb_file=wordEmb_file,
                              wordDict_file=wordDict_file,
                              userDict_file=userDict_file,
                              vertDict_file=vertDict_file,
                              subvertDict_file=subvertDict_file,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              max_steps=args.max_steps,
                              model_path=args.model_path)
    print(hparams)

    iterator = MINDAllIterator

    model = NAMLModel(hparams, iterator, seed=seed)

    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

    #res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
    #print(res_syn)

    # sb.glue("res_syn", res_syn)

    os.makedirs(args.model_path, exist_ok=True)

    #model.model.save_weights(os.path.join(args.model_path, "naml_ckpt"))
    #model.model.save(os.path.join(args.model_path, "naml_new.h5"))

    sess = tf.compat.v1.keras.backend.get_session()
    sess.close()


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="""batchsize""")
    parser.add_argument('--epochs', type=int, default=1,
                        help="""epoch""")
    parser.add_argument('--model_path', type=str, default='./',
                        help="""pb path""")
    parser.add_argument('--data_path', type=str, default='./',
                        help = """the preprocess path of output data""")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="""the max train steps""")
    parser.add_argument('--MIND_type', default='small', choices=["demo", "small", "large"],
                        help = """the type of MIND data""")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

if __name__ == '__main__':
    main()
