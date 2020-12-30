import os
import collections
import tensorflow as tf
import json
import squad_utils
import pickle

n_best_size=20
max_answer_length=384
start_n_top=5
end_n_top=5

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir", None,
    "The inputput directory where the model checkpoints will be written.")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "idx_file", None,
    "The output directory where the model checkpoints will be written.")



predict_file=os.path.join(FLAGS.input_dir, "dev-v2.0.json")
predict_feature_left_file=os.path.join(FLAGS.input_dir, "pred_left_file.pkl")
idx_file=FLAGS.idx_file
output_dir=FLAGS.output_dir


output_path=os.path.join(FLAGS.output_dir,"output")
files = os.listdir(output_path)
files = sorted(files)

def get_list(file_path, dtype="int"):
    with open(file_path) as f:
        for line in f.readlines():
            results = (line.strip().split()) # only one line in file
    if dtype == "int":
        result = [int(x) for x in results]
    elif dtype == "float":
        result= [float(x) for x in results]
    if len(results) == 1:
        return results[0]
    else:
        return result

def get_file(files, idx_file):
    unique_idxs = []
    with open(idx_file, "r") as f:
        for line in f.readlines():
            unique_idxs.append(int(line.strip()))
    with tf.gfile.Open(predict_file) as pre_file:
      prediction_json = json.load(pre_file)["data"]
    with tf.gfile.Open(predict_feature_left_file, "rb") as fin:
        eval_features = pickle.load(fin)
    eval_examples = squad_utils.read_squad_examples(
        input_file=predict_file, is_training=False)


    all_results = []
    for idex in range(int(len(files)/ 5)):
        start_top_log_probs_path=os.path.join(output_path, files[idex * 5])
        start_top_index_path=os.path.join(output_path, files[idex * 5 + 1])
        end_top_log_probs_path=os.path.join(output_path, files[idex * 5 + 2])
        end_top_index_path=os.path.join(output_path, files[idex * 5 + 3])
        cls_logits_path=os.path.join(output_path, files[idex * 5 + 4])

        start_top_index = get_list(start_top_index_path, "int")
        start_top_log_probs = get_list(start_top_log_probs_path,"float")
        end_top_index = get_list(end_top_index_path, "int")
        end_top_log_probs = get_list(end_top_log_probs_path,"float")
        cls_logits = get_list(cls_logits_path,"float")
        all_results.append(
            squad_utils.RawResultV2(
                unique_id = unique_idxs[idex],
                start_top_log_probs=start_top_log_probs,
                start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs,
                end_top_index=end_top_index,
                cls_logits=cls_logits))
    output_prediction_file = os.path.join(
        output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        output_dir, "null_odds.json")
    result_dict = {}
    cls_dict ={}
    squad_utils.accumulate_predictions_v2(
        result_dict, cls_dict, eval_examples, eval_features,
        all_results, n_best_size, max_answer_length,
        start_n_top, end_n_top)
    result = squad_utils.evaluate_v2(
        result_dict, cls_dict, prediction_json, eval_examples,
        eval_features, all_results, n_best_size,
        max_answer_length, output_prediction_file, output_nbest_file,
        output_null_log_odds_file)
    print("***** Final Eval results *****")
    for key in sorted(result.keys()):
        print("  {} = {}".format(key, str(result[key])))

get_file(files, idx_file)
