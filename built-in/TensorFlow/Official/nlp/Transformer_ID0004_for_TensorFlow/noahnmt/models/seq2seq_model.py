# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import functools
import six
import math
import os

import tensorflow as tf

from noahnmt.decoders.beam_search_decoder import BeamSearchDecoder
from noahnmt.configurable import Configurable
from noahnmt.graph_module import GraphModule
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.utils import parallel_utils
from noahnmt.utils import graph_utils
from noahnmt.utils import vocab_utils
from noahnmt.utils import loss_utils
from noahnmt.utils import optimize_utils
from noahnmt.utils import fp16_utils
from noahnmt.utils import quantize_utils as quant_utils
from noahnmt.utils import learning_rate_utils as lr_utils
from noahnmt.layers.modality import SymbolModality
from noahnmt.layers import common_layers as common_utils
from noahnmt.inference import beam_search_utils as bs_utils


@registry.register_model("seq2seq_model")
class Seq2seqModel(Configurable, GraphModule):
    """base class for models.

    Args:
      params: A dictionary of hyperparameter values
      name: A name for this model to be used as a variable scope
    """

    def __init__(self, params, mode, name="seq2seq_model",
                 hparams=None,
                 data_parallelism=None):

        # if mode == tf.estimator.ModeKeys.TRAIN:
        params["mixed_precision"] = hparams.use_fp16
        if hparams.use_fp16:
            params["pad_to_eight"] = hparams.use_fp16

        Configurable.__init__(self, params, mode)
        GraphModule.__init__(self, name)

        self.data_parallelism = data_parallelism or parallel_utils.DataParallelism([""])
        self.hparams = hparams
        self.num_data_shards = self.data_parallelism.n
        self._built_modality = False

        # self.fp16_helper = None
        # if self.params["use_fp16"]:
        #   self.fp16_helper = fp16_utils.FP16Helper()

        self.source_vocab_info = vocab_utils.get_vocab_info(
            self.params["vocab_source"],
            self.params["special_vocabs"].copy(),
            pad_to_eight=self.params["pad_to_eight"])

        self.target_vocab_info = vocab_utils.get_vocab_info(
            self.params["vocab_target"],
            self.params["special_vocabs"].copy(),
            pad_to_eight=self.params["pad_to_eight"])

    @staticmethod
    def default_params():
        """Returns a dictionary of default parameters for this model."""
        return {
            # variable initializer
            "init_scale": 0.1,
            "initializer": "uniform",

            # vocabulary file
            "disable_vocab_table": False,
            "vocab_source": "",
            "vocab_target": "",
            "special_vocabs": {
                "unk": vocab_utils.UNK,
                "sos": vocab_utils.SOS,
                "eos": vocab_utils.EOS},
            "pad_to_eight": False,

            # embedding config
            "embedding.dim": 512,
            "embedding.share": False,
            "embedding.num_shards": 1,
            "src.embedding.multiply_mode": None,
            "src.embedding.initializer": None,
            "tgt.embedding.multiply_mode": None,
            "tgt.embedding.initializer": None,

            # encoder and decoder
            "encoder.class": "",
            "encoder.params": {},
            "decoder.class": "",
            "decoder.params": {},

            # beam search config
            "inference.use_sampling": False,
            "inference.beam_search.return_top_beam": True,
            "inference.beam_search.keep_finished": True,
            "inference.beam_search.stop_early": False,
            "inference.beam_search.beam_width": 4,
            "inference.beam_search.length_penalty_weight": 1.0,
            "inference.beam_search.coverage_penalty_weight": 0.0,

            # loss
            "word_level_loss": True,
            "label_smoothing_factor": 0.1,
            "weight_tying": False,
            "softmax_bias": False,

            # optimizer
            "optimizer.name": "sgd",
            "optimizer.params": {
                # Arbitrary parameters for the optimizer
                # for Momentum optimizer
                "momentum": 0.99,
                "use_nesterov": True,
                # for Adam optimizer, tf default values
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
                # for MultistepAdam
                "accumulate_steps": 1
            },

            # learning_rate
            "learning_rate.decay_steps": 10000,
            "learning_rate.decay_rate": 0.98,
            "learning_rate.start_decay_step": 0,
            "learning_rate.stop_decay_at": tf.int32.max,
            "learning_rate.min_value": 1e-12,
            "learning_rate.decay_staircase": True,
            "learning_rate.warmup_steps": 0,
            "learning_rate.constant": 1.0,
            "learning_rate.schedule": "",

            # gradients clip
            "max_grad_norm": None,

            # guided attention loss
            "guided_attention.weight": 0.,
            "guided_attention.loss_type": "ce",  # ce, mse or sqrt_mse
            "guided_attention.decay_steps": 10000,
            "guided_attention.decay_rate": 0.95,
            "guided_attention.start_decay_step": 0,  # default: no decay
            "guided_attention.stop_decay_at": tf.int32.max,

            # clip gemm: MatMul and BatchMatMul (optional)
            "clip_gemm.value": None,
            "clip_gemm.decay_steps": 10000,
            "clip_gemm.decay_rate": 0.95,
            "clip_gemm.start_decay_step": 0,  # default: no decay
            "clip_gemm.stop_decay_at": tf.int32.max,
            "clip_gemm.min_value": 1.0,
            "clip_gemm.staircase": True,
            "clip_gemm.batch_matmul": False,

            # quantization
            "quant_bits": None,  # 16 or 8
            "quant16_mul_bits": 9,  # 10 in Marian but overflow, so we use 9

            # float16 training
            "mixed_precision": False,
            "mixed_precision.params": {
                "init_loss_scale": 2.0 ** 10,
                "incr_every_n_steps": 2000,
                "decr_every_n_nan_or_inf": 2,
                "incr_ratio": 2,
                "decr_ratio": 0.5,
                "fix_loss_scale": False,
            },

            # mixture of softmax:
            "mos_n_experts": 0,  # recommend 15
        }

    def _create_vocab_tables(self):
        """ create vocab table and add to graph
        """

        def _create_table(vocab_info):
            vocab_table, reverse_vocab_table \
                = vocab_utils.create_vocab_tables(vocab_info.path)
            return vocab_table, reverse_vocab_table

        src_vocab_table, src_reverse_vocab_table \
            = _create_table(self.source_vocab_info)
        tgt_vocab_table, tgt_reverse_vocab_table \
            = _create_table(self.target_vocab_info)

        # Add vocab tables to graph colection so that we can access them in
        # other places, e.g., hooks.
        graph_utils.add_dict_to_collection({
            "source_vocab_to_id": src_vocab_table,
            "source_id_to_vocab": src_reverse_vocab_table,
            "target_vocab_to_id": tgt_vocab_table,
            "target_id_to_vocab": tgt_reverse_vocab_table,
        }, "vocab_tables")

    def create_modality(self):
        if self._built_modality:
            return

        if len(graph_utils.get_dict_from_collection("vocab_tables")) < 1 and not self.params["disable_vocab_table"]:
            tf.logging("create vocab table #######")
            assert False
            self._create_vocab_tables()

        self._built_modality = True
        partitioner = None
        if self.params["embedding.num_shards"] > 1:
            partitioner = tf.fixed_size_partitioner(
                self.params["embedding.num_shards"], axis=0)

        self.source_modality = SymbolModality(
            vocab_info=self.source_vocab_info,
            num_units=self.params["embedding.dim"],
            weight_tying=self.params["weight_tying"],
            partitioner=partitioner,
            embedding_multiply_mode=self.params["src.embedding.multiply_mode"],
            mos_n_experts=self.params["mos_n_experts"],
            embedding_initializer=self.params["src.embedding.initializer"],
            name="source_modality",
            softmax_bias=self.params["softmax_bias"])

        if self.params["embedding.share"]:
            self.target_modality = self.source_modality
        else:
            self.target_modality = SymbolModality(
                vocab_info=self.target_vocab_info,
                num_units=self.params["embedding.dim"],
                weight_tying=self.params["weight_tying"],
                partitioner=partitioner,
                embedding_multiply_mode=self.params["src.embedding.multiply_mode"],
                mos_n_experts=self.params["mos_n_experts"],
                embedding_initializer=self.params["tgt.embedding.initializer"],
                name="target_modality",
                softmax_bias=self.params["softmax_bias"])

    @property
    def _custom_getter(self):
        def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                            initializer=None, regularizer=None,
                                            trainable=True,
                                            *args, **kwargs):
            """Custom variable getter that forces trainable variables to be stored in
              float32 precision and then casts them to the training precision.
            """
            storage_dtype = tf.float32 if trainable else dtype
            variable = getter(name, shape, dtype=storage_dtype,
                              initializer=initializer, regularizer=regularizer,
                              trainable=trainable,
                              *args, **kwargs)
            if trainable and dtype != tf.float32:
                variable = tf.cast(variable, dtype)
            return variable

        return float32_variable_storage_getter if self.params["mixed_precision"] else None

    def optimize(self, loss, num_async_replicas=1):
        """Return a training op minimizing loss."""
        # learning rate
        lr = lr_utils.learning_rate_schedule(self.params, self.hparams, self.data_parallelism.n)
        if num_async_replicas > 1:
            tf.logging.info("Dividing learning rate by num_async_replicas: %d",
                            num_async_replicas)
            lr /= math.sqrt(float(num_async_replicas))

        # optimize
        train_op, hooks = optimize_utils.optimize(
            loss=loss,
            learning_rate=lr,
            params=self.params,
            hparams=self.hparams,
            mixed_precision=self.params["mixed_precision"],
            mixed_precision_params=self.params["mixed_precision.params"],
            is_finite=None)
            # is_finite=self.is_finite)

        # summary
        tf.summary.scalar("learning_rate", lr)

        # for logging learning rate
        hooks.append(
            tf.train.LoggingTensorHook(
                {"learning_rate": lr},
                every_n_iter=100)
        )

        return train_op, hooks

    def _preprocess(self, features):
        """Model-specific preprocessing for features and labels:

        - Creates vocabulary lookup tables for source and target vocab
        - Converts tokens into vocabulary ids
        """
        """def _pad_to_eight(tensor, axis, dtype=tf.string, pad_value="</s>", pad_more=0):
          shape = common_utils.shape_list(tensor)
          max_len = shape[axis]
          extra_len = tf.mod(8 - tf.mod(max_len, 8), 8)
          extra_len += pad_more
    
          paddings = [[0,0]] * len(shape)
          paddings[axis] = [0, extra_len]
          paddings = tf.convert_to_tensor(paddings)
    
          tensor = tf.pad(
              tensor, paddings,
              constant_values=pad_value)
    
          return tensor
    
    
        if self.params["pad_to_eight"]:
          features["source_tokens"] = _pad_to_eight(features["source_tokens"], axis=1)
          if "target_tokens" in features:
            features["target_tokens"] = _pad_to_eight(features["target_tokens"], axis=1, pad_more=1)"""

        # features["source_tokens"] = tf.Print(features["source_tokens"], [tf.shape(features["source_tokens"])], message="source shape: ", summarize=10)
        # if "target_tokens" in features:
        #   features["target_tokens"] = tf.Print(features["target_tokens"], [tf.shape(features["target_tokens"])], message="target shape: ", summarize=10)

        """# Look up the source ids in the vocabulary
        if "source_ids" not in features:
          src_vocab_table = graph_utils.get_dict_from_collection("vocab_tables")["source_vocab_to_id"]
          features["source_ids"] = src_vocab_table.lookup(features["source_tokens"])

        # Look up the target ids in the vocabulary
        # target_tokens are started with <s> and ended with </s>
        # <s> belongs to target input
        # </s> belongs to target output
        if "target_tokens" in features or "target_ids" in features:
            if "target_ids" not in features:
              tgt_vocab_table = graph_utils.get_dict_from_collection("vocab_tables")["target_vocab_to_id"]
              features["target_ids"] = tgt_vocab_table.lookup(features["target_tokens"])

            # if use selected target vocab
            # map target_ids to selected vocab
            if "selected_vocab_ids" in features:
                lookup_table = tf.contrib.lookup.index_table_from_tensor(
                    features["selected_vocab_ids"],
                    default_value=vocab_utils.UNK_ID,
                    dtype=constant_utils.DT_INT(),
                )
                features["target_ids"] = lookup_table.lookup(features["target_ids"])"""

        # reshape tensor to orig shape
        # bs = features["batch_size"]
        # maxlen = features["seq_length"]
        # del features["batch_size"]
        # del features["seq_length"]
        # for name in features:
        #     features[name] = tf.reshape(features[name], tf.stack([bs, maxlen]))
        
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            graph_utils.add_dict_to_collection({"source_ids": features["source_ids"]}, "SAVE_TENSOR")
            graph_utils.add_dict_to_collection({"target_ids": features["target_ids"]}, "SAVE_TENSOR")

        return features

    def _create_predictions(self, decoder_output, features):
        """Creates the dictionary of predictions that is returned by the model.
        """
        # batch_size = self.batch_size(features, labels)
        # predictions = {"batch_size": batch_size}
        predictions = {}

        # Add features to predictions
        predictions.update(features)

        # decoder outputs, batch-major
        # decoder_output.pop("logits", None)
        predictions.update(decoder_output)

        # If we predict the ids also map them back into the vocab and process them
        if self.mode != tf.estimator.ModeKeys.TRAIN:
            assert "predicted_ids" in predictions
        if "predicted_ids" in predictions and not self.params["disable_vocab_table"]:
            vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
            target_id_to_vocab = vocab_tables["target_id_to_vocab"]
            predicted_tokens = target_id_to_vocab.lookup(
                tf.to_int64(predictions["predicted_ids"]))
            # Raw predicted tokens
            predictions["predicted_tokens"] = predicted_tokens
        return predictions

    def _clip_and_quantize(self):
        if self.params["clip_gemm.value"]:
            # if self.mode == tf.estimator.ModeKeys.TRAIN:
            quant_utils.clip_matmul_inputs(self.params, self.mode)

        if self.mode != tf.estimator.ModeKeys.TRAIN and self.params["quant_bits"]:
            if self.params["quant_bits"] == 8:
                quant_utils.quantize8(
                    self.params["clip_gemm.value"],
                    batch_matmul=self.params["clip_gemm.batch_matmul"])
            elif self.params["quant_bits"] == 16:
                quant_utils.quantize16(
                    self.params["quant16_mul_bits"],
                    batch_matmul=self.params["clip_gemm.batch_matmul"])
            else:
                raise ValueError("Unknown quantize bits!")

    def _build(self, features, **kwargs):
        """Subclasses should implement this method. See the `model_fn` documentation
        in tf.contrib.learn.Estimator class for a more detailed explanation.
        """
        del kwargs

        # set custom getter
        # multiple getter are composed
        set_custom_getter_compose(self._custom_getter)

        # global variable initializer
        tf.get_variable_scope().set_initializer(
            optimize_utils.get_variable_initializer(
                initializer=self.params["initializer"],
                initializer_gain=self.params["init_scale"]))

        # source and target modality
        if not self._built_modality:
            self.create_modality()

        # preprocess features
        # including tokens-to-ids, selected_vocab etc.
        features = self._preprocess(features)

        # summarize features
        common_utils.summarize_features(features)

        # if multi-gpus, shard features
        # each gpu processes one shard
        # Models can override this function to custom gpu assignments
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            sharded_outputs, losses = self.model_fn_sharded(features)

            # logits are concatenated
            if isinstance(sharded_outputs, dict):
                concat_outputs = {}
                for k, v in six.iteritems(sharded_outputs):
                    concat_outputs[k] = tf.concat(v, 0)
                return concat_outputs, losses
            else:
                return tf.concat(sharded_outputs, 0), losses
        else:
            return self.model_fn(features)

    def model_fn_sharded(self, features):
        dp = self.data_parallelism
        # shard features
        sharded_features = common_utils.shard_features(features, dp)
        # exec on multi-gpus
        sharded_outputs, sharded_losses = dp(self.model_fn, sharded_features)

        if isinstance(sharded_outputs[0], dict):
            temp_dict = {k: [] for k, _ in six.iteritems(sharded_outputs[0])}
            for k, _ in six.iteritems(sharded_outputs[0]):
                for l in sharded_outputs:
                    temp_dict[k].append(l[k])
            sharded_outputs = temp_dict

        losses = None
        if sharded_losses and sharded_losses[0]:
            tf.logging.info(self.mode)
            losses = common_utils.average_sharded_losses(sharded_losses)

        return sharded_outputs, losses

    def model_fn(self, features):
        with tf.variable_scope(tf.get_variable_scope(), use_resource=False) as vs:
            # bottom
            transformed_features = self.bottom(features)
            # add modalities for convenience
            transformed_features["source_modality"] = self.source_modality
            transformed_features["target_modality"] = self.target_modality

            # body
            # with tf.variable_scope("body") as body_vs:
            tf.logging.info("Building model body")
            outputs = self.body(transformed_features)

            # top
            if "logits" not in outputs and self.mode != tf.estimator.ModeKeys.PREDICT:
                logits = self.top(outputs["output"], transformed_features)
                outputs["logits"] = logits

            # loss if not in PREDICT mode
            losses = {}
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                losses_ = self.loss(outputs, transformed_features)
                for key in losses_:
                  losses[key] = losses_[key][0] / tf.to_float(losses_[key][1])

            return outputs, losses

    @property
    def use_beam_search(self):
        """Returns true if the model should perform beam search.
        """
        return self.mode == tf.estimator.ModeKeys.PREDICT \
               and self.params["inference.beam_search.beam_width"] > 0 \
               and not self.params["inference.use_sampling"]

    def _get_beam_search_decoder(self, decoder):
        """Wraps a decoder into a Beam Search decoder.

        Args:
          decoder: The original decoder

        Returns:
          A BeamSearchDecoder with the same interfaces as the original decoder.
        """
        config = bs_utils.BeamSearchConfig(
            beam_width=self.params["inference.beam_search.beam_width"],
            vocab_size=self.source_vocab_info.vocab_size,
            eos_token=self.target_vocab_info.special_vocab.eos,
            length_penalty_weight=self.params["inference.beam_search.length_penalty_weight"],
            coverage_penalty_weight=self.params["inference.beam_search.coverage_penalty_weight"],
            stop_early=self.params["inference.beam_search.stop_early"],
            keep_finished=self.params["inference.beam_search.keep_finished"],
            return_top_beam=self.params["inference.beam_search.return_top_beam"])

        return BeamSearchDecoder(decoder_=decoder, config=config)

    def body(self, features):
        """Computes the targets' logits for one shard given transformed inputs.

        Most `ModelBase` subclasses will override this method.

        Args:
          features: dict of str to Tensor, where each Tensor has shape [batch_size,
            ..., hidden_size].

        Returns:
          outputs: dict of tensors outputed by decoders.
        """
        encoder_outputs = self.encode(features)
        encoder_outputs["memory_mask"] = features["source_mask"]
        return self.decode(features, encoder_outputs.copy())

    def encode(self, features):
        """ create encoder and encode
        """
        self.encoder = registry.encoder(
            self.params["encoder.class"])(
            self.params["encoder.params"], self.mode)

        return self.encoder(
            inputs=features["source_embeded"],
            mask=features["source_mask"],
            segment_ids=features.get("source_segids", None),
            position_ids=features.get("source_posids", None))

    def decode(self, features, encoder_outputs):
        """ create decoder and decode
        """
        self.decoder = registry.decoder(
            self.params["decoder.class"])(
            self.params["decoder.params"], self.mode)

        if self.use_beam_search:
            self.decoder = self._get_beam_search_decoder(self.decoder)

        return self.decoder(features, encoder_outputs,
                            use_sampling=self.params["inference.use_sampling"])

    def bottom(self, features):
        """Transform features to feed into body.
        """
        transformed_features = collections.OrderedDict()
        for key, value in six.iteritems(features):
            transformed_features[key] = value
        #transformed_features["source_ids"] = tf.Print(transformed_features["source_ids"], [transformed_features["source_ids"][:5,:5]], summarize=100, message="xxxxxxxxxxxxxxxxxxxxxxxx\n")
        #graph_utils.add_dict_to_collection({"source_ids": transformed_features["source_ids"]}, "SAVE_TENSOR")
        #graph_utils.add_dict_to_collection({"source_mask": transformed_features["source_mask"]}, "SAVE_TENSOR")

        transformed_features["source_embeded"] = self.source_modality.bottom(
            transformed_features["source_ids"])

        if "target_ids" in transformed_features:
            transformed_features["target_embeded"] = self.target_modality.target_bottom(
                transformed_features["target_ids"])
        else:
            self.target_modality._build_bottom()

        return transformed_features

    def top(self, output, features):
        """Returns `logits` given body output and features.
        """
        return self.target_modality.top(output)

    def loss(self, outputs, features):
        if self.params["mixed_precision"]:
            outputs["logits"] = tf.cast(outputs["logits"], tf.float32)
            outputs["attention_scores"] = tf.cast(outputs["attention_scores"], tf.float32)

        losses = {}
        graph_utils.add_dict_to_collection({"logits": outputs["logits"]}, "SAVE_TENSOR")
        # nce loss
        nce_loss, outputs["crossent"] = loss_utils.compute_nce_loss(
            logits=outputs["logits"],
            target=features["label_ids"],
            target_weight=features["label_weight"],
            label_smoothing=self.params["label_smoothing_factor"])

        if self.params["word_level_loss"]:
            nce_loss_len = tf.reduce_sum(features["label_weight"])
        else:
            nce_loss_len = tf.shape(features["label_ids"])[0]
        losses["nce"] = (nce_loss, nce_loss_len)

        # alignment loss
        if self.params["guided_attention.weight"] > 0:
            align_loss, _ = loss_utils.compute_alignment_loss(
                attention_scores=outputs["attention_scores"],
                features=features,
                params=self.params)
            losses["align"] = (align_loss, nce_loss_len)

        return losses

    @staticmethod
    def make_estimator_model_fn(model_name, params, hparams):
        model_cls = registry.model(model_name)

        def wrapping_model_fn(features, labels, mode, config=None):
            return model_cls.estimator_model_fn(
                features,
                labels,
                mode,
                config=config,
                params=params,
                hparams=hparams)

        return wrapping_model_fn

    @classmethod
    def estimator_model_fn(cls,
                           features,
                           labels,
                           mode,
                           config=None,
                           params=None,
                           hparams=None):
        """Model fn for Estimator.

        Args:
          features: dict<str name, Tensor feature>
          labels: Tensor
          mode: tf.estimator.ModeKeys
          config: RunConfig, possibly with data_parallelism attribute
          params: dict, may include batch_size, use_tpu
          decode_hparams: HParams, used when mode == PREDICT.

        Returns:
          TPUEstimatorSpec if use tpu else EstimatorSpec
        """
        mode_ = mode
        if hparams.eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
            mode_ = tf.estimator.ModeKeys.PREDICT

        # Instantiate model
        data_parallelism = config.data_parallelism
        model = cls(
            params,
            mode_,
            data_parallelism=data_parallelism,
            hparams=hparams)

        # # for overflow detection
        # from npu_bridge.estimator.npu.npu_loss_scale_optimizer import gen_npu_ops
        # from npu_bridge.hccl import hccl_ops
        # float_status = gen_npu_ops.npu_alloc_float_status()
        # # end

        # TRAIN and EVAL modes
        decoder_output, losses_dict = model(features)

        # predictions
        predictions = model._create_predictions(
            decoder_output=decoder_output,
            features=features)
        # add to graph for convinence
        graph_utils.add_dict_to_collection(predictions, "predictions")

        # possibly clip matmul and quantize
        model._clip_and_quantize()

        # log trainable variables
        # if mode == tf.estimator.ModeKeys.TRAIN:
        common_utils.print_trainable_variables(mode)

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return model.estimator_spec_predict(predictions)

        loss = 0.
        if losses_dict:
            # Accumulate losses
            loss = tf.add_n([losses_dict[key] for key in sorted(losses_dict.keys())])
            tf.summary.scalar("loss", loss)
        else:
            loss = tf.constant(loss, dtype=constant_utils.DT_FLOAT())
        
        # # for overflow detection
        # with tf.get_default_graph().control_dependencies([loss]):
        #     local_float_status = gen_npu_ops.npu_get_float_status(float_status)
        #     cleared_float_status = gen_npu_ops.npu_clear_float_status(local_float_status)
        
        # RANK_SIZE = int(os.environ.get('RANK_SIZE', '1').strip())
        # RANK_ID = int(os.environ.get('DEVICE_ID', '0').strip())

        # if RANK_SIZE > 1:
        #     with tf.get_default_graph().control_dependencies([local_float_status]):
        #         aggregated_float_status = hccl_ops.allreduce([float_status], "sum", fusion=0)
        #         is_overall_finite = tf.reduce_all(tf.equal(aggregated_float_status,
        #                                                     cleared_float_status))
        # else:
        #     is_overall_finite = tf.reduce_all(tf.equal(float_status,
        #                                                     cleared_float_status))
        # model.is_finite = is_overall_finite
        # # end


        # EVAL mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return model.estimator_spec_eval(predictions, loss)

        # TRAIN mode
        assert mode == tf.estimator.ModeKeys.TRAIN
        assert loss is not None
        num_async_replicas = 1 if not config else config.nmt_device_info["num_async_replicas"]
        return model.estimator_spec_train(
            predictions, loss, num_async_replicas=num_async_replicas)

    def estimator_spec_train(self, predictions, loss, num_async_replicas=1, is_finite=None):
        """Constructs `tf.estimator.EstimatorSpec` for TRAIN (training) mode."""
        train_op, hooks = self.optimize(loss, num_async_replicas=num_async_replicas)

        if self.params["mixed_precision"]:
            loss = tf.cond(tf.is_nan(loss), lambda: tf.zeros_like(loss), lambda: loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=hooks)

    def estimator_spec_eval(self, predictions, loss):
        """Constructs `tf.estimator.EstimatorSpec` for EVAL (evaluation) mode.
        """
        # eval_metrics_fns = metrics.create_evaluation_metrics(task_list, hparams)
        # eval_metrics = {}
        # for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
        #   eval_metrics[metric_name] = metric_fn(logits, features,
        #                                         features["targets"])

        if self.params["mixed_precision"]:
            loss = tf.cond(tf.is_nan(loss), lambda: tf.zeros_like(loss), lambda: loss)

        eval_metrics = {}
        if self.hparams.metrics is not None:
            for dict_ in self.hparams.metrics:
                # metric instance
                metric_cls = registry.class_ins(dict_["class"])
                params = dict_.get("params", {})
                metric = metric_cls(params)
                # metric ops
                eval_metrics[metric.name] = metric.create_metric_ops(predictions)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            predictions=predictions,
            eval_metric_ops=eval_metrics,
            loss=loss)

    def estimator_spec_predict(self, predictions, scaffold=None):
        """Constructs `tf.estimator.EstimatorSpec` for PREDICT (inference) mode.
        """
        if self.params["disable_vocab_table"]:
          export_out = {
              "outputs": predictions["predicted_ids"],
              "attention_scores": predictions["attention_scores"]}
        else:
          export_out = {
              "outputs": predictions["predicted_tokens"],
              "attention_scores": predictions["attention_scores"]}

        common_utils.remove_summaries()

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(export_out)
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            scaffold=scaffold,
            export_outputs=export_outputs)


def _compose_custom_getters(getter_a, getter_b):
    """Compose two custom getters.

    Example use:
    tf.get_variable_scope().set_custom_getter(
      compose_custom_getters(tf.get_variable_scope().custom_getter, new_getter))

    This composes getters in the same way as creating a new variable scope with
    the new_getter, but it does not actually create a new variable scope.

    Args:
      getter_a: a custom getter - generally from the existing variable scope.
      getter_b: a custom getter

    Returns:
      a custom getter
    """
    if not getter_a:
        return getter_b
    if not getter_b:
        return getter_a

    def getter_fn(getter, *args, **kwargs):
        return getter_b(functools.partial(getter_a, getter), *args, **kwargs)

    return getter_fn


def set_custom_getter_compose(custom_getter):
    """Set a custom getter in the current variable scope.

    Do not overwrite the existing custom getter - rather compose with it.

    Args:
      custom_getter: a custom getter.
    """
    tf.get_variable_scope().set_custom_getter(
        _compose_custom_getters(tf.get_variable_scope().custom_getter,
                                custom_getter))
