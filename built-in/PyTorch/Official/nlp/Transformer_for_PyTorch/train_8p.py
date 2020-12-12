#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# -------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the BSD 3-Clause License (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#     https://spdx.org/licenses/BSD-3-Clause.html
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
import collections
import itertools
import os
import math
import torch
import torch.npu
import time
import ctypes

import sys
import threading

from copy import deepcopy
from utils import distributed_utils, options, utils
from utils.ddp_trainer import DDPTrainer
from utils.meters import StopwatchMeter, TimeMeter
from utils.sequence_generator import SequenceGenerator
import data
from data import tokenizer, dictionary, data_utils, load_dataset_splits
from models import build_model
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import sacrebleu
import dllogger as DLLogger
from utils.log_helper import AggregatorBackend, setup_logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + ']'


def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    print(args)
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '90000'
    mp.spawn(main_worker, nprocs=args.distributed_world_size, args=(args.distributed_world_size, args))



def main_worker(pid_idx, device_nums_per_node, args):
    setup_logger(args)

    args.distributed_rank = pid_idx
    args.device_id = args.distributed_rank
    dist.init_process_group(backend=args.dist_backend, world_size=args.distributed_world_size, rank=args.distributed_rank)
    loc = 'npu:{}'.format(args.device_id)
    torch.npu.set_device(loc)

    if args.max_tokens is None:
        args.max_tokens = 6000

    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    add_extra_items_to_checkpoint({'src_dict': src_dict, 'tgt_dict': tgt_dict})
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)
    model = build_model(args)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    trainer = DDPTrainer(args, model)
    print('| model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))

    if (args.online_eval or args.target_bleu) and not args.remove_bpe:
        args.remove_bpe = '@@ '
    # print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        max_positions_num=96,

    )
    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)

    # Sanity check
    if args.do_sanity_check:
        sanity_score = score(args, trainer, datasets['test'], src_dict, tgt_dict, 'test.raw.de')
        DLLogger.log(step='SANITY_CHECK', data={'sanity_check_score': sanity_score}, verbosity=1)

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    tgt_bleu = args.target_bleu or math.inf
    current_bleu = 0.0
    best_bleu = -1.0
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    run_summary = {'loss': float('inf'),
                   'val_loss': float('inf'),
                   'speed': 0,
                   'accuracy': 0}

    # max_update
    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update and current_bleu < tgt_bleu:
        DLLogger.log(step=trainer.get_num_updates(), data={'epoch': epoch_itr.epoch}, verbosity=0)
        # train for one epoch
        train(args, trainer, datasets, epoch_itr)
        with torch.autograd.profiler.profile(enabled=args.profile, use_cuda=True) as prof:
            train(args, trainer, datasets, epoch_itr)
        if args.profile:
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            if args.profiler_file:
                with open(os.path.join(args.save_dir, args.profiler_file),'w') as f:
                    f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            exit(0)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, datasets, valid_subsets)
            valid_bleu = score(args, trainer, datasets[valid_subsets[0]], src_dict, tgt_dict, 'valid.raw.de')
            DLLogger.log(step=trainer.get_num_updates(), data={'val_loss': valid_losses[0], 'val_bleu': valid_bleu},
                         verbosity=1)

        # Eval BLEU score
        if args.online_eval or (not tgt_bleu is math.inf):
            current_bleu = score(args, trainer, datasets[args.gen_subset], src_dict, tgt_dict, 'test.raw.de')
            DLLogger.log(step=trainer.get_num_updates(), data={'test_bleu': current_bleu}, verbosity=1)
            print(f'test_bleu on test subset: {current_bleu}')
            if current_bleu > best_bleu:
                best_bleu = current_bleu
                DLLogger.log(step='RUN', data={'BLEU': best_bleu}, verbosity=0)
                save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if valid_losses[0] < run_summary['val_loss']:
            run_summary['val_loss'] = valid_losses[0]
            if best_bleu < 0:
                run_summary['accuracy'] = valid_bleu
            else:
                run_summary['accuracy'] = best_bleu
        run_summary['loss'] = valid_losses[0]
        run_summary['speed'] = trainer.throughput_meter.u_avg

        # Only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # Save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    train_meter.stop()
    DLLogger.log(step=[], data=run_summary, verbosity=0)
    DLLogger.log(step='RUN', data={'walltime': train_meter.sum}, verbosity=0)
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, datasets, epoch_itr):
    """Train the model for one epoch."""

    itr = epoch_itr.next_epoch_itr()

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    if args.enable_parallel_backward_allred_opt and update_freq > 1:
        raise RuntimeError('--enable-parallel-backward-allred-opt is incompatible with --update-freq > 1')

    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    begin = time.time()

    # reset meters
    DLLogger.flush()
    trainer.get_throughput_meter().reset()
    end = time.time()

    for i, sample in enumerate(itr):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False, last_step=(i == len(itr) - 1))
            continue
        else:
            begin = time.time()
            if i != 10:
                trainer.train_step(sample, update_params=True, last_step=(i == len(itr) - 1))
            else:
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    trainer.train_step(sample, update_params=True, last_step=(i == len(itr) - 1))
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                prof.export_chrome_trace("transformer_npu_1013.txt")

            end = time.time()
            print("time:", end-begin)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_throughput_meter().reset()
            for backend in DLLogger.GLOBAL_LOGGER.backends:
                if isinstance(backend, AggregatorBackend):
                    backend._reset_perf_meter('tokens')
                    backend._reset_perf_meter('updates')
                    break

        # Mid epoch checkpoint
        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, datasets, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if (i + 1) % args.log_interval == 0:
            DLLogger.flush()

        if num_updates >= max_update:
            break

    print('Epoch time:', time.time() - begin)

    # Print epoch stats and reset training meters
    DLLogger.log(step=trainer.get_num_updates(), data={'speed': trainer.get_throughput_meter().avg}, verbosity=0)
    DLLogger.flush()


def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    # Reset value iterations counter
    trainer._num_val_iterations = 0

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=datasets[subset],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=args.max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            max_positions_num=1024,
        ).next_epoch_itr(shuffle=False)

        # reset validation loss meters
        DLLogger.flush()

        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        subset_loss = sum(subset_losses) / len(subset_losses)

        DLLogger.flush()

        valid_losses.append(subset_loss)
        print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses


def score(args, trainer, dataset, src_dict, tgt_dict, ref_file):
    begin = time.time()

    src_dict = deepcopy(src_dict)  # This is necessary, generation of translations
    tgt_dict = deepcopy(tgt_dict)  # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    # Initialize data iterator
    itr = data.EpochBatchIterator(
        dataset=dataset,
        max_tokens=None,
        # max_sentences=max(8, min(math.ceil(1024 / 8), 128)),
        max_sentences=max(8,min(math.ceil(1024/args.distributed_world_size),128)),
        max_positions=args.max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        max_positions_num=1024,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    translator = SequenceGenerator(
        args,
        [model],
        tgt_dict.get_metadata(),
        maxlen=args.max_target_positions - 1, #do not include EOS token
        beam_size=args.beam,
        stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
    )

    # Generate and compute BLEU
    dict = dictionary.Dictionary()
    num_sentences = 0
    predictions = []
    translations = translator.generate_batched_itr(
        itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
        npu=True, timer=gen_timer, prefix_size=args.prefix_size,
    )
    for sample_id, src_tokens, target_tokens, hypos in translations:
        # # Process input and grount truth
        # src_tokens = src_tokens.cpu().int()
        target_tokens = target_tokens.int().cpu()

        src_str = src_dict.string(src_tokens, args.remove_bpe)
        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

        # Process top predictions
        for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe
            )
            # Score only the top hypothesis
            if i == 0:
                hypo_str = tokenizer.Tokenizer.detokenize(hypo_str, 'de')
                predictions.append('{}\t{}'.format(sample_id, hypo_str))
        num_sentences += 1
    if args.distributed_world_size > 1:
        predictions = _all_gather_predictions(predictions)

    with open(os.path.join(args.data, ref_file), 'r') as reference:
        # print(os.path.join(args.data, ref_file))
        refs = [reference.readlines()]

    # reducing indexed predictions as strings is more memory efficient than reducing tuples
    predictions = [tuple(item.split('\t')) for item in predictions]
    predictions = [(int(item[0]), item[1]) for item in predictions]
    predictions.sort(key=lambda tup: tup[0])
    predictions = [hypo[1] + ('\n' if hypo[1][-1] != '\n' else '') for hypo in predictions]
    sacrebleu_score = sacrebleu.corpus_bleu(predictions, refs, lowercase=not args.test_cased_bleu).score
    if args.save_predictions:
        os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
        with open(os.path.join(args.save_dir, 'predictions', ref_file + '.pred.update_{}'.format(trainer._num_updates)),
                  'w') as f:
            f.write(''.join(predictions))

    DLLogger.log(step=trainer.get_num_updates(),
                 data={
                     'inference tokens/s': float(args.distributed_world_size) / gen_timer.avg
                 },
                 verbosity=0)
    DLLogger.flush()
    if gen_timer.sum != 0:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            len(predictions), gen_timer.n, gen_timer.sum, len(predictions) / gen_timer.sum,
                                                          float(args.distributed_world_size) / gen_timer.avg))

    print('| Eval completed in: {:.2f}s | {}CASED BLEU {:.2f}'.format(time.time() - begin,
                                                                      '' if args.test_cased_bleu else 'UN',
                                                                      sacrebleu_score))

    return sacrebleu_score


def _all_gather_predictions(predictions):
    ready = False
    all_ready = False
    reduced_predictions = []
    max_size = 65000
    while not all_ready:
        lst_len = len(predictions)
        size = 2000  # some extra space for python stuff
        n = 0
        while n < lst_len:
            str_len = len(predictions[n].encode('utf8')) + 8  # per string pickle overhead
            if size + str_len >= max_size:
                break
            size += str_len
            n += 1
        chunk = predictions[:n]
        predictions = predictions[n:]
        if not predictions:
            ready = True
        chunk = (ready, chunk)
        gathered = distributed_utils.all_gather_list(chunk, max_size=65000)
        reduced_predictions += [t[1] for t in gathered]
        all_ready = all([t[0] for t in gathered])

    reduced_predictions = [item for sublist in reduced_predictions for item in sublist]

    return reduced_predictions


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(os.path.join(args.save_dir, 'checkpoints'),
                                             pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def add_extra_items_to_checkpoint(dict):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(dict)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    print(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


if __name__ == '__main__':
    main()
