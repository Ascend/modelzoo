# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DifferentialAlgorithm."""
import copy
import logging
import os
from collections import namedtuple
import numpy as np
import vega
from .conf import CARSConfig
from vega.core.common import Config
from vega.algorithms.nas.darts_cnn import DartsNetworkTemplateConfig
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.report.nsga_iii import SortAndSelectPopulation
from .nsga3 import CARS_NSGA
from .utils import eval_model_parameters

if vega.is_torch_backend():
    import torch
    from vega.core.metrics.pytorch import Metrics
elif vega.is_tf_backend():
    import tensorflow as tf
    from vega.core.metrics.tensorflow import Metrics

logger = logging.getLogger(__name__)
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class CARSAlgorithm(SearchAlgorithm):
    """Differential algorithm.

    :param search_space: Input search_space.
    :type search_space: SearchSpace
    """

    config = CARSConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init CARSAlgorithm."""
        super(CARSAlgorithm, self).__init__(search_space, **kwargs)
        self.network_momentum = self.config.policy.momentum
        self.network_weight_decay = self.config.policy.weight_decay
        self.parallel = self.config.policy.parallel
        self.sample_num = self.config.policy.sample_num
        self.sample_idx = 0
        self.completed = False
        self.trainer = None
        self.alg_policy = None

    def set_model(self, model):
        """Set model."""
        self.model = model
        self.steps = self.model.steps
        self.num_ops = self.model.num_ops
        self.len_alpha = self.model.len_alpha

    def search(self):
        """Search function."""
        logging.debug('====> {}.search()'.format(__name__))
        self.completed = True
        return self.sample_idx, self.search_space

    def update(self, record):
        """Update function.

        :param record: record from shared memory.
        """
        self.sample_idx += 1

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.

        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.25:
                alphas_c = self.mutation(alphas[np.random.randint(0, alphas.shape[0])])
            elif rand < 0.5:
                alphas_c = self.crossover(
                    alphas[np.random.randint(0, alphas.shape[0])],
                    alphas[np.random.randint(0, alphas.shape[0])])
            else:
                alphas_c = self.random_sample_path()
            if self.judge_repeat(alphas, alphas_c) == 0:
                offsprings.append(alphas_c)
        # offsprings = torch.cat([offspring.unsqueeze(0) for offspring in offsprings], dim=0)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = np.reshape(np.absolute(alphas - np.expand_dims(new_alphas, axis=0)), (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    @property
    def is_completed(self):
        """Check if the search is finished."""
        return (self.sample_idx >= self.sample_num) or self.completed

    def search_evol_arch(self, epoch, alg_policy, trainer, alphas):
        """Update architectures.

        :param epoch: The current epoch
        :type epoch: int
        :param valid_queue: valid dataloader
        :type valid_queue: dataloader
        :param model: The model to be trained
        :type model: nn.Module
        """
        self.trainer = trainer
        self.alg_policy = alg_policy
        model = self.trainer.model
        if epoch >= alg_policy.start_ga_epoch and \
                (epoch - alg_policy.start_ga_epoch) % alg_policy.ga_interval == 0:
            if vega.is_torch_backend():
                self.save_model_checkpoint(model, 'weights_{}.pt'.format(epoch))
            for generation in range(alg_policy.num_generation):
                fitness = np.zeros(
                    int(alg_policy.num_individual * (1 + alg_policy.expand)))
                model_sizes = np.zeros(
                    int(alg_policy.num_individual * (1 + alg_policy.expand)))
                genotypes = []
                # generate offsprings using mutation and cross-over
                offsprings = self.gen_offspring(alphas)
                alphas = np.concatenate((alphas, offsprings), axis=0)
                # calculate fitness (accuracy) and #parameters
                for i in range(int(alg_policy.num_individual * (1 + alg_policy.expand))):
                    pfm = self.search_infer_step(alphas[i])
                    fitness[i] = pfm.get(self.config.objective_keys)
                    model_sizes[i] = pfm.get('kparams')
                    genotypes.append(self.genotype_namedtuple(alphas[i]))
                    logging.info('Valid_acc for invidual {} %f, size %f'.format(i), fitness[i], model_sizes[i])
                # update population using pNSGA-III (CARS_NSGA)
                logging.info('############## Begin update alpha ############')
                if alg_policy.nsga_method == 'nsga3':
                    _, _, keep = SortAndSelectPopulation(
                        np.vstack((1 / fitness, model_sizes)), alg_policy.num_individual)
                elif alg_policy.nsga_method == 'cars_nsga':
                    nsga_objs = [model_sizes]
                    keep = CARS_NSGA(fitness, nsga_objs, alg_policy.num_individual)
                drop = list(set(list(
                    range(int(alg_policy.num_individual * (1 + alg_policy.expand))))) - set(keep.tolist()))
                logging.info('############## KEEP ############')
                fitness_keep = []
                size_keep = []
                genotype_keep = []
                for i in keep:
                    logging.info('KEEP Valid_acc for invidual {} %f, size %f, genotype %s'.format(i), fitness[i],
                                 model_sizes[i], genotypes[i])
                    fitness_keep.append(fitness[i])
                    size_keep.append(model_sizes[i])
                    genotype_keep.append(genotypes[i])
                logging.info('############## DROP ############')
                for i in drop:
                    logging.info('DROP Valid_acc for invidual {} %f, size %f, genotype %s'.format(i), fitness[i],
                                 model_sizes[i], genotypes[i])
                if alg_policy.select_method == 'uniform':
                    selected_genotypes, selected_acc, selected_model_sizes = self.select_uniform_pareto_front(
                        np.array(fitness_keep), np.array(size_keep), genotype_keep)
                else:  # default: first
                    selected_genotypes, selected_acc, selected_model_sizes = self.select_first_pareto_front(
                        np.array(fitness_keep), np.array(size_keep), genotype_keep)

                ga_epoch = int((epoch - alg_policy.start_ga_epoch) / alg_policy.ga_interval)
                self.save_genotypes(selected_genotypes, selected_acc, selected_model_sizes,
                                    'genotype_selected_{}.txt'.format(ga_epoch))
                self.save_genotypes(genotype_keep, np.array(fitness_keep), np.array(size_keep),
                                    'genotype_keep_{}.txt'.format(ga_epoch))
                alphas = alphas[keep].copy()
                self._broadcast(selected_genotypes, selected_acc)
                logging.info('############## End update alpha ############')
        return alphas

    def search_infer_step(self, alpha):
        """Infer in search stage.

        :param valid_queue: valid dataloader
        :type valid_queue: dataloader
        :param model: The model to be trained
        :type model: nn.Module
        :param alpha: encoding of a model
        :type alpha: array
        :return: Average top1 acc and loss
        :rtype: nn.Tensor
        """
        if vega.is_torch_backend():
            metrics = Metrics()
            alpha_tensor = torch.from_numpy(alpha).cuda()
            self.trainer.model.eval()
            with torch.no_grad():
                for step, (input, target) in enumerate(self.trainer.valid_loader):
                    input = input.cuda()
                    target = target.cuda(non_blocking=True)
                    logits = self.trainer.model(input, alpha_tensor)
                    metrics(logits, target)
        elif vega.is_tf_backend():
            # self.trainer.valid_alpha = tf.convert_to_tensor(alpha)
            metrics = self.trainer.valid_metrics
            setattr(self.trainer, 'valid_alpha', alpha)
            eval_results = self.trainer.estimator.evaluate(input_fn=self.trainer.valid_loader.input_fn,
                                                           steps=len(self.trainer.valid_loader))
            metrics.update(eval_results)
        performance = metrics.results
        objectives = metrics.objectives
        # support min
        for key, mode in objectives.items():
            if mode == 'MIN':
                performance[key] = -1 * performance[key]
        performance.update({'kparams': self.eval_model_sizes(alpha)})
        return performance

    def select_first_pareto_front(self, fitness, obj, genotypes):
        """Select models in the first pareto front.

        :param fitness: fitness, e.g. accuracy
        :type fitness: ndarray
        :param obj: objectives (model sizes, FLOPS, latency etc)
        :type obj: ndarray
        :param genotypes: genotypes for searched models
        :type genotypes: list
        :return: The selected samples
        :rtype: list
        """
        F, _, selected_idx = SortAndSelectPopulation(np.vstack(
            (1 / fitness, obj)), self.alg_policy.pareto_model_num)
        selected_genotypes = []
        selected_acc = []
        selected_model_sizes = []
        for idx in selected_idx:
            selected_genotypes.append(genotypes[idx])
            selected_acc.append(fitness[idx])
            selected_model_sizes.append(obj[idx])
        return selected_genotypes, selected_acc, selected_model_sizes

    def select_uniform_pareto_front(self, fitness, obj, genotypes):
        """Select models in the first pareto front.

        :param fitness: fitness, e.g. accuracy
        :type fitness: ndarray
        :param obj: objectives (model sizes, FLOPS, latency etc)
        :type obj: ndarray
        :param genotypes: genotypes for searched models
        :type genotypes: list
        :return: The selected samples
        :rtype: list
        """
        # preprocess
        max_acc = fitness.max()
        keep = (fitness > max_acc * 0.5)
        fitness = fitness[keep]
        obj = obj[keep]
        genotypes = [i for (i, v) in zip(genotypes, keep) if v]
        max_obj = obj.max()
        min_obj = obj.min()
        grid_num = self.alg_policy.pareto_model_num
        grid = np.linspace(min_obj, max_obj, num=grid_num + 1)
        selected_idx = []
        for idx in range(grid_num):
            keep = (obj <= grid[idx]) | (obj > grid[idx + 1])
            sub_fitness = np.array(fitness)
            sub_fitness[keep] = 0
            selected_idx.append(sub_fitness.argmax())
        selected_genotypes = []
        selected_acc = []
        selected_model_sizes = []
        for idx in selected_idx:
            selected_genotypes.append(genotypes[idx])
            selected_acc.append(fitness[idx])
            selected_model_sizes.append(obj[idx])
        return selected_genotypes, selected_acc, selected_model_sizes

    def crossover(self, alphas_a, alphas_b, ratio=0.5):
        """Crossover for two individuals.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param alphas_b: An individual
        :type alphas_b: nn.Tensor
        :param ratio: Probability to crossover
        :type ratio: float
        :return: The offspring after crossover
        :rtype: nn.Tensor
        """
        # alpha a
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_a[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_a[self.len_alpha:])
        new_alphas_normal_node0 = alphas_normal_node.copy()
        new_alphas_normal_ops0 = alphas_normal_ops.copy()
        new_alphas_reduce_node0 = alphas_reduce_node.copy()
        new_alphas_reduce_ops0 = alphas_reduce_ops.copy()
        # alpha b
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_b[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_b[self.len_alpha:])
        new_alphas_normal_node1 = alphas_normal_node.copy()
        new_alphas_normal_ops1 = alphas_normal_ops.copy()
        new_alphas_reduce_node1 = alphas_reduce_node.copy()
        new_alphas_reduce_ops1 = alphas_reduce_ops.copy()
        # crossover index
        for i in range(new_alphas_normal_node0.shape[0]):
            if np.random.rand() < 0.5:
                new_alphas_normal_node0[i] = new_alphas_normal_node1[i].copy()
                new_alphas_normal_ops0[i] = new_alphas_normal_ops1[i].copy()
                new_alphas_reduce_node0[i] = new_alphas_reduce_node1[i].copy()
                new_alphas_reduce_ops0[i] = new_alphas_reduce_ops1[i].copy()
        alphas_normal = self._node_ops_to_alpha(new_alphas_normal_node0, new_alphas_normal_ops0).copy()
        alphas_reduce = self._node_ops_to_alpha(new_alphas_reduce_node0, new_alphas_reduce_ops0).copy()
        # alphas = torch.cat([alphas_normal, alphas_reduce], dim=0)
        alphas = np.concatenate([alphas_normal, alphas_reduce], axis=0)
        return alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for An individual.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param ratio: Probability to mutation
        :type ratio: float
        :return: The offspring after mutation
        :rtype: nn.Tensor
        """
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_a[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_a[self.len_alpha:])
        new_alphas_normal_node0 = alphas_normal_node.copy()
        new_alphas_normal_ops0 = alphas_normal_ops.copy()
        new_alphas_reduce_node0 = alphas_reduce_node.copy()
        new_alphas_reduce_ops0 = alphas_reduce_ops.copy()
        # random alpha
        random_node, random_ops = self._random_one_individual()
        new_alphas_normal_node1 = random_node.copy()
        new_alphas_normal_ops1 = random_ops.copy()
        random_node, random_ops = self._random_one_individual()
        new_alphas_reduce_node1 = random_node.copy()
        new_alphas_reduce_ops1 = random_ops.copy()
        for i in range(new_alphas_normal_node0.shape[0]):
            if np.random.rand() < 0.5:
                new_alphas_normal_node0[i] = new_alphas_normal_node1[i].copy()
                new_alphas_normal_ops0[i] = new_alphas_normal_ops1[i].copy()
                new_alphas_reduce_node0[i] = new_alphas_reduce_node1[i].copy()
                new_alphas_reduce_ops0[i] = new_alphas_reduce_ops1[i].copy()
        alphas_normal = self._node_ops_to_alpha(new_alphas_normal_node0, new_alphas_normal_ops0).copy()
        alphas_reduce = self._node_ops_to_alpha(new_alphas_reduce_node0, new_alphas_reduce_ops0).copy()
        alphas = np.concatenate([alphas_normal, alphas_reduce], axis=0)
        return alphas

    def _random_one_individual(self):
        """Randomly initialize an individual."""
        random_individual_node = np.zeros((self.steps, 2), dtype=int)
        random_individual_ops = np.zeros((self.steps, 2), dtype=int)
        num_ops = self.num_ops
        for i in range(self.steps):
            n = i + 2
            random_idx = np.random.randint(0, n, 2)
            while (np.min(random_idx) == np.max(random_idx)):
                random_idx = np.random.randint(0, n, 2)
            for j in range(2):
                random_individual_node[i][j] = random_idx[j].item()
            random_idx = np.random.randint(0, num_ops, 2)
            for j in range(2):
                random_individual_ops[i][j] = random_idx[j].item()
        return random_individual_node, random_individual_ops

    def _node_ops_to_alpha(self, node, ops):
        """Calculate alpha according to the connection of nodes and operation.

        :param node: node
        :type node: Tensor
        :param ops: operation
        :type ops: Tensor
        """
        random_alpha = np.zeros((self.len_alpha, self.num_ops), dtype=np.float32)
        start = 0
        n = 2
        for i in range(self.steps):
            end = start + n
            for j in range(2):
                random_alpha[start + node[i][j]][ops[i][j]] = 1
            start = end
            n += 1
        return random_alpha

    def _alpha_to_node_ops(self, alpha):
        """Calculate the connection of nodes and operation specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        random_individual_node = np.zeros((self.steps, 2), dtype=int)
        random_individual_ops = np.zeros((self.steps, 2), dtype=int)
        start = 0
        for i in range(4):
            n = i + 2
            end = start + n
            idx = np.argmax(alpha[start:end, :], axis=1)
            cnt = 0
            if np.sum(alpha[start:end, :]) != 2:
                logger.error("Illegal alpha.")
            for j in range(n):
                if alpha[start + j, idx[j]] > 0:
                    random_individual_node[i][cnt] = j
                    random_individual_ops[i][cnt] = idx[j]
                    cnt += 1
            start = end
        return random_individual_node, random_individual_ops

    def random_sample_path(self):
        """Randomly sample a path.

        :param depth: the number of paths
        :type depth: int
        :param n_primitives: the number of operations
        :type n_primitives: int
        :return: The randomly sampled path
        :rtype: nn.Tensor
        """
        random_node, random_ops = self._random_one_individual()
        alphas_normal = self._node_ops_to_alpha(random_node, random_ops)
        random_node, random_ops = self._random_one_individual()
        alphas_reduce = self._node_ops_to_alpha(random_node, random_ops)
        alphas = np.concatenate([alphas_normal, alphas_reduce], axis=0)
        return alphas

    def eval_model_sizes(self, alpha):
        """Calculate model size for a genotype.

        :param genotype: genotype for searched model
        :type genotype: list
        :return: The number of parameters
        :rtype: Float
        """
        if vega.is_torch_backend():
            from vega.search_space.networks.pytorch import CARSDartsNetwork
        elif vega.is_tf_backend():
            from vega.search_space.networks.tensorflow import CARSDartsNetwork
        normal = alpha[:self.len_alpha]
        reduce = alpha[self.len_alpha:]
        child_desc = self.codec.calc_genotype([normal, reduce])
        child_cfg = copy.deepcopy(self.codec.darts_cfg.super_network)
        child_cfg.search = False
        child_cfg.normal.genotype = child_desc[0]
        child_cfg.reduce.genotype = child_desc[1]
        net = CARSDartsNetwork(child_cfg)
        model_size = eval_model_parameters(net)
        return model_size

    def genotype_namedtuple(self, alpha):
        """Obtain genotype.

        :param alpha: alpha for cell
        :type alpha: Tensor
        :return: genotype
        :rtype: Genotype
        """
        normal = alpha[:self.len_alpha]
        reduce = alpha[self.len_alpha:]
        child_desc = self.codec.calc_genotype([normal, reduce])
        _multiplier = 4
        concat = range(2 + self.steps - _multiplier, self.steps + 2)
        genotype = Genotype(
            normal=child_desc[0], normal_concat=concat,
            reduce=child_desc[1], reduce_concat=concat
        )
        return genotype

    def _broadcast(self, genotype, performance):
        """Broadcast performance."""
        self.trainer.performance = performance
        self.trainer.config.codec = self.genotypes_to_json(genotype)

    def save_model_checkpoint(self, model, model_name):
        """Save checkpoint for a model.

        :param model: A model
        :type model: nn.Module
        :param model_name: Path to save
        :type model_name: string
        """
        worker_path = self.trainer.get_local_worker_path()
        save_path = os.path.join(worker_path, model_name)
        _path, _ = os.path.split(save_path)
        if not os.path.isdir(_path):
            os.makedirs(_path)
        torch.save(model, save_path)
        logging.info("checkpoint saved to %s", save_path)

    def save_genotypes(self, genotypes, acc, obj, save_name):
        """Save genotypes.

        :param genotypes: Genotype for models
        :type genotypes: namedtuple Genotype
        :param acc: accuracy
        :type acc: ndarray
        :param obj: objectives, etc. FLOPs or number of parameters
        :type obj: ndarray
        :param save_name: Path to save
        :type save_name: string
        """
        worker_path = self.trainer.get_local_worker_path()
        save_path = os.path.join(worker_path, save_name)
        _path, _ = os.path.split(save_path)
        if not os.path.isdir(_path):
            os.makedirs(_path)
        with open(save_path, "w") as f:
            for idx in range(len(genotypes)):
                f.write('{}\t{}\t{}\n'.format(
                    acc[idx], obj[idx], genotypes[idx]))
        logging.info("genotypes saved to %s", save_path)

    def genotypes_to_json(self, genotypes):
        """Transfer genotypes to json.

        :param genotypes: Genotype for models
        :type genotypes: namedtuple Genotype
        """
        desc_list = []
        if self.trainer.config.darts_template_file == "{default_darts_cifar10_template}":
            template = DartsNetworkTemplateConfig.cifar10
        elif self.trainer.config.darts_template_file == "{default_darts_imagenet_template}":
            template = DartsNetworkTemplateConfig.imagenet

        for idx in range(len(genotypes)):
            template_cfg = Config(template)
            template_cfg.super_network.normal.genotype = genotypes[idx].normal
            template_cfg.super_network.reduce.genotype = genotypes[idx].reduce
            desc_list.append(template_cfg)
        return desc_list
