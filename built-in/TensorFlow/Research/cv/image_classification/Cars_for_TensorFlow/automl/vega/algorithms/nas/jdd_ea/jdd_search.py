# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""search algorithm for JDD_EA."""
import csv
import logging
from bisect import bisect_right
from random import random, sample
import numpy as np
from .conf import JDDSearchConfig
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from .jdd_ea_individual import JDDIndividual


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class JDDSearch(SearchAlgorithm):
    """Evolutionary search algorithm of the efficient JDD model."""

    config = JDDSearchConfig()

    def __init__(self, search_space=None, **kwargs):
        """Construct the JDD EA search class.

        :param search_space: Config of the search space
        :type search_space: dictionary
        """
        super(JDDSearch, self).__init__(search_space, **kwargs)
        self.individual_num = self.config.policy.num_individual
        self.generation_num = self.config.policy.num_generation
        self.elitism_num = self.config.policy.num_elitism
        self.mutation_rate = self.config.policy.mutation_rate
        self.min_active = self.config.range.min_active
        self.max_flops = self.config.range.max_flops
        self.min_flops = self.config.range.min_flops
        self.indiv_count = 0
        self.evolution_count = 0
        self.initialize_pop()
        self.elitism = [JDDIndividual(self.codec, self.config.range, search_space) for _ in range(self.elitism_num)]
        self.elit_fitness = [0] * self.elitism_num
        self.fitness_pop = [0] * self.individual_num
        self.fit_state = [0] * self.individual_num

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        :rtype: Bool
        """
        return self.indiv_count > self.generation_num * self.individual_num

    def update_fitness(self, evals):
        """Update the fitness of each individual.

        :param evals: the evalution
        :type evals: list
        """
        for i in range(self.individual_num):
            self.pop[i].update_fitness(evals[i])

    def update_elitism(self, evaluations):
        """Update the elitism and its fitness.

        :param evaluations: evaluations result
        :type evaluations: list
        """
        popu_all = [
            JDDIndividual(self.codec, self.config.range, self.search_space)
            for _ in range(self.elitism_num + self.individual_num)
        ]
        for i in range(self.elitism_num + self.individual_num):
            if i < self.elitism_num:
                popu_all[i].copy(self.elitism[i])
            else:
                popu_all[i].copy(self.pop[i - self.elitism_num])
        fitness_all = self.elit_fitness + evaluations
        sorted_ind = sorted(range(len(fitness_all)), key=lambda k: fitness_all[k])
        for i in range(self.elitism_num):
            self.elitism[i].copy(popu_all[sorted_ind[len(fitness_all) - 1 - i]])
            self.elit_fitness[i] = fitness_all[sorted_ind[len(fitness_all) - 1 - i]]
        logging.info('Generation: {}, updated elitism fitness: {}'.format(self.evolution_count, self.elit_fitness))

    def _log_data(self, net_info_type='active_only', pop=None, value=0):
        """Get the evolution and network information of children.

        :param net_info_type:  defaults to 'active_only'
        :type net_info_type: str
        :param pop: defaults to None
        :type pop: list
        :param value:  defaults to 0
        :type value: int
        :return: log_list
        :rtype: list
        """
        log_list = [value, pop.parameter, pop.flops]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    def save_results(self):
        """Save the results of evolution contains the information of pupulation and elitism."""
        arch_file = self.local_output_path + '/arch.txt'
        arch_child = self.local_output_path + '/arch_child.txt'
        sel_arch_file = self.local_output_path + '/selected_arch.npy'
        sel_arch = []
        with open(arch_file, 'a') as fw_a, open(arch_child, 'a') as fw_ac:
            writer_a = csv.writer(fw_a, lineterminator='\n')
            writer_ac = csv.writer(fw_ac, lineterminator='\n')
            writer_ac.writerow(['Population Iteration: ' + str(self.evolution_count + 1)])
            for c in range(self.individual_num):
                writer_ac.writerow(
                    self._log_data(net_info_type='active_only', pop=self.pop[c],
                                   value=self.pop[c].fitness))

            writer_a.writerow(['Population Iteration: ' + str(self.evolution_count + 1)])
            for c in range(self.elitism_num):
                writer_a.writerow(self._log_data(net_info_type='active_only',
                                                 pop=self.elitism[c],
                                                 value=self.elit_fitness[c]))
                sel_arch.append(self.elitism[c].gene)
        sel_arch = np.stack(sel_arch)
        np.save(sel_arch_file, sel_arch)
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_output_path, self.backup_base_path)

    def parent_select(self, parent_num=2, select_type='Tournament'):
        """Select parent from a population with Tournament or Roulette.

        :param parent_num: number of parents
        :type parent_num: int
        :param select_type: select_type, defaults to 'Tournament'
        :type select_type: str
        :return: the selected parent
        :rtype: class
        """
        popu_all = [
            JDDIndividual(self.codec, self.config.range, self.search_space)
            for _ in range(self.elitism_num + self.individual_num)
        ]
        parent = [
            JDDIndividual(self.codec, self.config.range, self.search_space)
            for _ in range(parent_num)
        ]
        fitness_all = self.elit_fitness
        for i in range(self.elitism_num + self.individual_num):
            if i < self.elitism_num:
                popu_all[i].copy(self.elitism[i])
            else:
                popu_all[i].copy(self.pop[i - self.elitism_num])
                fitness_all = fitness_all + [popu_all[i].fitness]
        fitness_all = np.asarray(fitness_all)
        if select_type == 'Tournament':
            for i in range(parent_num):
                tourn = sample(range(len(popu_all)), 2)
                if fitness_all[tourn[0]] >= fitness_all[tourn[1]]:
                    parent[i].copy(popu_all[tourn[0]])
                    fitness_all[tourn[0]] = 0
                else:
                    parent[i] = popu_all[tourn[1]]
                    fitness_all[tourn[1]] = 0
        elif select_type == 'Roulette':
            eval_submean = fitness_all - np.min(fitness_all)
            eval_norm = eval_submean / sum(eval_submean)
            eva_threshold = np.cumsum(eval_norm)
            for i in range(parent_num):
                ran = random()
                selec_id = bisect_right(eva_threshold, ran)
                parent[i].copy(popu_all[selec_id])
                eval_submean[selec_id] = 0
                eval_norm = eval_submean / sum(eval_submean)
                eva_threshold = np.cumsum(eval_norm)
        else:
            logging.info('Wrong selection type')
        return parent

    def initialize_pop(self):
        """Initialize the population of first generation."""
        self.pop = [JDDIndividual(self.codec, self.config.range, self.search_space) for _ in range(self.individual_num)]
        for i in range(self.individual_num):
            while self.pop[i].active_num < self.min_active:
                self.pop[i].mutation_using(self.mutation_rate)
            while self.pop[i].flops > self.max_flops or self.pop[i].flops < self.min_flops:
                self.pop[i].mutation_node(self.mutation_rate)

    def get_mutate_child(self, muta_num):
        """Generate the mutated children of the next offspring with mutation operation.

        :param muta_num: number of mutated children
        :type muta_num: int
        """
        for i in range(muta_num):
            if int(self.individual_num / 2) == len(self.elitism):
                self.pop[i].copy(self.elitism[i])
            else:
                self.pop[i].copy(sample(self.elitism, 1)[0])
            self.pop[i].mutation_using(self.mutation_rate)
            while self.pop[i].active_num < self.min_active:
                self.pop[i].mutation_using(self.mutation_rate)
            self.pop[i].mutation_node(self.mutation_rate)
            while self.pop[i].flops > self.max_flops or self.pop[i].flops < self.min_flops:
                self.pop[i].mutation_node(self.mutation_rate)

    def get_cross_child(self, muta_num):
        """Generate the children of the next offspring with crossover operation.

        :param muta_num: number of mutated children
        :type muta_num: int
        """
        for i in range(int(self.individual_num / 4)):
            pop_id = muta_num + i * 2
            father, mother = self.parent_select(2, 'Roulette')
            length = np.random.randint(4, int(father.gene.shape[0] / 2))
            location = np.random.randint(0, father.gene.shape[0] - length)
            gene_1 = father.gene.copy()
            gene_2 = mother.gene.copy()
            gene_1[location:(location + length), :] = gene_2[location:(location + length), :]
            gene_2[location:(location + length), :] = father.gene[location:(location + length), :]
            self.pop[pop_id].update_gene(gene_1)
            self.pop[pop_id + 1].update_gene(gene_2)
            while self.pop[pop_id].active_num < self.min_active:
                self.pop[pop_id].mutation_using(self.mutation_rate)
            self.pop[pop_id].mutation_resolutions()
            flops = self.pop[pop_id].flops
            while flops > self.max_flops or flops < self.min_flops:
                self.pop[pop_id].mutation_node(self.mutation_rate)
                flops = self.pop[pop_id].flops
            while self.pop[pop_id + 1].active_num < self.min_active:
                self.pop[pop_id + 1].mutation_using(self.mutation_rate)
            self.pop[pop_id + 1].mutation_resolutions()
            flops = self.pop[pop_id + 1].flops
            while flops > self.max_flops or flops < self.min_flops:
                self.pop[pop_id + 1].mutation_node(self.mutation_rate)
                flops = self.pop[pop_id + 1].flops

    def reproduction(self):
        """Generate the new offsprings."""
        muta_num = self.individual_num - (self.individual_num // 4) * 2
        self.get_mutate_child(muta_num)
        self.get_cross_child(muta_num)

    def update(self, record):
        """Update function.

        :param step_name: step name.
        """
        worker_id = record.get('worker_id')
        performance = record.get("rewards")
        self.fitness_pop[(worker_id - 1) % self.individual_num] = performance
        self.fit_state[(worker_id - 1) % self.individual_num] = 1

    def search(self):
        """Search one model.

        :return: current number of samples, and the model
        :rtype: int and class
        """
        if self.indiv_count > 0 and self.indiv_count % self.individual_num == 0:
            if np.sum(np.asarray(self.fit_state)) < self.individual_num:
                return None, None
            else:
                self.update_fitness(self.fitness_pop)
                self.update_elitism(self.fitness_pop)
                self.save_results()
                self.reproduction()
                self.evolution_count += 1
                self.fitness_pop = [0] * self.individual_num
                self.fit_state = [0] * self.individual_num
        current_indiv = self.pop[self.indiv_count % self.individual_num]
        indiv_cfg = self.codec.decode(current_indiv)
        self.indiv_count += 1
        logging.info('model parameters:{}, model flops:{}'.format(current_indiv.parameter, current_indiv.flops))
        logging.info('model arch:{}'.format(current_indiv.active_net_list()))
        return dict(worker_id=self.indiv_count, desc=indiv_cfg)
