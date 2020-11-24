"""The first stage of SMNAS."""

import copy
import hashlib
import logging
import os
import random

from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps
from vega.search_space.search_algs.pareto_front import ParetoFront
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm

from .mmdet_desc import MMDetDesc
from .mmdet_meta_cfgs import Dataset, Optimizer
from .mmdet_meta_cfgs import search_space as whole_space
from .utils import CfgGenerator
from .conf import SMNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SMNasS(SearchAlgorithm):
    """First Stage of SMNAS."""

    config = SMNasConfig()

    def __init__(self, search_space=None):
        super(SMNasS, self).__init__(search_space)
        self.components = ['detector', 'backbone', 'neck', 'rpn_head',
                           'roi_extractor', 'shared_head', 'bbox_head']
        # compose the search_space
        space = copy.deepcopy(whole_space)
        for component, choice in self.config.search_space.items():
            if component == 'dataset':
                setattr(Dataset, 'attr_space', choice)
            elif component == 'optimizer':
                setattr(Optimizer, 'attr_space', choice)
            else:
                module_names = list(space[component].keys())
                for module_name in module_names:
                    if module_name in choice.keys():
                        setattr(space[component][module_name],
                                'attr_space', choice[module_name])
                    else:
                        space[component].pop(module_name)
        self.search_space = space
        self.train_setting = self.config.train_setting
        self.data_setting = self.config.data_setting
        self.sampled_md5s = []

        # coarse code
        self.max_sample = self.config.max_sample
        self.min_sample = self.config.min_sample
        self.sample_count = 0
        logging.info("inited SMNasS")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def generate_sample(self, **kwargs):
        """Generate samples."""
        self.subspace = {
            component: list(
                self.search_space[component].keys()) for component in self.components}
        model = dict()
        model.update(optimizer=Optimizer(**self.train_setting))
        model.update(dataset=Dataset.sample(self.data_setting, fore_part=model))

        for component in self.components:
            if len(self.subspace[component]) == 0:
                model.update({component: None})
                continue
            module_name = random.choice(self.subspace[component])
            module = self.search_space[component][module_name].sample(
                fore_part=model)
            model.update({component: module})
            self._update_subspace((component, module_name))

        return model

    def _update_subspace(self, cue):
        """Update subspace."""
        cue_component, cue_module = cue
        update = self.search_space[cue_component][cue_module].module_space
        for component, module_space in update.items():
            if len(self.subspace[component]) == 0:
                continue
            if module_space is None:
                self.subspace[component] = []
            elif not isinstance(module_space, list):
                module_space = [module_space]
                cur_space = self.subspace[component]
                intersection = [i for i in cur_space if i in module_space]
                if set(module_space) > set(self.search_space[component].keys()):
                    raise ValueError(
                        "Get a wrong module space {} from {}".format({cue_component, module_space}, cue_module))
                self.subspace[component] = intersection

    def generate_mmdet_config(self, model):
        """Generate mmdet config."""
        # No pretrained as default
        pretrained = 'default'
        cfgs = dict(pretrained=pretrained, load_from=None)
        cfgs['data_root'] = model['dataset'].data_root
        # model setting
        for component in self.components:
            module = model[component]
            if component == 'detector':
                cfgs['train_cfg'] = module.train_cfg
                cfgs['test_cfg'] = module.test_cfg
                cfgs['num_stages'] = getattr(module, 'num_stages', None)
            if component == 'backbone' and pretrained == 'default':
                cfgs['pretrained'] = module.pretrained
            cfgs[component] = module.config if module is not None else None
        # dataset setting.
        cfgs.update(model['dataset'].config)
        # training setting
        cfgs.update(model['optimizer'].config)
        # generate cfg
        cfg_generator = CfgGenerator(**cfgs)
        config = cfg_generator.config
        return config

    def search(self):
        """Search a sample."""
        sample = None
        while sample is None:
            sample = self.generate_sample()

            sample_desc = self.generate_mmdet_config(sample)
            sample_cost = float(sample['backbone'].size_info['FLOPs']) / 1e9
            md5_value = self.md5(sample_desc)
            if md5_value in self.sampled_md5s:
                continue
            self.sampled_md5s.append(md5_value)
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample_desc):
                sample = None
        self.sample_count += 1
        model_desc = MMDetDesc({'desc': sample_desc, 'cost': sample_cost})
        return self.sample_count, model_desc

    def md5(self, model_desc):
        """Get md5 for model_desc."""
        m = hashlib.md5()
        m.update(model_desc.encode())
        return m.hexdigest()

    def update(self, worker_info):
        """Update generator."""
        if not worker_info:
            return
        worker_id = worker_info["worker_id"]
        performance_file = self.performance_path(worker_info)
        logging.info(
            "SMNasS.update(), performance file={}".format(performance_file))
        if os.path.isfile(performance_file):
            with open(performance_file, 'r') as f:
                performance = []
                for line in f.readlines():
                    performance.append(float(line.strip()))
                logging.info("update performance={}".format(performance))
                self.pareto_front.add_pareto_score(worker_id, performance)
        else:
            logging.info("SMNasS.update(), file is not exited, "
                         "performance file={}".format(performance_file))
        self.save_output()

    def performance_path(self, worker_info):
        """Get performance path."""
        step_name = worker_info["step_name"]
        worker_id = worker_info["worker_id"]
        worker_path = self.get_local_worker_path(step_name, worker_id)
        performance_dir = os.path.join(worker_path, 'performance')
        if not os.path.exists(performance_dir):
            FileOps.make_dir(performance_dir)
        return os.path.join(performance_dir, 'performance.txt')

    def save_output(self):
        """Save performance."""
        try:
            board_path = os.path.join(self.local_output_path, 'smnas_s_score_board.csv')
            self.pareto_front.sieve_board.to_csv(
                board_path,
                index=None, header=True)
        except Exception as e:
            logging.error("write score_board.csv error:{}".format(str(e)))
        try:
            pareto_dict = self.pareto_front.get_pareto_front()
            if len(pareto_dict) > 0:
                id = list(pareto_dict.keys())[0]
                net_desc = pareto_dict[id]
                config_path = os.path.join(
                    self.local_output_path, 'smnas_s_best_config.py')
                with open(config_path, 'w') as fp:
                    fp.write(net_desc)
        except Exception as e:
            logging.error("write best model error:{}".format(str(e)))
