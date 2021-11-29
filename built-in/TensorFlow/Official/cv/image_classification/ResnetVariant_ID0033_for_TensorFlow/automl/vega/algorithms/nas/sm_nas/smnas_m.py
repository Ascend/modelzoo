"""The second stage of SMNAS."""

import copy
import logging
import os
import random
import mmcv
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
class SMNasM(SearchAlgorithm):
    """Second Stage of SMNAS."""

    config = SMNasConfig()

    def __init__(self, search_space=None):
        super(SMNasM, self).__init__(search_space)
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
        self.sample_setting = self.config.sample_setting
        self.data_setting = self.config.data_setting

        # load the base model
        self.base_model = self.load_base_model()
        self.discard_archs = []
        self.sampled_archs = []

        # ea or random
        self.num_mutate = self.config.num_mutate
        self.random_ratio = self.config.random_ratio
        self.max_sample = self.config.max_sample
        self.min_sample = self.config.min_sample
        self.sample_base = self.config.sample_base
        if self.sample_base:
            self.max_sample += 1
            self.min_sample += 1
        self.sample_count = 0
        logging.info("inited SMNasM")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)

    def load_base_model(self):
        """Load base model."""
        base_model_path = self.get_base_model_path()
        base_model_info = mmcv.Config.fromfile(base_model_path)
        base_model_dict = {}
        base_model_dict['optimizer'] = {
            'optimizer': base_model_info.optimizer,
            'optimizer_config': base_model_info.optimizer_config,
            'lr_config': base_model_info.lr_config}
        img_scale_train = None
        for item in base_model_info.train_pipeline:
            if 'img_scale' in item:
                img_scale_train = item['img_scale']
                break
        img_scale_test = None
        for item in base_model_info.test_pipeline:
            if 'img_scale' in item:
                img_scale_test = item['img_scale']
                break
        base_model_dict['dataset'] = {
            'type': base_model_info.dataset_type,
            'img_scale': {
                'train': img_scale_train,
                'val': img_scale_test,
                'test': img_scale_test
            }
        }
        detection_type = base_model_info.model['type']
        rpn_head_type = base_model_info.model['rpn_head']['type']
        if detection_type == 'CascadeRCNN' and rpn_head_type == 'GARPNHead':
            detection_type = 'GACascadeRCNN'
        base_model_dict['detector'] = {
            'type': detection_type,
        }
        base_model_dict['roi_extractor'] = base_model_info.model['bbox_roi_extractor']['roi_layer']
        for component in base_model_info.model.keys():
            if component not in self.components:
                continue
            base_model_dict[component] = base_model_info.model[component]

        return mmcv.Config(base_model_dict)

    def get_base_model_path(self):
        """Get the path of base model."""
        return FileOps.join_path(self.local_output_path, 'smnas_s_best_config.py')

    def generate_sample(self, model_info, method='random',
                        base_mode=False, **kwargs):
        """Generate a sample."""
        all_components = ['optimizer', 'dataset'] + self.components
        model = dict()
        for component in all_components:
            attr = model_info.get(component, None)
            if attr is None:
                continue
            if isinstance(attr, list):
                attr = attr[0]
            attr_type = attr.get('type')
            if component == 'dataset':
                attr.update(self.data_setting)
                module = Dataset.set_from_model_info(attr, fore_part=model)
            elif component == 'optimizer':
                attr.update(self.train_setting)
                module = Optimizer.set_from_model_info(attr, fore_part=model)
            elif component == 'backbone':
                backbone_space = self.search_space['backbone']
                backbone_type = copy.copy(model_info.backbone.get('type'))
                base_depth = attr.get('depth')
                if base_mode:
                    module = backbone_space[backbone_type].set_from_model_info(
                        attr, fore_part=model)
                else:
                    if 'Variant' not in backbone_type:
                        backbone_type += '_Variant'
                    base_arch = kwargs['base_arch'] if 'base_arch' in kwargs else None
                    sample = backbone_space[backbone_type].sample(
                        method=method,
                        base_depth=base_depth,
                        base_arch=base_arch,
                        fore_part=model,
                        sampled_archs=self.discard_archs + self.sampled_archs,
                        **self.sample_setting)
                    module, discard = sample['arch'], sample['discard']
                    self.discard_archs.extend(discard.get('arch'))
                    if module is None:
                        logging.info('Sample finished(failed)!')
                        return
            else:
                if attr_type == 'FPN_':
                    attr_type = 'FPN'
                    attr.type = 'FPN'
                module = self.search_space[component][attr_type].set_from_model_info(
                    attr, fore_part=model)
            model.update({component: module})
        for rest_component in (set(self.components) - set(model.keys())):
            model.update({rest_component: None})

        return model

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

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search a sample."""
        sample = None
        pareto_arch_cfg = None
        while sample is None:
            pareto_dict = self.pareto_front.get_pareto_front()
            pareto_list = list(pareto_dict.values())
            if self.sample_base:
                sample = self.generate_sample(self.base_model, base_mode=True)
                self.sample_base = False
            if self.pareto_front.size < self.min_sample or random.random() < self.random_ratio or len(
                    pareto_list) == 0:
                sample = self.generate_sample(self.base_model, method='random')
            else:
                pareto_arch_cfg = pareto_list[0]
            if sample is None and pareto_arch_cfg is not None:
                with open('tmp.py', 'w') as f:
                    f.write(str(pareto_arch_cfg))
                pareto_cfg = mmcv.Config.fromfile('tmp.py')
                os.remove('tmp.py')
                pareto_model = pareto_cfg.model.backbone
                if 'ResNet' in pareto_model.type:
                    if 'arch' in pareto_model:
                        pareto_arch = 'r{}_{}_{}'.format(pareto_model.base_depth,
                                                         pareto_model.base_channel, pareto_model.arch)
                    else:
                        pareto_arch = 'ResNet{}'.format(pareto_model.depth)
                elif 'ResNeXt' in pareto_model.type:
                    if 'arch' in pareto_model:
                        pareto_arch = 'x{}({}x{}d)_{}_{}'.format(
                            pareto_model.base_depth,
                            pareto_model.groups, pareto_model.base_width,
                            pareto_model.base_channel, pareto_model.arch)
                    else:
                        pareto_arch = 'ResNeXt{}'.format(pareto_model.depth)
                else:
                    print('Not supported currently')
                sample = self.generate_sample(
                    self.base_model, method='EA', base_arch=pareto_arch)

            sample_desc = self.generate_mmdet_config(sample)
            sample_cost = float(sample['backbone'].size_info['FLOPs']) / 1e9
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample_desc):
                sample = None
        self.sample_count += 1

        model_desc = MMDetDesc({'desc': sample_desc, 'cost': sample_cost})
        return self.sample_count, model_desc

    def update(self, worker_info):
        """Update sampler."""
        worker_id = worker_info["worker_id"]
        performance_file = self.performance_path(worker_info)
        logging.info(
            "SmnasEA.update(), performance file={}".format(performance_file))
        if os.path.isfile(performance_file):
            with open(performance_file, 'r') as f:
                performance = []
                for line in f.readlines():
                    performance.append(float(line.strip()))
                logging.info("update performance={}".format(performance))
                self.pareto_front.add_pareto_score(worker_id, performance)
        else:
            logging.info("SmnasEA.update(), file is not exited, "
                         "performance file={}".format(performance_file))
        self.save_output(self.local_output_path)

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
        """Save results."""
        try:
            board_path = os.path.join(self.local_output_path, 'smnas_m_score_board.csv')
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
                    self.local_output_path, 'smnas_m_best_config.py')
                with open(config_path, 'w') as fp:
                    fp.write(net_desc)
        except Exception as e:
            logging.error("write best model error:{}".format(str(e)))
