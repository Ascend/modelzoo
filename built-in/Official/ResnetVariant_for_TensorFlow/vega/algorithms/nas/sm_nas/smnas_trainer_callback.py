"""Trainer for SMNas."""

import logging
import os
import mmcv
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from vega.core.trainer.callbacks import Callback
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.CALLBACK)
class SMNasTrainerCallback(Callback):
    """Trainer for SMNas."""

    disable_callbacks = ["ModelStatistics", "MetricsEvaluator", "ModelCheckpoint", "PerformanceSaver",
                         "LearningRateScheduler", "ProgressLogger", "ReportCallback"]

    def __init__(self):
        super(SMNasTrainerCallback, self).__init__()
        self.alg_policy = None

    def set_trainer(self, trainer):
        """Set trainer object for current callback."""
        self.trainer = trainer
        self.trainer._train_loop = self._train_process
        self.cfg = self.trainer.config
        self._worker_id = self.trainer._worker_id
        self.gpus = self.cfg.gpus
        if hasattr(self.cfg, "kwargs") and "smnas_sample" in self.cfg.kwargs:
            self.sample_result = self.cfg.kwargs["smnas_sample"]
        self.local_worker_path = self.trainer.get_local_worker_path()
        self.output_path = self.trainer.local_output_path
        config_path = os.path.join(self.local_worker_path, 'config.py')
        with open(config_path, 'w') as f:
            f.write(self.trainer.model.desc)
        self.config_path = config_path
        self.cost_value = self.trainer.model.cost if self.trainer.model is not None else 0.0
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self._train_script = os.path.join(dir_path, 'tools/dist_train.sh')
        self._eval_script = os.path.join(dir_path, 'tools/dist_test.sh')
        self.epochs = self.cfg.epochs

    def _train_process(self):
        """Process of train and test."""
        logging.info("start training")
        self._train()
        torch.cuda.empty_cache()
        logging.info("start evaluation")
        performance = self._valid()
        performance.append(self.cost_value)
        self.save_performance(performance)

    def _train(self):
        """Train the network."""
        cmd = ['bash', self._train_script, self.config_path, str(self.gpus),
               '--total_epochs', str(self.epochs),
               '--work_dir', self.local_worker_path]
        cmd_str = ''
        for item in cmd:
            cmd_str += (item + ' ')
        logging.info(cmd_str)
        os.system(cmd_str)

    def _valid(self):
        """Get performance on validate dataset."""
        checkpoint_path = os.path.join(self.local_worker_path, 'latest.pth')
        eval_prefix = os.path.join(self.local_worker_path, 'eval.pkl')
        cmd = ['bash', self._eval_script, self.config_path, checkpoint_path,
               str(self.gpus),
               '--out', eval_prefix, '--eval', 'bbox']
        cmd_str = ''
        for item in cmd:
            cmd_str += (item + ' ')
        logging.info(cmd_str)

        os.system(cmd_str)
        eval_file = os.path.join(self.local_worker_path, 'eval.pkl.bbox.json')
        model_desc = mmcv.Config.fromfile(self.config_path)
        try:
            performance = self.coco_eval(
                eval_file, model_desc.data.test.anno_file)
        except BaseException:
            performance = 0.0
        return [performance]

    def save_performance(self, performance):
        """Save performance results."""
        if isinstance(performance, int) or isinstance(performance, float):
            performance_dir = os.path.join(self.local_worker_path,
                                           'performance')
            if not os.path.exists(performance_dir):
                FileOps.make_dir(performance_dir)
            with open(os.path.join(performance_dir, 'performance.txt'),
                      'w') as f:
                f.write("{}".format(performance))
        elif isinstance(performance, list):
            performance_dir = os.path.join(self.local_worker_path,
                                           'performance')
            if not os.path.exists(performance_dir):
                FileOps.make_dir(performance_dir)
            with open(os.path.join(performance_dir, 'performance.txt'),
                      'w') as f:
                for p in performance:
                    if not isinstance(p, int) and not isinstance(p, float):
                        logging.error("performance must be int or float!")
                        return
                    f.write("{}\n".format(p))

    def coco_eval(self, result_file, coco):
        """Eval result_file by coco."""
        if mmcv.is_str(coco):
            coco = COCO(coco)
        assert isinstance(coco, COCO)
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        cocoEval = COCOeval(coco, coco_dets, 'bbox')
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval.stats[0] * 100
