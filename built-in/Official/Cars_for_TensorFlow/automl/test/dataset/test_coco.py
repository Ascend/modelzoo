# -*- coding:utf-8 -*-
"""This is an test to copy."""
import unittest
import copy
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakeTFPipeStep(PipeStep, unittest.TestCase):
    """Fake TF PipeStep."""

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        """Do train."""
        data_cls = ClassFactory.get_cls(ClassType.DATASET)
        data_cfg = copy.deepcopy(ClassFactory.__configs__.get(ClassType.DATASET))
        data_cfg.pop('type')
        train_data, valid_data = [
            data_cls(**data_cfg, mode=mode) for mode in ['train', 'val']
        ]


class TestDataset(unittest.TestCase):
    """Test Dataset."""

    def test_coco(self):
        """Test coco."""
        vega.run('./coco_tf.yml')


if __name__ == "__main__":
    unittest.main()
