# -*- coding:utf-8 -*-
"""This is an exmple to TF."""
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
        train_steps = train_data.data_len
        self.assertEqual(train_steps, 781)
        valid_steps = valid_data.data_len
        self.assertEqual(valid_steps, 156)
        for data_file in train_data.data_files:
            print("train file:", data_file)
        for data_file in valid_data.data_files:
            print("valid file:", data_file)


class TestDataset(unittest.TestCase):
    """Test Dataset."""

    def test_cifar10(self):
        """Test cifar10."""
        vega.run('./cifar10_tf.yml')


if __name__ == "__main__":
    unittest.main()
