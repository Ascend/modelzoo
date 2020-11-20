import unittest
import os
from roma.env import init_env
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        self.assertEqual(self.cfg.task.local_base_path, "/efs/cache/local/")
        dataset = ClassFactory.get_cls("dataset")
        train_dataset = dataset(mode='train')
        self.assertEqual(len(train_dataset), 25000)


class TestDataset(unittest.TestCase):

    def test_roma(self):
        init_env('hb1_y')
        vega.run('./roma_config.yml')


if __name__ == "__main__":
    unittest.main()
