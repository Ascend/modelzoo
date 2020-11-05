import unittest
import os
import torchvision.transforms as tf
from vega.datasets.common.dataset import Dataset
from roma.env import init_env
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.task_ops import TaskOps
from vega.core.common.file_ops import FileOps
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, TaskOps, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        FileOps.copy_folder("s3://automl-hn1/liuzhicheng/test_roma/", "/cache/test/")
        test_file = len([x for x in os.listdir(os.path.dirname("/cache/test/"))])
        self.assertEqual(test_file, 3)
        self.assertEqual(self.local_base_path, "/efs/{}".format(self.task_id))
        self.assertEqual(self.output_subpath, "output/")
        self.assertEqual(self.get_worker_subpath("1", "10"), "workers/1/10/")


class TestDataset(unittest.TestCase):

    def test_roma(self):
        init_env()
        vega.run('./roma.yml')


if __name__ == "__main__":
    unittest.main()
