import unittest
import vega
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.user_config import Config


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        self.assertEqual(self.cfg.task.local_base_path, "/efs/cache/local/")
        dataset = ClassFactory.get_cls("dataset")
        train_dataset = dataset(mode='train')
        self.assertEqual(len(train_dataset), 800)


class TestNetworkDesc(unittest.TestCase):

    def test_hb(self):
        cfg = Config("./bj4.yml")
        # DefaultConfig().data = cfg
        vega.run("./esr_ea.yml")


if __name__ == "__main__":
    unittest.main()
