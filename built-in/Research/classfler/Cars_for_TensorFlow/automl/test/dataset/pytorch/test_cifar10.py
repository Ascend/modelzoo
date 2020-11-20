import unittest
import torchvision.transforms as tf
from vega.datasets.pytorch.common.dataset import Dataset
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        dataset = Dataset(mode="train",
                          transforms=[{'type': 'tf.RandomCrop', 'size': 32, 'padding': 4},
                                      {'type': 'tf.RandomHorizontalFlip'}])
        self.assertEqual(len(dataset.transform.transforms), 2)
        print(dataset.transform)
        dataset.transforms.append(tf.ToTensor())
        self.assertEqual(len(dataset.transform.transforms), 3)
        dataset.transforms.insert(2, "Color", level=2)
        self.assertEqual(len(dataset.transform.transforms), 4)
        dataset.transforms.remove("Color")
        self.assertEqual(len(dataset.transform.transforms), 3)
        train = dataset.dataloader
        self.assertEqual(len(train), 25000)
        for input, target in train:
            self.assertEqual(len(input), 1)
            break


class TestDataset(unittest.TestCase):

    def test_cifar10(self):
        vega.run('./cifar10.yml')


if __name__ == "__main__":
    unittest.main()
