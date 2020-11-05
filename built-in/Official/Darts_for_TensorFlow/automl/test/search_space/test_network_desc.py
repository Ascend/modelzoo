import unittest
import torch.nn as nn
from vega.core.common import Config
from vega.search_space.networks.network_desc import NetworkDesc


class TestNetworkDesc(unittest.TestCase):

    def test_prune_resnet(self):
        from vega.search_space.networks.backbones import PruneResNet
        cfg = Config('./prune_resnet.yml')
        net_desc = NetworkDesc(cfg)
        model = net_desc.to_model()
        self.assertEqual(isinstance(model, PruneResNet), True)


if __name__ == "__main__":
    unittest.main()
