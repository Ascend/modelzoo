import unittest
import torch.nn as nn
from vega.search_space.networks import NetTypes, NetTypesMap, Network, NetworkFactory


@NetworkFactory.register(NetTypes.BACKBONE)
class SimpleNetwork(Network):

    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.net = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.net(x)


class TestNetworkFactory(unittest.TestCase):

    def test_simple_network(self):
        module_type = NetTypesMap['backbone']
        net_class = NetworkFactory.get_network(module_type, 'SimpleNetwork')
        self.assertEqual(isinstance(net_class(), SimpleNetwork), True)


if __name__ == "__main__":
    unittest.main()
