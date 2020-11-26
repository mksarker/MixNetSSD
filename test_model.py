import unittest
import torch
import torch.nn as nn
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.efficientnet_ssd import create_efficientnet
from vision.ssd.mixnet_ssd import create_mixnet_ssd


from vision.ssd.fpn_squeezenet_ssd import create_fpn_squeezenet_ssd

from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.config import efficientnet_config
from vision.ssd.config import mixnet_config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestResNetWrapper(unittest.TestCase):

    def test_resnet_wrapper(self):
        net = create_mixnet_ssd(num_classes=90)
        
        print(net)
        params = list(net.parameters()) 
        count=count_parameters(net)
        print (count)

        x = torch.rand(2, 3, 224, 224)
        x = torch.autograd.Variable(x)
        x = net(x)
        # print(x.size())
        self.assertTrue(x is not None)

if __name__ == '__main__':
    unittest.main()