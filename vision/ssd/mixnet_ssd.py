import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU

from ..nn.mixnet import MixNet

from .ssd import SSD
from .predictor import Predictor
from .config import mixnet_config as config
from .attention import CAM_Module


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mixnet_ssd(num_classes, is_test=False):
    base_net = MixNet().layers 
    # print("*********************************************************************")
    # print(base_net)
    source_layer_indexes = [
        (len(base_net), Conv2d(in_channels=1536, out_channels=256, kernel_size=1)),
        
     ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1536, out_channels=256, kernel_size=1),
            ReLU(),
            # CAM_Module(256),
            SeperableConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ), 
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            # CAM_Module(128),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            # CAM_Module(128),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            # CAM_Module(128),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),   
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1), 
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_efficientnet_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor