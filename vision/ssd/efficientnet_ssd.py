import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
# from ..nn.efficientnet import EfficientNet 

from ..nn.efficientnet import (EfficientNet, efficientnet_b0, efficientnet_b1,
                           efficientnet_b2, efficientnet_b3, efficientnet_b4,
                           efficientnet_b5, efficientnet_b6, efficientnet_b7)

from .ssd import SSD
from .predictor import Predictor
from .config import efficientnet_config as config
from .attention import CAM_Module, DUAttn


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_efficientnet(num_classes, is_test=False):
    # base_net = EfficientNet.from_pretrained('efficientnet-b5').extract_features   # disable dropout layer
    base_net = efficientnet_b5(pretrained=True).features  # version2
    # print("*********************************************************************")
    # print(len(base_net))
    source_layer_indexes = [
        (40, Conv2d(in_channels=512, out_channels=256, kernel_size=1)),
        (len(base_net), Conv2d(in_channels=2048, out_channels=256, kernel_size=1)),
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=2048, out_channels=256, kernel_size=1),
            ReLU(),
            DUAttn(256),
            SeperableConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()       
        ), 
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            DUAttn(128),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            DUAttn(128),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            DUAttn(128),
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