import torch.nn as nn
# from torchvision.models import resnet101
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign


class FasterRCNNModel(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super(FasterRCNNModel, self).__init__()
        self.num_classes = num_classes

        # self.backbone = nn.Sequential(
        #     *list(resnet101(pretrained=False).children())[:-3]
        # )

        self.backbone = nn.Sequential(
            *list(fcn_resnet101(pretrained=use_pretrained).backbone.children())[:-1]
        )

        self.backbone.out_channels = 1024

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        self.roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                             output_size=7,
                                             sampling_ratio=2)

        self.faster_rcnn = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=self.roi_pooler
        )

    def forward(self, images, targets):
        output = self.faster_rcnn(images, targets)
        return output
