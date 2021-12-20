from transformers import LayoutLMModel
import torch.nn as nn
from collections import OrderedDict

from RCNN import FasterRCNNModel


class LayoutLMImageEmbeddingModel(nn.Module):
    def __init__(self, num_classes=4,
                 pretrained_layoutlm='microsoft/layoutlm-base-uncased',
                 pretrained_rcnn_model=None,
                 use_pretrained_coco_rcnn=True):
        super(LayoutLMImageEmbeddingModel, self).__init__()

        self.layoutlm = LayoutLMModel.from_pretrained(pretrained_layoutlm)
        if pretrained_rcnn_model is not None:
            faster_rcnn = FasterRCNNModel(use_pretrained=use_pretrained_coco_rcnn)
            self.faster_rcnn_backbone = faster_rcnn.backbone
            self.roi_pooler = faster_rcnn.roi_pooler
        else:
            self.faster_rcnn_backbone = pretrained_rcnn_model.backbone
            self.roi_pooler = pretrained_rcnn_model.roi_pooler

        self.roi_feat_proj = nn.Linear(
            in_features=1024 * 7 * 7,
            out_features=768)

        self.classifier = nn.Linear(
            in_features=768,
            out_features=num_classes
        )

        self.output = LayoutLMImageEmbeddingOutput(num_classes=num_classes)

    def forward(self, input_ids, bbox_layoutlm, bbox_rcnn, images, attention_mask=None, labels=None):
        x = self.layoutlm(input_ids, bbox=bbox_layoutlm, attention_mask=attention_mask)
        x = x.last_hidden_state

        batch_size = x.size(0)
        seq_length = x.size(1)

        feat_maps = self.faster_rcnn_backbone(images)
        feat_maps_od = OrderedDict([
            ('0', feat_maps)
        ])

        boxes = [bbox_ for bbox_ in bbox_rcnn]
        image_shapes = [(512, 512)] * batch_size

        roi_feat = self.roi_pooler(feat_maps_od, boxes=boxes, image_shapes=image_shapes)
        roi_feat = roi_feat.view(batch_size, seq_length, -1)
        roi_feat = self.roi_feat_proj(roi_feat)

        x = x + roi_feat
        x = self.classifier(x)

        if labels is not None:
            self.output.compute_loss(logits=x, labels=labels, attention_mask=attention_mask)
        else:
            self.output.logits = x

        return self.output


class LayoutLMImageEmbeddingOutput:
    def __init__(self, num_classes):
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.logits = None
        self.loss = None
        return

    def compute_loss(self, logits, labels, attention_mask=None):
        self.logits = logits
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_classes)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            self.loss = self.loss_fn(active_logits, active_labels)
        else:
            self.loss = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        return
