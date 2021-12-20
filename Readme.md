<h1><b>LayoutLMV1</b></h1>

This project illustrates how we can use LayoutLM Version 1 with image embeddings
from documents' feature maps to improve performance.

This project tries to pretrain an RCNN model with a ResNet101 as it's backbone and the uses this 
backbone to generate feature maps, which are documents' embeddings. 

This feature map from documents is used in addition to LayoutLM Transformer to classify tokens and extract fields from 
the document.

The project was inspired from the paper https://arxiv.org/abs/1912.13318.