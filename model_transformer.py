from PIL import Image
import requests
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

class crowdcounting_tr(nn.Module):

    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,load_weights=False):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_output = nn.Linear(hidden_dim, num_classes)
#        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(1, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._init_weights()
    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        print('conv.shape',x.shape)
        x = self.backbone.bn1(x)
        print('bn1.shape',x.shape)
        x = self.backbone.relu(x)
        print('relu.shape',x.shape)
        x = self.backbone.maxpool(x)
        print('maxpool.shape',x.shape)
        x = self.backbone.layer1(x)
        print('layer1.shape',x.shape)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        print('layer4.shape',x.shape)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        print('self.conv.shape',h.shape)
        # construct positional encodings
        H, W = h.shape[-2:]
        pos_temp=torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1)
        print('pos_temp',pos_temp.shape)
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        print('pos.shape',pos.shape)
        temp=self.query_pos.unsqueeze(1)
        print('self.query_pos.shape',temp.shape)
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        print('transformer_output',h.shape)
        h=self.linear_output(h)
        print('linear_output.shape',h.shape)
        b,_,h_temp,w_temp=inputs.shape
        print('b',b)
        print('h_temp',h_temp)
        h= h.view(h,(b,h_temp//8,w_temp//8))
        print('output',h.shape)
        return h
        # finally project transformer outputs to class labels and bounding boxes
#        return {'pred_logits': self.linear_class(h), 
   #     'pred_boxes': self.linear_bbox(h).sigmoid()}
  
    def _init_weights(self):
      """ Initialize the weights """
      for m in self.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
 #       elif isinstance(m, BertLayerNorm):
   #         module.bias.data.zero_()
    #        module.weight.data.fill_(1.0)
        elif isinstance(m, nn.Conv2d):
              nn.init.normal_(m.weight, std=0.01)
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
  #      if isinstance(m, nn.Linear) and module.bias is not None:
     #       module.bias.data.zero_()
