import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()

        self.conv_layer = torchvision.models.efficientnet_v2_s(pretrained=True).features
        self.conv_layer[:int(len(self.conv_layer)*0.5)].requires_grad_(False)

        self.linear_layer_general = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1280, out_features=20),
        )

        self.linear_layer_special = nn.Linear(in_features=1300, out_features=100)

    def forward(self, x):
        out = self.conv_layer(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        features = torch.flatten(out, 1)
        out_general = self.linear_layer_general(features)
        out_special = self.linear_layer_special(torch.cat((features, out_general), dim=1))

        return out_general, out_special
