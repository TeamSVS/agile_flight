import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CompassModel(BaseFeaturesExtractor):
    def __init__(self, observation_space, linear_prob, pretrained_encoder_path, feature_size):
        super(CompassModel, self).__init__(observation_space, feature_size)
        self.pretrained_encoder_path = pretrained_encoder_path
        self.linear_prob = linear_prob
        from .select_backbone import select_resnet
        self.encoder, _, _, _, param = select_resnet('resnet18')

        if self.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size'], 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)

        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(self.pretrained_encoder_path)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path,map_location=torch.device('cpu'))['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt2[key.replace('module.backbone.', '')] = ckpt[key]
            self.encoder.load_state_dict(ckpt2)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')

    def forward(self, x):




        # x: B, C, SL, H, W
        x = torch.reshape(x, (32 ,3, 224, 224))

        x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x = self.encoder(x)  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
        B, N, T, H, W = x.shape
        x = x.mean(dim=(2, 3,4))
        #x=torch.flatten(x,3,4)
        #x = torch.flatten(x, 0,2)
        """if self.linear_prob:
            x = x.mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].
            x = self.pred(x)  # Shape: [B,C'] -> [B,C''].

        else:
            B, N, T, H, W = x.shape
            x = x.view(B, T, N, H, W)
            x = x.view(B * T, N, H, W)
            x = self.pred(x)
            x = x.mean(dim=(1, 2, 3))
        """
        print("x")
        return x
