import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _initialize_weights(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, 0.1)


class CompassModel(BaseFeaturesExtractor):
    def __init__(self, observation_space, linear_prob, pretrained_encoder_path, feature_size):
        super(CompassModel, self).__init__(observation_space, feature_size)
        self.pretrained_encoder_path = pretrained_encoder_path
        from .select_backbone import select_resnet
        self.encoder, _, _, _, param = select_resnet('resnet18')
        self.load_pretrained_encoder_weights(self.pretrained_encoder_path)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            if torch.cuda.is_available():
                print("Compass CUDA")
                ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            else:
                print("Compass CPU")
                ckpt = torch.load(pretrained_path, map_location=torch.device('cpu'))['state_dict']

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
        if x.shape.__len__() == 3:
            x = x.unsqueeze(0)  # FIX used for train env with 1 drone(yaml num env). compass need 5D tensor
        x = x.unsqueeze(2)  # Shape: [B,C,H,W] -> [B,C,1,H,W].

        x = self.encoder(x)  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
        """
        if self.linear_prob:
            x = x.mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].
            x = self.pred(x)  # Shape: [B,C'] -> [B,C''].

        else:
            B, N, T, H, W = x.shape
            x = x.view(B, T, N, H, W)
            x = x.view(B * T, N, H, W)
            x = self.pred(x)
            x = x.mean(dim=(1, 2, 3))

        # old x shape (2,4)
        #              b,f
        # new x shape (2,256,1,7,7)
        #              b,c'  ,1,h',w'
        """

        # x = torch.randint(20, size=(2, 256), device=0) / 20
        x = x.mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].
        return x
