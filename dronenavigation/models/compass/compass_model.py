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
        self.model_rgb, _, self.model_depth, _, param = select_resnet('resnet18')
        self.load_pretrained_encoder_weights(self.pretrained_encoder_path)
        self.mode = "both"

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            if torch.cuda.is_available():
                print("Compass CUDA")
                ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            else:
                print("Compass CPU")
                ckpt = torch.load(pretrained_path, map_location=torch.device('cpu'))['state_dict']

            ckpt_rgb = {}
            ckpt_depth = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt_rgb[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt_rgb[key.replace('module.backbone.', '')] = ckpt[key]
                elif key.startswith('backbone_depth.'):
                    ckpt_depth[key.replace('backbone_depth.', '')] = ckpt[key]
            self.model_rgb.load_state_dict(ckpt_rgb)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')

    def __tune_rgb_tensor(self, x):
        x["rgb"] = x["rgb"].unsqueeze(2)  # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x["rgb"] = self.model_rgb(
            x["rgb"])  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
        x["rgb"] = x["rgb"].mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].

    def __tune_depth_tensor(self, x):
        x["depth"] = x["depth"].unsqueeze(2)  # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x["depth"] = self.model_depth(
            x["depth"])  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
        print("_")
        # x = torch.randint(20, size=(2, 256), device=0) / 20
        x["depth"] = x["depth"].mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].

    def forward(self, x):

        # x: B, C, SL, H, W
        if self.mode == "rgb":
            self.__tune_rgb_tensor(x)
            return torch.cat((x["rgb"], x["state"]), 1)
        elif self.mode == "depth":
            self.__tune_depth_tensor(x)
            return torch.cat((x["depth"], x["state"]), 1)
        else:
            self.__tune_rgb_tensor(x)
            self.__tune_depth_tensor(x)
            return torch.cat((x["rgb"], x["depth"], x["state"]), 1)

