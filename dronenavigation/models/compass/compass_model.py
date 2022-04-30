import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _initialize_weights(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, 0.1)


class CompassModel(BaseFeaturesExtractor):
    def __init__(self, observation_space, mode, pretrained_encoder_path, feature_size):
        super(CompassModel, self).__init__(observation_space, feature_size)
        self.pretrained_encoder_path = pretrained_encoder_path
        self.mode = mode
        from .select_backbone import select_resnet
        self.model_rgb, _, self.model_depth, _, param = select_resnet('resnet18')
        self.load_pretrained_encoder_weights(self.pretrained_encoder_path)

        for param in self.model_rgb.parameters():
            param.requires_grad = False  # not update by gradient
        for param in self.model_depth.parameters():
            param.requires_grad = False  # not update by gradient

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            if torch.cuda.is_available():
                logging.info("Compass CUDA")
                ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            else:
                logging.info("Compass CPU")
                ckpt = torch.load(pretrained_path, map_location=torch.device('cpu'))['state_dict']

            ckpt_rgb = {}
            ckpt_depth = {}
            for key in ckpt:
                if key.startswith('backbone_rgb') and (self.mode == "rgb" or self.mode == "both"):
                    ckpt_rgb[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('backbone_depth.') and (self.mode == "depth" or self.mode == "both"):
                    ckpt_depth[key.replace('backbone_depth.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt_rgb[key.replace('module.backbone.', '')] = ckpt[key]
                    ckpt_depth[key.replace('module.backbone.', '')] = ckpt[key]

            if self.mode == "rgb" or self.mode == "both":
                self.model_rgb.load_state_dict(ckpt_rgb)

            if self.mode == "depth" or self.mode == "both":
                self.model_depth.load_state_dict(ckpt_depth)
            logging.info('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            logging.info('Train from scratch.')

    def forward(self, x):
        logging.info("_")
        # x: B, C, SL, H, W
        # concat order: state-rgb-depth

        tensor_concat = nn.Flatten(x["state"])

        if "rgb" in x:
            # x["rgb"] = x["rgb"].unsqueeze(2)  # Shape: [B,C,H,W] -> [B,C,1,H,W].
            x["rgb"] = self.model_rgb(
                x["rgb"])  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
            x["rgb"] = x["rgb"].mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].
            tensor_concat = torch.cat((tensor_concat, x["rgb"]), 1)
        if "depth" in x:
            # x["depth"] = x["depth"].unsqueeze(2)  # Shape: [B,C,H,W] -> [B,C,1,H,W].
            x["depth"] = self.model_depth(
                x["depth"])  # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.
            # x = torch.randint(20, size=(2, 256), device=0) / 20
            x["depth"] = x["depth"].mean(dim=(2, 3, 4))  # Shape: [B,C',1,H',W'] -> [B,C'].
            tensor_concat = torch.cat((tensor_concat, x["depth"]), 1)

        return tensor_concat
