# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import os
import torch
from torch import nn
from torch.nn import functional as F


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        ## WTN section
        cur_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        self.clsnet_weights = torch.load(cur_file_dir + "/../../../../datasets/openimages/clsnet_weights.pth")

        # standard normalization
        self.clsnet_weights = self.clsnet_weights-self.clsnet_weights.mean(0,keepdim=True)
        self.clsnet_weights = self.clsnet_weights/self.clsnet_weights.std(0,keepdim=True)
        self.clsnet_weights.requires_grad_(False)
        # inds on the clsnet side
        self.clsnet_inds = torch.load(cur_file_dir + "/../../../../datasets/openimages/clsnet_inds_common.pth")
        # WTN
        self.wtn_enc = nn.Sequential(nn.Linear(2048, 2048, bias=False),
                                nn.GroupNorm(32, 2048),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(2048, 2048))
        self.wtn_dec = nn.Sequential(nn.Linear(2048, 2048, bias=False),
                                nn.GroupNorm(32, 2048),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(2048, 2048))
        self.common_bias = nn.Parameter(torch.zeros(1))
        self.cls_score = nn.Linear(representation_size, num_classes-len(self.clsnet_inds))
        num_bbox_reg_classes = 1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        self.reconst_loss = cfg.MODEL.ROI_BOX_HEAD.RECONST_LOSS
        self.reconst_w = cfg.MODEL.ROI_BOX_HEAD.RECONST_W


        nn.init.constant_(self.wtn_enc[1].weight, 0)
        nn.init.constant_(self.wtn_enc[1].bias, 0)
        nn.init.constant_(self.wtn_enc[-1].bias, 0)
        nn.init.constant_(self.wtn_dec[1].weight, 1)
        nn.init.constant_(self.wtn_dec[1].bias, 0)
        nn.init.normal_(self.wtn_dec[-1].weight, std=0.001)
        nn.init.constant_(self.wtn_dec[-1].bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.001)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        bias_value = -torch.log(torch.Tensor([(1 - 0.01) / 0.01]))
        nn.init.constant_(self.common_bias, bias_value.item())
        nn.init.constant_(self.cls_score.bias, bias_value.item())

    def forward(self, x, clsnet_inds):
        # move wtn stuffs to GPU
        if not self.clsnet_weights.is_cuda:
            self.clsnet_weights = self.clsnet_weights.cuda(x.get_device())
            self.clsnet_inds = self.clsnet_inds.cuda(x.get_device())

        # WTN autoencoding
        if self.training or clsnet_inds is None:
            det_weights = self.wtn_enc(self.clsnet_weights)
            common_weights = det_weights[self.clsnet_inds,...]
            reconst = self.wtn_dec(det_weights)
            if self.reconst_loss == "L2":
                reconst_loss = ((reconst-self.clsnet_weights)**2).mean()
            elif self.reconst_loss == "smoothL1":
                reconst_loss = F.smooth_l1_loss(reconst, self.clsnet_weights)

            scores_common = F.linear(x, common_weights, self.common_bias.repeat(len(self.clsnet_inds)))
            scores = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)

            return torch.cat((scores_common, scores),1), bbox_deltas, reconst_loss*self.reconst_w
        elif clsnet_inds is not None:
            weights = self.wtn_enc(self.clsnet_weights)[clsnet_inds,...]
            scores = F.linear(x, weights, self.common_bias.repeat(len(clsnet_inds)))
            bbox_deltas = self.bbox_pred(x)

            return scores, bbox_deltas, None

@registry.ROI_BOX_PREDICTOR.register("FPNPredictor_2nd")
class FPNPredictor_2nd(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor_2nd, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.001)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        bias_value = -torch.log(torch.Tensor([(1 - 0.01) / 0.01]))
        nn.init.constant_(self.cls_score.bias, bias_value.item())

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def make_roi_box_predictor(cfg, is_2nd=False):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR + ("_2nd" if is_2nd else "")]
    return func(cfg)