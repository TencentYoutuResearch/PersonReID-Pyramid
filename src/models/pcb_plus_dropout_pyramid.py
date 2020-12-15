import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
from .resnet import resnet101


class PCB_plus_dropout_pyramid(nn.Module):
    def __init__(
        self,
        last_conv_stride=1,
        last_conv_dilation=1,
        num_stripes=6,  # number of sub-parts
        used_levels=(1, 1, 1, 1, 1, 1),
        num_conv_out_channels=128,
        num_classes=0
    ):

        super(PCB_plus_dropout_pyramid, self).__init__()

        print("num_stripes:{}".format(num_stripes))
        print("num_conv_out_channels:{},".format(num_conv_out_channels))

        self.base = resnet101(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)

        self.dropout_layer = nn.Dropout(p=0.2)

        # ==============================================================================
        # ============================================================================== pyramid
        self.num_classes = num_classes
        self.num_stripes = num_stripes
        self.used_levels = used_levels

        # =========================================
        input_size0 = 2048
        self.pyramid_conv_list0 = nn.ModuleList()
        self.pyramid_fc_list0 = nn.ModuleList()
        PCB_plus_dropout_pyramid.basic_branch(self, num_conv_out_channels,
                                              input_size0,
                                              self.pyramid_conv_list0,
                                              self.pyramid_fc_list0)

        # =========================================
        input_size1 = 1024
        self.pyramid_conv_list1 = nn.ModuleList()
        self.pyramid_fc_list1 = nn.ModuleList()
        PCB_plus_dropout_pyramid.basic_branch(self, num_conv_out_channels,
                                              input_size1,
                                              self.pyramid_conv_list1,
                                              self.pyramid_fc_list1)
        # ==============================================================================pyramid
        # ==============================================================================

    def forward(self, x):
        """
        Returns:
        feat_list: each member with shape [N, C]
        logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat0 = self.base(x)

        assert feat0.size(2) % self.num_stripes == 0
        # assert feat1.size(2) % self.num_stripes == 0
        # ==============================================================================
        # ============================================================================== pyramid
        feat_list = []
        logits_list = []

        PCB_plus_dropout_pyramid.pyramid_forward(self, feat0,
                                                 self.pyramid_conv_list0,
                                                 self.pyramid_fc_list0,
                                                 feat_list,
                                                 logits_list)

        """
        PCB_plus_dropout_pyramid.pyramid_forward(self, feat1,
                        self.pyramid_conv_list1,
                        self.pyramid_fc_list1,
                        feat_list,
                        logits_list)
        """
        return feat_list, logits_list
        # ============================================================================== pyramid
        # ==============================================================================

    @ staticmethod
    def basic_branch(self, num_conv_out_channels,
                     input_size,
                     pyramid_conv_list,
                     pyramid_fc_list):
        # the level indexes are defined from fine to coarse,
        # the branch will contain one more part than that of its previous level
        # the sliding step is set to 1
        self.num_in_each_level = [i for i in range(self.num_stripes, 0, -1)]
        self.num_levels = len(self.num_in_each_level)
        self.num_branches = sum(self.num_in_each_level)

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            pyramid_conv_list.append(nn.Sequential(
                nn.Conv2d(input_size, num_conv_out_channels, 1),
                nn.BatchNorm2d(num_conv_out_channels),
                nn.ReLU(inplace=True)))

        # ========================
        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            fc = nn.Linear(num_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            pyramid_fc_list.append(fc)

    @staticmethod
    def pyramid_forward(self, feat,
                        pyramid_conv_list,
                        pyramid_fc_list,
                        feat_list,
                        logits_list):

        basic_stripe_size = int(feat.size(2) / self.num_stripes)

        idx_levels = 0
        used_branches = 0
        for idx_branches in range(self.num_branches):

            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            idx_in_each_level = idx_branches - \
                sum(self.num_in_each_level[0:idx_levels])

            stripe_size_in_level = basic_stripe_size * (idx_levels+1)

            st = idx_in_each_level * basic_stripe_size
            ed = st + stripe_size_in_level

            local_feat = F.avg_pool2d(feat[:, :, st: ed, :],
                                      (stripe_size_in_level, feat.size(-1))) + F.max_pool2d(feat[:, :, st: ed, :],
                                                                            (stripe_size_in_level, feat.size(-1)))

            local_feat = pyramid_conv_list[used_branches](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)

            local_logits = pyramid_fc_list[used_branches](
                self.dropout_layer(local_feat))
            logits_list.append(local_logits)

            used_branches += 1
