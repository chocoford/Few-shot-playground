"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from .resnet import resnet


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, encoder="vgg"):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        if encoder == "vgg":
            # Encoder: VGG-16
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        elif encoder == "fpn" :
            fpn = resnet()
            fpn.create_architecture()
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', fpn), 
            ]))

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, segments):
        """
        Args
        -------------
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            qry_imgs: super piexel segment mask
                N x [H x W], list of tensors
        Return
        ---------------
            query_pred: distance array 
                [N+B x (1+way) x H x W]  eg. [1, 2, 417, 417]
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(
            way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]  # [W, H]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            query_fts = qry_fts[:, epi]
            ###### Extract prototype 获得公式1、2中右半边的值######
            supp_fg_fts = [
                [self.getFeatures(supp_fts[way, shot, [epi]], fore_mask[way, shot, [
                                  epi]]) for shot in range(n_shots)]
                for way in range(n_ways)
            ]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(
                supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            if self.config['no_pbg']: 
                prototypes = fg_prototypes
            else:
                prototypes = [bg_prototype, ] + fg_prototypes
            
            if self.config['superpixel_preSeg']:
                # 先放大再比较
                query_fts = F.interpolate(query_fts, size=img_size, mode='bilinear')
                if self.config['no_pbg']:
                    dist = [self.calDist(query_fts, prototype, threshold=0.5)
                            for prototype in prototypes]
                else:
                    qry_prototypes = self.getQrySuperPrototype(query_fts, segments)
                    # print(f'line 118: each qry_prototype\'s shape: {qry_prototypes[0].shape}')
                    dist = [F.cosine_similarity(torch.cat(qry_prototypes), prototype)
                            for prototype in prototypes] # num_protoypes x [num_qry_prototypes]
                    dist = [self.restorePredMask(pred, segments) for pred in dist]
                pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H x W
                outputs.append(pred)
            else:
                dist = [self.calDist(query_fts, prototype)
                        for prototype in prototypes]
                pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
                if self.config['no_pbg']:
                    zeros = torch.zeros_like(pred)
                    pred = F.interpolate(pred, size=img_size, mode='bilinear')
                    pred = torch.where(pred > 0.5, pred, zeros)
                    outputs.append(pred)
                else:
                    outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            
                
            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(query_fts, pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size

    def calDist(self, fts, prototype, threshold=None, scaler=20):
        """
        Calculate the distance between features and prototypes
        --------
        Args
        --------
            fts: input features
                expect shape: N x C x H' x W'
            prototype: prototype of one semantic class
                expect shape: 1 x C
            threshold: only valid when ignoring background. Define whether 
        Return
        --------
            dist: cosine distance.
                expect shape: N x H' x W' or N x H x W if up_first
        """
        dist = F.cosine_similarity(
            fts, prototype[..., None, None], dim=1) * scaler
        if threshold:
            zeros = torch.zeros_like(dist)
            dist = torch.where(dist > 0.5, dist, zeros)
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(
            fts, size=mask.shape[-2:], mode='bilinear')  # 采样到与mask一样的大小

        # playground
        if self.config['cwa']:
            valid_fts = fts * mask[None, ...]
            ones = torch.ones_like(valid_fts)
            zeros = torch.zeros_like(valid_fts)
            masked_fts = torch.sum(valid_fts, dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)).repeat(fts.shape[:2]) \
            + torch.where(valid_fts > 0, ones, zeros).sum(dim=(2,3)) + 1e-5)
        elif self.config['cwwa']: 
            masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        
        # 把整个输入变成一个向量
        else:
            masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Parameters
        ----------
            fg_fts: lists of list of foreground features for each way/shot, 就是论文中公式(1)中的右边
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot, 就是论文中公式(2)中的右边
                expect shape: Wa x Sh x [1 x C]

        Returns
        -----------
            fg_prototypes: expected shape: way x [1 x C]
            bg_prototype:
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W' or N x C x H x W if us_first
            pred: predicted segmentation score，其中pred[:, 0]是背景
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        # N x 1 x H' x W' 在每个像素的层面上判断这个像素最可能是哪个类
        pred_mask = pred.argmax(dim=1, keepdim=True)
        # binary_masks是一个list，包含(1 + n_ways)个tensor，每个tensor表示每个像素是否属于该类。
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(
            n_ways) if binary_masks[i + 1].sum() == 0]
        # N x (1 + Wa) x 1 x H' x W'
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(
            qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / \
            (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            if self.config['no_pbg']:
                prototypes = [qry_prototypes[[way + 1]]]
            else:
                prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]] #[[0]]相当于增加了一个维度
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                if self.config['us_first']:
                    img_fts = F.interpolate(img_fts, size=fore_mask.shape[-2:], mode='bilinear')
                    if self.config['no_pbg']:
                        supp_dist = [self.calDist(img_fts, prototype, threshold=0.5)
                                    for prototype in prototypes]
                    else: 
                        supp_dist = [self.calDist(img_fts, prototype)
                                    for prototype in prototypes]
                    supp_pred = torch.stack(supp_dist, dim=1)
                else:
                    supp_dist = [self.calDist(img_fts, prototype)
                                for prototype in prototypes]
                    supp_pred = torch.stack(supp_dist, dim=1)
                    supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                            mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss

    def getQrySuperPrototype(self, qry_fts, segments):
        """
        根据超像素分割获得每个超像素的prototype

        Parameters
        ----------
            qry_fts: features of query image, expected shape: [N, C, H, W]
            segments: superpixel segmentation mask, expected shape: [H, W]

        Return
        ---------
            prototypes: expected shape: n x [1, C]
        """
        ones = torch.ones(1).cuda()
        zeros = torch.zeros(1).cuda()
        n_segments = torch.max(segments) + 1
        # print(f'num of segments: {n_segments}')
        prototypes = []
        for i in range(n_segments):
            mask = torch.where(segments == i, ones, zeros)
            prototype = self.getFeatures(qry_fts, mask[None, ...])
            prototypes.append(prototype)
        return prototypes

    def restorePredMask(self, superpixel_pred, segments):
        """
        根据计算的超像素prototype相似度，还原预测掩模
        Parameters:
        -----------
            superpixel_pred: expected shape: [num_qry_prototypes(i.e. n_segments)]
            segments: superpixel segmentation mask, expected shape: [H, W]

        Return:
        -------
            mask: expected shape: [H, W]
        """
        zeros = torch.zeros(1).cuda()
        n_segments = torch.max(segments) + 1
        prototypes = []
        mask = torch.zeros_like(segments).cuda().float()
        # print(f'mask: {mask}')
        for i in range(n_segments):
            mask += torch.where(segments == i, superpixel_pred[i], zeros)

        # print(f'mask shape: {mask.shape}')
        return mask[None, ...]

