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

        # Encoder: VGG-16
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))

        self.cat_layer = nn.Sequential(
            nn.Conv2d(in_channels=512 * 2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),)


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, mode='train'):
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


            fg_prototypes = fg_prototypes[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, fts_size[0], fts_size[1])  # tile for cat
            out = torch.cat([qry_fts[:, 0], fg_prototypes], dim=1)

            # [N, 2, H', W']
            out = self.cat_layer(out)

            # ###### Compute the distance ######

            # prototypes = [bg_prototype, ] + fg_prototypes

            # dist = [self.calDist(query_fts, prototype)
            #         for prototype in prototypes]
            # pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
    
            outputs.append(F.interpolate(out, size=img_size, mode='bilinear'))

        output = torch.stack(outputs, dim=1)  # N x B x 2 x H x W
        output = output.view(-1, *output.shape[2:])
        return output

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

    def augmentFeatureMutually(self, supp_fts, qry_fts):
        """
            Transfer Features
            2020年6月28日，目前只handle 1 way 1 shot的情况


        """
        # [B, C, H', W'] i.e. [1, C, H', W']
        supp_fts = supp_fts[0, 0, :]
        # [B, C, H', W'] i.e. [1, C, H', W']
        qry_fts = qry_fts[0]

        # [B, C, H', W'] i.e. [1, C, H', W']
        s_fts = F.relu(supp_fts)
        # [B, C, H', W'] i.e. [1, C, H', W']
        q_fts = F.relu(qry_fts)

        _, c, h, w = s_fts.shape

        s_query = self.query_conv(s_fts)
        q_query = self.query_conv(q_fts)
        s_key = self.key_conv(s_fts)
        q_key = self.key_conv(q_fts)
        # s_query = s_fts
        # q_query = q_fts
        # s_key = s_fts
        # q_key = q_fts

        ######归一化#####
        def emb_normalize(emb):
            _, _, h, w = emb.shape
            buffer = torch.pow(emb, 2)
            norm = torch.sqrt(torch.sum(buffer, 1).add_(1e-10))

            return torch.div(emb, norm.view(-1, 1, h, w).expand_as(emb))

        s_query = emb_normalize(s_query)
        q_query = emb_normalize(q_query)
        s_key = emb_normalize(s_key)
        q_key = emb_normalize(q_key)
        ####结束归一化###


        # [1, C, H' x W']
        s_key = s_key.view(1, -1, h*w)
        q_key = q_key.view(1, -1, h*w)
        # [1, H' x W', C]
        s_query = s_query.view(1, -1, h*w).permute(0, 2, 1)
        q_query = q_query.view(1, -1, h*w).permute(0, 2, 1)
        # [1, H' x W', H' x W']
        s2q_similarity_map = F.softmax(torch.bmm(s_query, q_key), dim=1)
        q2s_similarity_map = F.softmax(torch.bmm(q_query, s_key), dim=1)

        #########可视化相似图###########
        # save_image(qry_imgs[0][0], name=f'qry_img', normalize=True)
        # for i, m in enumerate(s2q_similarity_map[0]):
        #     result = m.view(h, w)
        #     if i < 2:
        #         print(result)
        #         print(result.max())
        #     save_image(result, name=f'result_{i%w}_{int(i/w)}')
        # exit()
        ##########结束end###########

        # similarity_map = torch.stack([F.cosine_similarity(s1[..., i][..., None], q, dim=0)
        #                   for i in range(h*w)])
        # [H' x W', C]
        # s2 = torch.stack([torch.sum(q * similarity_map[i, :], dim=1) for i in range(h*w)])
        
        # [1, C, H' x W']
        s_value = self.value_conv(s_fts).view(1, -1, h*w)
        q_value = self.value_conv(q_fts).view(1, -1, h*w)
        # [1, C, H' x W']
        s = torch.bmm(q_value, s2q_similarity_map.permute(0, 2, 1))
        q = torch.bmm(s_value, q2s_similarity_map.permute(0, 2, 1))
        # [C, H', W']
        # s2 = s2.view(h, w, c).permute(2, 0, 1) / (h*w)
        s = s.view(1, -1, h, w)
        q = q.view(1, -1, h, w)
        # print(f's2\'s shape: {s2.shape}')
        s = self.gamma_s * s + supp_fts
        q = self.gamma_q * q + qry_fts
        supp_fts = s.unsqueeze(dim=0).unsqueeze(dim=0)
        qry_fts = q.unsqueeze(dim=0)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        ###### End Transfer Features ######
        return supp_fts, qry_fts
