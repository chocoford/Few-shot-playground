"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from .resnet import resnet18, resnet50


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

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, encoder="vgg", gpu_tracker=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        self.gpu_tracker = gpu_tracker

        self.cat_layer = nn.Sequential(
            nn.Conv2d(in_channels=512 * 2, out_channels=512, kernel_size=1, stride=1, padding=1, dilation=1,
                      bias=True),)

        # 新加的
        self.rcu_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.rcu_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layer_final = nn.Conv2d(
            512, 2, kernel_size=1, stride=1, bias=True)

        self.attention_channels = 256

        self.query_conv = nn.Conv2d(
            in_channels=512, out_channels=self.attention_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=512, out_channels=self.attention_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=512, out_channels=self.attention_channels, kernel_size=1)
        self.de_conv = nn.Conv2d(
            in_channels=self.attention_channels, out_channels=512, kernel_size=1)
        # self.q_shotcut_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        # 为新加的层初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # print('Conv2d', m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print('nn.BatchNorm2d: ', m)

        # Encoder: VGG-16
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', resnet50(True))]))

        # exit()

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, mode='train', mutual_enhancement=True):
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

        if mutual_enhancement == True:
            self.augmentFeatureMutually(supp_fts, qry_fts)

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
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

            prototypes = [bg_prototype, ] + fg_prototypes

            bg_prototype = bg_prototype.unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, fts_size[0], fts_size[1])  # tile for cat
            fg_prototype = fg_prototypes[0].unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, fts_size[0], fts_size[1])  # tile for cat

            fg_out = torch.cat([qry_fts[:, 0], fg_prototype], dim=1)
            # [N, C, H', W']
            fg_out = self.cat_layer(fg_out)
            fg_out = self.rcu_1(fg_out)
            fg_out = self.rcu_2(fg_out) + fg_out
            # [N, 1, H', W']
            fg_out = self.layer_final(fg_out)
            pred = fg_out

            # bg_out = torch.cat([qry_fts[:, 0], bg_prototype], dim=1)
            # # [N, C, H', W']
            # bg_out = self.cat_layer(bg_out)
            # bg_out = self.rcu_1(bg_out)
            # # [N, 1, H', W']
            # bg_out = self.layer_final(bg_out)

            # pred = torch.cat([bg_out, fg_out], dim=1)
            # ###### Compute the distance ######

            # qry_fts_s = [bg_out, fg_out]
            # dist = [self.calDist(qry_fts, prototype)
            #         for qry_fts, prototype in zip(qry_fts_s, prototypes)]
            # pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'

            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))

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
            fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)  # 采样到与mask一样的大小

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

    def transferGlobally(self, supp_fts, qry_fts):
        # [B, C, H', W'] i.e. [1, C, H', W']
        supp_fts = supp_fts[0, 0, :]
        # [B, C, H', W'] i.e. [1, C, H', W']
        qry_fts = qry_fts[0]

        _, _, hs, ws = supp_fts.shape
        _, _, hq, wq = qry_fts.shape

        ###直接把support全局迁移###
        s_p = torch.sum(supp_fts, dim=(2, 3)) / (hs*ws)
        q_p = torch.sum(qry_fts, dim=(2, 3)) / (hq*wq)

        bias = s_p - q_p

        supp_fts = supp_fts - self.gamma_t * bias[..., None, None]
        return supp_fts.unsqueeze(dim=0).unsqueeze(dim=0)

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
        s2q_similarity_map = torch.bmm(s_query, q_key)
        q2s_similarity_map = torch.bmm(q_query, s_key)

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
        s = self.de_conv(s.view(1, -1, h, w))  # s.view(1, -1, h, w)
        q = self.de_conv(q.view(1, -1, h, w))  # q.view(1, -1, h, w)
        # print(f's2\'s shape: {s2.shape}')
        s = s + supp_fts
        q = q + qry_fts
        supp_fts = s.unsqueeze(dim=0).unsqueeze(dim=0)
        qry_fts = q.unsqueeze(dim=0)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        ###### End Transfer Features ######
        return supp_fts, qry_fts
