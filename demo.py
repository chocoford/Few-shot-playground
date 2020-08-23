import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import torchvision

# 测试模型的参数
from models.resnet import resnet18
from models.fewshot import FewShotSeg

# from config import ex
if __name__ == '__main__':
    # model = FewShotSeg(pretrained_path='./pretrained_model/vgg16-397923af.pth', cfg={'align': True,}, encoder="vgg")

    # # Find total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    resnet50 = torchvision.models.resnet50(pretrained=True)
    saved_state_dict = resnet50.state_dict()
    # print(saved_state_dict.keys())
    for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer
        # print(i)
        # print(saved_state_dict[i])
        i_parts = i.split('.')
        print('.'.join(i_parts))
        # print(saved_state_dict['.'.join(i_parts)])
        # break

    def load_resnet_param(model, pretrained_model, stop_layer='layer4'):
        """
        加载指定参数
        原理：根据state_dict()的返回值（如"layer2.3.bn3.running_var"）判断哪些层加载，忽略哪些层
        """
        saved_state_dict = pretrained_model.state_dict()
        new_params = model.state_dict().copy()

        for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer
            i_parts = i.split('.')
            if not i_parts[0] == stop_layer:
                new_params['.'.join(i_parts)] = saved_state_dict[i]
            else:
                break
        model.load_state_dict(new_params)

        return model

    # model = Res_Deeplab(num_classes=2).cuda()
    # model=load_resnet_param(model)

    # query_rgb = torch.FloatTensor(1,3,321,321).cuda()
    # support_rgb = torch.FloatTensor(1,3,321,321).cuda()
    # support_mask = torch.FloatTensor(1,1,321,321).cuda()

    # history_mask=(torch.zeros(1,2,50,50)).cuda()

    # pred = (model(query_rgb,support_rgb,support_mask,history_mask))
