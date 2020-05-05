# 测试模型的参数
from models.resnet import resnet
from models.fewshot import FewShotSeg

# from config import ex

# @ex.automain
# def main(_run, _config, _log):
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])

    # model = resnet(20)
    # model.create_architecture()


    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')