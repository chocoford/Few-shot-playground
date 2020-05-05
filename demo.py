# 测试模型的参数
from models.resnet import resnet
from models.fewshot import FewShotSeg

# from config import ex
if __name__ == '__main__':
    model = FewShotSeg(pretrained_path='./pretrained_model/vgg16-397923af.pth', cfg={'align': True,}, encoder="vgg")

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')