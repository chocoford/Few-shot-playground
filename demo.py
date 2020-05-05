# 测试模型的参数
from models.resnet import resnet
from models.fewshot import FewShotSeg

# from config import ex
if __name__ == '__main__':
    model = FewShotSeg(pretrained_path='./pretrained_model/vgg16-397923af.pth', cfg={'align': True,})

    # model = resnet(20)
    # model.create_architecture()


    # Find total parameters and trainable parameters
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total num: ", total_num)
    print("trainable num: ",trainable_num)
