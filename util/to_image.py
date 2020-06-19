import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  
unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def save_image(tensor, name='result', normalize=False, file_path='quick_look'):
    os.makedirs(file_path, exist_ok=True)
    if normalize:
        transforms.functional.normalize(tensor, mean=[0, 0, 0], std=[1, 1, 1])
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(f'{file_path}/{name}.jpg')

if __name__ == "__main__":
    a = torch.ones(24, 24)
    # a[0, :] = 2
    # a[1, :] = 2
    a[2, :] = 0
    a[3, :] = 0
    a[4, :] = 0
    a[5, :] = 0
    a[6, :] = 0
    a[7, :] = 0
    save_image(a)
