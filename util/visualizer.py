import numpy as np
import os
from PIL import Image

class Visualizer():
    """
    使结果可视化
    """

    def __init__(self):
        self.rootDir = "./results"
        self.imgDir = os.path.join(self.rootDir, "img")
        self.i = 0


    def visualize(self, query_images, query_pred, target, name, labels=None, n_run=None):
        """
        通过blend原图和mask从而可视化结果。

        Args
        ----------
            query_images: expect shape [B x 3 x H x W]

            query_pred:
                predicted mask array, expected shape is 1 x (1+way) x H x W

            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        # 获得预测mask即每一个像素最可能的类，[H, W]
        pred = np.array(query_pred.argmax(dim=1)[0].cpu()) 
        img_size = pred.shape[-2:]
        # print(img_size)
        assert pred.shape == target.shape

        pred_visual = np.zeros((img_size[0], img_size[1], 3))

        color = [np.array([254, 67, 101]), np.array([30, 41, 61])]

        query_image = query_images[0]

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_visual[idx[0], idx[1], :] = color[j]
            # pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # # Get the location of the pixels that are class j in ground truth
            # idx = np.where(target == j)
            # target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

        im_mask = Image.fromarray(np.uint8(pred_visual))
        im_query = Image.fromarray(query_image.cpu().numpy().transpose(1, 2, 0))
        im_blend = self.blend(im_query, im_mask)
        self.saveImgs(im_mask, name)


    def blend(self, img1, img2):
        return Image.blend(img1, img2, 0.618)

    def saveImgs(self, im, name):
        """
        save query image with mask

        Parameters
        -----------
            imgs: 

        """
        im.save(f'{self.imgDir}/{name}.png')


    # def mkdir(path):
    #     isExist = os.path.isExist(path)
    #     if not isExist:
    #         os.mkdir(path)