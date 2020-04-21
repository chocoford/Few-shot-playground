import numpy as np
import os
import matplotlib

class Visualizer():
    """
    使结果可视化
    """

    def __init__(self):
        self.rootDir = "./results"
        self.imgDir = os.path.join(self.rootDir, "img")
        self.i = 0


    def record(self, query_image, query_pred, target, labels=None, n_run=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args
        ----------
            query_pred:
                predicted mask array, expected shape is 1 x (1+way) x H x W

            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        # 获得预测mask即每一个像素最可能的类，[H, W]
        pred = np.array(query_pred.argmax(dim=1)[0].cpu()) 

        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0 

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            dirName = f'{rootDir}/{label}'
            np.save(f'{dirName}/', query_image.numpy())


    def saveImg(self, img, name):
        """
        save query image

        Parameters
        -----------
            img: 

        """
        print(img.cpu().numpy())
        print(img.cpu().shape)
        # matplotlib.image.imsave(f'{self.imgDir}/{name}.png', img.cpu().numpy())


    # def mkdir(path):
    #     isExist = os.path.isExist(path)
    #     if not isExist:
    #         os.mkdir(path)