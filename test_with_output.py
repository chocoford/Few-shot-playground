"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.visualizer import Visualizer
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex
from skimage.segmentation import slic, mark_boundaries

@ex.automain
def main(_run, _config, _log):
    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']] # {1-5}
    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)

    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    visualizer = Visualizer()
    with torch.no_grad():
        # for run in range(_config['n_runs']):
        run = 0
        _log.info(f'### Run {run + 1} ###')
        set_seed(_config['seed'] + run)

        _log.info(f'### Load data ###')
        dataset = make_data(
            base_dir=_config['path'][data_name]['data_dir'],
            split=_config['path'][data_name]['data_split'],
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            max_iters=_config['n_steps'] * _config['batch_size'],
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            n_queries=_config['task']['n_queries']
        )
        if _config['dataset'] == 'COCO':
            coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
        testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                num_workers=1, pin_memory=True, drop_last=False)
        _log.info(f"Total # of Data: {len(dataset)}")


        for i, sample_batched in enumerate(tqdm.tqdm(testloader)):
            if _config['dataset'] == 'COCO':
                label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
            else:
                label_ids = list(sample_batched['class_ids'])
            support_images = [[shot.cuda() for shot in way]
                              for way in sample_batched['support_images']]
            suffix = 'scribble' if _config['scribble'] else 'mask'

            if _config['bbox']:
                support_fg_mask = []
                support_bg_mask = []
                for i, way in enumerate(sample_batched['support_mask']):
                    fg_masks = []
                    bg_masks = []
                    for j, shot in enumerate(way):
                        fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                    sample_batched['support_inst'][i][j])
                        fg_masks.append(fg_mask.float().cuda())
                        bg_masks.append(bg_mask.float().cuda())
                    support_fg_mask.append(fg_masks)
                    support_bg_mask.append(bg_masks)
            else:
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

            if _config['superpixel_preSeg'] == True:
                std= [0.229, 0.224, 0.225]
                mean=[0.485, 0.456, 0.406]
                query_image = torch.cat([query_image[0] for query_image in sample_batched['query_images']])
                query_image[0] = query_image[0,:] * std[0] + mean[0]
                query_image[1] = query_image[1,:] * std[1] + mean[1]
                query_image[2] = query_image[2,:] * std[2] + mean[2]
                image = query_image.permute(1,2,0).double().numpy()
                segments = torch.from_numpy(slic(image, n_segments=500)) # [H, W]


            query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                  query_images, segments)
            # _log.info(f'shape of query_images: {query_images[0].shape}')
            query_images_t = [query_image_t.cuda()
                            for query_image_t in sample_batched['query_images_t']]
            # _log.info(f'shape of query_images_t: {query_images_t[0].shape}')


            

            visualizer.visualize(query_images_t[0], query_pred, np.array(query_labels[0].cpu()), str(i), labels=label_ids)
                

    #             metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
    #                           np.array(query_labels[0].cpu()),
    #                           labels=label_ids, n_run=run)