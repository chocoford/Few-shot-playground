# -*- coding: utf-8 -*-
"""Training Script"""
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from util.loss import entropy_loss
from util.metric import Metric
from config import ex
from util.gpu_mem_track import MemTracker
import inspect

import matplotlib.pyplot as plt


@ex.automain
def main(_run, _config, _log):
    """
    从15个训练类里随机抽取一个类，然后抽取两张图片分别作为support和query
    """
    frame = inspect.currentframe()
    gpu_tracker = MemTracker(frame, path=_run.observers[0].dir+'/')

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(
        pretrained_path=_config['path']['init_path'], cfg=_config['model'], encoder=_config['encoder'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    # 和OSLSM的机制一样，labels[i]即pascal-5i
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    train_ds = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,  # 与val唯一不同
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )

    # valid_ds = make_data(
    #     base_dir=_config['path'][data_name]['data_dir'],
    #     split=_config['path'][data_name]['data_split'],
    #     transforms=transforms,
    #     to_tensor=ToTensorNormalize(),
    #     labels=labels[:3],
    #     max_iters=_config['n_steps'] * _config['batch_size'],
    #     n_ways=_config['task']['n_ways'],
    #     n_shots=_config['task']['n_shots'],
    #     n_queries=_config['task']['n_queries']
    # )

    train_dl = DataLoader(
        train_ds,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    # valid_dl = DataLoader(
    #     train_ds,
    #     batch_size=_config['batch_size'],
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=True
    # )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(
        optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0, 'val_loss': 0}
    train_losses = []
    align_losses = []
    avg_train_losses = []
    avg_align_losses = []
    val_losses = []
    m_iou = []
    metric = Metric(max_label=20, n_runs=1)

    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(train_dl):
        model.train()
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
        # query_labels = torch.cat(
        #     [((query_label + 255) % 255).float().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()

        # # entropy loss
        support_label = [[shot[f'fg_mask'].long().cuda() for shot in way]
                    for way in sample_batched['support_mask']]
        support_label = torch.cat(support_label[0], dim=0)
        support_pred = model(support_images, support_fg_mask, support_bg_mask,
                                support_images[0], gpu_tracker, mutual_enhancement=True)
        # ent_loss = entropy_loss(support_pred)
        support_loss = criterion(support_pred, support_label)
        # query_loss = support_loss
        loss = support_loss #+ 0.001 * ent_loss
        loss.backward()
        # end

        query_pred = model(support_images, support_fg_mask, support_bg_mask,
                           query_images, gpu_tracker)
        # query_pred = query_pred[0]
        query_loss = criterion(query_pred, query_labels)
        # + 0.001 * entropy_loss(F.softmax(query_pred, dim=0))
        loss = query_loss
        # with torch.no_grad():
        #     # if query_loss > 1:

        #     # print(f'query_loss: {query_loss}, entropy_loss: {entropy_loss(F.softmax(query_pred))}')
        #     print(f'query_loss: {query_loss}')
        #     # print(query_labels[0].nonzero().shape[0])
        #     # print(query_pred[0].nonzero().shape[0])
        loss.backward()
        optimizer.step()
        scheduler.step()
        # query_pred = model(support_images, support_fg_mask, support_bg_mask,
        #                         query_images)

        # loss = entropy_loss(query_pred)
        # loss.backward()
        # with torch.no_grad():
        #     print(f'query_loss: {query_loss}, entropy_loss: {entropy_loss(query_pred)}')
        #     print(query_labels[0].nonzero().shape[0])
        #     print(query_pred[0].nonzero().shape[0])
        # optimizer.step()
        # scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        # align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        # _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        # log_loss['align_loss'] += align_loss
        # train_losses.append(query_loss)
        # align_losses.append(align_loss)
        # avg_train_losses.append(log_loss['loss'] / (i_iter + 1))
        # avg_align_losses.append(log_loss['align_loss'] / (i_iter + 1))

        iou = metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                            np.array(query_labels[0].cpu()),
                            labels=list(sample_batched['class_ids']), n_run=0)

        # val loss
        # model.eval()
        # with torch.no_grad():
        #     val_query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
        #                                            query_images)
        #     val_loss = criterion(val_query_pred, query_labels)

        #     val_loss = val_loss.detach().data.cpu().numpy()
        #     _run.log_scalar('val_loss', val_loss)
        #     log_loss['val_loss'] += val_loss
        #     val_losses.append(log_loss['val_loss']/(i_iter + 1))

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            avg_loss = log_loss['loss'] / (i_iter + 1)
            # avg_align_loss = log_loss['align_loss'] / (i_iter + 1)
            train_losses.append(query_loss)
            avg_train_losses.append(log_loss['loss'] / (i_iter + 1))

            _, _, meanIoU, _ = metric.get_mIoU(labels=sorted(labels))
            print(
                f'step {i_iter+1}: loss: {query_loss}, avg_loss: {avg_loss}, m_iou: {meanIoU}')

            m_iou.append(meanIoU)

            # save the loss pic
            x = [i for i in range(1, len(avg_train_losses)+1)]
            fig = plt.figure(figsize=(19.2, 10.8))
            plt.plot(x, m_iou, label='mean iou')
            plt.plot(x, avg_train_losses, label='average train loss')
            if _config['model']['align'] == True:
                plt.plot(x, align_losses, label='align loss')
                plt.plot(x, avg_align_losses, label='average align loss')
            plt.xlabel('iteration (hundreds)')
            plt.ylabel('loss')
            plt.title("training loss")
            plt.legend()
            plt.savefig(f'{_run.observers[0].dir}/loss.png')
            plt.close()

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
