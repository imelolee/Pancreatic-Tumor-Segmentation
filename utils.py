import os
import torch.distributed as dist
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import torch


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
    

def read_split_data(train_dir, val_dir):
    train_img_list, train_label_list=[], []
    with open(os.path.join(train_dir, 'ct.txt'),'r') as f:
        for line in f:
            train_img_list.append(line.strip('\n'))
    
    with open(os.path.join(train_dir, 'seg.txt'),'r') as f:
        for line in f:
            train_label_list.append(line.strip('\n'))

    
    val_img_list, val_label_list=[], []
    with open(os.path.join(val_dir, 'ct.txt'),'r') as f:
        for line in f:
            val_img_list.append(line.strip('\n'))
    
    with open(os.path.join(val_dir, 'seg.txt'),'r') as f:
        for line in f:
            val_label_list.append(line.strip('\n'))
    
    return train_img_list, train_label_list, val_img_list, val_label_list



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


