import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from models.mlp import MLP
from models.resnet import ResNet18
from models.densenet import DenseNet121


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_and_restore_model(dataset, arch, snum, path=None):
    if dataset in ['MNIST', 'EMNIST', 'USPS', 'Kuzushiji', 'Fashion']:
        if arch == 'MLP':
            model = MLP(28*28, snum)
        elif arch == 'ResNet18':
            model = ResNet18(snum, channals=1)
        elif arch == 'DenseNet121':
            model = DenseNet121(snum, channals=1)
    elif dataset in ['CIFAR', 'SVHN']:
        if arch == 'MLP':
            model = MLP(3*32*32, snum)
        elif arch == 'ResNet18':
            model = ResNet18(snum, channals=3)
        elif arch == 'DenseNet121':
            model = DenseNet121(snum, channals=3)
    
    if path is not None:
        print('\n=> Loading checkpoint {}'.format(path))
        checkpoint = torch.load(path)
        info_keys = ['epoch', 'train_acc', 'val_acc', 'test_acc']
        info = {k: checkpoint[k] for k in info_keys}
        pprint(info)
        model.load_state_dict(checkpoint['model'])
    
    model = model.cuda()
    return model


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)


def accuracy_logic(logits, targets, knowledge):
    pseudo_facts = logits.argmax(dim=-1)
    outputs = knowledge(pseudo_facts)
    correct = outputs.eq(targets).sum().item()
    return correct * 100. / targets.size(0)


def visualise_images(images, labels, targets, savepath='./tmp.png'):
    num_images, num_columns, num_channels, height, width = images.shape
    fig, axes = plt.subplots(num_images, num_columns + 1, figsize=((num_columns + 1)*3, num_images*3))
    for i in range(num_images):
        for j in range(num_columns):
            if num_channels == 3:
                image = images[i, j].permute(1, 2, 0)
            else:
                image = images[i, j].squeeze(0)
            axes[i, j].imshow(image, cmap='gray' if num_channels == 1 else None)
            axes[i, j].set_title('Label: {}'.format(labels[i, j].item()), fontsize=35)
            axes[i, j].axis('off')
        axes[i, -1].axis('off')
        axes[i, -1].text(0.5, 0.5, 'Target: {}'.format(int(targets[i].item())), 
                         fontsize=35, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(savepath)


def save_results(args):
    keys = ['model_path', 'final_acc', 'real_init_acc', 'init_acc', 'dataset', 'arch', 'seed', 'repeat', 'kb', 'rank', 'tnum', 'snum']
    values = []
    for key in keys:
        values.append(args.__dict__[key])

    import csv, os
    csv_fn = os.path.join(args.out_dir, args.exp_name + '.csv')
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)
    print('=> csv file is saved at [{}]'.format(csv_fn))