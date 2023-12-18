import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
from pprint import pprint

from utils import set_seed, make_and_restore_model, save_results
from knowledge import KnowledgeBase
from data import prepare_dataset, generate_dataset, RawDataset
from train import train_model, eval_model, pre_train_model


def main(args):
    set_seed(args.seed)
    knowledge = KnowledgeBase(name=args.kb)

    args.snum, args.tnum, args.rank = knowledge.snum, knowledge.tnum, knowledge.rank

    train_loader, val_loader, test_loader, full_train_loader = prepare_dataset(args.dataset, args.data_path, args.batch_size, knowledge)
    raw_data, ground_facts, targets = generate_dataset(knowledge, full_train_loader, args.sample_size)
    raw_train_set = RawDataset(raw_data, ground_facts, targets)
    raw_loader = DataLoader(dataset=raw_train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    knowledge.cuda()

    set_seed(args.repeat)
    model = make_and_restore_model(args.dataset, args.arch, args.snum)
    model = pre_train_model(args, model, val_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loaders = (raw_loader, train_loader, val_loader, test_loader)
    train_model(args, knowledge, model, optimizer, loaders)

    model = make_and_restore_model(args.dataset, args.arch, args.snum, path=args.model_path)
    loss, acc = eval_model(args, model, test_loader)

    args.final_acc = acc
    save_results(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Abductive learning on benchmark tasks')

    parser.add_argument('--train_loss', default='TL', type=str, choices=['AVG', 'RAND', 'MAXP', 'MIND', 'TL'])

    parser.add_argument('--dataset', default='MNIST', type=str, choices=['MNIST', 'EMNIST', 'USPS', 'Kuzushiji', 'Fashion'])
    parser.add_argument('--arch', default='ResNet18', type=str, choices=['MLP', 'ResNet18', 'DenseNet121'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--repeat', default=0, type=int)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--init_acc', default=0, type=int)

    parser.add_argument('--kb', default='ConjEq', type=str, choices=['ConjEq', 'Conjunction', 'Addition'])
    args = parser.parse_args()

    args.lr = 1e-3
    args.batch_size = 256
    args.log_gap = 1
    args.sample_size = 500000
    args.epochs = 100

    args.data_path = os.path.join('../datasets/', args.dataset)
    args.out_dir = os.path.join('./results/', args.dataset)
    args.exp_name = '{}-{}-{}-Init{}-seed{}-r{}'.format(args.train_loss, args.kb, args.arch, args.init_acc, args.seed, args.repeat)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')

    pprint(vars(args))
    args.writer = SummaryWriter(args.tensorboard_path)
    args.writer.add_text('arg', str(args))

    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gpuid = gpus[args.gpuid % 8]

    torch.cuda.set_device(gpuid)
    main(args)






