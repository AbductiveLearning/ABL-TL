import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import AverageMeter, accuracy, accuracy_logic, make_and_restore_model
    

def avg_loss(args, outputs, targets, knowledge):
    p_z = F.softmax(outputs, dim=-1)
    losses = []
    loss = 0
    for t in range(knowledge.tnum):
        indices = (t == targets)
        pseudo_label_probs = p_z[indices]
        pseudo_label_probs = pseudo_label_probs.unsqueeze(1)
        abduced_labels = knowledge.rules[t]
        abduced_labels_onthot = F.one_hot(abduced_labels, num_classes=knowledge.snum)
        abduced_labels_onthot = abduced_labels_onthot.expand((pseudo_label_probs.shape[0],) + abduced_labels_onthot.shape)

        p_y = abduced_labels_onthot * pseudo_label_probs
        tloss = -torch.log(p_y.sum(-1) + 1e-32)
        tloss = tloss.mean(-1).mean(-1)
        losses.append(tloss)

        # One way of entropy regularisation
        if args.kb in ['ConjEq']:
            p_y_t = p_y.sum(-1).prod(-1)
            p_y_t_normalised = p_y_t / p_y_t.sum(-1).view(-1, 1)
            p_y_t_averaged = p_y_t_normalised.mean(-2)
            entropy = -(p_y_t_averaged * torch.log(p_y_t_averaged + 1e-32)).mean()
            loss -= entropy * 100
    
    losses = torch.cat(losses)
    loss += losses.mean()

    return loss


def maxp_loss(args, outputs, targets, knowledge):
    p_z = F.softmax(outputs, dim=-1)
    losses = []
    loss = 0
    for t in range(knowledge.tnum):
        indices = (t == targets)
        pseudo_label_probs = p_z[indices]
        pseudo_label_probs = pseudo_label_probs.unsqueeze(1)
        abduced_labels = knowledge.rules[t]
        abduced_labels_onthot = F.one_hot(abduced_labels, num_classes=knowledge.snum)
        abduced_labels_onthot = abduced_labels_onthot.expand((pseudo_label_probs.shape[0],) + abduced_labels_onthot.shape)

        p_y = abduced_labels_onthot * pseudo_label_probs
        p_y = p_y.sum(-1)
        p_Y = p_y.prod(-1)
        p_Y_max = p_Y.max(-1)[0]
        loss_per_target = -torch.log(p_Y_max + 1e-32) / knowledge.rules[t].shape[1]
        losses.append(loss_per_target)

        # One way of entropy regularisation
        if args.kb in ['ConjEq']:
            p_y_t = p_Y
            p_y_t_normalised = p_y_t / p_y_t.sum(-1).view(-1, 1)
            p_y_t_averaged = p_y_t_normalised.mean(-2)
            entropy = -(p_y_t_averaged * torch.log(p_y_t_averaged + 1e-32)).mean()
            loss -= entropy * 100
    
    losses = torch.cat(losses)
    loss += losses.mean()

    return loss


def rand_loss(args, outputs, targets, knowledge):
    p_z = F.softmax(outputs, dim=-1)
    losses = []
    loss = 0
    for t in range(knowledge.tnum):
        indices = (t == targets)
        pseudo_label_probs = p_z[indices]
        pseudo_label_probs = pseudo_label_probs.unsqueeze(1)
        abduced_labels = knowledge.rules[t]
        abduced_labels_onthot = F.one_hot(abduced_labels, num_classes=knowledge.snum)
        abduced_labels_onthot = abduced_labels_onthot.expand((pseudo_label_probs.shape[0],) + abduced_labels_onthot.shape)

        p_y = abduced_labels_onthot * pseudo_label_probs
        p_y = p_y.sum(-1)
        p_Y = p_y.prod(-1)

        rand_indices = torch.randint(0, p_Y.size(1), (p_Y.size(0),), device=p_Y.device)
        p_Y_rand = torch.gather(p_Y, -1, rand_indices.unsqueeze(-1)).squeeze(-1)
        loss_per_target = -torch.log(p_Y_rand + 1e-32) / knowledge.rules[t].shape[1]
        losses.append(loss_per_target)

        # One way of entropy regularisation
        if args.kb in ['ConjEq']:
            p_y_t = p_Y
            p_y_t_normalised = p_y_t / p_y_t.sum(-1).view(-1, 1)
            p_y_t_averaged = p_y_t_normalised.mean(-2)
            entropy = -(p_y_t_averaged * torch.log(p_y_t_averaged + 1e-32)).mean()
            loss -= entropy * 100
    
    losses = torch.cat(losses)
    loss += losses.mean()

    return loss


def mind_loss(args, outputs, targets, knowledge):
    p_z = F.softmax(outputs, dim=-1)
    pseudo_labels_all = outputs.argmax(-1)
    losses = []
    loss = 0
    for t in range(knowledge.tnum):
        indices = (t == targets)
        pseudo_label_probs = p_z[indices]
        pseudo_label_probs = pseudo_label_probs.unsqueeze(1)
        abduced_labels = knowledge.rules[t]
        abduced_labels_onthot = F.one_hot(abduced_labels, num_classes=knowledge.snum)
        abduced_labels_onthot = abduced_labels_onthot.expand((pseudo_label_probs.shape[0],) + abduced_labels_onthot.shape)

        p_y = abduced_labels_onthot * pseudo_label_probs
        p_y = p_y.sum(-1)
        p_Y = p_y.prod(-1)

        pseudo_labels = pseudo_labels_all[indices]
        edit_distances = (pseudo_labels[:, None] != abduced_labels[None, :]).sum(dim=-1)
        mind_indices = edit_distances.argmin(dim=-1)

        p_Y_rand = torch.gather(p_Y, -1, mind_indices.unsqueeze(-1)).squeeze(-1)
        loss_per_target = -torch.log(p_Y_rand + 1e-32) / knowledge.rules[t].shape[1]
        losses.append(loss_per_target)

        # One way of entropy regularisation
        if args.kb in ['ConjEq']:
            p_y_t = p_Y
            p_y_t_normalised = p_y_t / p_y_t.sum(-1).view(-1, 1)
            p_y_t_averaged = p_y_t_normalised.mean(-2)
            entropy = -(p_y_t_averaged * torch.log(p_y_t_averaged + 1e-32)).mean()
            loss -= entropy * 100
    
    losses = torch.cat(losses)
    loss += losses.mean()

    return loss


def tl_loss(args, outputs, targets, knowledge):
    py = F.softmax(outputs, dim=-1)
    Q = knowledge.matrix
    po = torch.matmul(py, Q)

    locations = torch.arange(py.shape[1], device=py.device).unsqueeze(0).repeat(py.shape[0], 1)
    target_location = targets.view(-1, 1) * py.shape[1] + locations

    po_flatten = po.flatten(0, 1)
    target_location_flatten = target_location.flatten(0, 1)

    loss = F.nll_loss(torch.log(po_flatten + 1e-32), target_location_flatten)

    return loss


LOSS_FUNC = {
    'AVG': avg_loss,
    'RAND': rand_loss,
    'MAXP': maxp_loss,
    'MIND': mind_loss,
    'TL': tl_loss,
}


def train(args, knowledge, model, optimizer, loader, epoch):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    
    iterator = tqdm(enumerate(loader), total=len(loader), ncols=100)
    for i, (inputs, facts, targets) in iterator:
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        inputs = inputs.reshape((-1,) + inputs.shape[2:])
        outputs = model(inputs)
        outputs = outputs.view(len(targets), -1, *outputs.shape[1:])
        loss = LOSS_FUNC[args.train_loss](args, outputs, targets, knowledge)
        acc = accuracy_logic(outputs, targets, knowledge)

        loss_logger.update(loss.item(), inputs.size(0))
        acc_logger.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)

    if args.writer is not None:
        descs, vals = ['loss', 'accuracy'], [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            args.writer.add_scalar('train_raw_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg


def train_model(args, knowledge, model, optimizer, loaders):
    raw_loader, train_loader, val_loader, test_loader = loaders
    best_acc = 0.
    for epoch in range(args.epochs):
        _, train_acc_target = train(args, knowledge, model, optimizer, raw_loader, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            train_loss, train_acc = eval_model(args, model, train_loader, epoch, 'train')
            val_loss, val_acc = eval_model(args, model, val_loader, epoch, 'val')
            test_loss, test_acc = eval_model(args, model, test_loader, epoch, 'test')

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }
            if is_best:
                torch.save(checkpoint, args.model_path)

        # Restart if necessary for MAXP and MIND
        if args.train_loss in ['MAXP', 'MIND'] and train_acc_target < 65:
            model = make_and_restore_model(args.dataset, args.arch, args.snum)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model


def eval_model(args, model, loader, epoch=0, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for i, (inp, label) in iterator:
        inp = inp.cuda()
        label = label.cuda()
        logits = model(inp)
        loss = nn.CrossEntropyLoss()(logits, label)
        acc = accuracy(logits, label)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        desc = ('[Evaluation {}] | Loss {:.4f} | Accuracy {:.4f} ||'.format(loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    if args.writer is not None:
        descs, vals = ['loss', 'accuracy'], [loss_logger, acc_logger]
        for k, v in zip(descs, vals):
            args.writer.add_scalar('{}_{}'.format(loop_type, k), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg


def pre_train_model(args, model, loader):

    iterator = iter(loader)
    for times in range(100000):
        loss, acc = eval_model(args, model, loader, times, 'pre')

        # If the flag is 0, no pretraining
        if args.init_acc == 0:
            args.real_init_acc = acc
            break

        if abs(loss) > 10:
            model = make_and_restore_model(args.dataset, args.arch, args.snum)
        if abs(acc - args.init_acc) <= 3:
            args.real_init_acc = acc
            print('\nPre-training finished with initial accuracy: {:.4f}\n'.format(acc))
            break
        elif acc - args.init_acc > 3:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, maximize=True)
        elif acc - args.init_acc < -3:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, maximize=False)

        try:
            inp, label = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inp, label = next(iterator)

        model.train()
        inp = inp.cuda()
        label = label.cuda()
        logits = model(inp)
        loss = nn.CrossEntropyLoss()(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model