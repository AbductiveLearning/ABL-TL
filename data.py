import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from tqdm import tqdm


transform = transforms.Compose([
    transforms.ToTensor(),
    ])


def prepare_dataset(dataset='MNIST', path='./data', batch_size=64, knowledge=None):
    if dataset == 'MNIST':
        trainset_ori = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        testset_ori = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        trainset_labels = trainset_ori.targets
        testset_labels = testset_ori.targets
    elif dataset == 'EMNIST':
        trainset_ori = datasets.EMNIST(root=path, split='mnist', train=True, download=True, transform=transform)
        testset_ori = datasets.EMNIST(root=path, split='mnist', train=False, download=True, transform=transform)
        trainset_labels = trainset_ori.targets
        testset_labels = testset_ori.targets
        trainset_ori.data = trainset_ori.data.transpose(-1, -2)
        testset_ori.data = testset_ori.data.transpose(-1, -2)
    elif dataset == 'USPS':
        trainset_ori = datasets.USPS(root=path, train=True, download=True, transform=transform)
        testset_ori = datasets.USPS(root=path, train=False, download=True, transform=transform)
        trainset_labels = torch.tensor(trainset_ori.targets)
        testset_labels = torch.tensor(testset_ori.targets)
        from scipy.ndimage import zoom
        trainset_ori.data = zoom(trainset_ori.data, (1, 28/16, 28/16))
        testset_ori.data = zoom(testset_ori.data, (1, 28/16, 28/16))
    elif dataset == 'Fashion':
        trainset_ori = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        testset_ori = datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        trainset_labels = trainset_ori.targets
        testset_labels = testset_ori.targets
    elif dataset == 'Kuzushiji':
        trainset_ori = datasets.KMNIST(root=path, train=True, download=True, transform=transform)
        testset_ori = datasets.KMNIST(root=path, train=False, download=True, transform=transform)
        trainset_labels = trainset_ori.targets
        testset_labels = testset_ori.targets
    elif dataset == 'CIFAR':
        trainset_ori = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        testset_ori = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
        trainset_labels = torch.tensor(trainset_ori.targets)
        testset_labels = torch.tensor(testset_ori.targets)
    elif dataset == 'SVHN':
        trainset_ori = datasets.SVHN(root=path, split='train', download=True, transform=transform)
        testset_ori = datasets.SVHN(root=path, split='test', download=True, transform=transform)
        trainset_labels = torch.tensor(trainset_ori.labels)
        testset_labels = torch.tensor(testset_ori.labels)
        
    symbol_set = knowledge.symbol_set.cpu()
    indices = symbol_set.unsqueeze(1) == trainset_labels
    indices = indices.sum(dim=0).nonzero().squeeze()
    trainset = Subset(trainset_ori, indices)
    indices = symbol_set.unsqueeze(1) == testset_labels
    indices = indices.sum(dim=0).nonzero().squeeze()
    testset = Subset(testset_ori, indices)

    val_num = int(0.8 * len(trainset))
    lengths = [val_num, len(trainset) - val_num]
    train_set, val_set = random_split(dataset=trainset, lengths=lengths, generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_train_loader = DataLoader(dataset=train_set, batch_size=len(train_set), shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader, full_train_loader


def generate_raw_data(ground_facts, data, sym_indices):
    # generate random indices according to ground facts
    random_indices = torch.zeros_like(ground_facts)
    iterator = tqdm(range(random_indices.shape[0]), total=random_indices.shape[0], ncols=100)
    for i in iterator:
        for j in range(random_indices.shape[1]):
            index = torch.randint(0, len(sym_indices[ground_facts[i][j]]), (1,))
            random_indices[i][j] = sym_indices[ground_facts[i][j]][index]

    # let the shape of data to be equal to indices
    data = data.unsqueeze(1)
    data = data.expand((-1, random_indices.shape[1]) + (len(data.shape) - 2) * (-1,))
    random_indices = random_indices.reshape(random_indices.shape + (len(data.shape) - len(random_indices.shape)) * (1,))
    random_indices = random_indices.expand((-1, -1) + data.shape[2:])
    raw_data = torch.gather(data, 0, random_indices)

    return raw_data


def generate_dataset(knowledge, full_loader, sample_size):
    data, labels = next(iter(full_loader))
    sym_indicator = knowledge.symbol_set.unsqueeze(1) == labels
    sym_indices = [sym_indicator[i].nonzero().squeeze() for i in range(len(sym_indicator))]

    sample_sizes = {}
    sample_size_up_to_now = 0
    for tconcept, num_rules in knowledge.num_rules.items():
        sample_sizes[tconcept] = int(num_rules / knowledge.total_rules * sample_size)
        sample_size_up_to_now += sample_sizes[tconcept]
    sample_sizes[tconcept] += sample_size - sample_size_up_to_now
    
    raw_data_all, ground_facts_all, targets_all = [], [], []
    for tconcept, trules in knowledge.rules.items():
        indices = torch.randint(high=knowledge.num_rules[tconcept], size=(sample_sizes[tconcept],))
        ground_facts = trules[indices]
        raw_data = generate_raw_data(ground_facts, data, sym_indices)
        targets = torch.ones(len(raw_data), device=raw_data.device).long() * tconcept

        raw_data_all.append(raw_data)
        ground_facts_all.append(ground_facts)
        targets_all.append(targets)

    raw_data = torch.cat(raw_data_all)
    ground_facts = torch.cat(ground_facts_all)
    targets = torch.cat(targets_all)

    return raw_data.cpu(), ground_facts.cpu(), targets.cpu()


class RawDataset(Dataset):
    def __init__(self, data, facts, targets):
        super().__init__()
        self.data = data
        self.facts = facts
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.data[index], self.facts[index], self.targets[index]


if __name__ == '__main__':
    from pprint import pprint
    from knowledge import KnowledgeBase
    from utils import visualise_images
    
    name = 'ConjEq'
    name = 'Conjunction'
    name = 'Addition'
    kb = KnowledgeBase(name)
    # kb.cuda()
    
    # _, _, _, full_train_loader = prepare_dataset('MNIST', '../datasets/MNIST/', 64, kb)
    # _, _, _, full_train_loader = prepare_dataset('CIFAR', '../datasets/CIFAR/', 64, kb)
    # _, _, _, full_train_loader = prepare_dataset('SVHN', '../datasets/SVHN/', 64, kb)
    # _, _, _, full_train_loader = prepare_dataset('Kuzushiji', '../datasets/Kuzushiji/', 64, kb)
    # _, _, _, full_train_loader = prepare_dataset('Fashion', '../datasets/Fashion/', 64, kb)
    # _, _, _, full_train_loader = prepare_dataset('EMNIST', '../datasets/EMNIST/', 64, kb)
    _, _, _, full_train_loader = prepare_dataset('USPS', '../datasets/USPS/', 64, kb)
    raw_data, ground_facts, targets = generate_dataset(kb, full_train_loader, 100)
    raw_train_set = RawDataset(raw_data, ground_facts, targets)

    raw_loader = DataLoader(dataset=raw_train_set, batch_size=8, shuffle=True, num_workers=0)
    inputs, facts, targets = next(iter(raw_loader))
    visualise_images(inputs, facts, targets)
    print()