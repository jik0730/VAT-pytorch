import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import Params


def split_datasets(train_dataset, n_labels, n_val):
    """
    Split train dataset into labeled one, unlabeled one, and validation set.
    """
    n_classes = 10
    n_labels_per_class = n_labels / n_classes
    n_val_per_class = n_val / n_classes
    labels_indices = {c: [] for c in range(n_classes)}
    val_indices = {c: [] for c in range(n_classes)}

    rand_indices = [i for i in range(len(train_dataset))]
    # NOTE need seed (fixed)
    np.random.seed(1)
    np.random.shuffle(rand_indices)
    for idx in rand_indices:
        target = int(train_dataset[idx][1])
        if len(labels_indices[target]) < n_labels_per_class:
            labels_indices[target].append(idx)
        elif len(val_indices[target]) < n_val_per_class:
            val_indices[target].append(idx)
        else:
            continue

    labels_set, val_set = [], []
    for indices in labels_indices.values():
        labels_set.extend(indices)
    for indices in val_indices.values():
        val_set.extend(indices)
    assert len(labels_set) == n_labels
    assert len(val_set) == n_val

    return labels_set, val_set


def fetch_dataloaders_MNIST(data_dir, params):
    """
    Fetches the DataLoader objects for MNIST.
    """

    # TODO "transform" for pertutation invariant MNIST
    train_dataset = torchvision.datasets.MNIST(
        data_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(
        data_dir, train=False, transform=transforms.ToTensor(), download=True)
    labels_set, val_set = split_datasets(train_dataset, params.n_labels,
                                         params.n_val)
    unlabels_set = list(set(range(len(train_dataset))) - set(val_set))
    labeled_dataset = torch.utils.data.Subset(train_dataset, labels_set)
    unlabeled_dataset = torch.utils.data.Subset(train_dataset, unlabels_set)
    val_dataset = torch.utils.data.Subset(train_dataset, val_set)

    dataloaders = {}
    dataloaders['label'] = DataLoader(
        labeled_dataset, batch_size=params.nll_batch_size, shuffle=True)
    dataloaders['unlabel'] = DataLoader(
        unlabeled_dataset, batch_size=params.vat_batch_size, shuffle=True)
    dataloaders['val'] = DataLoader(
        val_dataset, batch_size=params.vat_batch_size, shuffle=False)
    dataloaders['test'] = DataLoader(
        test_dataset, batch_size=params.vat_batch_size, shuffle=False)

    return dataloaders


if __name__ == '__main__':
    data_dir = 'data/'
    json_path = os.path.join('experiments/base_model', 'params.json')
    params = Params(json_path)
    dataloaders = fetch_dataloaders_MNIST(data_dir, params)
    dl = dataloaders['label']
    # x, y = dl.__iter__().next()
    # print(x, y)
    # x, y = dl.__iter__().next()
    # print(x, y)
