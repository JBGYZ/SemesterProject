import torch
from .histogenerator import generate_vocabulary, HistoDataset


def dataset_initialization(args):
    """
    Initialize train and test loaders for chosen dataset and transforms.
    :param args: parser arguments (see main.py)
    :return: trainloader, testloader, image size, number of classes.
    """
    torch.manual_seed(args.seed_init)
    vocab = generate_vocabulary(args.embedding_dim, args.one_hot, args.dummy)
    trainset = HistoDataset(args.data_size, vocab)
    return vocab, trainset
