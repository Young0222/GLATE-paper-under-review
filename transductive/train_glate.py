import argparse
import os.path as osp
import random
import pickle as pkl
import networkx as nx
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import sys
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn.functional import conv2d
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon, WikiCS
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification, LREvaluator

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train(model: Model, x, edge_index, epoch, dataset):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    if dataset == 'PubMed':
        loss, tau = model.loss(z1, z2, epoch, batch_size=1024)
    else:
        loss, tau = model.loss(z1, z2, epoch, batch_size=0)

    loss.backward()
    optimizer.step()

    return loss.item(), tau
    

def visualization(encoder_model: Model, x, edge_index, y, seed, index):
    encoder_model.eval()
    z = encoder_model(x, edge_index).to("cpu").detach().numpy()
    z = np.squeeze(z)
    y = y.to("cpu").detach().numpy()
    x_tsne = TSNE(n_components=2).fit_transform(z)
    np.savetxt(str(index)+"_x.txt", x_tsne)
    np.savetxt(str(index)+"_y.txt", y)


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    if torch.cuda.is_available():
        assert args.gpu_id in range(0, 8)
        torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    for _ in range(1, 2):
        seed = random.randint(1,999999)
        print("seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        learning_rate = config['learning_rate']
        num_hidden = config['num_hidden']
        num_proj_hidden = config['num_proj_hidden']
        activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
        base_model = ({'GCNConv': GCNConv})[config['base_model']]
        num_layers = config['num_layers']
        drop_edge_rate_1 = config['drop_edge_rate_1']
        drop_edge_rate_2 = config['drop_edge_rate_2']
        drop_feature_rate_1 = config['drop_feature_rate_1']
        drop_feature_rate_2 = config['drop_feature_rate_2']
        num_epochs = config['num_epochs']
        weight_decay = config['weight_decay']
        tau1 = config['tau1']
        tau2 = config['tau2']

        def get_dataset(path, name):
            assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'CS', 'Physics', 'Computers', 'Photo', 'Wiki']
            name = 'dblp' if name == 'DBLP' else name
            if name in ['Cora', 'CiteSeer', 'PubMed']: 
                return Planetoid(
                path,
                name)
            elif name in ['CS', 'Physics']:
                return Coauthor(
                path,
                name,
                transform=T.NormalizeFeatures())
            elif name in ['Computers', 'Photo']:
                return Amazon(
                path,
                name,
                transform=T.NormalizeFeatures())
            elif name in ['Wiki']:
                return WikiCS(
                path,
                transform=T.NormalizeFeatures())
            else:
                return CitationFull(
                path,
                name)

        path = osp.join(osp.expanduser('~'), 'datasets')
        print("path: ", path)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau1, tau2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        start = t()
        prev = start
        tau_list = []
        for epoch in range(1, num_epochs + 1):
            loss, tau = train(model, data.x, data.edge_index, epoch, args.dataset)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            tau_list.append(tau)
            prev = now
        # visualization(model, data.x, data.edge_index, data.y, seed, epoch)