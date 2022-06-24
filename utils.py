import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric

import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GINConv, GCNConv, Sequential
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn import dense_diff_pool
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader

import matplotlib.pyplot as plt
import networkx as nx
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from math import ceil
from tqdm import tqdm


class GAT(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout,
        return_embeds=False,
    ):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList(
            [GATConv(in_channels=input_dim, out_channels=hidden_dim)]
            + [
                GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
                for i in range(num_layers - 2)
            ]
            + [GATConv(in_channels=hidden_dim, out_channels=output_dim)]
        )

        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for i in range(num_layers - 1)
            ]
        )

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, edge_index)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1

        out = self.convs[-1](x, edge_index)

        return out


class GAT_Graph(torch.nn.Module):
    def __init__(self, args):
        super(GAT_Graph, self).__init__()

        self.gnn_node = GAT(
            args["input_dim"],
            args["hidden_dim"],
            args["hidden_dim"],
            args["num_layers"],
            args["dropout"],
            return_embeds=True,
        )

        self.linear1 = torch.nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.linear2 = torch.nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.linear3 = torch.nn.Linear(args["hidden_dim"], args["output_dim"])

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch

        x = self.gnn_node(x, edge_index)
        # features = torch_scatter.scatter(embed, torch.tensor([0]), dim=0, reduce='max')
        x = global_max_pool(x, batch)  # max

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # out = F.softmax(out, dim=1)

        return x


##############################################################################################################################


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout,
        return_embeds=False,
    ):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)]
            + [
                GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
                for i in range(num_layers - 2)
            ]
            + [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )

        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for i in range(num_layers - 1)
            ]
        )

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, edge_index)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1

        out = self.convs[-1](x, edge_index)

        return out


class GCN_Graph(torch.nn.Module):
    def __init__(self, args):
        super(GCN_Graph, self).__init__()

        self.gnn_node = GCN(
            args["input_dim"],
            args["hidden_dim"],
            args["hidden_dim"],
            args["num_layers"],
            args["dropout"],
            return_embeds=True,
        )

        self.linear1 = torch.nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.linear2 = torch.nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.linear3 = torch.nn.Linear(args["hidden_dim"], args["output_dim"])

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch

        x = self.gnn_node(x, edge_index)
        # features = torch_scatter.scatter(embed, torch.tensor([0]), dim=0, reduce='max')
        x = global_max_pool(x, batch)  # max

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # out = F.softmax(out, dim=1)

        return x


##############################################################################################################################


class GIN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout,
        return_embeds=False,
    ):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList(
            [GINConv(in_channels=input_dim, out_channels=hidden_dim)]
            + [
                GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
                for i in range(num_layers - 2)
            ]
            + [GINConv(in_channels=hidden_dim, out_channels=output_dim)]
        )

        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for i in range(num_layers - 1)
            ]
        )


###########################################################################################################################


###########################################################################################################################


class GNN(torch.nn.Module):
    def __init__(
        self,
        args,
        in_channels,
        hidden_channels,
        out_channels,
        normalize=False,
        lin=True,
    ):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(args["source"]))

        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(args["source"]))

        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            try:
                x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))

            except:
                x = F.relu(self.convs[step](x, adj, mask))

        return x


###########################################################################################################################


def make_batch(dataset_sub, args, num_point):
    np.random.seed(args["seed"])
    if args["transform"] == "radius":
        transform = T.Compose(
            [
                T.FixedPoints(num_point, replace=False),
                T.RadiusGraph(args["threshold"], max_num_neighbors=2000000000),
            ]
        )
    elif args["transform"] == "knn":
        transform = T.Compose(
            [
                T.FixedPoints(num_point, replace=False),
                T.KNNGraph(args["threshold"], force_undirected=True),
            ]
        )
    else:
        raise Exception("need proper transformation ('radius', 'knn')")
    dataset_sub.dataset.transform = transform
    batch = DataLoader(dataset_sub, batch_size=args["batch_size"])
    return batch


def train(
    dataset_train,
    dataset_val,
    model,
    optimizer,
    loss_fn,
    args,
    num_point,
    save_model=False,
):
    filename = (
        f"{args['dataset']}_{args['transform']}_{args['threshold']}_{args['source']}"
    )
    losses = []
    train_acc_list = []
    val_acc_list = []
    best_acc = args["best"]
    pbar = tqdm(range(args["epochs"]))
    for epoch in pbar:
        ## train
        model.train()
        total_loss = 0
        train_correct = 0

        batch_train = make_batch(dataset_train, args, num_point)
        assert next(iter(batch_train)).edge_index != None
        for batch in tqdm(batch_train, leave=False):
            # print(batch)
            _ = batch.to(args["device"])
            optimizer.zero_grad()
            pred = model(batch)
            try:
                label = batch.category
            except:
                label = batch.y
            loss = loss_fn(pred, label)
            total_loss += loss.item() * batch.num_graphs
            pred = torch.argmax(pred, dim=1)
            train_correct += pred.eq(label).sum().item()
            loss.backward()
            optimizer.step()

        train_acc = train_correct / len(batch_train.dataset)
        total_loss /= len(batch_train.dataset)
        losses.append(total_loss)
        train_acc_list.append(train_acc)

        ## evaluate
        model.eval()
        val_correct = 0

        batch_val = make_batch(dataset_val, args, num_point)
        assert next(iter(batch_val)).edge_index != None
        for batch in tqdm(batch_val, leave=False):
            # print(batch)
            _ = batch.to(args["device"])
            with torch.no_grad():
                pred = model(batch)
                pred = torch.argmax(pred, dim=1)
                try:
                    label = batch.category
                except:
                    label = batch.y
            val_correct += pred.eq(label).sum().item()

        val_acc = val_correct / len(batch_val.dataset)
        val_acc_list.append(val_acc)

        if save_model and val_acc > best_acc:
            pbar.write("saving best model...")
            torch.save(model.state_dict(), "model/{}_best.pt".format(filename))
            best_acc = val_acc

        if epoch % 5 == 0:
            pbar.write(
                f"Epoch: {epoch+1}, train_loss: {total_loss:.3f}, train_acc: {train_acc * 100:.2f}%, test_acc: {val_acc * 100:.2f}%"
            )
        pbar  # .set_postfix({'train_loss': format(total_loss, '.3f'), 'train_acc': format(train_acc * 100, '.2f'), 'val_acc': format(val_acc * 100, '.2f')})

    torch.save(
        model.state_dict(), "model/{}_last.pt".format(filename)
    ) if save_model else None
    return losses, train_acc_list, val_acc_list


def test(dataset_val, model, args, num_point):
    model.eval()
    val_correct = 0

    batch_val = make_batch(dataset_val, args, num_point)
    assert next(iter(batch_val)).edge_index != None
    for batch in batch_val:
        # print(batch)
        _ = batch.to(args["device"])
        with torch.no_grad():
            pred = torch.argmax(model(batch), dim=1)
            try:
                label = batch.category
            except:
                label = batch.y
        val_correct += pred.eq(label).sum().item()

    val_acc = val_correct / len(batch_val.dataset)

    print(f"test_acc: {val_acc * 100:.2f}%")

    return val_acc
