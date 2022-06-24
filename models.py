import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, GCNConv, DenseGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


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


###


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


###


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
