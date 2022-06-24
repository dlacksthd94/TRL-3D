from fnmatch import translate
import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import joblib as jl
import plotly
import plotly.graph_objects as go

# print("import done")
# print("num_cpus:", jl.cpu_count())

# DATA_DIR = os.path.join(os.getcwd(), "data")
# SHAPENETCORE_DIR = os.path.join(DATA_DIR, "ShapeNetCore.v2")
# list_cat = os.listdir(SHAPENETCORE_DIR)
# list_cat.remove("taxonomy.json")
# print("list done")


# def get_pos(SHAPENETCORE_DIR, cat, obj):
#     file_path = os.path.join(
#         SHAPENETCORE_DIR, cat, obj, "models", "model_normalized.obj"
#     )
#     if not os.path.exists(file_path):
#         file_path = os.path.join(SHAPENETCORE_DIR, cat, obj, "model_normalized.obj")
#     if not os.path.exists(file_path):
#         return [None, None, None, cat, obj]
#     with open(file_path, "r") as f:
#         list_line = f.readlines()
#     list_pos = list(filter(lambda line: line.startswith("v "), list_line))
#     list_pos = list(map(lambda line: line[2:].strip().split() + [cat, obj], list_pos))
#     return list_pos


# print("function done")

# list_pos = []
# for cat in tqdm(list_cat[0:1]):
#     list_obj = os.listdir(os.path.join(SHAPENETCORE_DIR, cat))
#     # list_pos_temp = jl.Parallel(n_jobs=-1)(
#     #     jl.delayed(get_pos)(SHAPENETCORE_DIR, cat, obj)
#     #     for obj in tqdm(list_obj, leave=False)
#     # )
#     # list_pos.extend(list_pos_temp)
# print("task done")


# """ """ """ exp """ """ """
# obj = list_obj[100]
# list_pos = get_pos(SHAPENETCORE_DIR, cat, obj)
# df = pd.DataFrame(list_pos, columns=["x", "y", "z", "class", "id"])
# df = df.astype({'x': float, 'y': float, 'z': float})

# def plot_3d(df, edge=True):
#     # G = tg.utils.convert.to_networkx(data, to_undirected=True)
#     # components = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
#     # df = pd.DataFrame(data.pos.numpy(), columns=["x", "y", "z"])
#     # df["parts"] = data.y.numpy()
#     data_fig = []
#     data_temp = go.Scatter3d(
#         x=df["x"],
#         y=df["y"],
#         z=df["z"],
#         # text = ['point #{}'.format(i) for i in range(X.shape[0])],
#         mode="markers",
#         marker=dict(
#             size=3,
#             # color=df["parts"],
#             colorscale=plotly.colors.qualitative.D3,
#             # line=dict(
#             #     color='rgba(217, 217, 217, 0.14)',
#             #     color='rgb(217, 217, 217)',
#             #     width=0.0
#             # ),
#             opacity=0.8,
#         ),
#     )
#     data_fig.append(data_temp)
#     # if edge != None:
#     #     if data.edge_index != None:
#     #         xs, ys, zs = [], [], []
#     #         for e1_idx, e2_idx in tqdm(data.edge_index.T):
#     #             x, y, z = zip(data.pos[e1_idx].numpy(), data.pos[e2_idx].numpy())
#     #             xs += [*x, None]
#     #             ys += [*y, None]
#     #             zs += [*z, None]
#     #         data_temp = go.Scatter3d(
#     #             x=xs,
#     #             y=ys,
#     #             z=zs,
#     #             mode="lines",
#     #             line=dict(color="black", width=0.5),
#     #         )
#     #     else:
#     #         raise Exception("no edges!")
#     # data_fig.append(data_temp)
#     layout = go.Layout(
#         autosize=True,
#         margin=go.layout.Margin(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#         ),
#         scene=dict(
#             xaxis=dict(
#                 nticks=10,
#                 range=[-1, 1],
#             ),
#             yaxis=dict(
#                 nticks=10,
#                 range=[-1, 1],
#             ),
#             zaxis=dict(
#                 nticks=10,
#                 range=[-1, 1],
#             ),
#             aspectmode="cube",
#             dragmode="orbit",
#         ),
#         # paper_bgcolor='#7f7f7f',
#         # plot_bgcolor='#c7c7c7'
#     )
#     fig = go.Figure(data=data_fig, layout=layout)
#     fig.show()
#     # fig.write_html(os.path.join(os.getcwd(), "vis.html"))

# # transform = T.RadiusGraph(r=0.015)
# # # transform = T.KNNGraph(k=5, force_undirected=True)
# # d = ShapeNet("/home/chansonglim/META/PROJECT/data/tg_ShapeNet", transform=transform)
# # for i in range(3):
# #     plot_3d(data=d[i], edge=True)
# plot_3d(df=df, edge=True)

# import plotly.io as pio

# pio.renderers
# pio.renderers.default = "chrome"
# """ """ """  """ """ """

# df = pd.DataFrame(
#     list(chain.from_iterable(list_pos)), columns=["x", "y", "z", "class", "id"]
# )
# df.to_csv(os.path.join(DATA_DIR, "df_pos.csv"), index=False)
# print("file saved")


""" """ """ """ """ """ """  """ """ """ """ """ """ """

from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T

def plot_3d(data, edge=True):
    # G = tg.utils.convert.to_networkx(data, to_undirected=True)
    # components = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
    df = pd.DataFrame(data.pos.numpy(), columns=["x", "y", "z"])
    df["parts"] = data.y.numpy()
    data_fig = []
    data_temp = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        # text = ['point #{}'.format(i) for i in range(X.shape[0])],
        mode="markers",
        marker=dict(
            size=3,
            color=df["parts"],
            colorscale=plotly.colors.qualitative.D3,
            # line=dict(
            #     color='rgba(217, 217, 217, 0.14)',
            #     color='rgb(217, 217, 217)',
            #     width=0.0
            # ),
            opacity=0.8,
        ),
    )
    data_fig.append(data_temp)
    if edge != False:
        if data.edge_index != None:
            xs, ys, zs = [], [], []
            for e1_idx, e2_idx in tqdm(data.edge_index.T):
                x, y, z = zip(data.pos[e1_idx].numpy(), data.pos[e2_idx].numpy())
                xs += [*x, None]
                ys += [*y, None]
                zs += [*z, None]
            data_temp = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="black", width=0.5),
            )
        else:
            raise Exception("no edges!")
    data_fig.append(data_temp)
    layout = go.Layout(
        autosize=True,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        scene=dict(
            xaxis=dict(
                nticks=10,
                range=[-1, 1],
            ),
            yaxis=dict(
                nticks=10,
                range=[-1, 1],
            ),
            zaxis=dict(
                nticks=10,
                range=[-1, 1],
            ),
            aspectmode="cube",
            dragmode="orbit",
        ),
        # paper_bgcolor='#7f7f7f',
        # plot_bgcolor='#c7c7c7'
    )
    fig = go.Figure(data=data_fig, layout=layout)
    fig.show()

transform = T.Compose(
    [
        T.FixedPoints(500),
        T.RadiusGraph(0.05, max_num_neighbors=2000000000),
    ]
)

# transform = T.Compose(
#     [
#         T.FixedPoints(500),
#         T.KNNGraph(5, force_undirected=True),
#     ]
# )

# dataset = ShapeNet(root="data/ShapeNet", transform=transform)
# data = dataset[1]
# data.num_edges
# plot_3d(data)

""" """ """ """ """ """ """  """ """ """ """ """ """ """

def plot_ModelNet(data, edge=True):
    # G = tg.utils.convert.to_networkx(data, to_undirected=True)
    # components = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
    df = pd.DataFrame(data.pos.numpy(), columns=["x", "y", "z"])
    data_fig = []
    data_temp = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        # text = ['point #{}'.format(i) for i in range(X.shape[0])],
        mode="markers",
        marker=dict(
            size=3,
            # line=dict(
            #     color='rgba(217, 217, 217, 0.14)',
            #     color='rgb(217, 217, 217)',
            #     width=0.0
            # ),
            opacity=0.8,
        ),
    )
    data_fig.append(data_temp)
    if edge != False:
        if data.edge_index != None:
            xs, ys, zs = [], [], []
            for e1_idx, e2_idx in tqdm(data.edge_index.T):
                x, y, z = zip(data.pos[e1_idx].numpy(), data.pos[e2_idx].numpy())
                xs += [*x, None]
                ys += [*y, None]
                zs += [*z, None]
            data_temp = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="black", width=0.5),
            )
        else:
            raise Exception("no edges!")
    data_fig.append(data_temp)
    layout = go.Layout(
        autosize=True,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        scene=dict(
            xaxis=dict(
                nticks=10,
                range=[-2, 2],
            ),
            yaxis=dict(
                nticks=10,
                range=[-2, 2],
            ),
            zaxis=dict(
                nticks=10,
                range=[-2, 2],
            ),
            aspectmode="cube",
            dragmode="orbit",
        ),
        # paper_bgcolor='#7f7f7f',
        # plot_bgcolor='#c7c7c7'
    )
    fig = go.Figure(data=data_fig, layout=layout)
    fig.show()
