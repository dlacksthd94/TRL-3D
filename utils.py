import torch
from torch_geometric.datasets import ShapeNet, ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np


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


def to_df(
    df_result, args, list_loss, list_train_acc, list_val_acc, finetune, num_point
):
    df_temp = pd.DataFrame(
        {
            "dataset": [args["dataset"]] * len(list_train_acc),
            # "model": [args["model"]] * len(list_train_acc),
            "transform": [args["transform"]] * len(list_train_acc),
            "finetune": [finetune] * len(list_train_acc),
            "threshold": [args["threshold"]] * len(list_train_acc),
            "num_source": [args["source"]] * len(list_train_acc),
            "num_target": [num_point] * len(list_train_acc),
            "seed": [args["seed"]] * len(list_train_acc),
            "loss": list_loss,
            "train_acc": list_train_acc,
            "val_acc": list_val_acc,
            "epoch": list(range(len(list_train_acc))),
        }
    )
    return pd.concat([df_result, df_temp], ignore_index=True)


def load_and_split(args):
    # load dataset
    torch.manual_seed(0)
    if args["dataset"] == "ShapeNet":
        print("Takes about 15 mins to download and preprocess.(only the first time)")
        dataset = ShapeNet(root="data/ShapeNet").shuffle()
        args["output_dim"] = 16
    elif args["dataset"] == "ModelNet":
        print("Takes about 30 mins to download and preprocess.(only the first time)")
        dataset = ModelNet(
            "data/ModelNet",
            name="40",
            pre_transform=T.Compose(
                [T.NormalizeScale(), T.RandomScale((0.5, 0.5)), T.SamplePoints(1000)]
            ),
        ).shuffle()
        args["output_dim"] = 40
    dataset = dataset[:25]  # test
    # dataset[0].category # check randomness

    # split dataset
    ratio_source = 0.7
    ratio_train = 0.7
    num_source = int(len(dataset) * ratio_source)
    num_target = len(dataset) - num_source
    num_source_train = int(num_source * ratio_train)
    num_source_test = num_source - num_source_train
    num_target_train = int(num_target * ratio_train)
    num_target_test = num_target - num_target_train

    (
        dataset_source_train,
        dataset_source_test,
        dataset_target_train,
        dataset_target_test,
    ) = torch.utils.data.random_split(
        dataset, [num_source_train, num_source_test, num_target_train, num_target_test]
    )
    # len(dataset_source_train) # check split
    # len(dataset_target_test)

    return (
        dataset_source_train,
        dataset_source_test,
        dataset_target_train,
        dataset_target_test,
    )
