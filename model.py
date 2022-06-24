import torch
from torch_geometric.datasets import ShapeNet, ModelNet
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import utils
import os

def load_and_split(args):
    # load dataset
    torch.manual_seed(0)
    if args["dataset"] == "ShapeNet":
        dataset = ShapeNet(root="data/ShapeNet").shuffle()
        args["output_dim"] = 16
    elif args["dataset"] == "ModelNet":
        dataset = ModelNet(
            "data/ModelNet",
            name="40",
            pre_transform=T.Compose(
                [T.NormalizeScale(), T.RandomScale((0.5, 0.5)), T.SamplePoints(1000)]
                # takes about 20 mins... must be pre-transformed to save time (not just transform)
            ),
        ).shuffle()
        args["output_dim"] = 40
    dataset = dataset[:25] # test
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


def experiment(args):
    if not os.path.exists(args['data_dir']):
        os.mkdir(args['data_dir'])
    if not os.path.exists(args['model_dir']):
        os.mkdir(args['model_dir'])
    if not os.path.exists(args['result_dir']):
        os.mkdir(args['result_dir'])
        
    print(
        f"Experiment: dataset={args['dataset']}, source={args['source']}, thres={args['threshold']}"
    )
    (
        dataset_source_train,
        dataset_source_test,
        dataset_target_train,
        dataset_target_test,
    ) = load_and_split(args)

    filename = (
        f"{args['dataset']}_{args['transform']}_{args['threshold']}_{args['source']}"
    )
    df_result = pd.DataFrame()

    # baseline 1
    print("===========================================================================")
    print("TEST WITHOUT PRETRAINING (baseline 1)")
    print("training with target_train & evaluation with target_test")
    print("===========================================================================")
    for num_point in args["target"]:
        print(f"sampling {num_point} point")
        model = utils.GCN_Graph(args).to(args['device'])
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        list_loss, list_train_acc, list_val_acc = utils.train(
            dataset_target_train,
            dataset_target_test,
            model,
            optimizer,
            loss_fn,
            args,
            num_point=num_point,
            save_model=False,
        )
        print(
            f"best train_acc: {max(list_train_acc)}, best test_acc: {max(list_val_acc)}"
        )
        df_result = to_df(
            df_result,
            args,
            list_loss,
            list_train_acc,
            list_val_acc,
            finetune="baseline1",
            num_point=num_point,
        )

    df_result.to_csv(f"result/{filename}.csv", index=False)

    # pretrain
    print("===========================================================================")
    print("PREPARE PRETRAINED MODEL")
    print("pretraining with source_train & validation with source_test")
    print("===========================================================================")
    model = utils.GCN_Graph(args).to(args['device'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    list_loss, list_train_acc, list_val_acc = utils.train(
        dataset_source_train,
        dataset_source_test,
        model,
        optimizer,
        loss_fn,
        args,
        num_point=args["source"],
        save_model=True,
    )
    print(f"best train_acc: {max(list_train_acc)}, best test_acc: {max(list_val_acc)}")
    df_result = to_df(
        df_result,
        args,
        list_loss,
        list_train_acc,
        list_val_acc,
        finetune="original",
        num_point=-1,
    )

    df_result.to_csv(f"result/{filename}.csv", index=False)

    # baseline 2
    print("===========================================================================")
    print("TEST WITHOUT FINETUNING (baseline 2)")
    print("evaluation with target_test on pretrained model")
    print("===========================================================================")
    for num_point in args["target"]:
        print(f"sampling {num_point} point")
        model.load_state_dict(torch.load(f"model/{filename}_best.pt"))
        val_acc = utils.test(dataset_target_test, model, args, num_point)
        df_result = to_df(df_result, args, [np.nan], [np.nan], [val_acc], "baseline2", num_point)

    df_result.to_csv(f"result/{filename}.csv", index=False)

    # finetune all layer
    print("===========================================================================")
    print("FINETUNE ALL LAYERS (experment 1)")
    print("finetuning all layers with target_train & evaluation with target_test")
    print("===========================================================================")
    for num_point in args["target"]:
        print(f"sampling {num_point} point")
        model.load_state_dict(torch.load(f"model/{filename}_best.pt"))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        list_loss, list_train_acc, list_val_acc = utils.train(
            dataset_target_train,
            dataset_target_test,
            model,
            optimizer,
            loss_fn,
            args,
            num_point=num_point,
            save_model=False,
        )
        print(
            f"best train_acc: {max(list_train_acc)}, best test_acc: {max(list_val_acc)}"
        )
        df_result = to_df(
            df_result,
            args,
            list_loss,
            list_train_acc,
            list_val_acc,
            finetune="all",
            num_point=num_point,
        )

    df_result.to_csv(f"result/{filename}.csv", index=False)

    # finetune output layer
    print("===========================================================================")
    print("FINETUNE OUTPUT LAYER (experiment 2)")
    print("finetuning output layer with target_train & evaluation with target_test")
    print("===========================================================================")
    for num_point in args["target"]:
        model.load_state_dict(torch.load(f"model/{filename}_best.pt"))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        for para in model.parameters():
            para.requires_grad = False
        for name, param in model.named_parameters():
            if "linear" in name:
                param.requires_grad = True
        list_loss, list_train_acc, list_val_acc = utils.train(
            dataset_target_train,
            dataset_target_test,
            model,
            optimizer,
            loss_fn,
            args,
            num_point=num_point,
            save_model=False,
        )
        print(
            f"best train_acc: {max(list_train_acc)}, best test_acc: {max(list_val_acc)}"
        )
        df_result = to_df(
            df_result,
            args,
            list_loss,
            list_train_acc,
            list_val_acc,
            finetune="output",
            num_point=num_point,
        )

    df_result.to_csv(f"result/{filename}.csv", index=False)

    return df_result
