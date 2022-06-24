import torch
from tqdm import tqdm
import utils
import os
import pandas as pd
import numpy as np
import models


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

        batch_train = utils.make_batch(dataset_train, args, num_point)
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

        batch_val = utils.make_batch(dataset_val, args, num_point)
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

    batch_val = utils.make_batch(dataset_val, args, num_point)
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


def experiment(args):
    if not os.path.exists(args["data_dir"]):
        os.mkdir(args["data_dir"])
    if not os.path.exists(args["model_dir"]):
        os.mkdir(args["model_dir"])
    if not os.path.exists(args["result_dir"]):
        os.mkdir(args["result_dir"])

    print(
        f"Experiment: dataset={args['dataset']}, source={args['source']}, thres={args['threshold']}"
    )
    (
        dataset_source_train,
        dataset_source_test,
        dataset_target_train,
        dataset_target_test,
    ) = utils.load_and_split(args)

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
        model = models.GCN_Graph(args).to(args["device"])
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        list_loss, list_train_acc, list_val_acc = train(
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
        df_result = utils.to_df(
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
    model = models.GCN_Graph(args).to(args["device"])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    list_loss, list_train_acc, list_val_acc = train(
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
    df_result = utils.to_df(
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
        val_acc = test(dataset_target_test, model, args, num_point)
        df_result = utils.to_df(
            df_result, args, [np.nan], [np.nan], [val_acc], "baseline2", num_point
        )

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
        list_loss, list_train_acc, list_val_acc = train(
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
        df_result = utils.to_df(
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
        list_loss, list_train_acc, list_val_acc = train(
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
        df_result = utils.to_df(
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
