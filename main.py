import torch
import model
import argparse

# common settings
NUM_LAYER = 3
INPUT_DIM = 3
HIDDEN_DIM = 32
DROPOUT = 0.1
LR = 0.005
EPOCH = 50
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TARGET = [40, 20, 10]
print("Device: {}".format(DEVICE))

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='transfer learning with 3D point cloud')

    parser.add_argument('-d', '--dataset', type=str, choices=['ShapeNet', 'ModelNet'], required=True,
                        help='The dataset to use')
    parser.add_argument('--transform', type=str, choices=['radius', 'knn'], default='radius',
                        help='The way to create edges')
    parser.add_argument('-t', '--threshold', type=float, choices=[0.1, 0.05, 5, 10], required=True,
                        help='The threshold for creating edges {radius: [0.1, 0.05], knn: [10, 5]}')
    parser.add_argument('-n', '--source', type=int, choices=[1000, 500], required=True,
                        help='The number of point to sample from the source task dataset')
    parser.add_argument('--target', nargs='*', default=[40, 20, 10],
                        help='The number of point to sample from the target task dataset')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYER,
                        help='The number of layers in the model')
    parser.add_argument('--input_dim', type=int, default=INPUT_DIM,
                        help='The size of dimension in input layers')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM,
                        help='The size of dimension in hidden layers')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='The dropout ratio')
    parser.add_argument('--lr', type=float, default=LR,
                        help='The learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCH,
                        help='The number of iterations')
    parser.add_argument('--best', type=int, default=50,
                        help='The minimum accuracy to save the model parameters')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='The batch size')
    parser.add_argument('-s', '--seed', type=int, default=1000,
                        help='The random seed number')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The path to save and load the dataset')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='The path to save and load the model parameter')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='The path to save and load the experiment result')
    return parser.parse_args()

args = dict(parse_args()._get_kwargs())
args['device'] = DEVICE

args = {
    "dataset": "ShapeNet", # ShapeNet or ModelNet
    "transform": "radius",  # radius or knn
    "threshold": 0.05,  # 0.05 or 0.1 for radius / 5 or 10 for knn
    "source": 500,
    "target": NUM_TARGET,
    "device": DEVICE,
    "num_layers": NUM_LAYER,
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "lr": LR,
    "epochs": EPOCH,
    "best": 0,
    "batch_size": BATCH_SIZE,
    "seed": 1000,
    "data_dir": './data',
    "model_dir": './model',
    "result_dir": './result'
}

df_result = model.experiment(args)
