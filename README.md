# Environment Setup and Package installation

### case1: GSDS remote server
1. create conda environment
2. run commands below

    conda install -y --file requirements.txt
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    conda install -y pyg -c pyg

- recommend to use 'conda install' instead of 'pip install' (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) **

### case2: other machine
1. create conda environment
2. run command

    conda install -y --file requirements.txt

3. refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


# Experiment

run command

    python main.py [-h] -d {ShapeNet,ModelNet} [--transform {radius,knn}] -t
               {0.1,0.05,5,10} -n {1000,500} [--target [TARGET [TARGET ...]]]
               [--num_layers NUM_LAYERS] [--input_dim INPUT_DIM]
               [--hidden_dim HIDDEN_DIM] [--dropout DROPOUT] [--lr LR]
               [--epochs EPOCHS] [--best BEST] [--batch_size BATCH_SIZE]
               [-s SEED] [--data_dir DATA_DIR] [--model_dir MODEL_DIR]
               [--result_dir RESULT_DIR]

for help

    python main.py -h