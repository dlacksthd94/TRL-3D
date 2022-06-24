# Environment Setup and Package installation

### case1: GSDS remote server
1. create conda environment
2. run commands below::
    conda install -y --file requirements.txt
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    conda install -y pyg -c pyg
** recommend to use 'conda install' instead of 'pip install' (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) **

### case2: other machine
1. create conda environment
2. run command::
    conda install -y --file requirements.txt
3. refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


# Experiment

run command