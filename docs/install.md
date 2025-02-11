# Installation Instruction

In this document, we provide instructions of how to properly install this codebase. We highly recommend using a conda environment to simplify set up.

(This document is still under active polishing and construction, ping me in discord if you need help or want to contribute).

## Setup Conda Environment

You can skip this section if you are not using conda virtual environment.

```
conda create -y -n hora python=3.8
conda activate hora
# (optional) install gdown for downloading
conda install -c conda-forge gdown urllib3
# pytorch, 1.10 and cu10.2 is used for development, higher version should also work
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

## IsaacGym

Download IsaacGym Preview 3.0 ([Download](https://drive.google.com/file/d/1oK-QMZ40PO60PFWWsTmtK5ToFDkbL6R0/)), then follow the installation instructions in the documentation. We provide the bash commands what we did during development.

```
gdown 1oK-QMZ40PO60PFWWsTmtK5ToFDkbL6R0 -O isaac3.tar.gz
tar -xzvf isaac3.tar.gz
cd isaacgym/python
pip install -e .
```

Note, there may be some errors about `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.` This is mainly caused by `dm-control` which can be ignored at this stage. As long as you can run some isaacgym examples, you are good to go:

```
python examples/joint_monkey.py
```

## Hora Repository

```
git clone https://github.com/HaozhiQi/hora
cd hora
pip install -r requirements.txt
```

Next, follow the main instructions to test if you install correctly.