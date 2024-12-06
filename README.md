# MEM679-HW03

The data are a combination of *my lab data* and some online data credited to Verheyen et al., *Integrated data-driven modeling and experimental optimization of granular hydrogel matrices*, Matter, 2023. (https://doi.org/10.1016/j.matt.2023.01.011)

## Installation

```bash
git clone https://github.com/simon596/MEM679-HWnFinal.git
```

## Create Python environment

```bash
# make sure you are in the root directory MEM679-HW03/HW03/
cd HW03
# create virtual environment
conda create -n MEM679 python=3.12
conda activate MEM679
# locate in the root directory and install the dependent libraries
conda install pip
pip install -r requirements.txt
```

## Run the visualization

```bash
# to run the visualization, either run the ipynb, or type
python ./src/hw03/visualization.py
```

## Final Project: Segmentation of Hydrogel

Source files are located in directory *"./HW_03_04_final/src/final"*.
```bash
conda activate MEM679
# check CUDA version before installing Pytorch
nvcc --version
# install Pytorch (GPU version, cuda12.1 for example)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# check installation
python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```