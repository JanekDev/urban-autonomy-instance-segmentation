# Urban Autonomy Instance Segmentation

## Introduction

"Urban Autonomy Instance Segmentation" is a project focused on comparing deep learning instance segmentation models in an urban environments autonomous driving technology setups. Utilizing the COCO dataset, this project applies instance segmentation methods to identify and segment various objects encountered in urban driving scenarios, such as vehicles, pedestrians, and street signs.

## Dataset
TODO

## Usage

### Clone the repository

```
git clone --recurse-submodules https://github.com/JanekDev/urban-autonomy-instance-segmentation.git
```

### Requirements
    
Create a virtual environment and install the required packages:

```
pip install -r requirements.txt
```

Or if you want to completely setup the repository (large GPU nodes for rent), including the datasets, run:

```
bash setup_enviroment.sh
```

### Checkpoints

Will be available soon.

### Training

```bash
python3 main.py --config-name config.yaml
```

### Testing and evaluation

TODO
