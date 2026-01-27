# PML: Proximal Mapping Loss

Official code for ICLR-2025 paper "[Proximal Mapping Loss: Understanding Loss Functions in Crowd Counting & Localization](http://visal.cs.cityu.edu.hk/static/pubs/conf/iclr25-pml.pdf)"

## Project Structure

```
pml/
├── config.py          # Configuration settings
├── datasets/          # Data loaders
├── losses/            # Loss functions (including PML)
│   ├── __init__.py
│   └── pmloss.py      # Proximal Mapping Loss implementation
├── logger.py          # Logging utilities
├── lr_scheduler.py    # Learning rate scheduling
├── main.py            # Main training and evaluation script
├── models/            # Model architectures
├── README.md          # This file
├── run.sh             # Run script
└── utils.py           # Utility functions
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/elin24/pml.git
   cd pml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Download the desired crowd counting dataset (e.g., SHHA, SHHB) and place it in the appropriate directory. Update the `DATA_PATH` in the configuration or pass it as a command-line argument.

## Dataset Structure

The project supports multiple crowd counting datasets. Here's the expected structure for the SHHA (ShanghaiTech Part A) dataset:

```
shha/
├── train/
│   ├── img_001.png       # Image file
│   ├── img_001.npy        # Annotation file (contains point coordinates)
│   ├── img_002.png
│   ├── img_002.npy
│   └── ...
└── test/
    ├── img_001.png
    ├── img_001.npy
    └── ...
```

### Annotation Format

The annotation files are in NumPy (.npy) format, containing a 2D array where each row represents a point coordinate (x, y). For example:

```python
# img_001.npy content
[[100.5, 200.3],  # First point
 [150.2, 250.7],  # Second point
 ...]
```

### Supported Datasets

Currently, the following datasets are supported:

- **SHHA**: ShanghaiTech Part A dataset

To add support for other datasets (e.g., SHHB, QNRF), you need to:

1. Create a new dataset class in the `datasets/` directory (similar to `shha.py`)
2. Update the dataset mapping in `datasets/__init__.py` to include your new dataset
3. Modify the `DATA.DATASET` setting in `config.py` or pass it as a command-line argument

### Dataset Loading

The dataset is loaded using the `build_loader` function in `datasets/__init__.py`, which creates a DataLoader with the appropriate batch size, workers, and collation function for the selected dataset.

### Training

To train a model with PML loss:

```bash
# Example: Train on SHHA dataset
python main.py \
     --data-path /path/to/shha \
     --batch-size 16 \
     --device cuda:0 \
     --tag experiment1
```

### Evaluation


modify the `run.sh` with your settings and then:

```bash
# Modify run.sh with your settings
bash run.sh
```

## Configuration

Key configuration options:

- `MODEL.NAME`: Backbone model (e.g., 'hrnet_w48', 'vgg19') - modify this in config.py to change the model architecture
- `TRAIN.EPOCHS`: Number of training epochs
- `TRAIN.BASE_LR`: Base learning rate

See `config.py` for all available options. You can also override these settings using command-line arguments.

## Citation

If you use PML in your research, please cite our paper:

```bibtex
@inproceedings{linproximal,
  title={Proximal mapping loss: Understanding loss functions in crowd counting \& localization},
  author={Lin, Wei and Wan, Jia and Chan, Antoni B},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```