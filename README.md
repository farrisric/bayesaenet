# BNN-AENET

BNN-AENET is a library designed for training neural networks to predict atomic energies and forces, primarily in the context of materials science or computational chemistry. The library leverages PyTorch for neural network operations and includes specific modules for handling atomic structures, descriptors, and their derivatives. It supports both energy and force training, normalizing and batching data for efficient neural network training.

## Table of Contents

- Installation
- Usage
- Project Structure
- Key Components
- Configuration
- Examples
- Contributing
- License

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To train a model, you need to prepare the input files and configure the training parameters. The main entry point for training is through the `train.py` script. Here is an example command to start training:

```sh
python train.py --config config.yaml
```

Make sure to replace `config.yaml` with your actual configuration file.

## Project Structure

The project is organized as follows:

```
/home/g15farris/bin/bayesaenet/
├── data/               # Directory for storing datasets
├── models/             # Pre-trained models and model checkpoints
├── scripts/            # Utility scripts for data processing and analysis
├── src/                # Source code for the library
│   ├── descriptors/    # Code for generating atomic descriptors
│   ├── models/         # Neural network architectures
│   ├── trainers/       # Training routines and utilities
│   └── utils/          # Helper functions and utilities
└── train.py            # Main script for training models
```

## Key Components

- **Descriptors**: Functions and classes for generating atomic descriptors.
- **Models**: Neural network architectures for energy and force prediction.
- **Trainers**: Utilities for training models, including data loading, normalization, and batching.
- **Utils**: Helper functions for various tasks such as logging and configuration management.

## Configuration

The training process is configured using YAML files. Here is an example configuration:

```yaml
model:
    type: BNN
    layers: [128, 128, 64]

training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001

data:
    train_path: data/train.csv
    val_path: data/val.csv
```

## Examples

Here are some example commands to get you started:

- Train a model:

    ```sh
    python train.py --config config.yaml
    ```

- Evaluate a model:

    ```sh
    python evaluate.py --model models/checkpoint.pth --data data/test.csv
    ```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
