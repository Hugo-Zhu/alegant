# Alegant

Alegant is an elegant training framework for PyTorch models.

## Install Alegant

Before installing Alegant, please make sure you have the following requirements:
- Python >= 3.7
- torch >= 1.9

Simple installation from PyPI
```bash
pip install alegant
```
To install Alegant and develop locally:
```bash
python setup.py develop
```

## Example

For examples on how to use elegant, please refer to the examples directory in this repository. It contains sample configuration files and code snippets to help you get started.

## Usage

To use Alegant, follow the steps below:

1. Define your Model.
2. Define your DataModule.
3. Define your Trainer.
4. Set your configuration.
5. Run the training script using the following command:

```bash
cd alegant/example
python example_main.py  # for simply use the DataModule and Trainer
python example_runner.py # for using the runner from Alegant
```

## Configuration
To customize the training process, you need to provide a configuration file. This file specifies various parameters such as dataset paths, model architecture, hyperparameters, etc. Make sure to create a valid configuration file before running the framework.

## Project Structure

```plaintext
alegant
├── alegant
│   ├── __init__.py
│   ├── runner.py
│   ├── trainer.py
│   ├── data_module.py
│   ├── utils.py
│   └── example
│       ├── data
│       ├── config.yaml
│       ├── example_main.py
│       ├── example_runner.py
│       ├── logs
│       ├── README.md
│       ├── src
│       │   ├── dataset.py
│       │   ├── loss.py
│       │   ├── model
│       │   │   ├── modeling.py
│       │   │   ├── poolers.py
│       │   ├── trainer.py
│       │   └── utils.py
│       └── tensorboard
└── setup.py
```

## Contact
If you have any questions or inquiries, please contact us at zhuhh17@qq.com

Thank you for using Alegant! Happy training!