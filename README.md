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
To install fairseq and develop locally:
```bash
python setup.py develop
```
## Usage
To use alegant, follow the steps below:

1. Define your Model.
2. Define your DataModule.
3. Define your Trainer.
4. Run the training script using the following command:

```python
python --config_file run.py
```
Make sure to replace config_file with the path to your configuration file.

## Configuration
To customize the training process, you need to provide a configuration file. This file specifies various parameters such as dataset paths, model architecture, hyperparameters, etc. Make sure to create a valid configuration file before running the framework.

## Project Structure

```plaintext
alegant
├── tensorboard
├── data
├── alegant
│   ├── data_module.py
│   ├── trainer.py
│   └── utils.py
├── src
│   ├── dataset.py
│   ├── loss.py
│   ├── model
│   │   ├── modeling.py
│   │   ├── poolers.py
│   ├── trainer.py
│   └── utils.py
├── config.yaml
├── run.py
└── setup.py
```

## Contact
If you have any questions or inquiries, please contact us at zhuhh17@qq.com

Thank you for using Alegant! Happy training!