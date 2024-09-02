import sys
from loguru import logger
from .runner import Runner
from .utils import parse_args, seed_everything
from .trainer import TrainingArguments, Trainer
from .data_module import DataModuleConfig, DataModule


logger.remove()
logger.add(sys.stdout, format='File "<cyan>{file.path}</cyan>", line <cyan>{line}</cyan>\n  {message}', level='DEBUG')
