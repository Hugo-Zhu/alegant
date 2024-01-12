import os
import torch
from typing import Optional, List
from abc import ABC,abstractmethod
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset, RandomSampler


@dataclass
class DataModuleConfig:
    """
    Data configuration class for defining data-related configuration parameters.

    Args:
        train_data_path (str): File path to the training data.
        val_data_path (str): File path to the validation data.
        test_data_path (str): File path to the test data.
        do_setup (bool): Whether prepare datasets from data_paths
        cache_dir (Optional[str]): Path to cache the datasets
        train_batch_size (Optional[int]): Batch size for training data. Defaults to 32.
        train_limit_batches (Optional[float]): Batch limit ratio for training data. Defaults to 1.0.
        val_batch_size (Optional[int]): Batch size for validation data. Defaults to 32.
        val_limit_batches (Optional[float]): Batch limit ratio for validation data. Defaults to 1.0.
        test_batch_size (Optional[int]): Batch size for test data. Defaults to 32.
        test_limit_batches (Optional[float]): Batch limit ratio for test data. Defaults to 1.0.
    """
    train_data_path: str
    val_data_path: str
    test_data_path: str
    do_setup: bool = field(default=True, metadata={"description": "Whether prepare datasets from data_paths"})
    cache_dir: str = field(default=None, metadata={"description": "Path to cache the datasets"})
    
    train_batch_size: int = field(default=32, metadata={"description": "Batch size for training data"})
    train_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for training data"})
    val_batch_size: int = field(default=32, metadata={"description": "Batch size for validation data"})
    val_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for validation data"})
    test_batch_size: int = field(default=32, metadata={"description": "Batch size for test data"})
    test_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for test data"})


class DataModule(ABC):
    def __init__(self, 
        config: DataModuleConfig):
        """
        Abstract base class for data modules.

        Args:
            config (DataModuleConfig): Data configuration object.
        """
        self.config = config
        if self.config.cache_dir and not self.config.do_setup:
            self.train_dataset = torch.load(os.path.join(self.config.cache_dir, "train_dataset.pt"))
            self.val_dataset = torch.load(os.path.join(self.config.cache_dir, "val_dataset.pt"))
            self.test_dataset = torch.load(os.path.join(self.config.cache_dir, "test_dataset.pt"))
        else:
            self.setup()
            if self.config.cache_dir:
                torch.save(self.train_dataset,os.path.join(self.config.cache_dir, "train_dataset.pt"))
                torch.save(self.val_dataset, os.path.join(self.config.cache_dir, "val_dataset.pt"))
                torch.save(self.test_dataset, os.path.join(self.config.cache_dir, "test_dataset.pt"))

    @abstractmethod
    def setup(self):
        """
        Prepare datasets from data_paths
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """
        Returns a data loader for training data.

        Returns:
            DataLoader: Data loader for training data.
        """
        full_size = len(self.train_dataset)
        real_size = int(full_size * self.config.train_limit_batches)
        train_dataset, _ = torch.utils.data.random_split(self.train_dataset, [real_size, full_size-real_size])
        train_dataloader = DataLoader(dataset = train_dataset, 
                                shuffle = True,
                                batch_size = self.config.train_batch_size)
        return train_dataloader
    
    @abstractmethod
    def val_dataloader(self):
        """
        Returns a data loader for validation data.

        Returns:
            DataLoader: Data loader for validation data.
        """
        full_size = len(self.val_dataset)
        real_size = int(full_size * self.config.val_limit_batches)
        val_dataset, _ = torch.utils.data.random_split(self.val_dataset, [real_size, full_size-real_size])
        val_dataloader = DataLoader(val_dataset, 
                                shuffle = False,
                                batch_size = self.config.val_batch_size)
        return val_dataloader

    @abstractmethod
    def test_dataloader(self):
        """
        Returns a data loader for test data.

        Returns:
            DataLoader: Data loader for test data.
        """
        full_size = len(self.test_dataset)
        real_size = int(full_size * self.config.test_limit_batches)
        test_dataset, _ = torch.utils.data.random_split(self.test_dataset, [real_size, full_size-real_size])
        test_dataloader = DataLoader(test_dataset, 
                                shuffle = False,
                                batch_size = self.config.test_batch_size)
        return test_dataloader