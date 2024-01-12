import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from typing import Optional, List
from abc import ABC,abstractmethod
from dataclasses import dataclass, field
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from .data_module import DataModule

@dataclass
class TrainingArguments:
    """
    TrainingArguments is a data class that represents the arguments for training a model.

    Attributes:
        amp (bool): Automatic mixed precision. If True, enables automatic mixed precision training. Defaults to False.
        do_save (bool): Whether to save the checkpoint during training. If True, saves the checkpoint. Defaults to False.
        epochs (int): Number of training epochs. Defaults to 10.
        accumulation_steps (int): Number of steps to accumulate gradients. Defaults to 5.
        eval_steps (int): Evaluate the model every eval_steps steps. Defaults to 500.
        max_norm (Optional[float]): Maximum value for gradient clipping. Defaults to None.
        device (str): Device for training. Defaults to "cuda".
        gpus (List[int]): GPU device IDs to be used. Defaults to [0].
        learning_rate (float): Learning rate. Defaults to 2.0e-3.
        learning_rate_plm (float): Learning rate for pre-trained models. Defaults to 2.0e-5.
        weight_decay (float): Weight decay. Defaults to 1.0e-4.
        checkpoint_save_path (Optional[str]): Model weight save path. Defaults to None.
        log_dir (str): TensorBoard log directory path. Defaults to "./tensorboard".
        do_test (bool): Whether to perform testing or training. If True, performs testing. Defaults to False.
        random_seed (int): Random seed for reproducibility. Defaults to 1.
        dataset (str): Dataset name. Defaults to "kaggle".
        log_file (str): Path to save the log file. Defaults to "train.log".
    """
    amp: bool = field(default=False, metadata={"help": "Automatic mixed precision"})
    do_save: bool = field(default=False, metadata={"help": "Whether to save the checkpoint during training"})
    epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    accumulation_steps: int = field(default=5, metadata={"help": "Number of steps to accumulate gradients"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluate the model every eval_steps steps"})
    max_norm: Optional[float] = field(default=None, metadata={"help": "Maximum value for gradient clipping"})
    device: str = field(default="cuda", metadata={"help": "Device for training"})
    gpus: List[int] = field(default_factory=lambda: [0], metadata={"help": "GPU device IDs to be used"})
    learning_rate: float = field(default=2.0e-3, metadata={"help": "Learning rate"})
    learning_rate_plm: float = field(default=2.0e-5, metadata={"help": "Learning rate for pre-trained models"})
    weight_decay: float = field(default=1.0e-4, metadata={"help": "Weight decay"})
    checkpoint_save_path: Optional[str] = field(default=None, metadata={"help": "Model weight save path"})
    log_dir: str = field(default="./tensorboard", metadata={"help": "TensorBoard log directory path"})
    do_test: bool = field(default=False, metadata={"help": "Whether to perform testing or training"})
    random_seed: int = field(default=1, metadata={"help": "Random seed for reproducibility"})
    dataset: str = field(default="kaggle", metadata={"help": "Dataset name"})
    log_file: str = field(default="train.log", metadata={"help": "Path to save the log file"})


class Trainer(ABC):
    """
    Abstract base class for trainers.

    """
    def __init__(self,
        args: TrainingArguments, 
        model: nn.Module, 
        data_module: DataModule): 
        """
        Initializes the Trainer object.

        Args:
            args (TrainingArguments): Training arguments.
            model (nn.Module): The neural network model.
            data_module (DataModule): The data module used for training.
        """
        self.args = args       
        self.model = model.to(self.args.device)
        self.data_module = data_module
        self.criterion = nn.CrossEntropyLoss()
        self.logger = SummaryWriter(self.args.log_dir)
        

    def fit(self, checkpoint=None):
        """
        Main training loop.

        Args:
            checkpoint (str): Path to a checkpoint file to resume training from.
        """
        # Initialize dataloader
        train_dataloader = self.data_module.train_dataloader()
        eval_dataloader = self.data_module.val_dataloader()
        
        # Initialize optimizer
        num_training_steps = len(train_dataloader) * self.args.epochs
        self.optimizer, self.scheduler = self.configure_optimizers(num_training_steps)
        
        # Load checkpoint or initialize training variables
        if checkpoint is not None:
            self.load_checkpoint(path=checkpoint)
        else:
            self.start_epoch = 0
            self.num_eval = 0
            self.num_train_step = 0
            self.num_eval_step = 0
            self.best_f1 = 0
        
        # Set up multi-GPU running
        if self.args.device == "cuda" and self.args.gpus is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus, output_device=self.args.device)
        else: self.model = self.model.to(self.args.device)
        
        scaler = GradScaler(enabled=self.args.amp)

        # Start train loop
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            for batch_idx, batch in tqdm(enumerate(train_dataloader), f"Epoch: {self.epoch}", total=len(train_dataloader), miniters=len(train_dataloader)*0.1, maxinterval=float("inf")):
                outputs = self.training_step(batch, batch_idx)
                loss = outputs.get("loss")/self.args.accumulation_steps     # 除不除accumulation_steps对结果有影响
                # loss.backward()
                scaler.scale(loss).backward()
                
                # Clip gradients for each parameter group in the optimizer
                if self.args.max_norm is not None:
                    for param_group in self.optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(param_group['params'], self.args.max_norm)
                
                self.num_train_step += 1
                self.logger.add_scalar(f"train_loss", loss.item(), self.num_train_step)
                
                # Update the learning rate each batch
                if self.scheduler is not None:
                    self.scheduler.step()

                    current_learning_rate = self.optimizer.param_groups[0]['lr']  # 这里只记录第一个参数组中的学习率
                    self.logger.add_scalar("learning_rate", current_learning_rate, self.num_train_step)

                # Gradient accumulation
                if (batch_idx+1) % self.args.accumulation_steps == 0 or (batch_idx+1)==len(train_dataloader):
                    # self.optimizer.step()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    self.optimizer.zero_grad()
                
                # Evaluation loop
                if (batch_idx+1) % self.args.eval_steps == 0 or (batch_idx+1)==len(train_dataloader):
                    self.validation(eval_dataloader)
            
            self.training_epoch_end()
        self.logger.close()
    

    @torch.no_grad()
    def validation(self, eval_dataloader):
        """
        Perform model validation.

        Args:
            eval_dataloader: Dataloader for evaluation.
        """
        self.model.eval()
        eval_outputs = []
        
        self.num_eval += 1
        for batch_idx_eval, batch_eval in tqdm(enumerate(eval_dataloader), "evalation", total=len(eval_dataloader), miniters=len(eval_dataloader)*0.1, maxinterval=float("inf")):
            outputs = self.validation_step(batch_eval, batch_idx_eval)
            loss = outputs.get("loss")
            eval_outputs.append(outputs)
            self.num_eval_step += 1
            self.logger.add_scalar(f"eval_loss", loss.item(), self.num_eval_step)
        
        self.validation_epoch_end(eval_outputs)
        self.model.train()

    @torch.no_grad()
    def test(self, checkpoint):
        """
        Perform model testing.

        Args:
            checkpoint (str): Path to the checkpoint file.
        Returns:
            float: Test accuracy.
        """
        self.model.to(self.args.device)
        # Load the checkpoint
        if checkpoint is None:
            logger.warning("Checkpoint is None during TESTING !!!")
        else:
            logger.info(f"Loading checkpoint from {checkpoint} ...")
            checkpoint = torch.load(checkpoint, map_location=self.args.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        # Initialize the test dataloader
        dataloader = self.data_module.test_dataloader()
        
        self.model.eval()
        test_outputs = []
        num_test_step = 0
        for batch_idx_test, batch_test in tqdm(enumerate(dataloader), "test", total=len(dataloader)):
            outputs = self.test_step(batch_test, batch_idx_test)
            loss = outputs.get("loss")
            test_outputs.append(outputs)
            num_test_step += 1
            self.logger.add_scalar(f"test_loss", loss.item(), num_test_step)

        self.test_epoch_end(test_outputs)
        self.model.train() 


    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch: Batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the training step outputs.
        """
        # 获取数据并移动到设备
        input_ids = batch['input_ids'].to(self.args.device)
        label = batch['label'].to(self.args.device)
        
        with autocast(enabled=self.args.amp):
            # 前向传播
            logits = self.model(input_ids)
            # 计算损失
            loss = self.criterion(logits, label)
        
        return {"loss": loss}

    def training_epoch_end(self):
        """
        Perform operations at the end of each training epoch.
        """
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch: Batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the validation step outputs.
        """
        # 获取数据并移动到设备 -> 前向传播 -> 计算损失 -> 记录评价指标
        input_ids = batch['input_ids'].to(self.args.device)
        label = batch['label'].to(self.args.device)
        with autocast(enabled=self.args.amp):
            logits = self.model(input_ids)
            loss = self.criterion(logits, label)
        predictions = torch.argmax(logits, dim=-1)
        batch_errors = (predictions != label)
        return {"loss": loss, "logits":logits, "label":label}

    @abstractmethod
    def validation_epoch_end(self, outputs):
        """
        Perform operations at the end of each validation epoch.

        Args:
            eval_outputs: List of outputs from validation step.
        """
        logits_list = []
        label_list = []
        for item in outputs:
            logits_list.append(item["logits"].detach().cpu())
            label_list.append(item["label"].detach().cpu())

        preds_list = torch.argmax(F.softmax(logits_list, dim=-1), dim=-1)
        preds_list = torch.cat(preds_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        metrics = self.compute_metrics(preds_list, label_list)
        accuracy = metrics.get("accuracy")
        self.logger.add_scalar("accuracy", accuracy, self.num_eval)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            if self.args.do_save:
                self.save_checkpoint()
        logger.success(f"acc: {accuracy}, best_acc: {self.best_acc}")


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        """
        Perform operations at the end of each test epoch.

        Args:
            eval_outputs: List of outputs from test step.
        """
        logits_list = []
        label_list = []
        for item in outputs:
            logits_list.append(item["logits"].detach().cpu())
            label_list.append(item["label"].detach().cpu())

        preds_list = torch.argmax(F.softmax(logits_list, dim=-1), dim=-1)
        preds_list = torch.cat(preds_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        metrics = self.compute_metrics(preds_list, label_list)
        accuracy = metrics.get("accuracy")
        logger.info(f"acc: {accuracy}")

    @abstractmethod
    def configure_optimizers(self, num_training_steps):
        """
        Configure the optimizer and learning rate scheduler.

        Args:
            num_training_steps (int): Total number of training steps.

        Returns:
            tuple: Tuple containing the optimizer and learning rate scheduler.
        """
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        params_plm = []
        no_decay_params_plm = []
        params = []
        no_decay_params = []        
        
        for name, param in self.model.named_parameters():
            if param.requires_grad == False:
                continue
            
            if "plm" in name:
                if any(nd in name for nd in no_decay):
                    no_decay_params_plm.append(param)
                else:
                    params_plm.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": params_plm,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate_plm
            },
            {   
                "params": no_decay_params_plm,
                "weight_decay": 0.0,
                "lr": self.args.learning_rate_plm
            },
            {
                "params": params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            }
        ]
        
        optimizer = Adam(optimizer_grouped_parameters)

        # # Define the warm-up steps
        # num_warmup_steps = int(0.1 * num_training_steps)  # You can adjust the percentage of warm-up steps as needed
        # # Create the learning rate scheduler
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        scheduler = None

        return optimizer, scheduler
    
    def save_checkpoint(self):
        """
        Save the current model checkpoint.

        Args:
            path (str): Path to save the checkpoint.
        """
        # TODO  save & load best metrics
        logger.info("Saving checkpoint ...")
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_eval': self.num_eval,
            'num_train_step': self.num_train_step,
            'num_eval_step': self.num_eval_step,
            }, self.args.checkpoint_save_path)

    
    def load_checkpoint(self, path):
        """
        Load a model checkpoint.

        Args:
            path (str): Path to the checkpoint file.
        """
        logger.info(f"Loading checkpoint from {path} ...")
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.num_eval = checkpoint['num_eval']
        self.num_train_step = checkpoint['num_train_step']
        self.num_eval_step = checkpoint['num_eval_step']
