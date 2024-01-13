import os
import json
import datetime
from abc import ABC,abstractmethod
from .trainer import TrainingArguments
from .utils import seed_everything, parse_args, logger


class Runner(ABC):
    """
    Abstract base class for a runner.

    Subclasses of `Runner` should implement the `setup` method to initialize the model, data module and trainer class.
    The `run` method can be called to start the training or testing process based on the config file or command line arguments.

    Example:
        class MyRunner(Runner):
            def setup(self):
                model = BERT(config=BertConfig(**self.args.bert_config))
                data_module = KaggleDataModule(DataModuleConfig(**self.args.data_module_config))
                trainer_class = BertTrainer
                return model, data_module, trainer_class

        runner = MyRunner()
        runner.run()
    """
    @abstractmethod
    def setup(self):
        """Initialize the model & data_module & trainer_class

        Returns:
            model: nn.Module
                An instance of nn.Module representing the initialized model.
            data_module: DataModule
                An instance of DataModule representing the initialized data module.
            trainer_class: Trainer
                A subclass of Trainer representing the trainer class available for training.
        """
        pass


    @logger.catch
    def run(self):
        """Run the training or testing process.

        This method initializes the model and data module, creates a trainer instance,
        and starts the training or testing based on the command line arguments.

        """
        self.args = parse_args()
        logger.add(f"logs/{datetime.datetime.now().date()}/{self.args.log_file}") # Set path for logs
        logger.info("Configuration:" + json.dumps(self.args, indent=4))
        logger.warning(f"python {os.path.basename(__file__)}!!!")
        seed_everything(self.args.random_seed)
        start_time = datetime.datetime.now()
        
        # Initialize model & data_module
        model, data_module, trainer_class = self.setup()
            
        # Initialize trainer
        trainer = trainer_class(
            args=TrainingArguments(**self.args.training_args),
            model=model,
            data_module=data_module
        )

        if not self.args.do_test:    # Start training
            trainer.fit()
        else:                   # Start testing
            trainer.test(checkpoint=self.args.training_args.checkpoint_save_path)
                    
        end_time = datetime.datetime.now()
        training_duration = end_time - start_time
        training_minutes = training_duration.total_seconds() / 60
        logger.success(f"Time Consuming: {training_minutes} minutes \n\n")

