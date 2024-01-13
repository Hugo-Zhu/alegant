from alegant import Runner
from src.trainer import BertTrainer
from src.model.modeling import BertConfig, BERT
from src.dataset import DataModuleConfig, KaggleDataModule


class MyRunner(Runner):
    def setup(self):
        model = BERT(config=BertConfig(**self.args.bert_config))
        if self.args.dataset == "kaggle":
            data_module = KaggleDataModule(DataModuleConfig(**self.args.data_module_config))
        else: # TODO
            raise NotImplementedError
        trainer_class = BertTrainer
        return model, data_module, trainer_class

if __name__ == "__main__":
    runner = MyRunner()
    runner.run()