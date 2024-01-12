import os
import json
import datetime
from tqdm import tqdm
from src.trainer import TrainingArguments,BertTrainer
from src.model.modeling import BertConfig, BERT
from src.dataset import DataModuleConfig, KaggleDataset, KaggleDataModule
from alegant.utils import seed_everything, parse_args, logger


@logger.catch
def main():
    args = parse_args()
    logger.add(f"{datetime.datetime.now().date()}/{args.log_file}") # 设置日志保存
    logger.info("Configuration:" + json.dumps(args, indent=4))
    logger.warning(f"python {os.path.basename(__file__)}!!!")
    seed_everything(args.random_seed)
    start_time = datetime.datetime.now()
    
    # 初始化 model & data_module
    model = BERT(config=BertConfig(**args.bert_config))
    if args.dataset == "kaggle":
        data_module = KaggleDataModule(DataModuleConfig(**args.data_module_config))
    else: # TODO
        raise NotImplementedError
        
    # 初始化 trainer
    trainer = BertTrainer(
        args=TrainingArguments(**args.training_args),
        model=model,
        data_module=data_module
    )

    if not args.do_test:    # 开始训练
        trainer.fit()
    else:                   # 开始测试
        trainer.test(checkpoint=args.training_args.checkpoint_save_path)
                
    end_time = datetime.datetime.now()
    training_duration = end_time - start_time
    training_minutes = training_duration.total_seconds() / 60
    logger.success(f"Time Consuming: {training_minutes} minutes \n\n")


if __name__ == "__main__":
    main()