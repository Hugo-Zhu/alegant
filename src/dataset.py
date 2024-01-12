import os
import torch
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from .utils import process_data, statistical_analysis
from transformers import  BertConfig, BertModel, BertTokenizer
from transformers import XLNetTokenizer, RobertaTokenizer
from alegant import DataModuleConfig, DataModule

def load_data(file_path):
    # data.keys(): ['annotations', 'posts_text', 'posts_num']
    data = pickle.load(open(file_path, 'rb'))
    text = data['posts_text']
    label = data['annotations']
    # print(statistical_analysis(train_label))
    processed_data = process_data(text, label)
    return processed_data
    

@ dataclass()
class KaggleConfig:
    data_path: str 
    pretrain_type: str = 'bert'
    model_dir: str = 'bert-base-cased'
    max_post: int = 50
    max_len: int = 70
    

class KaggleDataset(Dataset):
    def __init__(self, config: KaggleConfig):
        self.config = config
        self.data = load_data(self.config.data_path)

        if self.config.pretrain_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]'])
        elif self.config.pretrain_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['<pad>', '<cls>'])
        elif self.config.pretrain_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['<pad>', '<s>'])
        else:
            raise NotImplementedError

        self.convert_features() # self.data.keys(): ['posts', 'label0', 'label1', 'label2', 'label3', 'post_tokens_id']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return: 单一用户的: post_tokens_ids, label0, label1, label2, label3
        """
        e = self.data[idx]
        items = e['post_tokens_id'], e['label0'], e['label1'], e['label2'], e['label3']
        items_tensor = tuple(torch.tensor(t) for i,t in enumerate(items))
        return items_tensor
    

    def _tokenize(self, text):
        """
        将文本转换为经过处理的ids（未pad，未truncate）
        """
        tokenized = self.tokenizer.tokenize(text)   # text --> tokenized
        ids = self.tokenizer.convert_tokens_to_ids(tokenized) # tokenized --> ids
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids) # 加CLS和SEP
        return input_ids
    
    def _pad_and_truncate(self, input_ids):
        pad_len = self.config.max_len - len(input_ids)
        if pad_len > 0:
            if self.config.pretrain_type == 'bert':
                input_ids += [self.pad] * pad_len
            elif self.config.pretrain_type == 'xlnet':
                input_ids = [input_ids[-1]] + input_ids[:-1]
                input_ids += [self.pad] * pad_len
            elif self.config.pretrain_type == 'roberta':
                input_ids += [self.pad] * pad_len
            else:
                raise NotImplementedError
        else:
            if self.config.pretrain_type == 'bert':
                input_ids = input_ids[:self.config.max_len - 1] + input_ids[-1:]
            elif self.config.pretrain_type == 'xlnet':
                input_ids = [input_ids[-1]]+ input_ids[:self.config.max_len - 2] + [input_ids[-2]]
            elif self.config.pretrain_type == 'roberta':
                input_ids = input_ids[:self.config.max_len - 1] + input_ids[-1:]
            else:
                raise NotImplementedError
        assert (len(input_ids) == self.config.max_len)
        return input_ids
        
    def convert_feature(self, i):
        """
        将单一用户的所有posts转为feature
        """
        post_tokens_id=[]
        for post in self.data[i]['posts'][:self.config.max_post]:
            input_ids = self._tokenize(post)
            input_ids = self._pad_and_truncate(input_ids)
            post_tokens_id.append(input_ids)

        # 如果post数量不足，则用pad填充
        real_post = len(post_tokens_id)
        for j in range(self.config.max_post-real_post):
            post_tokens_id.append([self.pad]*self.config.max_len)
        # 将单一用户的post_tokens_id添加到self.data
        # post_tokens_id: (num_posts, seq_len)
        self.data[i]['post_tokens_id'] = post_tokens_id

    def convert_features(self):
        '''
        将所有用户的posts转为feature
        '''
        for i in tqdm(range(len(self.data))):
            self.convert_feature(i)


class KaggleDataModule(DataModule):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        kaggle_config = KaggleConfig(data_path=self.config.train_data_path)
        self.train_dataset = KaggleDataset(config=kaggle_config)
        kaggle_config = KaggleConfig(data_path=self.config.val_data_path)
        self.val_dataset = KaggleDataset(config=kaggle_config)
        kaggle_config = KaggleConfig(data_path=self.config.test_data_path)
        self.test_dataset = KaggleDataset(config=kaggle_config)
    
    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
