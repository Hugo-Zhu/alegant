import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel
from dataclasses import dataclass, field
from .poolers import AveragePooler


@dataclass
class BertConfig:
    dropout_prob: float = 0.0
    pretrained_model_name_or_path: str = "bert-base-cased"


class BERT(nn.Module):
    def __init__(self, 
                config: BertConfig ):
        super(BERT, self).__init__()
        self.config = config
        
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_name_or_path, output_hidden_states=True)
        self.pooler = AveragePooler()
        self.activation = nn.ReLU()
        self.fc = nn.Linear(self.bert.config.hidden_size, 8)
        self.dropout = nn.Dropout(self.config.dropout_prob)

                
    def forward(self, post_tokens_ids):
        pad_id = 0
        batch_size, num_subsequence, max_len = post_tokens_ids.size()
        
        ### B: batch size, N: num of posts, L: seq length
        attention_mask = (post_tokens_ids != pad_id).float()    # (B, N, L)
        post_mask = (attention_mask.sum(-1) > 0).float()        # (B, N)
        input_ids = post_tokens_ids.view(-1, max_len)           # (B*N, L)
        attention_mask = attention_mask.view(-1, max_len)       # (B*N, L)

        embeded = self.bert(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1]
        embeded_cls = embeded[:, 0, :].reshape([batch_size, num_subsequence, -1])  # (B, N, d)
        out = embeded_cls

        pooled_out = self.pooler(out, post_mask)                    # (B, d)
        pooled_out = self.dropout(pooled_out)
        logits_list = self.fc(pooled_out)
        
        batch_size, _ = logits_list.size()
        logits_list = logits_list.view(batch_size, -1, 2).transpose(0,1)  # (real_num_traits, B, 2)
        return {"logits_list":[logits for logits in logits_list]}
