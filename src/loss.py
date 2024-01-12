import torch.nn as nn

class ce_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_all_class, targets_all_class):
        """
        Inputs: logits_all_class: [logits1, logits2, logits3, logits4]
        其中每个logits包含单个标签的logits

        Return: 所有标签，所有样本的loss的平均值

        """
        num_samples = 0
        loss_all_class = []
        for i, logits in enumerate(logits_all_class):
            num_samples += logits.size(0)
            loss = nn.functional.cross_entropy(logits, targets_all_class[i])
            loss_all_class.append(loss)
        return sum(loss_all_class) / num_samples