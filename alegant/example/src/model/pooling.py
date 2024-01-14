import torch
import torch.nn as nn


class RnnPooler(nn.Module):
    def __init__(self, in_features, out_features):
        super(RnnPooler, self).__init__()
        self.bilstm = nn.LSTM(in_features,
                    int(out_features/2),
                    batch_first=True,
                    bidirectional=True)

    def forward(self, x):
        """
        Example:
            >>> rnn = RNN_Pooler(768, 128)
            >>> x = torch.randn([3, 4, 768])
            >>> out = rnn(x)
            >>> out.size()
            torch.Size([3, 128])
        """
        out, (hidden, cell) = self.bilstm(x)
        out = torch.cat ([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
        
        return out

class AveragePooler(nn.Module):
    def __init__(self):
        super(AveragePooler, self).__init__()
    
    def forward(self, x, mask):
        """
        :param x: [N, L, d]
        :param mask: [N, L]
        :return pooling_output: [N, d]

        Example: 
        >>> x = torch.randn([3, 4, 768])
        >>> out = Avg_Pooler(x)
        >>> out.size()
        torch.Size([3, 768])
        """
        # pooling_output = torch.mean(x, dim=1)
        mask = mask.unsqueeze(-1).expand_as(x).float()
        masked_x = x * mask

        # Average
        pooling_output = masked_x.sum(dim=1) / (mask+1e-8).sum(dim=1)
        return pooling_output

class AttentionPooler(nn.Module): 
    def __init__(self, input_dim):
        super(AttentionPooler, self).__init__()
        self.input_dim = input_dim
        self.attention_layer = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        """
        :param x: [N, L, d]
        :param mask: [N, L]
        :return pooling_output: [N, d]

        Example: 
        >>> x = torch.randn([3, 4, 768])
        >>> mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
        >>> out = Attention_Pooler(768)(x, mask)
        >>> out.size()
        torch.Size([3, 768])
        """
        # Calculate attention scores
        attention_scores = self.attention_layer(x).squeeze(-1)  # [N, L]
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  # Masked elements get -inf attention scores
        attention_weights = torch.softmax(attention_scores, dim=1)  # [N, L]

        # Apply attention weights to the input sequence
        masked_x = x * mask.unsqueeze(-1).float()  # [N, L, d]
        attended_sequence = torch.bmm(attention_weights.unsqueeze(1), masked_x).squeeze(1)  # [N, d]

        return attended_sequence



class CnnPooler(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1):
        super(CnnPooler, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Example:
            >>> pooler = CNN_Pooler(768, 128)
            >>> x = torch.randn([3, 4, 768])
            >>> out = pooler(x)
            >>> out.size()
            torch.Size([3, 128])
        """
        x = x.transpose(1, 2)    # permute to [N, d, L]
        out = self.conv(x)
        out = self.relu(out)
        out, _ = torch.max(out, dim=-1)  # global max pooling
        return out