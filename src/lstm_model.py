import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class TextAutoCompleteLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout,num_layers, bidirectional, tokenizer, device) -> None:
        self.device = device
        super().__init__()
        self.tokenizer = tokenizer
        self.emb = nn.Embedding(self.tokenizer.vocab_size, input_dim)
        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           batch_first=True,
                           bidirectional=bidirectional,
                           num_layers=num_layers)
        self.norm = nn.LayerNorm(2*hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.Linear(2*hidden_dim, self.tokenizer.vocab_size)

        torch.nn.init.xavier_uniform_(self.ln.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


    def forward(self,x_padded, lengths):
        embeddings = self.emb(x_padded)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.dropout(outputs)
        normed = self.norm(outputs)
        lengths = lengths.to(self.device)
        mean_output = torch.sum(normed, dim=1) / lengths.unsqueeze(1)
        logits  = self.ln(mean_output)
        return logits
    
    def generate(self, max_new_tokens, input_ids, **kwargs): 
        seq = input_ids.to(self.device)
        length = torch.tensor(input_ids.shape[1]).unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = self.forward(seq,
                                    length)
            next_token_id = torch.argmax(logits, dim=1, keepdim=True)
            seq = torch.concatenate([seq, next_token_id], dim=1)
            length += 1
            if next_token_id == self.tokenizer.sep_token_id:
                break
        return seq
