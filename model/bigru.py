import torch
import torch.nn as nn

from dataset import word2id
from model.Word2Vec import build_embdding_matrix

embedding_path = "word2vec.bin"


class BiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, drop_rate, word2vec_embedding=True):
        super(BiGRU, self).__init__()
        if word2vec_embedding:
            embedding_matrix = build_embdding_matrix(
                word_dict=word2id,
                embedding_path=embedding_path,
                embedding_dim=embedding_dim)
            embedding_weight = torch.from_numpy(embedding_matrix).float()
            self.embeds = nn.Embedding.from_pretrained(embedding_weight)
        else:
            self.embeds = nn.Embedding(len(word2id), embedding_dim)
            nn.init.uniform_(self.embeds.weight)
        self.gru = nn.GRU(
            bidirectional=True,
            num_layers=2,
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=drop_rate
        )
        self.batchnorm = nn.BatchNorm1d(84)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embeds(x)
        x, _ = self.gru(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc(torch.mean(x, dim=1))
        return x
