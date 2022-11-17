import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import word2id
from model.Word2Vec import build_embdding_matrix

embedding_path = "word2vec.bin"


class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, drop_prob, output_size, word2vec_embedding):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
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
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=drop_prob
        )

        self.weight_W = nn.Parameter(torch.Tensor(2*hidden_size, 2*hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(2*hidden_size, 1))

        self.decoder1 = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder2 = nn.Linear(hidden_size, output_size)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        embeddings = self.embeds(inputs)
        states, _ = self.encoder(embeddings.permute([0, 1, 2]))
        # attention
        u = torch.tanh(torch.matmul(states, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)
        return outputs
