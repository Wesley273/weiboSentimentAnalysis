import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import word2id
from model.gensim_word2vec import build_embdding_matrix

embedding_path="word2vec.bin"

class BiGRU(nn.Module):
    def __init__(self,embedding_dim,hidden_size,output_size,drop_prob,extra_embedding=True):
        super(BiGRU,self).__init__()
        if extra_embedding:
            embedding_matrix = build_embdding_matrix(
                word_dict=word2id,
                embedding_path=embedding_path,
                embedding_dim=embedding_dim)
            embedding_weight = torch.from_numpy(embedding_matrix).float()
            self.embeds = nn.Embedding.from_pretrained(embedding_weight)
        else:
            self.embeds = nn.Embedding(len(word2id),embedding_dim)
            nn.init.uniform_(self.embeds.weight)
        self.gru = nn.GRU(
            bidirectional=True, 
            num_layers=2, 
            input_size=embedding_dim, 
            hidden_size=hidden_size,
            batch_first=True,
            dropout=drop_prob
        )
        self.batchnorm = nn.BatchNorm1d(84)
        self.dropout = nn.Dropout(drop_prob)
        self.decoder = nn.Linear(hidden_size * 2,output_size)
        
    def forward(self,x):
        x = self.embeds(x)
        x,_ = self.gru(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.decoder(torch.mean(x,dim=1))
        return x
