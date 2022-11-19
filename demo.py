import json

import torch

from config import args
from dataset import WeiBoDataset
from model.BiGRU import BiGRU
from model.BiLSTMAttention import BiLSTMAttention
label2id = {'neural': 0,
            'happy': 1,
            'angry': 2,
            'sad': 3,
            'fear': 4,
            'surprise': 5}
if __name__ == "__main__":
    model = BiLSTMAttention(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        output_size=6,
        drop_rate=args.drop_prob,
        word2vec_embedding=args.extra_embedding)
    model.load_state_dict('./weight/model_49.pth')
    model.eval()
    dataset = WeiBoDataset("datasets/test.txt")
    data, label = dataset[0]
    print(model(data))
