import json

import torch
import torch.utils.data as data

word2id = json.load(open("datasets/word2id.json", "r", encoding="utf-8"))

label2id = {
    'neural': 0,
    'happy': 1,
    'angry': 2,
    'sad': 3,
    'fear': 4,
    'surprise': 5
}

# maxlen为每条文本的平均单词数+2倍标准差


class WeiBoDataset(data.Dataset):
    def __init__(self, data_path, maxlen=84) -> None:
        super(WeiBoDataset, self).__init__()
        self.maxlen = maxlen
        self.sents, self.labels = self.loadDataset(data_path)

    def loadDataset(self, data_path):
        sents, labels = [], []
        with open(data_path, "r", encoding="utf-8") as fp:
            for item in json.load(fp):
                ids = []
                for ch in item['content'][:self.maxlen]:
                    ids.append(word2id.get(ch, word2id["UNK"]))
                ids = ids[:self.maxlen] if len(ids) > self.maxlen else ids + [0] * (self.maxlen - len(ids))
                sents.append(ids)
                labels.append(label2id.get(item['label']))
        f = torch.LongTensor
        return f(sents), f(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]


if __name__ == "__main__":
    train_path = "datasets/train.txt"
    wddataset = WeiBoDataset(train_path)
    print(len(wddataset))
    train_iter = data.DataLoader(
        dataset=wddataset,
        batch_size=64
    )
    for x, y in train_iter:
        print(x.shape, y.shape)
