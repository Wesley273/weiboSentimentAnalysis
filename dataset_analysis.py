import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def drawPie(x, labels, fname):
    """
    功能：绘制饼图
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("trainset")
    plt.pie(x=x[0], labels=labels[0], autopct="%1.2f%%")
    plt.subplot(1, 2, 2)
    plt.title("testset")
    plt.pie(x=x[1], labels=labels[1], autopct="%1.2f%%")
    plt.savefig("images/{}".format(fname))


def commengLengthAnalysis(data_path):
    """
    功能：统计微博的平均长度
    """
    counts = []
    length, count = 0, 0
    with open(data_path, "r", encoding="utf-8") as fp:
        for weibo in json.load(fp):
            length += len(weibo['content'])
            counts.append(len(weibo['content']))
            count += 1

    print("Datapath:{} Avg length: {}, std: {}".format(data_path, length / count, np.std(counts)))


if __name__ == "__main__":
    commengLengthAnalysis(data_path="datasets/train.txt")
    commengLengthAnalysis(data_path="datasets/test.txt")
    traindataset = json.load(open("datasets/train.txt", "r", encoding="utf-8"))
    testdataset = json.load(open("datasets/test.txt", "r", encoding="utf-8"))
    print(len(traindataset), len(testdataset))
    train_df = pd.DataFrame(traindataset)
    test_df = pd.DataFrame(testdataset)
    train_group = train_df.loc[:, ['content', 'label']].groupby("label").count()
    test_group = test_df.loc[:, ['content', 'label']].groupby("label").count()
    x1, label1 = train_group.values.flatten(), train_group.index.values
    x2, label2 = test_group.values.flatten(), test_group.index.values
    drawPie([x1, x2], [label1, label2], "label_distrubution.png")
