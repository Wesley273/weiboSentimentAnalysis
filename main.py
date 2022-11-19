import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import args
from dataset import WeiBoDataset
from model.BiGRU import BiGRU
from model.BiLSTMAttention import BiLSTMAttention


def drawPlot(heights, fname, ylabel, legends=None):
    """
    功能：绘制训练集上的准确率和测试集上的loss和acc变化曲线
    heights: 纵轴值列表
    fname：保存的文件名
    marker：曲线上每个点的形状设置
    """
    plt.figure()
    x = [i for i in range(1, len(heights[0]) + 1)]
    # 绘制训练集和测试集上的loss变化曲线子图
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    # 设置横坐标的刻度间隔
    plt.xticks([i for i in range(0, len(heights[0]) + 1, 5)])
    for i in range(len(heights)):
        plt.plot(x, heights[i])
    if legends:
        plt.legend(legends)
    plt.savefig("images/{}".format(fname))
    plt.show()


def train(model, train_iter, loss_fn, optimizer, iscuda=False):
    """
    功能：训练模型
    train_iter：训练集迭代器
    test_iter: 测试机迭代器
    loss_fn: 损失函数
    lr: 学习率
    epochs：数据集迭代次数
    optimizer: 优化器
    params：手动实现模型时的可学习参数
    """
    model.train()
    train_l_sum, train_acc_sum, n, c = 0, 0, 0, 0
    iteration = 0
    for x, y in train_iter:
        if iscuda:
            x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        l = loss_fn(y_hat, y)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        l.backward()
        # 梯度裁剪
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        # 更新参数
        optimizer.step()
        train_l_sum += l.item()
        c += 1
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        iteration += 1
        if iteration % 20 == 0:
            print("iteration:{}, train loass:{}".format(iteration, l.item()))

    return train_l_sum / c, train_acc_sum / n


def test(model, test_iter, loss_fn, iscuda=False):
    """
    功能：对数据集上进行预测并返回准确率和平均loss
    """
    model.eval()
    test_l_sum, acc_sum, n, c = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in test_iter:
            if iscuda:
                x, y = x.cuda(), y.cuda()
            y_hat = model.forward(x)
            test_l_sum += loss_fn(y_hat, y)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            c += 1

    return test_l_sum / c, acc_sum / n


if __name__ == "__main__":
    output_size = 6

    train_dataset, test_dataset = WeiBoDataset("datasets/train.txt"),\
        WeiBoDataset("datasets/test.txt")

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("loading dataset done!!!")
    if args.model == "bigru":
        model = BiGRU(
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            output_size=output_size,
            drop_rate=args.drop_prob,
            word2vec_embedding=args.extra_embedding)
    elif args.model == "bilstm_attention":
        model = BiLSTMAttention(
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            output_size=output_size,
            drop_rate=args.drop_prob,
            word2vec_embedding=args.extra_embedding)
    print("loading model done!!!")

    iscuda = False
    if torch.cuda.is_available():
        model = model.cuda()
        iscuda = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=10,min_lr=1e-8)
    loss_fun = nn.CrossEntropyLoss()
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    for e in range(args.epochs):
        print("==========epoch {}==========".format(e + 1))
        train_loss, train_acc = train(model, train_dataloader, loss_fun, optimizer, iscuda)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss, test_acc = test(model, test_dataloader, loss_fun, iscuda)
        # scheduler.step(test_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print("epoch: {} train_loss: {:.4} test_loss: {:.4} train_acc: {:.4} test_acc: {:.4}".format(
            e + 1, train_loss, test_loss, train_acc, test_acc))
    # draw loss curve
    save_path = "e{}_b{}_h{}_lr{}_drop{}_wd{}_Loss.png".format(args.epochs,
                                                               args.batch_size, args.hidden_size, args.lr, args.drop_prob, args.weight_decay)
    drawPlot([train_loss_list, test_loss_list], save_path, "Loss", ["train loss", "test loss"])
    # draw acc curve
    save_path = "e{}_b{}_h{}_lr{}_drop{}_wd{}_Acc.png".format(args.epochs,
                                                              args.batch_size, args.hidden_size, args.lr, args.drop_prob, args.weight_decay)
    drawPlot([train_acc_list, test_acc_list], save_path, "Acc", ["train acc", "test acc"])
