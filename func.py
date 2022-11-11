import matplotlib.pyplot as plt

def drawPlot(heights,fname,ylabel,legends=None):
    """
    功能：绘制训练集上的准确率和测试集上的loss和acc变化曲线
    heights: 纵轴值列表
    fname：保存的文件名
    marker：曲线上每个点的形状设置
    """
    plt.figure()
    x = [i for i in range(1,len(heights[0]) + 1)]
    # 绘制训练集和测试集上的loss变化曲线子图
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    # 设置横坐标的刻度间隔
    plt.xticks([i for i in range(0,len(heights[0]) + 1,5)])
    for i in range(len(heights)):
        plt.plot(x,heights[i])
    if legends:
        plt.legend(legends)
    plt.savefig("images/{}".format(fname))
    plt.show()

if __name__ == "__main__":
    pass