import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from d2l import torch as d2l
import time
T1 = time.time()
# Pytorch 基于NiN的服饰识别（使用Fashion-MNIST数据集）
# https://developer.aliyun.com/article/1069741


# 2.定义 NiN 网络结构
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    # 定义NiN块
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

# 定义NiN网络
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())

# 3.下载并配置数据集和加载器
# 这里 NiN 输入图片尺寸应为 224*224，我们将 28*28 的 Fashion-MNIST 图片拉大到 224*224。
# 下载并配置数据集
trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root=r'NiNdataset', train=True,
                                      transform=trans, download=True)
test_dataset = datasets.FashionMNIST(root=r'NiNdataset', train=False,
                                     transform=trans, download=True)

# 配置数据加载器
batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# 4.定义训练函数
# 训练完成后会保存模型，可以修改模型的保存路径。

def train(net, train_iter, test_iter, epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print(f'Training on:[{device}]')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 30) == 0 or i == num_batches - 1:
                print(f'Epoch: {epoch+1}, Step: {i+1}, Loss: {train_l:.4f}')
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print(
            f'Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%')
    print(f'{metric[2] * epochs / timer.sum():.1f} examples/sec '
          f'on: [{str(device)}]')
    torch.save(net.state_dict(),
               f"./Model/NiN_Fashion-MNIST_Epoch{epochs}_Accuracy{test_acc*100:.2f}%.pth")


# 5.训练模型（或加载模型）
# 如果环境正确配置了CUDA，则会由GPU进行训练。加载模型需要根据自身情况修改路径。
epochs, lr = 1, 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train(net, train_loader, test_loader, epochs, lr, device)
# 加载保存的模型
# net.load_state_dict(torch.load(r".\Model\NiN_Fashion-MNIST_Epoch20_Accuracy89.41%.pth"))

# 6.可视化展示
def show_predict():
    # 预测结果图像可视化
    net.to(device)
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    plt.figure(figsize=(12, 8))
    name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(9):
        (images, labels) = next(iter(loader))
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        title = f"Predicted: {name[int(predicted[0])]}, True: {name[int(labels[0])]}"
        plt.subplot(3, 3, i + 1)
        plt.imshow(images.cpu()[0].squeeze())
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()


show_predict()

T2 = time.time()
print('程序运行时间:%s分钟' % ((T2 - T1)/60))
# 7.预测图