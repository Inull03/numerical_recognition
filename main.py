import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义卷积神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络的全连接层
        self.fc1 = torch.nn.Linear(28 * 28, 64)  # 输入层到隐藏层1
        self.fc2 = torch.nn.Linear(64, 64)       # 隐藏层1到隐藏层2
        self.fc3 = torch.nn.Linear(64, 64)       # 隐藏层2到隐藏层3
        self.fc4 = torch.nn.Linear(64, 10)       # 隐藏层3到输出层（10个类别）

    def forward(self, x):
        # 定义前向传播过程
        x = torch.nn.functional.relu(self.fc1(x))  # 通过第一层并应用ReLU激活
        x = torch.nn.functional.relu(self.fc2(x))  # 通过第二层并应用ReLU激活
        x = torch.nn.functional.relu(self.fc3(x))  # 通过第三层并应用ReLU激活
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 通过输出层并应用log_softmax激活
        return x

# 定义数据加载函数
def get_data_loader(is_train):
    # 定义数据转换方式，将图像转换为Tensor
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载MNIST数据集，is_train决定加载训练集还是测试集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 返回DataLoader，设置批量大小为15，并进行洗牌
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 定义评估函数
def evaluate(test_data, net):
    n_correct = 0  # 记录正确预测的数量
    n_total = 0    # 记录总预测数量
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))  # 将输入展平并通过网络
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:  # 判断预测结果是否正确
                    n_correct += 1  # 正确预测数量加1
                n_total += 1  # 总数量加1
    return n_correct / n_total  # 返回准确率

# 主函数
def main():
    train_data = get_data_loader(is_train=True)  # 获取训练数据
    test_data = get_data_loader(is_train=False)   # 获取测试数据
    net = Net()  # 初始化神经网络

    print("initial accuracy:", evaluate(test_data, net))  # 输出初始准确率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器
    for epoch in range(2):  # 进行2个epoch的训练
        for (x, y) in train_data:
            net.zero_grad()  # 清空梯度
            output = net.forward(x.view(-1, 28 * 28))  # 通过网络得到输出
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))  # 输出当前epoch的准确率

    # 可视化预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 2:  # 只展示前4个测试样本
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))  # 预测结果
        plt.figure(n)  # 创建新图形
        plt.imshow(x[0].view(28, 28))  # 显示图像
        plt.title("prediction: " + str(int(predict)))  # 设置标题为预测结果
    plt.show()  # 展示所有图形

if __name__ == "__main__":
    main()  # 执行主函数
