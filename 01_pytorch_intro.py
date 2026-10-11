"""第一个 PyTorch 示例：让模型从数据中学习 y = 2x + 1。"""

import torch
from torch import nn


def main():
    # 固定随机种子，让每次运行的结果便于比较。
    torch.manual_seed(42)
    print(f"PyTorch 版本：{torch.__version__}")

    # 1. 张量（Tensor）：可以理解为支持自动求导的多维数组。
    a = torch.tensor([1.0, 2.0, 3.0])
    print("\n张量 a：", a)
    print("a * 2：", a * 2)
    print("a 的形状：", a.shape)

    # 2. 准备训练数据。每一行是一条样本，每条样本只有一个输入。
    # x 的形状是 [5, 1]，也就是 5 条样本、1 个特征。
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 2 * x + 1

    # 3. 模型：一个输入、一个输出，计算 prediction = weight * x + bias。
    # weight 和 bias 一开始是随机值，训练会不断调整它们。
    model = nn.Linear(in_features=1, out_features=1)
    loss_fn = nn.MSELoss()  # 均方误差：预测值与正确答案的差距。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("\n开始训练：")
    for epoch in range(1000):
        prediction = model(x)           # 前向计算：模型给出预测。
        loss = loss_fn(prediction, y)    # 计算预测误差。
        optimizer.zero_grad()           # 清除上一轮的梯度。
        loss.backward()                 # 自动计算每个参数的梯度。
        optimizer.step()                # 根据梯度更新参数。

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"第 {epoch + 1:4d} 轮 | 误差：{loss.item():.6f}")

    print(f"\n学到的 weight：{model.weight.item():.4f}（目标是 2）")
    print(f"学到的 bias：  {model.bias.item():.4f}（目标是 1）")

    # 4. 使用模型预测新输入。预测时不需要计算梯度。
    with torch.no_grad():
        test_x = torch.tensor([[5.0], [10.0]])
        test_y = model(test_x)
        for input_value, output_value in zip(test_x, test_y):
            print(f"输入 {input_value.item():.0f} → 预测 {output_value.item():.4f}")


if __name__ == "__main__":
    main()
