# PyTorch 入门

第一个例子：从 5 条数据中学习 `y = 2x + 1`。使用 CPU，不需要显卡或下载数据集。

## 运行

在项目目录打开终端，依次执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_pytorch_intro.py
```

如果使用 VS Code，请选择 `.venv` 中的 Python 解释器。

## 观察结果

- 误差会逐渐下降。
- `weight` 接近 2，`bias` 接近 1。
- 输入 5 时，预测接近 11；输入 10 时，预测接近 21。

训练循环的五步：预测 → 计算误差 → 清空梯度 → 反向传播 → 更新参数。

## 自己动手

1. 将 `y = 2 * x + 1` 改成 `y = 3 * x + 2`，观察学到的参数。
2. 将训练轮数 `1000` 改成 `10`，观察训练不足时的预测。
3. 将学习率 `lr=0.01` 改成 `lr=0.001`，观察误差下降速度。

每次只改一处，方便比较结果。
