# PyTorch 入门

## 微型语言模型（第二个例子）

`tiny_lm/` 包含一个从零训练的字符级 Transformer。它具有字符嵌入、位置嵌入、因果注意力、前馈网络和下一个字符预测目标，展示 GPT 类语言模型的核心流程。规模约十万参数，适合 CPU 学习；它不是具备通用能力的大模型。

在项目目录运行：

```bash
source .venv/bin/activate
python tiny_lm/train.py
python tiny_lm/demo.py
```

如果还没有安装环境，先按下面“运行”章节安装 PyTorch。

- `tiny_lm/data.txt`：人工编写的 26 组中文问答，供教学使用。
- `tiny_lm/model.py`：模型结构和逐字符生成。
- `tiny_lm/train.py`：随机采样、训练、保存参数及词表。
- `tiny_lm/demo.py`：加载 `checkpoint.pt`，在终端输入问题。

默认训练 1000 步，可用 `python tiny_lm/train.py --steps 2000` 增加训练步数。每次训练从随机参数开始，并覆盖本例的 checkpoint.pt。

例如输入 `什么是机器学习` 或 `春天是什么样的`。这个数据集很小，模型主要记忆训练语句，可能生成混乱或重复文字；训练 loss 下降只说明拟合训练数据，本例没有独立验证集，不能据此判断泛化能力。词表仅包含训练文件中出现的字符，输入其他字符时 demo 会提示。修改数据后需重新训练。

建议阅读顺序：先看 train.py 中的 x/y 如何错开一个字符，再看 model.py 的因果遮罩，最后看 generate 如何循环生成。

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
