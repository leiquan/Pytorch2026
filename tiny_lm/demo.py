"""加载训练好的模型，尝试中文问答式续写。"""
from pathlib import Path
import torch
from model import TinyLanguageModel


def main():
    torch.set_num_threads(2)
    path = Path(__file__).resolve().parent / "checkpoint.pt"
    if not path.exists():
        print("请先运行 python tiny_lm/train.py 训练模型。")
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    chars = checkpoint["chars"]
    char_to_id = {char: i for i, char in enumerate(chars)}
    model = TinyLanguageModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    print("模型已加载！输入问题，输入 /quit 退出。")
    print("试试：你好 / 什么是机器学习 / 为什么要清空梯度 / 春天是什么样的")
    print("这是小数据上的续写演示，回答可能重复或出错。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/quit":
            break
        if not question:
            continue
        prompt = f"问：{question}\n答："
        unknown = sorted(set(prompt) - set(chars))
        if unknown:
            print("训练词表没有这些字符：" + "".join(unknown) + "。请换个问题，或扩充 data.txt 后重新训练。")
            continue
        tokens = torch.tensor([[char_to_id[c] for c in prompt]])
        result = model.generate(tokens, count=100)[0].tolist()[len(prompt):]
        answer = "".join(chars[i] for i in result).split("\n")[0]
        print("模型：" + (answer or "（生成了换行，请再试一次）"))


if __name__ == "__main__":
    main()
