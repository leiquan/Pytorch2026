"""运行：python tiny_lm/train.py；训练结果保存在本目录 checkpoint.pt。"""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from model import TinyLanguageModel

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    text = (ROOT / "data.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    char_to_id = {char: i for i, char in enumerate(chars)}
    data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    config = dict(vocab_size=len(chars), context=64, width=64, heads=4, layers=2)
    model = TinyLanguageModel(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print(f"字符数：{len(data)}；词表：{len(chars)}；参数量：{sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("使用 CPU 训练，loss 越低代表预测下一个字符越准确。", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        # 随机截取 16 段文字，标签是输入向右移动一个字符。
        starts = torch.randint(len(data) - model.context, (16,))
        x = torch.stack([data[i:i + model.context] for i in starts])
        y = torch.stack([data[i + 1:i + model.context + 1] for i in starts])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f}", flush=True)
    torch.save(dict(config=config, chars=chars, state_dict=model.state_dict()), ROOT / "checkpoint.pt")
    prompt = "问：你好\n答："
    tokens = torch.tensor([[char_to_id[c] for c in prompt]])
    result = model.generate(tokens, count=80)[0].tolist()
    print("\n训练后续写示例：\n" + "".join(chars[i] for i in result))
    print("\n模型已保存。运行 python tiny_lm/demo.py 进行交互。")


if __name__ == "__main__":
    main()
