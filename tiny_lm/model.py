"""字符级、自回归 Transformer：根据前面的字符预测下一个字符。"""
import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, context=64, width=64, heads=4, layers=2):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab_size))

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        # 上三角遮罩：每个位置只能看到自己及之前的字符，不能偷看答案。
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.output(self.transformer(hidden, mask=mask))

    @torch.no_grad()
    def generate(self, tokens, count=100, temperature=0.7):
        self.eval()
        for _ in range(count):
            logits = self(tokens[:, -self.context:])[:, -1, :]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
