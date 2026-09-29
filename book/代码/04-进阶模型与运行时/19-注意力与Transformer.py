"""训练一个最小 Transformer 编码器完成合成序列分类。"""

from pathlib import Path
import sys

import numpy as np


PART3_CODE = Path(__file__).resolve().parents[1] / "03-神经网络训练"
if str(PART3_CODE) not in sys.path:
    sys.path.insert(0, str(PART3_CODE))

from mini_nn import (  # noqa: E402
    CrossEntropyLoss,
    Embedding,
    Linear,
    Module,
    SGD,
    Tensor,
    TransformerEncoderLayer,
)


class TinyTransformerClassifier(Module):
    def __init__(self, vocab_size=12, d_model=8):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoderLayer(
            d_model=d_model, num_heads=2, dim_feedforward=16
        )
        self.classifier = Linear(d_model, 2)

    def forward(self, token_ids):
        hidden = self.embedding(token_ids)
        hidden = self.encoder(hidden)
        pooled = hidden.mean(axis=1)
        return self.classifier(pooled)


def make_dataset(seed=19, samples_per_class=20, sequence_length=5):
    rng = np.random.default_rng(seed)
    sequences = []
    labels = []
    for label, token_range in enumerate((range(0, 4), range(8, 12))):
        for _ in range(samples_per_class):
            sequences.append(
                rng.choice(list(token_range), size=sequence_length)
            )
            labels.append(label)
    return (
        Tensor(np.asarray(sequences, dtype=np.int64)),
        Tensor(np.asarray(labels, dtype=np.int64)),
    )


def main():
    np.random.seed(19)
    token_ids, labels = make_dataset()
    model = TinyTransformerClassifier()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.06)

    initial_loss = loss_fn(model(token_ids), labels).item()
    for _ in range(120):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    logits = model(token_ids)
    final_loss = loss_fn(logits, labels).item()
    accuracy = np.mean(np.argmax(logits.data, axis=1) == labels.data)
    print(f"token 形状: {token_ids.shape}")
    print(f"嵌入形状: {model.embedding(token_ids).shape}")
    print(f"初始交叉熵: {initial_loss:.6f}")
    print(f"最终交叉熵: {final_loss:.6f}")
    print(f"合成序列准确率: {accuracy:.3f}")

    repeated_tokens = Tensor(np.array([[1, 1]], dtype=np.int64))
    tiny_embedding = Embedding(num_embeddings=4, embedding_dim=2)
    tiny_embedding(repeated_tokens).sum().backward()
    print(f"重复 token 对词向量行的梯度: {tiny_embedding.weight.grad[1].tolist()}")


if __name__ == "__main__":
    main()
