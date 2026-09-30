# 19 注意力与 Transformer 编码器

卷积在空间邻近的窗口间移动；自注意力则让序列中的每个位置根据内容汇总其他位置的信息。本章实现词向量查表、多头缩放点积注意力、LayerNorm 和一个最小 Transformer 编码器层，并在合成 token 序列上训练分类器。

## 从 token 到向量

输入序列通常是整数 token id，例如形状 `(batch, time)`。Embedding 把每个 id 映射到一行可训练向量表，输出 `(batch, time, embedding_dim)`。同一个 token 若在输入中出现多次，它的梯度也要把各个位置的贡献累加到同一行。

脚本单独演示了 token `1` 在一条序列里出现两次的情况。对嵌入向量求和后，该行梯度为 `[2, 2]`，说明重复索引的更新不能互相覆盖。

## 缩放点积注意力

将每个位置的向量分别投影为查询 `Q`、键 `K` 和值 `V`。单个注意力头计算：

```text
scores = Q @ K.T / sqrt(head_dim)
weights = softmax(scores, axis=-1)
context = weights @ V
```

减去每行最大分数后再计算 softmax，可以降低指数运算溢出的风险。除以 `sqrt(head_dim)` 则控制分数随维度增长的尺度。反向时先从 `context = weights @ V` 得到对权重和值的梯度，再经过 softmax 雅可比，最后传回 Q 和 K。

多头注意力把特征维切成多个较小的头，每个头独立计算注意力，再拼接并经过输出投影。实现内部把张量按 `(batch, time, heads, head_dim)` 整理，在注意力计算中转为 `(batch, heads, time, head_dim)`。

## 编码器层

本章的编码器层按以下顺序计算：

```text
h = LayerNorm(x + MultiHeadSelfAttention(x))
y = LayerNorm(h + Linear2(ReLU(Linear1(h))))
```

残差连接保留原输入的信息；LayerNorm 沿最后的特征维归一化；前馈网络对每个位置分别应用两层线性变换。`Embedding`、注意力和前馈层都参与同一条自动微分计算图。

## 运行并观察训练

运行[配套脚本](../代码/04-进阶模型与运行时/19-注意力与Transformer.py)：

```bash
python 'book/代码/04-进阶模型与运行时/19-注意力与Transformer.py'
```

脚本用两个互不重叠的 token 子集构造分类任务，标签只取决于序列属于哪一组，因此这是一个便于观察训练闭环的合成任务。固定随机种子下，交叉熵从 `1.277568` 降到 `0.003678`，准确率为 `1.000`；嵌入输出形状是 `(40, 5, 8)`。

## 实现边界

当前实现支持 batch-first 的等长自注意力，不支持 attention mask、因果遮罩、dropout、交叉注意力或位置编码。示例任务刻意不依赖 token 顺序，因此不能据此判断模型已经学会顺序信息。LayerNorm 只归一化最后一维；编码器仅有一个注意力层和一个前馈层，也没有完整 Transformer 的训练与推理接口。

## 练习

1. 调整头数和 `head_dim`，观察 Q、K、V 的形状变化。
2. 改造合成标签，让类别与 token 顺序有关，再加入可训练的位置向量。
3. 为注意力增加 padding mask，确保被屏蔽的位置权重为零。

## 对应到仓库

仓库的 [`nlearn/nn/transformer.py`](../../nlearn/nn/transformer.py) 包含多头注意力和编码器层；[`nlearn/nn/sparse.py`](../../nlearn/nn/sparse.py) 与 [`nlearn/nn/normalization.py`](../../nlearn/nn/normalization.py) 提供 Embedding 和 LayerNorm。`examples/example6_transformer_train_compare.py` 展示了一个端到端序列分类例子。
