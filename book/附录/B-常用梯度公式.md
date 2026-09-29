# B 常用梯度公式

这里的 `g` 表示从后续计算传回来的上游梯度。多输入算子的每条输入边都要按链式法则计算自己的梯度；同一个 Tensor 被多次使用时，各条路径的梯度相加。

## 标量与逐元素运算

| 前向 | 对输入的梯度 |
| --- | --- |
| `z = x + y` | `dz/dx = 1`，`dz/dy = 1`；上游为 `g` 时两侧均为 `g` |
| `z = x * y` | `dz/dx = y`，`dz/dy = x`；反向为 `gx = g*y`、`gy = g*x` |
| `z = x**p` | `dz/dx = p*x**(p-1)`；须满足该幂函数在输入处有定义 |
| `y = sigmoid(x)` | `dy/dx = y*(1-y)` |
| `y = tanh(x)` | `dy/dx = 1-y**2` |
| `y = ReLU(x)` | `x>0` 时为 `1`，`x<0` 时为 `0`；本书在 `x=0` 取 `0` |

反向传播将局部导数乘以上游梯度。例如 `z=x*y` 时，`gx=g*y`。若损失由 `z` 经多条支路依赖 `x`，则把每条支路传回的 `gx` 加起来。

## 求和、均值和广播

若 `y = sum(x, axis=A)`，则 `gy` 需要沿被约简轴扩展回 `x` 的形状；被求和的每个输入位置接收对应输出位置的梯度。若使用 `keepdims=True`，被约简轴保留为长度 `1`，扩展更直接。

若 `y = mean(x)` 且 `x` 有 `N` 个元素，则每个元素梯度为 `gy/N`。沿指定轴求均值时，`N` 是该轴或这些轴包含的元素数。

广播运算的反向要把扩展出来的维度求和。例如：

```text
A: (N, D)   b: (D,)
y = A + b   y: (N, D)
```

若 `gy` 形状为 `(N,D)`，则 `gb = sum(gy, axis=0)`，恢复偏置的 `(D,)` 形状。通用做法是：先对梯度比目标多出的前导轴求和，再对目标中长度为 `1`、梯度中长度大于 `1` 的轴求和并保留维度，最后恢复原形状。

## 矩阵乘法、线性层和 MSE

对二维矩阵 `Y = A @ B`，若 `A:(M,K)`、`B:(K,N)`、`G=dL/dY:(M,N)`：

```text
dL/dA = G @ B.T       # (M, K)
dL/dB = A.T @ G       # (K, N)
```

线性层 `Y = X @ W + b` 的权重和偏置梯度为：

```text
dW = X.T @ G
db = sum(G, axis=0)    # 对 batch 轴求和
```

若 `L = mean((prediction-target)**2)`，共有 `N` 个参与均值的元素，则：

```text
dL/dprediction = 2 * (prediction-target) / N
```

这里的 `N` 取决于均值覆盖的元素总数，不一定只是批次大小。

## softmax 与交叉熵

对一行 logits `z`，`p=softmax(z)`。给定类别 `c` 的交叉熵为 `L=-log(p_c)`，其 logits 梯度为 `p-one_hot(c)`。对 `B` 个样本取平均时，再除以 `B`：

```text
dL/dlogits = (softmax(logits) - one_hot(target)) / B
```

稳定计算先减去每行最大值 `m`：

```text
logsumexp(z) = m + log(sum(exp(z-m)))
```

减去 `m` 不改变 softmax 概率，却能减小指数溢出的风险。

## 卷积与最大池化

二维卷积一个输出位置可以写为：

本书及仓库的 `Conv2d` 按深度学习库常见约定直接将输入窗口与卷积核相乘，不翻转卷积核（严格说是互相关形式）。

```text
Y[n,o,i,j] = sum over c,u,v of X_patch[n,c,u,v] * W[o,c,u,v] + bias[o]
```

对应窗口的梯度贡献为：

```text
dW[o,c,u,v] += X_patch[n,c,u,v] * G[n,o,i,j]
dX_patch[n,c,u,v] += W[o,c,u,v] * G[n,o,i,j]
dbias[o] += G[n,o,i,j]
```

若窗口重叠，落到同一输入元素或卷积核元素的贡献必须累加。最大池化在每个窗口保存 `argmax`，反向只将梯度加回获胜位置；并列最大值采用实现规定的一个位置。

## 注意力与 SGD

对单头缩放点积注意力：

```text
S = Q @ K.T / sqrt(d)
P = softmax(S, axis=-1)
C = P @ V
```

反向按计算顺序逐步应用矩阵乘法和 softmax 的导数：

```text
gV = P.T @ gC
gP = gC @ V.T
gS = P * (gP - sum(gP*P, axis=-1, keepdims=True))
gQ = gS @ K / sqrt(d)
gK = gS.T @ Q / sqrt(d)
```

最后，基础 SGD 的一次参数更新为：

```text
parameter = parameter - learning_rate * parameter.grad
```

没有梯度的参数不更新。每个新批次开始前通常先清理上一批次累积的梯度。
