# 半精度规约与归一化的中间精度

日期：2026-10-25。对应 PLAN-063。

## 问题与契约

CPU 基线为 702 项通过，但 float16 的 70000 类均匀 Softmax 输出全零，LayerNorm 在形状 (2, 128)、范围 [-256, 254] 的有限输入上输出全零。前者的指数和、后者的平方和先在半精度中溢出，末尾恢复 dtype 无法补救。

Softmax、LogSoftmax、交叉熵以及 LayerNorm/BatchNorm 的半精度输入应在 float32 中完成指数、统计量和归约，并在公开输出与回传边恢复原有浮点精度。显式混合精度参数仍按公共 dtype 输出；float32/float64 路径不降精度。真正超过输出 dtype 表示范围的最终结果不承诺有限。

## 实现范围

- Softmax/LogSoftmax 在减最大值之前转换半精度输入，反向也在 float32 中做乘积与求和，保持 Tensor 表达式支持高阶导数。
- 交叉熵对 logits、类别权重、概率、损失和均值分母使用累加精度，最终输出才恢复输入 dtype；忽略标签、类别权重、标签平滑和三种 reduction 保持现有规则。
- LayerNorm 在 float32 中求均值、方差及梯度中间量，输出按输入和仿射参数的公共 dtype 返回。
- BatchNorm 对半精度输入采用可微 float32 转换，running statistics 按缓冲区本身 dtype 写回，不改变整数/布尔缓冲区。

## 验收

大类别、大批次、大幅值及多轴形状的前向与梯度对照 float32/float64 参考；覆盖 float16/32/64、模块/函数式入口、连续优化及状态精度。半精度 PyTorch 自身溢出的交叉熵用 float32 参考验证，不复制参考实现的溢出。保留 702 项既有测试，并在 NumPy 1.24/2.2 和可用 CUDA 上复验。
