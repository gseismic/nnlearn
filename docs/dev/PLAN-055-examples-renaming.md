# PLAN-055：示例文件分组重命名

## 目标

将 `examples/` 中现有的 40 个示例统一改为带类别和类内序号的文件名，让文件列表能体现学习主题，并保持仓库内文档和命令引用可用。

## 命名规则

采用 `{num1}{num2}_{xx}_{yyy}.py`，例如 `010_xx_yyy.py`：

- `num1` 是类别序号，从 0 开始，只占一位；它与类别名 `xx` 一一对应。
- `num2` 是类别内序号，从 0 开始，以两位数字补齐；每个类别内连续递增。`010` 表示类别 `0` 中的第 `10` 个示例。
- `xx` 是固定的小写英文类别名；类别改变时，`num1` 和 `xx` 同时改变。
- `yyy` 是示例主题，使用小写 snake_case。
- 移除旧的 `example` 前缀，不修改示例实现逻辑。

例如：`100_training_linear_reg_simple.py` 表示类别 `1` 中的第 `00` 个示例。

## 类别规划

| num1 | xx | 内容 |
|---:|---|---|
| 0 | `tutorial` | 入门教程 |
| 1 | `training` | 线性回归、MLP、CNN 与 Transformer 训练示例 |
| 2 | `data` | DataLoader、Sampler 与批次数据示例 |
| 3 | `autograd` | 自动微分与梯度开关 |
| 4 | `nnapi` | nn、优化器与损失 API |
| 5 | `tensorapi` | Tensor 创建、形状、索引、归约等 API |
| 6 | `einsum` | einsum 基础用法及扩展行为 |
| 7 | `compat` | PyTorch 脚本兼容与 Torch Protocol |

## 文件映射

| 旧文件 | 新文件 |
|---|---|
| `example38_beginner_tutorial.py` | `000_tutorial_beginner_tutorial.py` |
| `example1_linear_reg_simple.py` | `100_training_linear_reg_simple.py` |
| `example2_linear_reg_compare.py` | `101_training_linear_reg_compare.py` |
| `example3_pytorch_compatible_train.py` | `102_training_pytorch_compatible_train.py` |
| `example4_mlp_train_compare.py` | `103_training_mlp_train_compare.py` |
| `example5_mnist_cnn_train_compare.py` | `104_training_mnist_cnn_train_compare.py` |
| `example6_transformer_train_compare.py` | `105_training_transformer_train_compare.py` |
| `example7_mnist_dataloader_train_compare.py` | `200_data_mnist_dataloader_train_compare.py` |
| `example19_dataloader_sampler_compare.py` | `201_data_dataloader_sampler_compare.py` |
| `example9_requires_grad_compare.py` | `300_autograd_requires_grad_compare.py` |
| `example8_functional_api_compare.py` | `400_nnapi_functional_api_compare.py` |
| `example10_dropout_train_eval_compare.py` | `401_nnapi_dropout_train_eval_compare.py` |
| `example11_softmax_logsoftmax_compare.py` | `402_nnapi_softmax_logsoftmax_compare.py` |
| `example14_nn_init_compare.py` | `403_nnapi_init_compare.py` |
| `example16_batchnorm_compare.py` | `404_nnapi_batchnorm_compare.py` |
| `example17_optimizer_param_groups_compare.py` | `405_nnapi_optimizer_param_groups_compare.py` |
| `example21_loss_functional_reduction_compare.py` | `406_nnapi_loss_functional_reduction_compare.py` |
| `example28_cross_entropy_advanced_compare.py` | `407_nnapi_cross_entropy_advanced_compare.py` |
| `example12_max_api_compare.py` | `500_tensorapi_max_api_compare.py` |
| `example13_argmax_api_compare.py` | `501_tensorapi_argmax_api_compare.py` |
| `example15_shape_api_compare.py` | `502_tensorapi_shape_api_compare.py` |
| `example18_cat_api_compare.py` | `503_tensorapi_cat_api_compare.py` |
| `example20_tensor_creation_compare.py` | `504_tensorapi_tensor_creation_compare.py` |
| `example22_shape_split_chunk_repeat_compare.py` | `505_tensorapi_shape_split_chunk_repeat_compare.py` |
| `example24_topk_compare.py` | `506_tensorapi_topk_compare.py` |
| `example26_sort_argsort_compare.py` | `507_tensorapi_sort_argsort_compare.py` |
| `example27_mask_indexing_numeric_compare.py` | `508_tensorapi_mask_indexing_numeric_compare.py` |
| `example30_scatter_compare.py` | `509_tensorapi_scatter_compare.py` |
| `example31_nonzero_where_compare.py` | `510_tensorapi_nonzero_where_compare.py` |
| `example33_amax_tuple_compare.py` | `511_tensorapi_amax_tuple_compare.py` |
| `example34_amin_aminmax_compare.py` | `512_tensorapi_amin_aminmax_compare.py` |
| `example35_min_compare.py` | `513_tensorapi_min_compare.py` |
| `example23_einsum_compare.py` | `600_einsum_basic_compare.py` |
| `example25_einsum_usage_compare.py` | `601_einsum_classic_usage_compare.py` |
| `example29_einsum_ellipsis_compare.py` | `602_einsum_ellipsis_compare.py` |
| `example32_einsum_repeated_labels_compare.py` | `603_einsum_repeated_labels_compare.py` |
| `example36_pytorch_training_baseline.py` | `700_compat_pytorch_training_baseline.py` |
| `example37_pytorch_script_entry_compat.py` | `701_compat_pytorch_script_entry.py` |
| `example39_training_mechanism_comparison.py` | `702_compat_training_mechanism_comparison.py` |
| `example40_torch_protocol.py` | `703_compat_torch_protocol.py` |

## 实施步骤

1. 按映射表使用 `git mv` 重命名 `examples/` 下的 40 个文件，只改路径，不顺带修改代码内容。
2. 在整个仓库中搜索旧文件名并更新引用，包括 README、教程、设计文档、计划与结果文档中的运行命令和链接。
3. 更新示例间和测试中的模块引用。新文件名以数字开头，不能写进普通 `import` 语句；需要改用 `importlib.import_module()` 按模块名加载，并同步更新测试脚本路径。
4. 复核重命名映射完整性：每个旧文件恰好出现一次，每个新文件唯一；首位类别号与类别名一致，后两位类内序号从 `00` 连续递增。
5. 全仓搜索旧文件名，确认除本计划“文件映射”表保留的源文件名外，没有失效引用；检查 `git diff --check`，并 review 文件改名与文档差异。

## 验收标准

- `examples/` 中原有 40 个脚本均使用新命名格式，文件内容和运行行为不变。
- 文档和脚本中所有指向示例文件的路径均指向新文件名。
- 全仓搜索旧示例文件名时，只有本计划“文件映射”表保留源文件名；其余文档和脚本引用均已更新。
- 重命名与引用更新的差异通过人工 review。

## 范围说明

本计划只整理示例文件名及其仓库内引用。若实施时发现已有示例间的文件路径依赖，则一并更新；不增加新示例、不调整 API。历史文档只替换失效的文件名和命令路径，保留其余历史说明。
