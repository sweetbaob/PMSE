# PMSE Benchmark

PMSE 是一种面向延迟反馈的相位记忆增强在线多步时间序列预测方法。它从短期和长期历史观测中构建预测候选，并按预测步长融合候选和骨干模型输出。完整的预测窗口标签到达后，模型执行在线更新，校准器利用已发布预测的残差修正后续预测。

本仓库提供 PMSE、Proceed 和其他在线预测基线的统一实验入口，支持 PatchTST、iTransformer、TCN 和 DLinear。

## 安装环境

在项目根目录执行。以下 Conda 环境使用 Python 3.10、PyTorch 2.1.2 和 CUDA 12.1。

```bash
conda env create -f environment.yml
conda activate pmse-benchmark
python run.py --help
```

CPU 环境可在 Python 3.10 环境中使用 `python -m pip install -r requirements.txt` 安装依赖，运行时指定 `--device cpu`。

## 准备数据

将所需的数据放入 `data/`。首次运行下面的 ETTh2 示例只需 `ETTh2.csv`。已有数据目录可通过 `--data-root /path/to/datasets` 指定。

| 数据集 | 相对于数据目录的文件路径 |
| --- | --- |
| ETTh2、ETTm1 | `ETTh2.csv`、`ETTm1.csv` |
| Weather、ECL、Traffic | `weather.csv`、`electricity.csv`、`traffic.csv` |
| ExchangeRate、Solar | `exchange_rate.csv`、`solar_AL.txt` |
| PEMS03 | `PEMS03/train.npz`、`PEMS03/val.npz`、`PEMS03/test.npz` |

```bash
python run.py check-data --datasets ETTh2
```

## 运行实验

先预览配置，再在 GPU 0 上运行 PMSE 和 Proceed。示例使用 PatchTST，并在 ETTh2 上预测未来 48 步。

```bash
python run.py run --methods PMSE Proceed --backbones PatchTST --datasets ETTh2 --horizons 48 --seeds 2021 --dry-run
python run.py run --methods PMSE Proceed --backbones PatchTST --datasets ETTh2 --horizons 48 --seeds 2021 --gpu 0 --run-name etth2_h48
python run.py collect --input outputs/etth2_h48 --output outputs/tables
```

每次运行会保存 `resolved.json`、`run.log` 和成功完成后的 `result.json`。汇总生成 `runs.csv` 和 `summary.csv`。省略 `--seeds` 会运行默认的 2021、2022、2023 三个种子，满足配置一致性检查的三种子结果才会标为完整。

用合成数据检查运行流程时，可执行 `python run.py smoke --methods PMSE Proceed --backbones PatchTST`。

## 评测协议与来源

训练段用于拟合标准化参数。在线评测以步长 1 滑动，仅统计完整预测窗口，并在全部 H 步标签到达后使用该样本更新。PMSE 主结果使用 PFTC。代码许可及第三方声明见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)，上游代码版本和配置来源见 [来源清单](docs/source_manifest.json)。

<details>
<summary>完整评测协议与实现说明</summary>

默认预测步长为 24、48、96，PEMS03 使用 12、24、48。FSNet 只支持 TCN。ETTh2 和 ETTm1 使用固定有效前缀，训练、验证和测试的边界分别为 2880/3600/14400 和 11520/14400/57600。PEMS03 从三个 NPZ 文件重建原始序列后重新划分。


### 评测协议与结果兼容性

协议 ID：`proceed_code_fixed_ett_v1`。原始配置依据和实际运行配置分开存档。
官方代码版本见 `upstream.json`；运行脚本快照在 `recipes_source/official`；
基线算法来源还包括经过必要修复的研究实现，见 `source_manifest.json`。

## 固定规则

1. 多变量直接多步预测；输入、目标分别为 `[t-L,t)` 与 `[t,t+H)`。
2. 顺序划分，训练段拟合 StandardScaler。MSE/MAE 在训练段标准化后的单位计算。
3. ETTh2/ETTm1 用官方固定有效前缀，其他数据集用 20/5/75 的整数边界。
4. 预测窗口步长 1；仅统计完整 H 步；全部测试窗口均纳入，不事后剔除高误差通道/窗口。
5. 在线样本 j 的完整目标在 `j+H<=t` 时才能用于时刻 t 的更新。GD/Proceed 等每次先更新成熟样本 `j=t-H`，再预测 t。
6. 验证集 warmup 的目标不得越过验证集终点。本项目显式禁用旧版 `take_post=H-1` 扩展。
7. 三种子 2021/2022/2023；所有超参数选择只能使用训练/验证信息。不得根据测试分数调参。

验证集最后完整预测原点为 `val_end-H`，测试集从 `val_end` 开始。
这与官方窗口切片方式一致，不额外创造跨验证/测试边界的预测目标。
Proceed warmup 可以用完整验证集标签训练适配器，因为在测试开始时这些历史标签均已可见。
测试阶段不会用尚未成熟的测试标签更新模型。

## 长度与扩展项

ETTh2 的 PatchTST/iTransformer 使用 **336**，遵循官方实际脚本。
论文附录的 512 不强制覆盖脚本。ETTm1 和其余非 PEMS03 数据集为 512，TCN 始终 60。
DLinear 的 ETTh2 为 336、PEMS03 为 96，其余为 512；它属于本项目扩展骨干，不能称为 Proceed 原文设置。
PEMS03 的非 TCN 输入 96、预测 12/24/48，同样属于扩展设置。
ExchangeRate/Solar/PEMS03 不属于 Proceed 原文的五数据集实验。

`recipes.json` 优先保存官方脚本的实际 LR、concept_dim、bottleneck_dim、batch_size 和 wo_clip。
iTransformer 的 Proceed 脚本位于官方 `Proact_XY` 目录，其 `online_method` 实为 Proceed。
脚本缺某个 H 时，显式记录 `transferred_from_horizon`；缺整个配置时标为 declared default。
存在多个脚本候选时保留 `script_candidates`，默认使用第一项，不根据本次测试分数挑选。
因此不能把整个矩阵都描述成作者逐项调好的超参数。

## 归一化、协变量与优化器

以实际代码而非论文简写为准：

| 骨干 | 基线归一化 | PMSE 预测路径归一化 | 基线优化器 |
|---|---|---|---|
| PatchTST | 内置 RevIN，affine=False | 同样的内置 RevIN | Adam |
| iTransformer | 原生按窗口 mean/std | 外置无仿射 RevIN，关闭重复内部归一化 | AdamW |
| TCN | 外置 RevIN，保留原实现 affine=True | PMSE 原有无仿射 RevIN | AdamW |
| DLinear | 原生 DLinear，无外置 RevIN | PMSE 原有无仿射 RevIN | Adam |

PMSE 的归一化路径属于其既有方法实现；未为追求基线成绩而统一改造 DLinear 或 TCN。
论文应如实描述此差异，不能声称所有方法归一化层逐参数相同。
PMSE 的优化器保留 Adam；其离线新模块 LR 从原配置候选中按验证 MSE 选择。
基线 AdamW 来自官方 `run.py` 对 need_x_mark 模型的实际覆盖，因此不宜笼统写“所有实验均 Adam”。

TCN 的七维日历特征和 iTransformer 的频率编码，在 PMSE 与对应基线间来自同一数据层。
OneNet 的 iTransformer 分支和 FSNet 分支分别接收各自预训练使用的编码。
Solar/PEMS03 没有原始时间戳时置零。修改此策略将产生新的结果指纹。
PMSE 在线沿用 eval 模式更新（含 BN 固定）；基线沿用原算法的 train/update、eval/predict 切换。
这些是方法内部策略，不将其误写成完全相同的网络训练状态。

## 历史结果核对

新项目不修改任何历史结果，也不自动从旧日志中猜测 checkpoint 或实验配置。
使用已有数值前至少核对：数据哈希、有效前缀、L/H、变量顺序、归一化、优化器、
离线训练已执行、完整 H 反馈、验证 warmup 边界、随机种子，以及本地必要修复是否已包含。
只要配置匹配且运行正常，保留已有结果。若有必要修复造成协议改变，需标注哪些结果不再同协议，
不可把新旧实现的种子混成一个均值。

对异常基线仅排查必要的代码、数据或协议错误；不以在新数据集上达到某个分数为交付条件。

### 实现边界与必要修复

`src/pmse_benchmark` 是新建的统一实验层；`src/pmse` 保留 PMSE 实现；
`vendor/onlinetsf` 保留基线算法和原始模块组织。新入口不执行旧的巨型 run.py，
不继承历史 checkpoint 命名猜测、注释掉的训练调用或共享日志路径。

## 公共实验层

- 单一配置解析器生成不可变的单次实验配置；薄 shell 脚本仅选择方法/骨干/数据集。
- 两个算法引擎使用同一个原始序列、训练 scaler、变量顺序、窗口原点和日历编码。
- 每次实验记录全部配置、数据与代码哈希、预训练来源、日志及原子写入的结果。
- 按骨干/数据集/H/种子/配置/数据/代码指纹复用预训练；不复用不匹配的历史权重。
- 成功且匹配的运行跳过；中断运行拒绝静默覆盖；目录锁阻止冲突写入。
- 离线训练与验证实际执行；补齐旧运行入口中被注释而没有执行的训练阶段。
- 汇总要求同协议、同实现、同配置；不足三个种子的组明确标为不完整。

## PMSE 集成改动

- 适配 src 目录路径，并允许 PEMS03 的 H=12。
- 将可选 `x_mark` 从数据层传到原生 TCN/iTransformer，避免 PMSE 与基线日历信息不一致。
  相应调用也经过原有 published_features 路径，不另增预测头。
- 保留既有因果相位记忆、训练/验证选择、全 H 延迟更新和成熟反馈校准。
- 主表提取 PFTC；PFT 与通道/分段诊断仍由 PMSE 引擎记录。

## 基线必要修复

- 通过 CommonExperiment 严格保存/恢复完整模型、优化器、requires_grad、Proceed 的
  non-persistent recent_batch、FSNet 的梯度/控制器状态及 OneNet 的两套额外优化器和 bias。
  验证适应后的状态不会污染后续离线训练。
- 最终模型构造完成后建立优化器并验证参数所有权，避免 Proceed 替换 Linear/Conv 后残留旧参数引用。
- 按参数名称恢复匹配预训练分支的 Adam 状态，适配器新参数保持新状态；OneNet 辅助 FSNet 独立预训练。
- warmup 传入显式有界 Dataset_Recent，禁止旧默认 take_post 进入测试目标。
- FSNet 梯度累积状态改为可随设备迁移的 buffer；移除强制 CUDA 构造；q_ema 历史状态断开旧梯度图。
- FSNet 检索由错误的“把内存槽编号当权重”修为用归一化 attention 值，保留 top-2 检索机制。
  这会与包含该错误的旧 FSNet/OneNet 结果不同，须按协议重新核对，不能宣称逐位一致。
- RevIN wrapper 转发 FSNet 的 store_grad 和触发开关，避免包装后钩子丢失。
- ER/DERpp 修复 list/tuple 拼接、输出解包和缓存预测 detach，保留仓库的回放损失形式。
  此处 DERpp 标签指 vendored 实现，不宣称重新实现其他论文版本的 DER++。
- SOLID++ 新增 Solar 的 144 步日周期，PEMS03 使用 288 步；属于显式数据集扩展。
- MSE/MAE 累加使用 float64；预测张量仍按模型 float32 计算。
- TCN/FSNet 尺寸参数增加可选读取，正式配置仍保持 320/64/10，小型 smoke 才缩小模型。
- 单进程单 GPU 的可见设备只在入口设定，不在模型内部重复映射物理编号。

## 与上游数值复现的距离

本项目以官方参数脚本、原生算法和一致数据协议为依据，并修复必要实现错误。
它采用单 GPU 顺序实验，训练 DataLoader 不丢弃最后一个不满 batch 的窗口；原仓库部分路径 drop_last=True。
原始论文可能使用多 GPU / 不同硬件和旧 PyTorch，因此这里不声称 bitwise reproduction。
注意力骨干的预训练显式使用 `pretraining.micro_batch` 累积梯度，保持官方全局 batch 的优化器步数；
验证 MSE 按全部元素加权，不对不满微批次等权平均。TCN/FSNet/OneNet 主训练不改其专有更新步骤。
BatchNorm 的统计粒度随微批次变化，因此与多卡原始训练不保证逐位相同；所有尺寸写入 resolved.json。
如仍需改全局 batch，可显式 `--batch-size`，覆盖会写入配置并隔离结果，不能隐藏调整。
完整运行的最终指标、耗时和显存仍需实际主实验确认。

</details>
