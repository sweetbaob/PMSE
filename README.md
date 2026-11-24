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
