# Motion Foundation · 研究框架

面向压力与 IMU 波形的可扩展运动时序框架，基于 V13。本项目位于 `testFM/motion-foundation/`，与仓库其他代码独立维护。不包含第三方数据和模型权重；公开可见不等于授予开源许可，使用与再分发须核实适用授权。

当前是**未预训练的研究框架**，默认编码器 533,760 参数。支持压力、acc-only、gyro-only、缺轴、变通道及混频；双向主干不是因果在线模型。视频/EMG未实现，计算兼容不代表迁移效果。

## 安装与自检

从仓库根目录进入 `motion-foundation/`，使用 Python 3.10+；下列命令均在此子目录执行：

```bash
python -m venv .venv
```

激活环境后，按机器硬件安装相容的 PyTorch，再执行：

```bash
python -m pip install -e ".[dev,ingest]"
motion-selfcheck
python -m pytest -q
```

`constraints-tested.txt`记录本次Python 3.11实测依赖版本，可用`pip install -c constraints-tested.txt -e ".[dev,ingest]"`复用；这是实测约束，不是跨平台完整锁文件。Python 3.10/3.12由CI配置覆盖，尚未在本次本地验收中运行。

核心推理只需 `python -m pip install -e .`；真实原始格式转换需要 `ingest` 依赖。CPU可运行所有合成测试，不需要真实数据、网络盘或GPU。GPU安装需匹配显卡架构，不能仅用 `cuda.is_available()` 判断相容。

## 代码入口

```text
src/motion_foundation/
  contracts.py        数据、时间、缺失及表征契约
  model.py            V13前端、器件池化、同步分组、共享时间主干及任务头
  registry.py         模型工厂和配置入口
  corpus.py           数据集无关的缓存写入器
  adapters.py         标准NPZ适配器注册与单流清单读取
  native_data.py      多器件清单、连续裁窗、均衡采样与预算
  objectives.py       掩码目标、准入与异构微批引擎
  denoising.py        观测空间去噪目标
  runtime.py          掩码模型包保存/加载
  train_mixed.py      native语料去噪训练入口
  train.py            单流清单掩码训练入口
  verify_ingest.py    真实清单哈希/形状/时间与抽样梯度验证
  selfcheck.py        无数据依赖的CPU结构自检
  datasets/           公共数据格式转换及显式清单合并
configs/v13.json       默认模型配置
examples/              新增数据与模型示例
tests/                 离线回归测试
```

## 最小使用

```python
from motion_foundation import FoundationV13, ModelConfig, Sample
from motion_foundation.selfcheck import signal

model = FoundationV13(ModelConfig())
representation = model.encode(Sample([signal()]))
features = representation.pooled()
```

`signal()`仅生成合成数据。真实输入使用`StreamRecord`：`values[T,C]`、严格递增秒时间、`observed`及通道元信息。压力为kPa，加速度为m/s^2，角速度为rad/s。未观测值不能作为观测送入模型。

新增数据集和模型的具体入口见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 数据准备

先运行一个完全合成的CSV适配示例，再验证：

```bash
python examples/add_dataset.py --out data/synthetic
motion-validate --manifest data/synthetic/manifest.json
python examples/model_variant.py
```

现有接入器包含公共压力/IMU格式，以及旧审计缓存的桥接。来源状态见`configs/datasets.json`。它们不包含下载器或原始包；旧Grouvel/ImbalanSense/UnderPressure/PD需要既有经审计缓存，不承诺从原始压缩包独立重建这些前序缓存。

路径通过环境变量配置，不能将个人绝对路径提交仓库：

- `MOTION_DATA_ROOT`：压力数据原始目录，默认`data`。
- `MOTION_LEGACY_ROOT`：既有审计缓存根目录，默认`legacy_data`。
- `MOTION_TWENTE_ARCHIVE`：Twente原包，默认数据根目录下`twente/Raw_data.rar`。
- `MOTION_UNRAR`：可选的unrar程序路径，否则从PATH查找。程序需自行安装。

在设置环境变量之后启动转换进程。例如：

```bash
python -m motion_foundation.datasets.build_corpus --sources physionet pedar second_skin xsensor --out corpora/base
python -m motion_foundation.datasets.extend_corpus --out corpora/tekscan --workers 4
python -m motion_foundation.datasets.remaining_ingest --sources twente whip --out corpora/remaining --workers 4
python -m motion_foundation.datasets.merge --inputs corpora/base/manifest.json corpora/tekscan/manifest.json corpora/remaining/manifest.json --out corpora/combined.json
```

合并器保留原划分，不自动重分受试者，不允许重复来源或跨划分同内容。正式训练池需显式准备训练/验证划分，不能挪用test。转换错误写在manifest中；有错误的清单不能合并。

## 训练入口与边界

首版保留两种明确区分的数据契约：

- `v13-native-corpus-1` → `motion-train` → `ObservationDenoising`，当前公共语料路径。
- `motion-stream-v13.1` → `python -m motion_foundation.train` → `MaskedReconstruction`，单流掩码路径。

两种清单和检查点不可互换。`runtime.load_package()`仅加载掩码格式；去噪检查点由`train_mixed`路径读取。共同加载协议及精确恢复仍是后续事项，不能把本版当作已验收训练平台。

训练命令必须显式加`--execute-training`。只有确定要训练时才运行：

```bash
motion-train --manifest corpora/combined.json --model-config configs/v13.json --out runs/experiment --device cuda:0 --steps 2000 --execute-training
```

此命令只是使用说明，本交付没有运行训练。去噪训练每次抽一个独立器件流，不证明跨器件融合学习；未知时间依赖不允许时间掩码目标。去噪/掩码头当前仍依赖V13标量前端，任意全新骨干需同时适配目标，不能声称所有注册模型都可直接使用现有目标。

## Git协作

本目录是独立 Python 子项目，Git 由上层 `testFM` 仓库统一管理，不包含嵌套仓库。提交本目录的`src/`、`configs/`、`examples/`、`tests/`、包配置与协作文件；数据、权重、环境和运行输出不提交。依赖与仓库根目录的其他项目分开安装。

同事在短期分支修改，通过PR审核后合并；不覆盖其他子项目。本子项目使用合成数据进行离线检查，不自动下载真实数据或启动训练。当前仓库未启用本子项目的 GitHub 自动检查，提交前手动执行 `CONTRIBUTING.md` 中的检查命令。任何第三方代码/数据仍遵循其原条款；仓库公开不自动消除数据使用限制或授予第三方资产再分发权。

原模型`model.py`、`contracts.py`、`objectives.py`、`denoising.py`保留V13算法及参数键。包版本为0.1.0，不代表已获得预训练权重。
