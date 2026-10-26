# Twente ViT-1D 模型

当前 Twente 九分类实验使用的模型代码，从头监督训练，无预训练。
本目录提供模型定义和调用示例，不包含数据、训练权重或完整数据处理/交叉验证入口。

## 文件

- `vit1d_model.py`：实验原始模型文件，支持 12 / 64 通道。
- `vit1d_variants.py`：从实验训练入口提取的模型子类，支持 32 / 44 / 76 通道；结构与参数键名不变。
- `requirements.txt`：本地实验使用的 PyTorch 版本。

## 输入配置

|输入|通道数|模型|
|---|---:|---|
|仅双脚 IMU（加速度和角速度）|12|`ViT1D(12)`|
|原始压力|32|`NoRatioViT1D(32)`|
|原始压力 + IMU|44|`NoRatioViT1D(44)`|
|压力 + 压力占比|64|`ViT1D(64)`|
|压力 + 压力占比 + IMU|76|`FusionViT1D(76)`|

输入为 `[B, C, 150]`，50 Hz 下对应 3 秒窗口。每 5 帧一个 patch，按时间优先、通道次之展平，共 30 个 patch。投影维度 32，前置 CLS token，31 个可学习位置向量，4 层 Encoder、4 个头、前馈维度 64、GELU、Pre-LayerNorm、Dropout 0.15；最终 LayerNorm 后取 CLS，输出 9 类 logits。

通道顺序必须与训练检查点保存的 `columns` 一致，不能只按通道数推断顺序。输入须除以训练受试者原始帧计算的逐通道 `P95(abs(x))`，不减均值、不截断；测试使用相同 P95。

## 调用示例

```python
import torch
from vit1d_model import ViT1D
from vit1d_variants import FusionViT1D, NoRatioViT1D

model = ViT1D(12).eval()
x = torch.randn(2, 12, 150)  # 示例输入；实际输入需要训练集 P95 归一化
with torch.no_grad():
    logits = model(x)
assert logits.shape == (2, 9)

# 对于本地实验的 model.pt，先按对应通道数创建模型：
# checkpoint = torch.load('model.pt', map_location='cpu', weights_only=False)
# model.load_state_dict(checkpoint['model'], strict=True)
# weights.pt 则直接保存 state_dict。
```

输出类别顺序：慢走、正常走、快走、慢跑、一般跑步、跳跃、落地、深蹲、弓步；排除 MVC。

## 实验设置参考

150 帧窗口、75 帧步长，不跨文件。三折测试受试者为 Subj04–06、Subj07–09、Subj10–12，各折其余 6 人训练。种子 20260917、20260918、20260919；折内种子为 seed + 100 × (fold − 1)。固定训练 30 轮，使用最后一轮，AdamW（lr=0.001、weight_decay=0.0001）、batch size 128、梯度裁剪 1.0、训练类别逆频率加权交叉熵，无早停或学习率调度。

旧 `transformer_model.py` 和 `mae_model.py` 已从当前版本移除，可从 Git 历史查阅。此处 ViT-1D 使用时间 patch，不是旧的逐帧 Transformer，也不包含 MAE 预训练模块。

## COP 计算脚本

独立的压力×面积加权 COP 计算脚本位于 [`cop_algorithm/`](cop_algorithm/)，包含坐标配置、输入示例、中文说明及验证记录。

- [使用说明](cop_algorithm/README_中文.md)
- [下载完整 COP 脚本包](downloads/cop_algorithm.zip)

计算输入为传感器压力、坐标和面积，无需测力板。已有核验确认脚本能复现原算法结果；这不等同于测力板精度验收通过。
