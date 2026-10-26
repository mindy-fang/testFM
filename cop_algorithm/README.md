# COP 计算脚本

根据各传感器的压力、坐标和有效面积，计算压力中心（COP）。不需要测力板或 Moticon 原生 COP 作为输入。

- [完整中文使用说明](README_中文.md)
- [下载完整脚本包](../downloads/cop_algorithm.zip)
- [主脚本](calculate_cop.py)
- [传感器坐标配置](sensor_coordinates_grouvel.csv)

## 快速运行

在本目录执行：

```bash
python -m pip install -r requirements.txt
python calculate_cop.py --input examples/pressure.csv --geometry examples/geometry.csv --output results/demo_cop.csv --diagnostics
```

只依赖 NumPy，支持 Python 3.8 及以上。原输入不被覆盖，输出同时保存计算参数和有效性统计。

## 验证范围

已完成 9 项测试，并复现原压缩包 259 个 CSV、233,123 行的 6 列 COP。这里验证的是计算实现与已有算法一致，不代表通过测力板精度验收。

`global_cop` 为可选输出；原参数使用固定双脚示意布局，不代表行走中的实验室全局 COP。实际坐标、面积和单位说明见完整中文文档。
