# 新算法 COP 计算脚本

输入每个传感器的压力、坐标和有效面积，输出 COP。**不需要测力板，也不使用 Moticon 自带 COP 参与计算。**本脚本整理了 `grouvel.zip` 末尾追加 COP 所用的压力×面积加权质心算法，没有修改原算法的传感器参数。

核心文件是 `calculate_cop.py`，仅依赖 NumPy。既可在命令行运行，也可作为 Python 函数导入。支持 Python 3.8 及以上。

## 1. 计算公式

每一帧、每一组传感器独立计算：

```text
weight_i = pressure_i × area_i
COP_x = sum(weight_i × x_i) / sum(weight_i)
COP_y = sum(weight_i × y_i) / sum(weight_i)
```

- 输出坐标的单位和坐标系与输入的 `x,y` 相同。
- 传感器面积必须提供。只有各传感器面积相同时，才可以全部填 `1`，此时等价于直接按压力加权。
- 同一组全部面积乘相同常数不会改变其 COP；各传感器的相对面积不能任意改动。
- 输入应是压力，不是已经换算出的单元力。若已有各单元力，直接按力加权即可，不能再次乘面积。
- 脚本不调整传感器位置，不平移或旋转 COP，不进行时间同步、平滑或滤波。

## 2. 快速运行

将交付包解压，终端进入 `calculate_cop.py` 所在目录。

```powershell
python -m pip install -r requirements.txt
python calculate_cop.py --input examples/pressure.csv --geometry examples/geometry.csv --output results/demo_cop.csv --diagnostics
```

`results` 文件夹会自动创建。原输入不会被覆盖；输出文件已存在时，请换一个输出名称。另生成 `demo_cop.csv.meta.json`，记录参数、来源哈希、有效/无效行数与实际传感器列名。

教学示例有 3 个传感器。第 2 行压力 `[10,20,30]`，面积 `[1,2,1]`，坐标 `[(0,0),(100,0),(0,100)]`，正确 COP 是 **`(50,37.5)`**。示例坐标单位为 mm，面积可视为 cm²，压力可视为 N/cm²；输出 COP 为 mm。

## 3. 换成自己的数据

压力表 `my_pressure.csv`：每行一个采样时刻，各通道须属于同一时刻。

```csv
time,p1,p2,p3
0.00,10,20,30
0.01,11,22,31
```

坐标表 `my_geometry.csv`：每行一个传感器。

```csv
group,pressure_column,x,y,area
left,p1,0,0,1
left,p2,100,0,2
left,p3,0,100,1
```

| 字段 | 含义 |
|---|---|
| `group` | 传感器组，例如 `left`、`right`、`foot`；分别计算 COP |
| `pressure_column` | 压力 CSV 中准确的列名；按名字匹配，不依赖列顺序 |
| `x,y` | 传感器中心坐标；同一组坐标必须属于同一坐标系和单位 |
| `area` | 有效面积或成比例的相对面积，必须为正数 |
| `x_shared,y_shared` | 可选，传感器在共同坐标系中的坐标，仅用于共同坐标 COP |

```powershell
python calculate_cop.py --input my_pressure.csv --geometry my_geometry.csv --output results/my_cop.csv
```

输出保留原输入列和采样行的顺序，并追加 `left_cop_x,left_cop_y`。如果坐标表还有 `right` 组，则同时追加 `right_cop_x,right_cop_y`。普通数值、时间及其他原有字段按原字符串保留；CSV 引号和换行形式可能重新规范化。物理空行不代表样本，会跳过。

`group` 使用英文字母开头，后面可含数字和下划线；`global` 保留给共同坐标输出。一个压力列只能分配一次。CSV 编码为 UTF-8，允许 BOM，分隔符为逗号。

## 4. 复现 grouvel.zip 的新算法

包内已提供两种等价的几何配置：

- `sensor_pe.json`：从原压缩包逐字节复制的参数文件。
- `sensor_coordinates_grouvel.csv`：转换为通用表格格式的 32 个传感器配置，方便查看和修改。

运行真实压力示例（306 帧，只有时间和 32 个压力通道）：

```powershell
python calculate_cop.py --input examples/grouvel_pressure.csv --geometry sensor_pe.json --output results/grouvel_cop.csv --shared-cop
```

也可以将 `--geometry sensor_pe.json` 换成 `--geometry sensor_coordinates_grouvel.csv`，结果相同。

输入若直接使用压缩包中已经带有末尾 COP 的 CSV，显式重算并写入新文件：

```powershell
python calculate_cop.py --input "P02_S01_Gait_01.csv" --geometry sensor_pe.json --output results/recomputed_cop.csv --shared-cop --replace-cop
```

`--replace-cop` 从输出副本中移除这些组已有的 `*_cop_x/y`、`*_cop_valid`、`*_load_weight` 和旧 `global` 计算字段，再追加本次请求的结果。**原生 `center of pressure X/Y[%]` 列不参与计算，也不会被替换。**原输入文件始终不变。

`--shared-cop` 开启时，Grouvel 配置将追加以下 6 列，与原压缩包顺序一致：

```text
left_cop_x, left_cop_y, right_cop_x, right_cop_y, global_cop_x, global_cop_y
```

原 JSON 支持压力列名 `left pressure 1[N/cm²]` 或 `left_pressure_1`，右脚同理；必须匹配唯一一种。其他列名使用坐标 CSV 显式指定。原 JSON 同时包含左右脚，因此要求两脚通道齐全；单脚任务请使用只含该脚的坐标 CSV。

### 原配置的坐标和面积含义

- 新算法 `cop_x`：内外方向 ML，向内侧为正；`cop_y`：前后方向 AP，脚跟 0、鞋尖 1。
- **两列均以鞋垫长度为单位**；单脚结果转 mm 时，x 和 y 都乘该脚实际鞋垫长度。不能把 x 乘鞋宽，也不需再除以 100。
- 原配置的 `area` 是从布局推得的 Voronoi 相对面积；并非本次重新标定的真实传感器敏感面积。几何配置仍是此前的近似参数，整理成脚本不改变其测量误差。
- 通用坐标表不强制这些方向定义；你输入什么坐标系，脚本就输出该坐标系的 COP。

### 关于 global_cop

默认只计算每组局部 COP。`--shared-cop` 要求每个传感器提供 `x_shared,y_shared`，并使用全部传感器的压力×面积在共同坐标系中加权。各组的面积比例和坐标单位必须相互兼容。

**原 Grouvel 配置的 shared 坐标只是固定双脚示意布局。因此该配置输出的 `global_cop_x/y` 不是走路时真实的实验室全局 COP。**真实步行全局 COP 还需要双脚逐帧位姿，把传感器或局部 COP 转到同一物理坐标系。本脚本的静态几何文件没有进行这一动态转换。

## 5. 零载荷、缺失值与阈值

| 情况 | 处理 |
|---|---|
| 一组压力全为 0 | 该组 COP 留空，valid=0 |
| 任一传感器压力缺失、非数值、NaN 或 Inf | 该组本帧无效，不忽略该传感器继续计算 |
| 任一压力为负 | 该组本帧无效，不取绝对值或裁为零 |
| 面积为零/负数、几何缺失、压力列缺失或重复分配 | 报错，修正输入后再计算 |
| 一只脚无效、另一只脚有效 | 两脚局部结果独立；共同 COP 不能使用缺失/负值的传感器 |

`--diagnostics` 额外输出 `*_cop_valid`（1/0）和 `*_load_weight`（压力×面积之和）。后者**不一定是牛顿**：原 JSON 使用相对面积，不能把它当成已经标定的力。

`--min-weight` 默认为 `0`，只要求总权重大于零。可根据自己的压力和面积单位设置阈值；边界为严格大于阈值。原 JSON 的相对面积不适合直接套用“50 N”阈值。

```powershell
python calculate_cop.py --input my_pressure.csv --geometry my_geometry.csv --output results/cop_threshold.csv --min-weight 5 --diagnostics
```

## 6. 直接在 Python 中调用

```python
import numpy as np
from calculate_cop import calculate_cop

# 二维数组：[帧数, 传感器数]，一帧也保留外层方括号
pressure = np.array([[10, 20, 30], [0, 0, 0]], dtype=float)
xy = np.array([[0, 0], [100, 0], [0, 100]], dtype=float)
area = np.array([1, 2, 1], dtype=float)

result = calculate_cop(pressure, xy, area)
print(result['cop'])         # [[50., 37.5], [nan, nan]]
print(result['valid'])       # [True, False]
print(result['weight_sum'])  # [80., 0.]
```

双脚或多组可分别调用此函数，也可使用脚本中的 `compute_groups`。大数据分块调用时，每帧独立计算，不依赖历史状态。

## 7. 验证记录

本次对原压缩包的 **259 个 CSV、233,123 行、全部 6 列 COP** 复算：数值和空值位置均匹配，最大归一化坐标差约 **8.88×10⁻¹⁶**。这是公式复现的浮点差异，**不是与测力板的物理误差**。

验证文件位于 `verification/`。独立可复现的检查命令：

```powershell
python tests/test_cop.py
python verify_archive.py --archive "D:\公司\数据库\grouvel.zip" --report-dir verification_rerun
```

测试覆盖不等面积已知答案、坐标平移/缩放、压力单位缩放、传感器重排、零载荷/缺失/负压力、双脚共同坐标、原生 COP 字段保留、禁止原地覆盖，以及 JSON/CSV 两种配置对真实样例的一致性。

## 文件清单

| 文件 | 用途 |
|---|---|
| `calculate_cop.py` | 独立主脚本与可导入函数 |
| `requirements.txt` | 依赖，只有 NumPy |
| `sensor_coordinates_grouvel.csv` | 原新算法的 32 个传感器坐标、相对面积与压力列映射 |
| `sensor_pe.json` | 原压缩包几何参数原件 |
| `examples/geometry.csv`、`examples/pressure.csv` | 可修改的通用输入模板 |
| `examples/grouvel_pressure.csv` | 真实压力示例，只有时间和压力，不含原生 COP |
| `examples/grouvel_expected_cop.csv` | 原压缩包同一示例的 6 列新算法 COP，供核验 |
| `verify_archive.py`、`tests/test_cop.py` | 全包复现检查与边界测试 |
| `example_results/` | 已运行的示例输出 |
| `verification/` | 本次核验结果 |
