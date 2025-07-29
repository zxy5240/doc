# NUREC 集成工具

本文档介绍了`tools/`目录中可用于 HUTB-NUREC 集成的实用工具。

## 提取蓝图尺寸工具

`extract_blueprint_sizes.py`工具用于测量 HUTB 中所有车辆和步行者蓝图的尺寸。在 HUTB 中重放 NUREC 场景时，这些尺寸对于正确调整 Actor 尺寸至关重要。

### 目的

在重放 NUREC 场景时，我们需要将真实世界中的物体尺寸与 HUTB actor 尺寸进行匹配。此工具：

1. 连接到正在运行的 HUTB 服务器
2. 生成每个可用的车辆和行人蓝图
3. 测量其边界框尺寸（宽度、长度、高度）
4. 将结果保存为 NUREC 集成可以使用的 JSON 文件

### 使用

```bash
python tools/extract_blueprint_sizes.py [options]
```

#### 先决条件：

- 正在运行的 HUTB 服务器
- 当前地图中有足够的生成点供所有蓝图使用

#### 命令行选项：

| 参数               | 长格式 | 默认                                   | 描述                                   |
|------------------|-----------|--------------------------------------|--------------------------------------|
| --host           | | 127.0.0.1                            | HUTB 主机服务器的 IP 地址                    |
| -p               | --port | 2000                                 | HUTB 服务器的 TCP 端口                     |
| -s               | --start | 0.0                                  | 测量开始时间                               |
| -d               | --duration | 200.0                                | 测量持续时间                               |
| -f               | --recorder-filename | test1.log                            | 记录器文件名                               |
| -c               | --camera | 0                                    | 摄像机跟随参与者（例如，82）                      |
| -x               | --time-factor | 0.2                                  | 时间因子（默认1.0）                          |
| -i               | --ignore-hero |                                      | 忽略英雄车辆                               |
| --move-spectator | |                                      | 移动观察者摄像机                             |
| --spawn-sensors  | | |  在重放世界中生成传感器 |

### 输出

该工具生成两个 JSON 文件：

1. `blueprint_sizes_vehicle.json`: 包含所有车辆蓝图的尺寸
2. `blueprint_sizes_walker.json`: 包含所有行人蓝图的尺寸 

每个文件包含一对列表：(blueprint_id，[x_extent，y_extent，z_extent]) ，其中：
- x_extent: 参与者沿其局部 X 轴的半长
- y_extent: 参与者沿其局部 Y 轴的半宽
- z_extent: 参与者沿其局部 Z 轴的半高

### 例子

```bash
# 连接到本地 HUTB 服务器并提取蓝图尺寸
python tools/extract_blueprint_sizes.py --host 127.0.0.1 --port 2000
```

### 笔记

- 该工具可能需要几分钟才能运行，因为它需要单独生成和测量每个蓝图
- 确保使用具有足够出生点的地图（推荐使用 Town10HD）
- 该工具自动过滤掉无效测量值（无穷大或 NaN 值）
- 生成的 JSON 文件应放置在 NUREC 集成目录中，以便与重放脚本一起使用 