# HUTB - NUREC 集成

该模块集成了 HUTB 与 NVIDIA 的 NUREC（NVIDIA 神经重建引擎）。它允许您通过基于 NUREC 数据重建交通场景，在 HUTB 中重放真实世界的记录。

## 概述

NUREC（NVIDIA 神经重建引擎）是一个利用传感器数据重建真实交通场景的框架。这种集成允许这些重建结果在 HUTB 模拟器中重现，从而在真实数据和模拟之间架起一座桥梁。

主要特性：
- 在 HUTB 中加载和回放 NUREC 场景
- 可视化车辆及其他交通参与者
- 支持摄像头视图，包括旁观者摄像头
- 坐标系转换

## 要求

要使用此集成，您需要：

- CARLA 0.9.16 或者更新
- Python 3.10
- 以下 Python 包：
  - pygame
  - numpy
  - scipy
  - grpc
  - carla
  - nvidia-nvimgcodec-cu12

## 安装

有关详细的安装说明，请参阅 [NUREC 安装指南](../nvidia-nurec.md) 。

有关详细的安装说明，请参阅。

## 使用 NUREC 运行 HUTB

### 步骤 1：启动 HUTB

按照标准流程启动 HUTB。

### 第 2 步：重放 NUREC 场景

请务必通过设置 NUREC_IMAGE 环境变量来指定要使用的 nurec 镜像。详情请参阅 [文档](../nvidia-nurec.md)。

一旦 HUTB 运行，您就可以使用`example_replay_recording.py`脚本重播 NUREC 场景：

```bash
python example_replay_recording.py --usdz-filename $(pwd)/maps/clipgt-9e849eeb-073f-424c-838c-493b56c806fb.usdz --move-spectator
```

## example_replay_recording.py 的命令行参数

下表解释了`example_replay_recording.py`脚本可用的命令行参数：

| 参数               | 长格式 | 默认         | 描述                                                                  |
|------------------|-----------|------------|---------------------------------------------------------------------|
| -h               | --host | 127.0.0.1  | HUTB 主机服务器的 IP 地址                                                   |
| -p               | --port | 2000       | HUTB 服务器的 TCP 端口                                                    |
| -np              | --nurec-port | 46435      | NUREC 服务器的端口                                                        |
| -u               | --usdz-filename | (必需的) | 包含 NUREC 场景的 USDZ 文件的路径                                             |
| --move-spectator | | False      | 移动观察者摄像机来跟随自主车辆 |

## 模块结构

- `nurec_integration.py`: 处理 NUREC 服务管理和场景重放的主要集成类
- `scenario.py`: 用于加载和管理 NUREC 场景的核心类
- `track.py`: 车辆轨迹的轨迹段表示和插值函数
- `constants.py`: 整个模块中使用的常量
- `projection_functions.py`: 坐标系变换函数
- `pygame_display.py`: 使用 Pygame 实现摄像头图像的可视化
- `example_*.py`: 演示不同用例的示例脚本：
  - `example_replay_recording.py`: 使用多台摄像机进行基本场景重放
  - `example_save_images.py`: 将相机图像保存到磁盘
  - `example_custom_camera.py`: 配置自定义相机参数
- `grpc_proto/`: 用于 NUREC 服务通信的协议缓冲区定义和生成的代码
- `tools/`: 用于蓝图尺寸提取和其他任务的附加实用工具

## 附加信息

- 蓝图大小 JSON 文件用于将 NUREC 对象尺寸与 HUTB 蓝图进行匹配
- 使用 `extract_blueprint_sizes.py` 工具生成自定义 CARLA 蓝图的尺寸信息

## 启动 NUREC 服务

创建 `NurecScenario` 实例时，NUREC 服务会自动启动。该服务提供以下主要功能：
- 从场景渲染 RGB 图像
- 获取版本信息
- 列出可用的场景、摄像机和轨迹

该服务默认运行在 46435 端口，可以使用 `--nurec-port` 参数进行配置。当 `NurecScenario` 实例被销毁时，该服务将自动关闭。

您可以在创建 `NurecScenario` 实例时使用 `reuse_container` 参数控制服务容器行为：
- 当 `reuse_container=True` （默认）时：如果服务容器已经存在，则会重复使用，防止多个实例同时运行
- 当 `reuse_container=False` 时：每次都会创建一个新的服务容器，这对于测试或需要确保干净状态时很有用

使用示例：
```python
# 重复使用现有容器（默认）
scenario = NurecScenario(client, usdz_filename)

# 强制新建容器
scenario = NurecScenario(client, usdz_filename, reuse_container=False)
```

## 工具

### 提取蓝图尺寸工具

有关蓝图提取工具的文档，请参阅 [tools.md](tools.md) 。


