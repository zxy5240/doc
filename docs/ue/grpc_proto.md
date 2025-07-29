# NVIDIA 神经重建引擎 gRPC 协议缓冲区

该软件包包含 NVIDIA 神经重建引擎 (NuRec) gRPC API 的协议缓冲区定义和 Python 生成工具。

## 内容

- `protos/`: 协议缓冲区定义（.proto 文件）
  - `common.proto`: 常见数据类型和结构
  - `sensorsim.proto`: 传感器模拟服务定义
- `update_generated.py`: Python 脚本从 proto 文件生成 Python 代码
- `requirements.txt`: 所需的 Python 依赖项

## 快速入门

### 1. 安装依赖项

使用 pip：
```bash
pip install -r requirements.txt
```

### 2. 生成 Python 代码

运行生成脚本以从 proto 定义创建 Python 文件：

```bash
python PythonAPI/examples/nvidia/grpc/update_generated.py
```

这将在 `PythonAPI/examples/nvidia/grpc/protos/` 中生成以下 Python 文件：
- `common_pb2.py` 和 `common_pb2.pyi`: common.proto 的 Python 代码
- `sensorsim_pb2.py` 和 `sensorsim_pb2.pyi`: sensorsim.proto 的 Python 代码
- `common_pb2_grpc.py` 和 `sensorsim_pb2_grpc.py`: gRPC 服务代码

### 3. 在您的代码中使用

生成 Python 文件后，您可以在 Python 代码中导入并使用它们：

```python
# 导入生成的模块
import grpc_proto.common_pb2 as common_pb2
import grpc_proto.sensorsim_pb2 as sensorsim_pb2
import grpc_proto.sensorsim_pb2_grpc as sensorsim_pb2_grpc

# 创建 gRPC 通道和存根
import grpc
channel = grpc.insecure_channel('localhost:50051')
stub = sensorsim_pb2_grpc.SensorsimServiceStub(channel)

# 使用服务
request = sensorsim_pb2.RGBRenderRequest()
# ... 配置您的请求
response = stub.render_rgb(request)
```

## 服务概述

NuRec gRPC API 提供以下主要服务：

### SensorsimService

- `render_rgb`: 从场景渲染 RGB 图像
- `render_lidar`: 渲染激光雷达点云 
- `get_version`: 获取版本信息
- `get_available_scenes`: 列出可用场景
- `get_available_cameras`: 列出场景可用的相机
- `get_available_trajectories`: 列出场景的可用轨迹
- `get_available_ego_masks`: 列出可用的自我掩膜
- `shut_down`: 正常关闭服务

## 部署

修改 .proto 文件时，始终重新生成 Python 代码：

```bash
python PythonAPI/examples/nvidia/grpc/update_generated.py
```

## 支持

如果对此 API 有疑问，请联系 NVIDIA 支持或参阅官方文档。

## 许可证

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
