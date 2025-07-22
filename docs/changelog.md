2.1版本主要特性：

1. 虚拟驾驶舱
	
    原生支持罗技方向盘，力反馈；

	通用键盘鼠标控制 WASD + 鼠标，按Z表示倒车

	集成 SteamVR 的完全可驾驶虚拟现实自主车辆，SteamVR HMD 头部跟踪 (朝向 & 位置)

	支持 HTC Vive Pro Eye，使用 HTC Vive Pro Eye VR 耳机进行实时眼动跟踪，包括：时间信息（基于耳机、世界和眼动仪）、三维眼睛凝视光线（左、右及组合）、2D 瞳孔位置（左和右）、瞳孔直径（左和右）、眼睛睁开度（左和右）、世界中的焦点及命中的参与者信息
	逼真的（可参数化的）后视镜和侧视镜

	车辆仪表板：速度计（默认单位为英里/小时）、档位指示器、转向信号
	“以自我为中心”的音频，响应式发动机转速（基于油门）、转向灯咔嗒声、档位切换

	非自我为中心的音频（非自主车辆的发动机转速）；世界环境音频、鸟鸣声、风声、烟雾声等

	能够将控制权移交给/接管 Carla 的 AI 轮式车辆控制器


2. 兼容最新版 Carla 0.9.16

	添加了 NVIDIA 神经重建引擎 (NuRec) 集成

	添加了 SimReady OpenUSD 和 MDL 转换器，以提供对 SimReady OpenUSD 阶段和 MDL 材质的导出和导入支持。

	添加了对左侧交通地图的支持

	在容器内运行 CARLA 时，支持从主机安装 UE4

	添加了对使用 Ubuntu 22 镜像的容器内 GUI 的支持

	在记录器中添加了车门

	添加了获取参与者组件变换的函数

	增加了数字孪生使用本地文件（osm 和 xodr）的可能性

	启用正确的材质合并以构建数字孪生

	添加了获取参与者骨骼变换的函数、添加了获取参与者骨骼和组件名称的函数、添加了获取参与者套接字变换的函数、添加了获取参与者套接字名称的函数

	扩展了调试绘图函数，允许在 HUD 层上绘制图元

	添加了更改加速度计 imui 传感器中重力变量的功能

	修复了系统中安装 ROS2 时出现的 ROS2 原生扩展构建错误。

    ROS2Native：强制下载 fast-dds 依赖项，以避免在 Linux 中未安装 boost_asio 和 tinyxml2 时构建崩溃。

	为车辆 Actor 添加了 API 函数 get_telemetry_data。

	添加了用于协作感知消息和自定义消息的 V2X 传感器，以支持车对车通信。

	为 BasicAgent.py 的检测结果添加了命名元组，以便使用类型提示和更好的语义。

	添加了对 PythonAPI 的类型提示支持。

	为 GlobalRoutePlanner 添加了类型提示，并使用 carla.Vector3D 代码替代 0.9.13 之前的 Numpy 代码。

	如果可用，请使用 ActorID 而不是 Unreal Engine ID 进行实例分割

	在服务器和客户端之间同步 Actor 边界框

	将 Actor_ID 添加到边界框

	现在可以从 carla.command 导入

	carla.ad 子包现在可以直接导入，以前不能直接导入（例如 import ad）

	修复了流量管理器在尝试访问不可用车辆时出现的段错误

	修复了 Python 示例/rss 中无效的比较

	更新了反向流量 PythonAPI 示例脚本，并添加了对航点引导的反向 AI 车辆的支持。

	修复了实例分割中样条线网格不可见的问题

	在 Windows 中同时兼容 VS 2019和 VS 2022，默认设置为 VS 2022

	添加了 env CARLA_CACHE_DIR，以便能够设置 CARLA CACHE 位置
	在实例分割中支持遮罩材质，从而可以对例如树叶或栅栏（类似于语义分割）

	添加了 API 函数 world.set_annotations_traverse_translucency，并实现了配置深度和语义+实例分割是否遍历半透明材质的功能。

	修复一些bug：包括

		修复了实例分割中地形不可见的问题

		修复了 SensorData 的帧、时间戳和变换与摄像头传感器实际发送的图像不匹配的问题。

		修复了切换地图时导航信息无法加载的错误

		修复了 waypoint.next 和 .previous 在地图中两条相反方向的车道相继时导致循环的问题。

		防止加载 OpenDrive 文件时因 SignalReference 识别失败而导致的段错误

		make PythonAPI Windows：修复了由于 py 命令导致的与 Anaconda 不兼容的问题。

		修复了 Python 代理中当车辆列表为空时导致检查所有车辆 (BasicAgent.py) 的错误，并在没有行人的情况下将行人检测为车辆 (BehaviourAgent.py)。

		PythonAPI Sensor.is_listening 被定义了两次（属性和方法），已将其清理并澄清为方法。

		
3. 兼容 AirSim 1.8

	手动控制无人机

	程序 API 控制，包括 C++、Python

	HDU 直接调控天气等参数

	建造四旋翼、六旋翼飞行器

	包括 CityEnviron、Coastline、LandscapeMountains、ZhangJiajie 等 fab 场景导入流

