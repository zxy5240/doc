# 虚幻引擎

## 蓝图
* [蓝图快速入门指南](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/blueprints-quick-start-guide?application_version=4.27)
* [持有棋子](https://dev.epicgames.com/documentation/en-us/unreal-engine/possessing-pawns?application_version=4.27)

## 编程
[`class ENGINE_API`](https://github.com/CarlaUnreal/UnrealEngine/pull/38/files) 是一个宏，用于控制类的导出行为，确保类在不同模块（DLL或共享库）之间正确使用。

## 杂项
* [语法](syntax.md)
* [像素流推送](./pixel_streaming.md)
* [配置说明](./config.md)
* [虚幻引擎commandlet](ue_commandlet.md)
* [虚幻引擎管线](ue_pipeline.md)
* [虚幻引擎相关效果](effect.md)
* [着色器](shader.md)
* [虚幻编辑器](unreal_editor.md)

## 源代码分析

* [USkeletalMeshComponent](https://zhuanlan.zhihu.com/p/637746453)
* [Actor生命周期](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/actor-lifecycle?application_version=4.27)

## 其他

* [PIE 和 SIE](https://blog.csdn.net/qq_43497224/article/details/129336509)

默认运行测试位置：菜单栏中央，三角形按钮。或者按

Play Modes
运行（测试）模式主要有两种，编辑器中运行（Play In Editor）或在编辑器中模拟（Simulate In Editor）。

PIE
PIE的意思是：Play，玩。你做的游戏是如何让玩家操控游玩的，在这里就是如何操控游玩的。（比如第一人称游戏，PIE下就是鼠标控制摄像机视角旋转，方向键移动,）

SIE
的意思是：模拟，测试。在这个模式下相当于是上帝视角。此时游戏在运行，但是我们仍然可以用之前做游戏（摆放地图关卡Actor、修改运行时参数）的方式修改这个世界。


PIE和SIE切换
在PIE模式下，shift + F1可以获得鼠标控制权（一般第一人称第三人称下鼠标用于旋转方向摄像头，不会出现光标）。然后点击工具栏的弹出（Eject）可以切换到SIE

你也可以点击 控制（Possess） 功能的快捷键（F10）来从在编辑器中模拟（SIE）切换到在编辑器中运行（PIE）。


## 加速构建过程

每次执行hutb的make的操作时，都会出现
```text
Using 'git status' to determine working set for adaptive non-unity build (D:\work\workspace\UnrealEngine).
Waiting for 'git status' command to complete
Target is up to date
Total execution time: 2.38 seconds
```
可以 [移除虚幻引擎的`.git`目录](https://github.com/adamrehn/ue4-docker/commit/9cf1375c33098a2787b5f514fd0ff36167a12b96) 来禁用 UBT 的 `git status` 调用，达到加速构建流程的作用。

此功能用于从 Unity 构建中排除大多数迭代文件。这应该会降低编译速度（如果你经常构建引擎，这个功能确实很有用，但大多数情况下并非如此）。

要禁用此行为，您可以更改 `Engine\Saved\UnrealBuildTool\BuildConfiguration.xml` 里的`<Configuration> </Configuration>`中添加：
```text
  <SourceFileWorkingSet> 
	<Provider>None</Provider> 
	<RepositoryPath></RepositoryPath> 
	<GitPath></GitPath> 
  </SourceFileWorkingSet>
```
改后生效，但是报错：
```text
Creating makefile for CarlaUE4Editor (BuildConfiguration.xml is newer)
UnrealBuildTool: WARNING: No Visual C++ installation was found. Please download and install Visual Studio 2017 with C++ components.
UnrealBuildTool: ERROR: Visual Studio 2019 must be installed in order to build this target.
make: *** [CarlaUE4Editor] 错误 6
```


## 参考链接
* [UE4初学者系列教程合集-全中文新手入门教程](https://www.bilibili.com/video/BV164411Y732/?share_source=copy_web&vd_source=d956d8d73965ffb619958f94872d7c57  )

* [ue4官方文档](https://docs.unrealengine.com/4.26/zh-CN/)

* [官方讨论社区](https://forums.unrealengine.com/categories?tag=unreal-engine)

* [知乎的虚幻引擎社区](https://zhuanlan.zhihu.com/egc-community)

