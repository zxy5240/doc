# [创建资产发行包](https://carla.readthedocs.io/en/latest/tuto_A_create_standalone/) 

使用独立包管理资产是 Carla 的常见做法。将它们放在一边可以减少构建的大小。这些资源包可以随时轻松导入到 Carla 包中。它们对于以有组织的方式轻松分配资产也非常有用。

- [__在源代码构建的 Carla 中导出包__](#export-a-package-from-the-ue4-editor)  
- [__使用 Docker 导出包__](#export-a-package-using-docker)
- [__将资源导入 Carla 包__](#import-assets-into-a-carla-package)  

---
## 在源代码构建的 Carla 中导出包 <span id="export-a-package-from-the-ue4-editor"></span>

将资产导入虚幻后，用户可以为其生成 __独立的包__。这将用于将内容分发到 Carla 包，例如 0.9.8。

要导出包，只需运行以下命令即可：
```sh
make package ARGS="--packages=Package1,Package2"
```

这将为列出的每个包创建一个压缩在 `.tar.gz` 文件中的独立包。


## 将特定地图导出为包

要将自定义映射导出为独立的内容包，我们使用`make package`命令。通过虚幻内容浏览器确定自定义地图在 CARLA 内容(Content) 目录中的位置。我们建议使用`Content > CARLA > maps`目录中的现有 CARLA 地图在中创建自定义地图。在本例中，我们将地图存储在其自己的文件夹*MyMap*中。

![mymap_to_export](img/tuto_content_authoring_maps/mymap_export.png)

请记住，地图需要一个关联的 OpenDRIVE 文件，该文件存储在与地图资源（`.umap`文件）相同级别的名为`OpenDRIVE`的目录中。OpenDRIVE 文件应与地图本身具有相同的名称，扩展名为`.xodr`。在本例中，我将地图命名为 *MyMap*（文件在文件浏览器中显示为`MyMap.umap`），因此我的OpenDRIVE文件名为`MyMap.xodr`。

![mymap_xodr](img/tuto_content_authoring_maps/mymap_xodr.png)

现在，我们需要为导出过程创建一个包配置 JSON 文件。在文件浏览器（不是虚幻内容浏览器）中，导航到 `CARLA_ROOT/Unreal/CarlaUE4/content/CARLA/Config` 。在该目录中，您将看到许多 JSON 配置文件。使用与地图对应的名称和后缀`.Package.json`。在这种情况下，我们将文件命名为`exportMyMap.Package.json`。

![export_mymap_config](img/tuto_content_authoring_maps/export_mymap_config.png)

JSON 文件应具有以下结构：

```json
{
    "props": [],
    "maps": [
        {
            "name": "MyMap",
            "path": "/Game/Carla/Maps/MyMap",
            "use_carla_materials": true
        }
    ]
}
```

`name`参数应与地图资源文件的名称相同（没有`.umap`后缀）。确保`path`参数指向地图资源（`.umap`文件）的正确目录位置。路径`CARLA_ROOT/Unreal/CarlaUE4/Content/`的第一部分应替换为`/Game/`。

现在，我们从`CARLA_ROOT`目录中的终端调用`make package`命令，参数如下：

```sh
make package ARGS="--packages=exportMyMap"
```
参数`--packages`的值应与 没有`.Package.json`后缀的 JSON配置文件的名称匹配。打包可能需要一些时间来构建，具体取决于地图的大小。

导出过程完成后，导出的地图包将另存为压缩存档：

* **Linux**: `.tar.gz` 存档在 `CARLA_ROOT/Dist` 目录中
* **Windows**: `.zip` 存档在 `CARLA_ROOT/Build/UE4Carla` 目录中

必须使用目标操作系统生成地图包。即，在 Linux 中构建的映射包不能导入到 Windows 的 CARLA 包中。


要将打包的地图导入到 CARLA 的打包版本中，请将`.tar.gz`或`.zip`存档放在提取的 CARLA 包的`CARLA_ROOT/import`目录中，然后运行`ImportAssets.bin/.sh`脚本。完成后，启动 CARLA，您将在可用地图列表中找到新的自定义地图。


!!! 笔记
    在虚幻编辑器的菜单“编辑->项目设置->项目->打包->(点下三角打开折叠的选项)打包版本中要包含的地图列表”中可以删除不需要的地图，加入需要打包的自定义地图。

---

## 使用 Docker 导出包 <span id="export-a-package-using-docker"></span>

虚幻引擎和 Carla 可以构建在 Docker 映像中，然后可以使用该映像创建包或导出资源以在包中使用。

要创建 Docker 映像，请按照 [此处](build_docker_unreal.md) 的教程进行操作。

准备好镜像后：

1. 导航至 `Util/Docker`。
2. 通过运行以下命令之一创建 Carla 包或准备在包中使用的资源：

```sh
# 创建独立的包
./docker_tools.py --output /output/path

# 烘培在 Carla 包中使用的资产
./docker_tools.py --input /assets/to/import/path --output /output/path --packages PkgeName1,PkgeName2
```

---
## 将资源导入 Carla 包 <span id="import-assets-into-a-carla-package"></span>

独立包包含在`.tar.gz`文件中。提取方式取决于平台。

*   __在 Windows 上，__ 将压缩文件解压到主根 Carla 文件夹中。
*   __在 Linux 上，__ 将压缩文件移至`Import`文件夹并运行以下脚本。

```sh
cd Import
./ImportAssets.sh
```

!!! 笔记
    独立包无法直接导入到 Carla 构建中。按照教程导入 [道具](tuto_A_add_props.md) 、[地图](tuto_M_custom_map_overview.md) 或 [车辆](tuto_A_add_vehicle.md)。

---

这总结了如何在 Carla 中创建和使用独立包。如果有任何意外问题，请随时在论坛中发帖。

<div class="build-buttons">
<p>
<a href="https://github.com/OpenHUTB/doc/issues" target="_blank" class="btn btn-neutral" title="Go to the CARLA forum">
讨论页面</a>
</p>
</div>