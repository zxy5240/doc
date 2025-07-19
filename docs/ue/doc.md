## 文档解读


## [注册安装构建版本](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/using-an-installed-build?application_version=4.27)

向团队分发安装构建版本时，请确保每个人的构建版本辨识符都相同。这将阻止编辑器提示用户选择版本，然后使用本地生成的唯一辨识符更新 .uproject 文件。可按照以下方法设置自定义辨识符：

在 Windows 中，向将你的辨识符用作其项的 `HKEY_CURRENT_USER\SOFTWARE\Epic Games\UnrealEngine\Builds` 添加注册表项，并将引擎路径作为其值。例如，项可以是 MyCustom419，值可以是 D:\\CustomUE4。

在 Mac 中，打开 /Users/MyName/Library/Application Support/Epic/UnrealEngine/Install.ini，然后向将你的辨识符用作项的 [Installations] 部分添加条目，并将引擎路径作为值。例如：

        [Installations]
        MyCustom419 = /Users/MyName/CustomUE4
