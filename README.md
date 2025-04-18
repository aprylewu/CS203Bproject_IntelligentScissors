# Intelligent Scissors

一个基于 LWJGL 实现的智能剪刀图像分割工具。

## 功能特点

- 使用 Dijkstra 算法实现智能路径规划
- 支持图像边缘检测和路径优化
- 实时路径预览
- 支持图像缩放和平移
- 支持闭环路径创建

## 环境要求

- Java 8 或更高版本
- LWJGL 3.3.6
- algs4 库

## 使用方法

1. 确保已安装 Java 环境
2. 运行 `run.sh` 启动程序
3. 使用鼠标左键点击选择起点
4. 移动鼠标预览路径
5. 再次点击确定路径
6. 按 ESC 键退出程序

## 项目结构

```
.
├── src/                    # 源代码目录
├── lwjgl-release-3.3.6-custom/  # LWJGL 库
├── algs4/                  # algs4 库
├── run.sh                  # 运行脚本
└── clean.sh               # 清理脚本
```

## 许可证

MIT License 