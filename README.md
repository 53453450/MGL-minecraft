Language: 中文 | [English](README_EN.md)


# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-3.0-orange.svg)]()

**MGL (Metal-GL)** 是一个将 OpenGL 4.6 和 OpenGL ES 3.x 转译到 Apple Metal 的图形驱动层。它允许现有的 OpenGL 应用无需修改即可在 macOS 上使用 Metal 后端运行。

## 前言

### 项目说明

- 这是一个纯粹的AI coding项目，如果你反感/厌恶AI代码，你可以离开此仓库
- 本项目分支于MGL：https://github.com/openglonmetal/MGL
- Minecraft(以下简称MC)是在mac上为数不多的运行较好的游戏之一，可是，MC可以长盛不衰的原因来自于它庞大的Mod社区，但是Apple 于2018年6 月在 WWDC 2018 上正式宣布弃用OpenGL与OpenCL，macOS的OpenGL支持永远停在了4.1版本，顶点着色器上限（GL_MAX_VERTEX_ATTRIBS）是16，这与现今的Mod社区严重脱节，部分mod与绝大多数的光影无法在macOS上运行。此项目将OpenGL提升至4.6，并将GL_MAX_VERTEX_ATTRIBS=30

## 要求

**前置**: 

- Xcode Command Line Tools
- Homebrew
- Cmake

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/53453450/MGL-minecraft.git
cd MGL-minecraft
```

### 2. 构建

```bash
#安装构建依赖
make install-pkgdeps
cd external
#克隆依赖
./clone_external.sh
# 依赖编译
./build_external.sh
```

### 3. 编译 MGL

```bash
#返回主目录
cd .. 
make
```

## 构建产物

编译完成后，将在 `build/` 目录生成：

| 文件 | 说明 |
|------|------|
| `libmgl.dylib` | OpenGL Core 动态库 |
| `libmgl_es.dylib` | OpenGL ES 动态库 |
| `libglfw.dylib` | 修改版 GLFW 库 |

## 使用方法

编译完成后在启动器的java参数中添加：
```JVM
-Dorg.lwjgl.opengl.libname="/yourpath/to/libmgl.dylib"
-Dorg.lwjgl.glfw.libname="/yourpath/to/libglfw.dylib"
-Dorg.lwjgl.opengles.libname="/yourpath/to/libmgl_es.dylib"
```
指向MGL-minecraft的产物，让它们接管渲染

## 现状

- 已知bug:1.21.11，1.21.10的世界亮度问题，1.21.11的UI选框消失（可能在其他版本有其他未知bug，但我并未测试）
- Forge modloader 的启动渲染动画不被加载，neoforge modloader 会崩溃，fabric modloader 不会有影响（在1.21.7测试）
- sodium 在与fabric modloader 一起加载时被支持（在以上所有被提及的版本）

## 项目结构

```
MGL-minecraft/
├── MGL/                          # 核心库源码
│   ├── src/                      # C/Objective-C 源文件
│   │   ├── gl_core.c            # OpenGL Core API 入口
│   │   ├── gl_es.c              # OpenGL ES API 入口
│   │   ├── shaders.c            # 着色器管理
│   │   ├── textures.c           # 纹理管理
│   │   ├── buffers.c            # 缓冲区管理
│   │   ├── programs.c           # 着色器程序管理
│   │   ├── rendering.c          # 渲染状态管理
│   │   ├── MGLRenderer.m        # Metal 渲染器实现
│   │   └── MGLTextures.m        # Metal 纹理实现
│   ├── include/                  # 头文件
│   │   ├── GL/                  # OpenGL 头文件
│   │   └── glm/                 # GLM 数学库
│   └── spirv_cross_c.cpp        # SPIRV-Cross C++ 桥接
│
├── external/                     # 外部依赖
│   ├── SPIRV-Cross/             # SPIR-V 到 MSL 转译器
│   ├── SPIRV-Tools/             # SPIR-V 工具链
│   ├── SPIRV-Headers/           # SPIR-V 头文件
│   ├── glslang/                 # GLSL 编译器
│   ├── OpenGL-Registry/         # OpenGL 规范
│   ├── glfw/                    # 修改版 GLFW
│   └── ezxml/                   # XML 解析器
│
├── test_mgl_glfw/               # 测试用例
├── MGL.xcodeproj/               # Xcode 项目文件
├── Makefile                     # 构建脚本
└── LICENSE                      # Apache 2.0 许可证
```

## 核心模块说明

### 着色器转译 (shaders.c)

着色器转译是 MGL 的核心功能，负责将 GLSL 着色器转换为 Metal Shading Language (MSL)：

```c
GLSL 源码 (330/420/450)
    │
    ▼
glslang 预处理与编译
    │
    ▼
SPIR-V 中间表示
    │
    ▼
SPIRV-Cross 转译
    │
    ▼
Metal Shading Language
```

**关键特性：**
- 自动升级旧版 GLSL (140/330) 到 420+
- 自动为 UBO 分配 binding 索引
- 添加必要的扩展声明 (`GL_ARB_shading_language_420pack`)

### 状态管理

OpenGL 状态通过脏标记系统同步到 Metal：

```c
// 状态变更标记
STATE(dirty_bits) |= DIRTY_RENDER_STATE;

// 在绘制时处理脏状态
processGLState(ctx, true);
```

### Metal 渲染器 (MGLRenderer.m)

Objective-C 实现的 Metal 渲染器，处理：
- RenderCommandEncoder 创建与管理
- 状态映射 (OpenGL → Metal)
- 绘制命令执行

## 调试与最小复现

### MGL_TRACE_LOG

设置 `MGL_TRACE_LOG=1` 可以启用 MGL 内部 trace 日志。日志默认写到 `libmgl.dylib` 所在目录，文件名格式为 `mgl-trace-<pid>.log`。

常用开关：

```bash
MGL_TRACE_LOG=1
MGL_TRACE_LOG_DRAW=1
MGL_TRACE_LOG_RESOURCES=1
MGL_TRACE_LOG_GUI=1
MGL_TRACE_LOG_PROGRAMS=91,92,93
```

这些变量可以按需加入启动器环境变量，用于捕获 draw、资源绑定、GUI 或指定 program 的调试信息。

### Minecraft GL 最小复现

仓库内提供了三个面向 Minecraft 1.21.11 渲染路径的最小复现 case，用于验证 MGL 是否正确覆盖相关 OpenGL 调用：

- `cloud-tbo-vertexid`: 云渲染路径，覆盖 `isamplerBuffer`、`GL_R8I` texture buffer、`gl_VertexID` 和大规模 indexed draw。
- `rt-pingpong-blur`: 后处理路径，覆盖 FBO ping-pong、RGBA8 render target 采样和 blur pass。
- `focused-tail-rendergraph`: 基于真实 trace tail 的渲染图路径，覆盖 DSA FBO attach/detach、mapped buffer flush、indexed/baseVertex GUI draw、运行时 render target 创建、六段 blur pass 和 final blit。

构建：

```bash
cd MGL-minecraft
make repro
```

运行全部 case：

```bash
./build/repro/minecraft_gl_repro all
```

单独运行：

```bash
./build/repro/minecraft_gl_repro cloud-tbo-vertexid
./build/repro/minecraft_gl_repro rt-pingpong-blur
./build/repro/minecraft_gl_repro focused-tail-rendergraph
```

输出图片默认写入当前运行目录下的 `mgl-repro-output/`。也可以用 `MGL_REPRO_OUTPUT_DIR` 指定输出目录：

```bash
MGL_REPRO_OUTPUT_DIR=./mgl-repro-output ./build/repro/minecraft_gl_repro all
```

注意：case 名是程序参数，不需要把可执行文件路径再作为参数传入。

## 致谢

- [Khronos Group](https://www.khronos.org/) - SPIRV-Cross, glslang, SPIRV-Tools,VK-GL-CTS
- [GLFW](https://www.glfw.org/) - 窗口管理库
- [openglonmetal/MGL](https://github.com/openglonmetal/MGL) - MGL框架，没有它就没有MGL-minecraft
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。
