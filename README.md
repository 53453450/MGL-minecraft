Language: 中文 | [English](README_EN.md)


# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-LGPL--3.0--only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-3.0-orange.svg)]()

**MGL (Metal-GL)** 是一个将 OpenGL 4.6 和 OpenGL ES 3.x 转译到 Apple Metal 的图形驱动层。它允许现有的 OpenGL 应用无需修改即可在 macOS 上使用 Metal 后端运行。

## 前言

### 项目说明

- 这是一个纯粹的AI coding项目，如果你反感/厌恶AI代码，你可以离开此仓库
- 本项目分支于MGL：https://github.com/openglonmetal/MGL
- Minecraft(以下简称MC)是在mac上为数不多的运行较好的游戏之一，可是，MC可以长盛不衰的原因来自于它庞大的Mod社区，但是Apple 于2018年6 月在 WWDC 2018 上正式宣布弃用OpenGL与OpenCL，macOS的OpenGL支持永远停在了4.1版本，顶点着色器上限（GL_MAX_VERTEX_ATTRIBS）是16，这与现今的Mod社区严重脱节，部分mod与绝大多数的光影无法在macOS上运行。此项目将OpenGL提升至4.6，并将GL_MAX_VERTEX_ATTRIBS=30

## 许可证

本仓库按代码来源分别适用许可证。基线提交
`79d38f666336141d962109a864a6744bf66e438c` 及其之前的原 MGL 代码继续使用
[Apache License 2.0](LICENSE-APACHE-2.0)；该基线之后本存储库中的修改由相应版权
持有人按 [LGPL-3.0-only](LICENSE) 授权。

这不是对原 Apache-2.0 贡献的重新许可。包含两类内容的文件必须分别遵守适用于
各部分的许可证。完整范围说明见 [LICENSING.md](LICENSING.md)，第三方组件继续使用
其各自许可证。

## 要求

**前置**: 

- macOS 14 或更新版本
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

`clone_external.sh` 只会在依赖目录缺失时拉取 OpenGL-Registry、ezxml 和 Apple
官方 [metal-cpp](https://github.com/apple/metal-cpp)。`external/glfw` 是本仓库的
本地修改版本，不会从远端克隆或更新，构建时始终使用该目录。

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

- 当前优先目标是 GL46CTS 规范兼容性；Minecraft/Mod 运行仍可能随 CTS 修复出现回归。

## 项目结构

```
MGL-minecraft/
├── MGL/                         # 核心库源码
│   ├── include/                 # OpenGL/MGL 头文件
│   └── src/                     # C/Objective-C 实现
│       ├── MGLRenderer.m        # Metal 渲染器与绘制路径
│       ├── MGLTextures.m        # Metal 纹理桥接
│       ├── buffers.c            # Buffer/UBO/SSBO 状态
│       ├── framebuffers.c       # FBO/RBO 与 completeness
│       ├── get.c                # glGet/internalformat 查询
│       ├── pixel_utils.c        # 像素格式与布局工具
│       ├── rendering.c          # 渲染状态与 draw 调度
│       ├── shaders.c            # GLSL frontend entry
│       ├── tex_param.c          # 纹理参数与 internalformat 查询
│       └── textures.c           # 纹理上传、清理、压缩格式路径
├── external/                    # SPIRV-Cross、SPIRV-Tools、glslang、GLFW 等依赖
├── benchmark/                   # 性能测试工具
├── test_mgl/                    # 本地 smoke/功能测试
├── MGL_Golden_Images/           # 图像回归基准
├── TestImages/                  # 测试纹理素材
├── enum_parser/                 # OpenGL enum 生成辅助
├── spec_parser/                 # 规范解析辅助
├── build/                       # 本地构建输出
├── Makefile                     # 唯一构建入口
├── README.md                    # 中文说明
├── README_EN.md                 # English README
├── LICENSE                      # 基线之后仓库修改的 LGPL 3.0 全文
├── LICENSE-APACHE-2.0           # 原 MGL 代码的 Apache 2.0 全文
├── LICENSE-GPL-3.0-only         # LGPL 3.0 引用的 GPL 3.0 全文
└── LICENSING.md                 # 许可证范围与提交边界说明
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

## 调试

### MGL_TRACE_LOG

设置 `MGL_TRACE_LOG=1` 可以启用 MGL 内部 trace 日志。日志默认写到 `libmgl.dylib` 所在目录，文件名格式为 `mgl-trace-<pid>.log`。

默认情况下，trace 不会再直接输出到终端或系统 Console，避免 Minecraft/启动器日志被大量诊断信息刷屏。需要同时镜像到 `stderr` 时，额外设置 `MGL_TRACE_LOG_STDERR=1`。

常用开关：

```bash
MGL_TRACE_LOG=1
MGL_TRACE_LOG_STDERR=1
MGL_TRACE_LOG_DRAW=1
MGL_TRACE_LOG_RESOURCES=1
MGL_TRACE_LOG_PROGRAMS=91,92,93
```

| 变量 | 说明 |
|------|------|
| `MGL_TRACE_LOG` | 启用 trace 文件输出 |
| `MGL_TRACE_LOG_STDERR` | 将 trace 同步镜像到 `stderr`，默认关闭 |
| `MGL_TRACE_LOG_DRAW` | 记录绘制调用与 draw replay 相关诊断 |
| `MGL_TRACE_LOG_RESOURCES` | 记录更详细的 buffer/texture/sampler 绑定诊断 |
| `MGL_TRACE_LOG_PROGRAMS` | 只重点追踪指定 program，支持逗号/空格/分号/冒号分隔 |

`MGL TRACE`、`MGL DUMP`、shader interface dump 等诊断输出统一进入 trace 文件。`MGL ERROR` / `MGL WARNING` 仍会保留在普通日志路径中，方便第一时间发现真实故障。

示例：

```bash
MGL_TRACE_LOG=1 MGL_TRACE_LOG_DRAW=1 MGL_TRACE_LOG_PROGRAMS=91,92
```

启动后，在 MGL dylib 所在目录查找 `mgl-trace-<pid>.log`。

### MGL_MIP_DIAG

设置 `MGL_MIP_DIAG=1` 报告被采样纹理的实际 mip 链和采样器状态，用于排查 mipmap 相关的画面异常。

它独立于 `MGL_TRACE_LOG`：逐绑定的 trace 行密度太高，帧率会掉到看不出随视角变化的瑕疵。输出走普通日志路径，前缀 `MGL MIP_DIAG`，并且只在状态发生变化时打印——画面稳定时完全静默，突然出现一批日志就说明某个状态翻转了。

三种记录：

| 记录 | 触发点 | 用途 |
|------|--------|------|
| `MIP_DIAG texture` | 纹理绑定 | GL 与 Metal 两侧的级数是否一致、`mipmapped`/`genmipmaps` 标志、`mtlTex` 指针变化（指针变了说明纹理被重建） |
| `MIP_DIAG frag` | 片元采样器解析 | 立即模式下生效的 filter、LOD 夹取、`BASE_LEVEL`/`MAX_LEVEL`；以及渲染目标是否经 Y-flip 副本采样（`viaCopy`）、副本级数、脏 mip 掩码和版本号 |
| `MIP_DIAG snapshot` | 延迟批次回放 | 开启延迟批处理时真正交给 Metal 的逐 draw 采样器，会覆盖上面 `frag` 记录的值 |

```bash
MGL_MIP_DIAG=1
```

## 致谢

- [Khronos Group](https://www.khronos.org/) - SPIRV-Cross, glslang, SPIRV-Tools,VK-GL-CTS
- [GLFW](https://www.glfw.org/) - 窗口管理库
- [openglonmetal/MGL](https://github.com/openglonmetal/MGL) - MGL框架，没有它就没有MGL-minecraft
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)
