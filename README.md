Language: 中文 | [English](README_EN.md)


# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-LGPL--3.0--only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-4.0-orange.svg)]()

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

- macOS 26 或更新版本（Metal 4 SDK；`make verify-toolchain` 会检查）
- Xcode Command Line Tools
- Homebrew
- `make install-pkgdeps` 安装 `llvm@15`、`cmake`、`glm`
  - LLVM 15：`make lib` 链接 `-lLLVM-15`
  - CMake：GLFW 与 `make gtest` 需要
  - glm：可选的 host-side 工具头文件
- 跑 AIR gtest 时还需要 GoogleTest（`make gtest`，默认克隆到 `~/googletest`）

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

`clone_external.sh` 会把 OpenGL-Registry 检出到 `MGL/generated/registry.lock`
记录的 commit；并在目录缺失时拉取 ezxml（固定 commit）、Apple 官方
[metal-cpp](https://github.com/apple/metal-cpp)，以及链接GL 4.6 core 的SPIRV所需的修改版
[SPIRV-Cross](https://github.com/53453450/SPIRV-Cross)（核心特性 ARB_gl_spirv的依赖）。`external/glfw` 是仓库的本地修改版本，不会从远端克隆或更新；`./build_external.sh` 只构建该 GLFW。

### 3. 编译 MGL

```bash
#返回主目录
cd ..
make          # 等同 make lib / make core
# make es     # 可选的单独 libmgl_es.dylib（默认上下文为 ES）
```

OpenGL ES 3.2 已合入 `libmgl.dylib`：同一库内用 `createGLMContextProfile(1, …)` 创建 ES 上下文（`ctx->is_es`），不必再编第二份 dylib。Minecraft / 桌面 GL 路径继续用默认的 Core 上下文。

本地非交互门禁：

```bash
make test-all          # 架构门禁 + AIR/Metal smoke + headless 回归等
make test-regression   # 仅跑 headless 图像/语义回归（golden 在 MGL_Golden_Images/）
make test              # 交互式 GLFW 冒烟（test_mgl），不是 CI 门禁
```

## 构建产物

默认 `make` 完成后：

| 路径 | 说明 |
|------|------|
| `build/libmgl.dylib` | OpenGL 4.6 Core（含可选 ES 3.2 上下文） |
| `build/libglfw.dylib` | 修改版 GLFW 库 |
| `libmgl.dylib` / `libglfw.dylib` | 指向 `build/` 的根目录符号链接（方便 `DYLD_LIBRARY_PATH=.`） |

中间对象在 `build/core/`。可选 `make es` 才会额外生成 `libmgl_es.dylib`（`-DMGL_GL_ES`；新代码优先用上面的 profile API）。

## 使用方法

编译完成后在启动器的 Java 参数中添加：
```JVM
-Dorg.lwjgl.opengl.libname="/yourpath/to/libmgl.dylib"
-Dorg.lwjgl.glfw.libname="/yourpath/to/libglfw.dylib"
```
指向 MGL-minecraft 的产物，让它们接管渲染。仅当外部仍 `dlopen` ES 专用库名时，才需要先 `make es` 并设置 `-Dorg.lwjgl.opengles.libname=…/libmgl_es.dylib`。

## 现状

- Minecraft 1.20.1-1.21.11可正常运行，现在目标是补充CTS扩展与性能优化。

## 项目结构

```
MGL-minecraft/
├── MGL/
│   ├── include/                 # OpenGL API、GLMContext、MGL C ABI
│   ├── generated/               # registry.lock、gl46 ABI 生成物
│   ├── aux_shaders/             # 构建期编入的辅助 Metal shader
│   ├── extensions/              # 可选扩展桩（如 xfb_stub）
│   └── src/                     # 生产实现：C 状态机 + C++（无引擎 .m）
│       ├── L5_api/              # GL 入口 / dispatch（gl_core.c、gl_es.c…）
│       ├── L4_context/          # GL 对象、状态、program/link、renderer 入口
│       ├── L3_validate/         # 校验、binding resolve、纯数据 plan
│       ├── L2_encode/           # batch / pass / draw issue-encode
│       ├── L1_backend/          # Metal 对象、PSO、cache、mgl_render*.cpp
│       ├── L0_platform/         # AppKit / CAMetalLayer / trace / shell
│       └── compiler/            # GLSL → MGLIR → AIR → metallib
├── external/
│   ├── metal-cpp/               # Apple 官方 header-only Metal C++ bindings
│   ├── SPIRV-Cross/             # Core 链接依赖
│   ├── glfw/                    # 本仓库维护的本地 GLFW（仍含 Cocoa .m）
│   ├── OpenGL-Registry/         # Khronos OpenGL registry
│   └── ezxml/                   # XML 解析依赖
├── test_legacy_compat/          # 架构门禁、AIR/Metal/gtest 等单测（make 目标驱动）
├── test_regression/             # headless OpenGL/Metal 回归
├── test_dirty_hash/             # dirty-state 批处理回归
├── test_mgl/                    # 交互式 GLFW 冒烟（make test）
├── tools/                       # 本地探测 / CTS 辅助工具
├── benchmark/                   # 性能测试
├── scripts/                     # 资产生成、回归、trace、benchmark
├── docs/                        # 仅 ARCHITECTURE.md（分层契约）
├── MGL_Golden_Images/           # 图像回归基准
├── TestImages/                  # 测试纹理
├── config.mk.example            # 本地 SDK/toolchain 模板
├── build/                       # Makefile 输出（core 对象树 + dylib）
└── Makefile                     # 唯一构建入口
```

分层职责与禁止捷径见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。`MGL/src` 根目录只保留跨层内部头，实现 TU 必须落在 L0–L5 或 `compiler/`。AppKit / CAMetalLayer 交互集中在 `L0_platform/mgl_platform_shell.cpp`（ObjC runtime + `objc_msgSend`）。
## 核心模块说明

### 着色器与 AIR 路径

```text
glCompileShader / glLinkProgram
        │
        ▼
L4  mgl_legacy_compat.c          # 旧 GLSL 语法兼容
        │
        ▼
compiler/ mgl_glsl_{lexer,parser,sema}.c
        │
        ▼
compiler/ mgl_ir.c · mgl_air_reflect.c
        │
        ▼
compiler/ mgl_air_backend.cpp    # → AIR bitcode → metallib
        │
        ▼
compiler/ mgl_air_loader.cpp
L1     mgl_render*.cpp           # Metal-cpp PSO / encoder
```

- 旧版 GLSL 兼容在进入 frontend 前完成
- AIR reflection 产出资源、varying、tessellation、geometry ABI
- 辅助 shader 以 metallib 字节表嵌入（`L1_backend/mgl_aux_assets.*`），运行时不读 Metal 源码
- 带 sampler 的 VS/FS 在 link 时可多编一份 soft minmax 变体；仅当绑定采样器为 `GL_MIN`/`GL_MAX` 时启用

### 状态管理

脏状态写在 `ctx->active_state` 上，draw 前由 L2 encode 路径刷到 Metal：

```c
mglMarkRendererDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
/* draw 路径内部会跑 processGLState / pass sync；不是对外公开的 C API */
```

### Metal backend 与平台壳

- `L1_backend/mgl_render.cpp` 是**唯一定义** Metal-cpp private implementation 宏的 TU；具体 draw/texture/binding 等拆在同目录 `mgl_render_*.cpp`。
- `L1_backend/mgl_renderer_backend.cpp` 持有 owner、cache、completion 与临时资源生命周期。
- `L0_platform/mgl_platform_shell.cpp` 持有 `NSView` / `CAMetalLayer` / drawable / device。
- L4（`mgl_renderer_entries.c`、`mgl_draw_entry.c` 等）做 GL 语义编排，经 value-state / opaque handle 进入 backend。

```text
L5  gl_core.c / glm_dispatch.c          # ABI shim → ctx->dispatch.*
        │
        ▼
L4  state / buffers / textures / program / renderer entries
        │
        ▼
L2  mgl_draw_issue.cpp / mgl_draw_encode.cpp / pass sync
        │ value-state / opaque handle
        ▼
L1  mgl_renderer_backend.cpp → mgl_render*.cpp → Metal
        ▲
        │ device / layer / drawable
L0  mgl_platform_shell.cpp
```

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

- [Khronos Group](https://www.khronos.org/) - OpenGL Registry and GL CTS
- [LLVM](https://llvm.org/) - AIR backend code generation
- [Apple metal-cpp](https://github.com/apple/metal-cpp) - Metal C++ bindings
- [GLFW](https://www.glfw.org/) - 窗口管理库
- [openglonmetal/MGL](https://github.com/openglonmetal/MGL) - MGL框架，没有它就没有MGL-minecraft
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)
