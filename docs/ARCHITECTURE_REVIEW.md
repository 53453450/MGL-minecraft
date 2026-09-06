# MGL 架构审查（落地对照）

规范基线：OpenGL 4.6 Core + GLSL 4.60。OpenGL ES 3.2 为第二阶段。Minecraft + Sodium/Iris 类路径是产品基线，与规范冲突时以 Khronos 为准。

本文件对照审查稿落地后的工作树。审查阶段的「产品代码未改」已经过时。批次 0–3 已合入；批次 4 只做了生产安全子集（不删 Compat、不补完 DrawExecutor）；批次 5 只有 ES 3.2 smoke，不是 ES 验收。

## 结论

当前设计已经选对了方向（自研 GLSL→AIR、C 状态机、Metal-cpp 单一实现 TU、CompileArtifact），但仍停在未完成的迁移上：生产 draw 仍走 Compat 桥和 ObjC category；`mglSeedUniformInitializers` 仍对源码再 parse；错误路径仍有直接 `STATE(error)=`。

不要把 Release 无线程 abort、缺 share context、PSO 无锁当成当前 P0——那些仍是 latent 或产品选择。

| 指标 | 审查稿 | 落地后 |
|------|--------|--------|
| P0 阻断项仍开放 | 3 | 0（F01 已 fail-closed；F02 为误报；F03 已进 CI） |
| P1 仍开放 | 16 | F04、F05（部分）、F10（seed）、F16（主体）、F20、F21 |
| 已落地 / 校准 | — | 见下方 F 状态表 |

**保留**：前后端分离、自研 GLSL→MGLIR→AIR、`CompileArtifact` 原子发布、C 状态机、Metal-cpp 作为唯一 GPU 实现、薄平台壳。

**不要保留**：ObjC 巨型 category 当 encoder、Compat 四跳、seed 路径重复 parse、手写 `gl_core`/`dispatch` 当唯一真相、散落 `getenv` 的 MC 吞错。

信任链只保留三件事：`CompileArtifact` 原子发布、CommandIR 与 live 状态隔离、C ABI 不泄漏 `MTL`。其它内部协议、env 开关、binaryarchive 格式都可以重做。

## 落地状态（按 F）

状态：`landed` 有代码与门禁；`partial` 做了审查要求的一部分；`open` 未做；`not-a-bug` / `misreported` 不需要改产品行为。

### P0

#### F01 DrawTransformFeedback* 静默 no-op — landed

未实现路径报 `INVALID_OPERATION`（`test-arch-correctness` F01）。Metal 捕获后的真实 Draw 仍未接通，禁止改回静默成功。

#### F02 24 项 golden 中缺 17 个 TGA — misreported

`test_regression/main.c` 现有 23 个 `GOLDEN_TEST`；`MGL_Golden_Images/` 有 23 个对应 `Reg_*.tga`（git 已跟踪）。审查稿的 24/7 计数与仓库不符。compare 模式不缺文件。

#### F03 CI 未安装 llvm@15 / googletest — landed

`.github/workflows/ci.yml` 安装 `llvm@15`、跑 `make gtest` 与 `make verify-gl-api`。README / `install-pkgdeps` 与之对齐。

### P1

#### F04 MSAA 以 2D array 仿真 — open

`max_image_samples=8` 与 `max_samples` 仍不一致；MS 纹理仍按 array 仿真。未改。

#### F05 错误双轨 — partial

`ERROR_RETURN` / `ERROR_CHECK_*` 已走 `mglDispatchError`。`mgl_unimplemented` 入队 `INVALID_OPERATION`。注释改为 §2.3.1。直接 `STATE(error)=` 仍散落（`mgl_gl_extensions.c` 等）。队列仍是 16 槽 FIFO，不是「只保留首错误」。

#### F06 4.6 Core 的 Getn* 全 stub — landed（Core 范围）

`GetnUniform*` 与 `GetnTexImage` 接到现有 Get 并检查 `bufSize`。`GetnMap*` / `GetnPixelMap*` 仍 unimplemented（Compatibility 命令，不是 Core 4.6 义务）。

#### F07 Shader subroutine 广告不可用能力 — landed

`max_subroutines` / `max_subroutine_uniform_locations` = 0；入口保持 unimplemented → `INVALID_OPERATION`。

#### F08 GLFW 扩展探测只认 11 条 — landed

`external/glfw/src/mgl_context.m` 的 `extensionSupportedMGL` 委托 `glGetStringi` + `GL_NUM_EXTENSIONS`。

#### F09 variant 编译绕过 CompileArtifact — landed

`mglCompileArtifactFromGLSLEx` 携带 `air_flags` / `iface_peers`。tess/cull/VS capture 经 `mglCompileCaptureVariant` → artifact `complete` 才发布。

#### F10 link 路径重复 parse — partial

`FrontendSession` 让 reflect + AIR codegen 共用一次 parse（gtest `FrontendSession.CompileReflectIsSingleParse`）。`mglSeedUniformInitializers` 仍对 `shader->src` 再 `mglGLSLParse`。

#### F11 gl_ClipDistance/gl_CullDistance 源码 strstr — landed

`mglFrontendBuiltinArrayCount`：IR 符号优先，否则 AST 常量子下标。源码启发式已删。

#### F12 legacy 翻译固定缓冲溢出 — landed

`mglFrontendRewriteLegacy` 动态缓冲；失败返回 compile log，不再静默跳过。

#### F13 MGL_MAX_TOKENS 未强制 — landed

lexer 超 `131072` token fail compile。`ShaderSource` 超过 8MiB → `INVALID_VALUE`。

#### F14 libmgl_es.dylib 零测试 — partial

`make test-es-smoke` 覆盖 context + `GL_VERSION` ES 3.2 + `DrawArrays`。`test-all` 包含该门。CI 的 `macos-gate` 经 `test-all` 间接触达。独立 ES limits 表与 GLES CTS 子集未做。`gl_es.c` 仍缺大量 3.2 入口。

#### F15 gl.xml codegen 已断开 — landed（验证层）

pin `MGL/generated/registry.lock` → OpenGL-Registry `9cb90ca`。`make verify-gl-api` 生成命令清单并对照 `gl_core.c` + overlay。overlay extra=361、missing=0。不自动生成 `mgl*` 实现体。`spec_parser/spec_parser.c` 仍保留；已删过期 `spec_parser/mgl.h` / `mgl_funcs.c`。

#### F16 生产 draw 仍是 Compat 桥 — partial（有意停在这里）

生产 `mglRendererBackendCreate` **不**安装 `MetalDrawExecutor`（vtable 空，走 ObjC Compat）。PSO miss 不再复用 previous PSO。不要补完 DrawExecutor 而不删 Compat。encode 下沉与删 Compat 仍开放。

#### F17 广告 GL_KHR_debug 但无消息存储 — landed

context 拥有 16 槽 debug ring；`DebugMessageInsert` / `GetDebugMessageLog` 可往返（arch F17）。

#### F18 VertexAttrib1/2/3* 为 no-op — landed

1/2/3 分量接到 current-attrib，缺省补 `(x,0,0,1)` 等。`GetVertexAttrib* CURRENT_VERTEX_ATTRIB` 不要求绑定 VAO（§10.2）。

#### F19 program pipeline 的 GS/tess/compute 未被 batch retain — landed

`MGLDrawBatch` retain 几何 / tess control / tess eval / compute，与 VS/FS 同一套 `mglRetainBatchProgram`。

### P2

#### F20 无 share group — open

`createGLMContext` 仍无 share 参数。GLFW 收到 share 时仍未显式失败。

#### F21 air_loader PSO cache 无锁 — open

`mgl_air_loader.cpp` 静态 `std::map` 仍无 mutex。

#### F22 空壳 TU 仍被 wildcard 链入 — landed

已删 `msl_patch_pipeline`、`mgl_toolchain`、`mgl_ir_postprocess`、`mgl_msl_compat`、`mgl_compute_pipeline_cache` 及 `enum_parser/`。Makefile 仍 `wildcard MGL/src/*.c`（空壳不在后无害）。未改为显式源列表。

#### F23 DeleteShader(未知名) 报 INVALID_VALUE — not-a-bug

保持现状。不要改成 silent ignore。

#### F24 Release 下 GL 线程断言为空 — not-a-bug

documented 设计。可选 `MGL_ENABLE_THREAD_CHECKS` 进 CI；不要当 P0。

## 当前架构 vs 目标

### 现在：迁移未完成的五层

```
gl* → dispatch → mgl* 状态
  → draw_command 录制 + live/replay 双代理
    → backend facade + Compat 桥
      → MGLRenderer categories 编排
        → mgl_render.cpp Metal-cpp encode
```

编译：`FrontendSession` 一次 parse 供 reflect+codegen；link 时 `SeedUniformInitializers` 仍再 parse。capture/variant 走 `CompileArtifact`。

### 目标：三层 + 一次编译

```
生成式 C ABI（profile-aware）
  → GL 语义核心：校验、可变 live 状态、不可变 CommandIR
      → C++ Metal backend：资源 + encoder + PSO
          → 薄 ObjC 端口：layer / drawable / swap
  → FrontendSession 一次 → Reflect + Codegen + uniform seed 共用 IR
      → CompileArtifact.complete 才发布
```

## 模块处置（仍有效）

| 模块 | 处置 | 落地备注 |
|------|------|----------|
| `gl_core` / `gl_es` / dispatch | 局部重写 | 薄 ABI 由 pin 的 `gl.xml` 验证 + overlay。ES 只补了 smoke 入口。 |
| GLMContext / 资源状态 | 保留并收口 | debug ring 进 context。share group 未做。 |
| GLSL frontend + MGLIR | 保留并收口 | `FrontendSession` 已存在；seed 仍旁路。 |
| `mgl_air_backend.cpp` | 局部重写 | clip/cull 改 IR/AST；TU 仍未按域拆开。 |
| CompileArtifact / reflect | 保留并收口 | variant/capture 已进同一门闩。 |
| `draw_command` 批处理 | 保留并收口 | GS/tess/compute retain 已齐。快照仍非全句柄 ID。 |
| `MGLRenderer+*.m` | 替换边界 | PSO miss 不再复用旧 PSO。encode 仍在 ObjC。 |
| `mgl_render.cpp` + backend | 局部重写 | 生产不装 DrawExecutor。Compat 仍在。 |
| Platform shell + GLFW fork | 保留并收口 | 扩展探测已委托 `glGetStringi`。 |
| `enum_parser` / stale spec_parser 输出 | 删除 | 已删。`spec_parser.c` 留下给 verify。 |
| 空壳 MSL/SPIRV TU | 删除 | 已删。 |
| OpenGL ES 3.2 路径 | 保留并收口 | smoke only。不要扩 ES 语义假装验收。 |

## 不要做的事

- 不要把 DrawExecutor VTable 再补完一层而不删除 Compat。
- 不要为「完整 4.6」去实现 display list / immediate mode。Core profile 应停止导出或保持 `INVALID_OPERATION`。
- 不要把 Minecraft compat 做成默认吞错误。MC 走显式 profile 对象，CTS 保持 strict。
- 不要把 TES 含 SSBO 一律强制 `tess_eval_compute`（会破坏 `air_tessellation_resources` 像素）。

## 分阶段路线

| 批次 | 审查停止条件 | 落地 |
|------|----------------|------|
| 0 安全网 | 干净 clone 上 `make test-all` 绿；CI 装 llvm@15+gtest | 完成。golden 本已齐；CI/gtest/`verify-gl-api` 已加。 |
| 1 codegen | pin `gl.xml`，CI diff；删空壳 TU | 完成（验证层 + 删空壳）。手写 `mgl*` 实现仍 overlay。Makefile 仍 wildcard。 |
| 2 GL 语义 | XFB 失败关闭；Getn*/Attrib1-3/debug 与广告一致；错误统一入队；GLFW 委托 | 除 F05 直写 `STATE(error)` 外完成。 |
| 3 编译链 | FrontendSession；clip/cull 改 IR；legacy 动态缓冲；variant 走 CompileArtifact | 除 F10 seed 再 parse 外完成。 |
| 4 backend | CommandIR 深句柄；encode 下沉；删 Compat | **未完成**。只做了 retain 面、PSO miss 不回退、生产不装 executor。 |
| 5 ES 3.2 | 独立 profile 表 + 最小 GLES CTS 子集 | **未完成**。只有 dylib smoke。 |

下一批应继续批次 4 的删除 Compat，或收口 F05/F10，而不是扩 ES。

## 验证矩阵

改造期间最低门禁。下列结果来自落地后本机跑通（macOS，Apple M4）：

| 门 | 命令 | 通过标准 | 本次证据 |
|----|------|----------|----------|
| A 构建 | CI brew llvm@15 + gtest；`make lib` | 干净 macOS 14 runner 成功 | workflow 已写；本机 `make -j8 lib` 成功 |
| B 状态 | `test-arch-correctness` + `test-dirty-hash` | 与已落地的 §2.3.1 / attrib / debug 探针一致 | arch all probes passed；dirty-hash PASS（默认 AIR tess） |
| C 编译 | `test-mglair-gtest`；`verify-gl-api` | 单次 parse（seed 除外）；reflection 槽位 assert | 51/51；`verify-gl-api: ok`（698 core，overlay extra=361 missing=0） |
| D 像素 | `test-regression` | 全 PASS 或 SKIP；golden 文件齐全 | **91 PASS / 0 FAIL / 2 SKIP / 93** |
| E 性能 | `test-benchmark` | P95 回退 ≤10% | 未在本次重跑；CI 仍有该 step |
| F 产品 | MC 1.21 + Sodium/Iris | 启动、世界、GUI、光影不花屏 | 未在本次验证 |
| ES | `test-es-smoke` | context + 3.2 字符串 + DrawArrays 不崩 | `es-smoke: ok` |

`make test-all` 现含 `verify-gl-api` 与 `test-es-smoke`。不要用 ES smoke 绿来声称 ES 3.2 已验收。

## 规范来源

- [OpenGL 4.6 Core Profile](https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf)（2022-05-05）
- [GLSL 4.60.8](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [OpenGL ES 3.2](https://registry.khronos.org/OpenGL/specs/es/3.2/es_spec_3.2.pdf)
- [GLSL ES 3.20.8](https://registry.khronos.org/OpenGL/specs/es/3.2/GLSL_ES_Specification_3.20.pdf)

章节号以 PDF 为准。代码注释里的 §2.5 已过时，现行错误模型在 §2.3.1。
