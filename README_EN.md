Language: [中文](README.md) | English


# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-LGPL--3.0--only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-4.0-orange.svg)]()

**MGL (Metal-GL)** is a graphics translation layer that converts OpenGL 4.6 and OpenGL ES 3.x calls into Apple Metal. It allows existing OpenGL applications to run on macOS using a Metal backend without modification.

## Introduction

### Project Notes

- This is a purely AI-generated coding project. If you dislike or are against AI-generated code, you may leave this repository.
- This project is forked from: https://github.com/openglonmetal/MGL
- Minecraft (MC) is one of the few games that run relatively well on macOS. However, its longevity largely comes from its massive modding community. Apple officially deprecated OpenGL and OpenCL at WWDC 2018 (June 2018), and macOS OpenGL support has been stuck at version 4.1 ever since. The vertex attribute limit (`GL_MAX_VERTEX_ATTRIBS`) is 16, which is far behind modern mod requirements. Many mods and most shader packs cannot run on macOS. This project upgrades OpenGL support to 4.6 and increases `GL_MAX_VERTEX_ATTRIBS` to 30.

## License

Licensing in this repository follows code provenance. The original MGL code at
and before baseline commit `79d38f666336141d962109a864a6744bf66e438c` remains
under the [Apache License 2.0](LICENSE-APACHE-2.0). Modifications made in this
repository after that baseline are licensed by their respective copyright
holders under [LGPL-3.0-only](LICENSE).

This does not relicense the original Apache-2.0 contributions. Files containing
both kinds of material must comply with the license applicable to each portion.
See [LICENSING.md](LICENSING.md) for the complete scope notice. Third-party
components remain under their own licenses.

## Requirements

**Prerequisites:**

- macOS 26 or newer (Metal 4 SDK; `make verify-toolchain` checks this)
- Xcode Command Line Tools
- Homebrew
- `make install-pkgdeps` installs `llvm@15`, `cmake`, and `glm`
  - LLVM 15: `make lib` links `-lLLVM-15`
  - CMake: required by GLFW and `make gtest`
  - glm: convenience headers for optional host-side tools
- GoogleTest for AIR unit tests (`make gtest`, clones into `~/googletest` by default)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/53453450/MGL-minecraft.git
cd MGL-minecraft
```

### 2. Build dependencies

```bash
# Install build dependencies
make install-pkgdeps
cd external
# Clone dependencies
./clone_external.sh
# Build dependencies
./build_external.sh
```

`clone_external.sh` checks out OpenGL-Registry at the commit in
`MGL/generated/registry.lock`, and fetches ezxml (pinned commit), Apple's official
[metal-cpp](https://github.com/apple/metal-cpp), and the modified
[SPIRV-Cross](https://github.com/53453450/SPIRV-Cross) needed for GL 4.6 core
SPIR-V (`ARB_gl_spirv`) when those directories are missing. `external/glfw` is the
repository's locally modified checkout (never cloned or pulled from upstream);
`./build_external.sh` builds that GLFW only.

### 3. Build MGL

```bash
# Return to the repository root
cd ..
make          # same as make lib / make core
# make es     # optional separate libmgl_es.dylib (ES-default context)
```

OpenGL ES 3.2 lives in `libmgl.dylib`: create an ES context with
`createGLMContextProfile(1, …)` (`ctx->is_es`). A second dylib is not required.
Minecraft / desktop GL keep using the default Core context.

Local non-interactive gates:

```bash
make test-all          # architecture gates + AIR/Metal smoke + headless regression
make test-regression   # headless image/semantics suite (goldens in MGL_Golden_Images/)
make test              # interactive GLFW smoke (test_mgl); not the CI gate
```

## Build Outputs

Default `make` produces:

| Path | Description |
|------|------|
| `build/libmgl.dylib` | OpenGL 4.6 Core (optional in-process ES 3.2 context) |
| `build/libglfw.dylib` | Modified GLFW library |
| `libmgl.dylib` / `libglfw.dylib` | Root symlinks into `build/` (for `DYLD_LIBRARY_PATH=.`) |

Object files live under `build/core/`. Optional `make es` also builds
`libmgl_es.dylib` (`-DMGL_GL_ES`). Prefer the profile API above for new code.

## Usage

After building, add the following JVM arguments in your launcher:
```JVM
-Dorg.lwjgl.opengl.libname="/yourpath/to/libmgl.dylib"
-Dorg.lwjgl.glfw.libname="/yourpath/to/libglfw.dylib"
```
Point them at the MGL-minecraft build products. Only if something still
`dlopen`s an ES-specific library name: run `make es` and set
`-Dorg.lwjgl.opengles.libname=…/libmgl_es.dylib`.

## Current Status

- Minecraft 1.20.1–1.21.11 runs. Current focus is filling out CTS coverage for remaining extensions and performance work.

## Project Structure

```
MGL-minecraft/
├── MGL/
│   ├── include/                 # OpenGL API, GLMContext, MGL C ABI
│   ├── generated/               # registry.lock, generated gl46 ABI
│   ├── aux_shaders/             # build-time embedded helper Metal shaders
│   ├── extensions/              # optional extension stubs (e.g. xfb_stub)
│   └── src/                     # production: C state machine + C++ (no engine .m)
│       ├── L5_api/              # GL entry / dispatch (gl_core.c, gl_es.c, …)
│       ├── L4_context/          # GL objects, state, program/link, renderer entries
│       ├── L3_validate/         # validation, binding resolve, pure-data plans
│       ├── L2_encode/           # batch / pass / draw issue-encode
│       ├── L1_backend/          # Metal objects, PSO, cache, mgl_render*.cpp
│       ├── L0_platform/         # AppKit / CAMetalLayer / trace / shell
│       └── compiler/            # GLSL → MGLIR → AIR → metallib
├── external/
│   ├── metal-cpp/               # Apple's official header-only Metal C++ bindings
│   ├── SPIRV-Cross/             # Core link dependency
│   ├── glfw/                    # local GLFW fork (still has Cocoa .m)
│   ├── OpenGL-Registry/         # Khronos OpenGL registry
│   └── ezxml/                   # XML parsing dependency
├── test_legacy_compat/          # architecture gates, AIR/Metal/gtest (make-driven)
├── test_regression/             # headless OpenGL/Metal regression
├── test_dirty_hash/             # dirty-state batch regression
├── test_mgl/                    # interactive GLFW smoke (make test)
├── tools/                       # local probes / CTS helpers
├── benchmark/                   # performance tools
├── scripts/                     # asset, regression, trace, benchmark scripts
├── docs/                        # ARCHITECTURE.md only (layering contract)
├── MGL_Golden_Images/           # image regression baselines
├── TestImages/                  # test textures
├── config.mk.example            # local SDK/toolchain template
├── build/                       # Makefile output (core object tree + dylibs)
└── Makefile                     # sole build entry point
```

Layer duties and forbidden shortcuts: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Implementation TUs must live under L0–L5 or `compiler/`; the `MGL/src` root keeps
cross-layer internal headers only. AppKit / CAMetalLayer access is concentrated in
`L0_platform/mgl_platform_shell.cpp` (ObjC runtime + `objc_msgSend`).

## Core Modules

### Shader and AIR Path

```text
glCompileShader / glLinkProgram
        │
        ▼
L4  mgl_legacy_compat.c          # legacy GLSL compatibility
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

- Legacy GLSL compatibility runs before the frontend
- AIR reflection produces resource, varying, tessellation, and geometry ABI
- Helper shaders are embedded as metallib bytes (`L1_backend/mgl_aux_assets.*`);
  runtime does not read Metal sources
- VS/FS stages with samplers may emit a soft minmax metallib at link; it is bound
  only when an active sampler uses `GL_MIN` / `GL_MAX`

### State Management

Dirty bits live on `ctx->active_state` and are flushed to Metal on the L2 encode path:

```c
mglMarkRendererDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
/* Draw paths run processGLState / pass sync internally; not a public C API */
```

### Metal Backend and Platform Shell

- `L1_backend/mgl_render.cpp` is the **only** TU that defines Metal-cpp private
  implementation macros; draw/texture/binding work is split across `mgl_render_*.cpp`
  in the same directory.
- `L1_backend/mgl_renderer_backend.cpp` owns handles, caches, completion, and
  temporary-resource lifetime.
- `L0_platform/mgl_platform_shell.cpp` owns `NSView` / `CAMetalLayer` / drawable /
  device.
- L4 (`mgl_renderer_entries.c`, `mgl_draw_entry.c`, …) owns GL semantic orchestration
  and enters the backend through value-state / opaque handles.

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

## Debugging

### MGL_TRACE_LOG

Set `MGL_TRACE_LOG=1` to enable MGL internal trace logging. Logs are written next to `libmgl.dylib` by default, using the file name format `mgl-trace-<pid>.log`.

By default, trace output is not written directly to the terminal or system Console, which keeps Minecraft/launcher logs readable. Set `MGL_TRACE_LOG_STDERR=1` when you also want the trace stream mirrored to `stderr`.

Useful switches:

```bash
MGL_TRACE_LOG=1
MGL_TRACE_LOG_STDERR=1
MGL_TRACE_LOG_DRAW=1
MGL_TRACE_LOG_RESOURCES=1
MGL_TRACE_LOG_PROGRAMS=91,92,93
```

| Variable | Description |
|------|------|
| `MGL_TRACE_LOG` | Enables trace file output |
| `MGL_TRACE_LOG_STDERR` | Mirrors trace output to `stderr`; disabled by default |
| `MGL_TRACE_LOG_DRAW` | Logs draw-call and draw-replay diagnostics |
| `MGL_TRACE_LOG_RESOURCES` | Logs more detailed buffer/texture/sampler binding diagnostics |
| `MGL_TRACE_LOG_PROGRAMS` | Focuses tracing on selected programs; accepts comma/space/semicolon/colon-separated values |

`MGL TRACE`, `MGL DUMP`, and shader interface dump diagnostics go through the trace file. `MGL ERROR` / `MGL WARNING` messages remain on the normal log path so real failures are still visible immediately.

Example:

```bash
MGL_TRACE_LOG=1 MGL_TRACE_LOG_DRAW=1 MGL_TRACE_LOG_PROGRAMS=91,92
```

After launch, look for `mgl-trace-<pid>.log` next to the MGL dylib.

### MGL_MIP_DIAG

Set `MGL_MIP_DIAG=1` to report the sampled texture's actual mip chain and sampler
state. This is for diagnosing mipmap-related visual artifacts.

It is independent of `MGL_TRACE_LOG`: per-binding trace lines are too dense, and
frame rate drops too far to see view-dependent defects. Output uses the normal log
path with the `MGL MIP_DIAG` prefix, and it only prints when state changes — silent
while the picture is stable, a burst of logs when something flips.

Three records:

| Record | Trigger | Purpose |
|------|--------|------|
| `MIP_DIAG texture` | Texture bind | Whether GL and Metal mip counts match, `mipmapped`/`genmipmaps` flags, `mtlTex` pointer changes (a new pointer means the texture was rebuilt) |
| `MIP_DIAG frag` | Fragment sampler resolve | Immediate-mode filter, LOD clamp, `BASE_LEVEL`/`MAX_LEVEL`; whether the render target is sampled through a Y-flip copy (`viaCopy`), copy mip count, dirty mip mask, and version |
| `MIP_DIAG snapshot` | Deferred batch replay | The per-draw sampler actually submitted to Metal when deferred batching is on; this overrides the `frag` record above |

```bash
MGL_MIP_DIAG=1
```

## Acknowledgements

- [Khronos Group](https://www.khronos.org/) - OpenGL Registry and GL CTS
- [LLVM](https://llvm.org/) - AIR backend code generation
- [Apple metal-cpp](https://github.com/apple/metal-cpp) - Metal C++ bindings
- [GLFW](https://www.glfw.org/) - Window management library
- [openglonmetal/MGL](https://github.com/openglonmetal/MGL) - Original MGL framework; MGL-minecraft would not exist without it
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn)
- [apitrace](https://github.com/apitrace/apitrace)
