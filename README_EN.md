Language: [中文](README.md) | English


# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-LGPL--3.0--only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-4.0-orange.svg)]()

**MGL (Metal-GL)** is a graphics translation layer that converts OpenGL 4.6 and OpenGL ES 3.x calls into Apple Metal. It allows existing OpenGL applications to run on macOS using a Metal backend without modification.

---

## Introduction

### Project Notes

- <span style="color:red;">This is a purely AI-generated coding project. If you dislike or are against AI-generated code, you may leave this repository.</span>

- This project is forked from: https://github.com/openglonmetal/MGL

- Minecraft (MC) is one of the few games that run relatively well on macOS. However, its longevity largely comes from its massive modding community. Apple officially deprecated OpenGL and OpenCL at WWDC 2018 (June 2018), and macOS OpenGL support has been stuck at version 4.1 ever since. The vertex attribute limit (GL_MAX_VERTEX_ATTRIBS) is 16, which is far behind modern mod requirements. Many mods and most shader packs cannot run on macOS.  

  This project upgrades OpenGL support to 4.6 and increases `GL_MAX_VERTEX_ATTRIBS` to 30.

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

---

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

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/53453450/MGL-minecraft.git
cd MGL-minecraft
```

### 2. Build dependencies

```bash
# Install dependencies

make install-pkgdeps

cd external

# Clone external dependencies

./clone_external.sh

# Build dependencies

./build_external.sh
```

`clone_external.sh` checks out OpenGL-Registry at the commit in
`MGL/generated/registry.lock`, and fetches ezxml (pinned commit) and Apple's official
[metal-cpp](https://github.com/apple/metal-cpp) only when those directories are missing.
`external/glfw` is the repository's locally modified checkout; it is never cloned
or pulled from upstream and is always used for the build.

### 3. Build MGL

```bash
# Return to the repository root
cd ..
make
```

## Build Outputs

After compilation, the following files will be generated in the build/ directory:

| File | Description |
|------|------|
| `libmgl.dylib` | OpenGL Core dynamic library |
| `libmgl_es.dylib` | OpenGL ES dynamic library |
| `libglfw.dylib` | Modified GLFW library |

## Usage

After building, add the following JVM arguments in your launcher:
```JVM
-Dorg.lwjgl.opengl.libname="/yourpath/to/libmgl.dylib"
-Dorg.lwjgl.glfw.libname="/yourpath/to/libglfw.dylib"
-Dorg.lwjgl.opengles.libname="/yourpath/to/libmgl_es.dylib"
```
Point them to the built libraries so they can take over rendering.

## Current Status

- The current priority is GL46CTS conformance. Minecraft/mod runtime behavior may still regress while CTS fixes are landing.

## Project Structure

```
MGL-minecraft/
├── MGL/
│   ├── include/                 # OpenGL API, GLMContext, and MGL C ABI headers
│   ├── src/                     # Engine: C state machine + C++ (no .m / .mm)
│   │   ├── gl_core.c / gl_es.c / glm_dispatch.c / glm_context.c
│   │   ├── mgl_glsl_*.c / mgl_ir.c / mgl_air_*.cpp   # GLSL → MGLIR → AIR
│   │   ├── mgl_renderer_entries.c / mgl_draw_entry.c # GL semantic entry points
│   │   ├── mgl_draw_issue.cpp / mgl_draw_encode.cpp  # tess / GS / draw orchestration
│   │   ├── mgl_render.cpp                            # Sole Metal-cpp implementation TU
│   │   ├── mgl_renderer_backend.cpp                  # Owner, cache, and transaction
│   │   ├── mgl_platform_shell.cpp                    # AppKit / CAMetalLayer platform shell
│   │   ├── mgl_objc_bridge.h                         # C++ wrappers around libobjc / objc_msgSend
│   │   └── mgl_aux_assets.*                          # Precompiled auxiliary metallib table
│   └── aux_shaders/             # Build-time compiled embedded helper shaders
├── external/
│   ├── metal-cpp/               # Apple's official header-only Metal C++ bindings
│   ├── glfw/                    # Local modified GLFW checkout (still contains Cocoa .m)
│   ├── OpenGL-Registry/         # Khronos OpenGL registry
│   └── ezxml/                   # XML parsing dependency
├── test_legacy_compat/          # GLSL compatibility, AIR, smoke, and gtest tests
├── test_regression/             # Headless OpenGL/Metal regression suite
├── test_dirty_hash/             # Dirty-state batch regression
├── test_mgl/                    # Local functional tests
├── benchmark/                   # Performance test tools
├── spec_parser/                 # Specification parsing helper (verify-codegen)
├── scripts/                     # Asset, regression, trace, and benchmark scripts
├── docs/                        # Architecture and spec reviews
├── MGL_Golden_Images/           # Image regression baselines
├── TestImages/                  # Test texture assets
├── config.mk.example            # Local SDK/toolchain configuration template
├── build/                       # Local build output generated by Makefile
├── Makefile                     # Sole build entry point
├── README.md                    # Chinese README
├── README_EN.md                 # English README
├── LICENSE                      # LGPL 3.0 text for repository changes after the baseline
├── LICENSE-APACHE-2.0           # Apache 2.0 text for original MGL code
├── LICENSE-GPL-3.0-only         # GPL 3.0 text incorporated by LGPL 3.0
└── LICENSING.md                 # License scope and commit boundary notice
```

Production code under `MGL/src` is C or C++ only. AppKit / CAMetalLayer access lives in `mgl_platform_shell.cpp`: classes are registered with the ObjC runtime at load, and messages go through `objc_msgSend`. There is no `.m` in the MGL engine. The GLFW fork still has Cocoa `.m` files; that is the window library, not the engine.

## Core Modules

### Shader and AIR Path

The production path is the project's GLSL frontend, MGLIR, AIR backend, and
precompiled metallib:

```c
OpenGL glCompileShader / program link
    │
    ▼
mgl_legacy_compat.c (legacy GLSL compatibility)
    │
    ▼
mgl_glsl_lexer.c -> mgl_glsl_parser.c -> mgl_glsl_sema.c
    │
    ▼
mgl_ir.c / mgl_air_reflect.c
    │
    ▼
mgl_air_backend.cpp -> AIR LLVM bitcode -> metallib
    │
    ▼
mgl_air_loader.cpp / mgl_render.cpp -> Metal-cpp PSOs and encoders
```

Key properties:
- Legacy GLSL compatibility is applied before the frontend.
- AIR reflection produces resource, varying, tessellation, and geometry ABI metadata.
- Auxiliary shaders are embedded as metallib bytes; runtime code does not read Metal sources.

### State Management

OpenGL state is synchronized to Metal using a dirty-flag system:

```c
// Status change mark
STATE(dirty_bits) |= DIRTY_RENDER_STATE;

// Deal with the dirty state when drawing
processGLState(ctx, true);
```

### Metal Backend and Platform Shell

- `mgl_render.cpp` is the only translation unit that defines Metal-cpp implementation
  macros. It owns resource creation, pipeline/render-pass, binding, draw/blit/compute,
  and command-buffer transactions.
- `mgl_renderer_backend.cpp` owns backend handles, caches, completion, and temporary
  resource lifetime.
- `mgl_platform_shell.cpp` is the platform shell. It owns `NSView`, `CAMetalLayer`,
  drawables, and device initialization, talking to AppKit through libobjc rather than
  an Objective-C source file.
- OpenGL state and semantic orchestration live in C (`mgl_renderer_entries.c`,
  `mgl_draw_entry.c`, and related files) and call the backend through value-state and
  opaque handles.

Core call path:

```text
OpenGL API (gl_core.c / glm_dispatch.c)
        |
        v
C state machine (state / buffers / textures / program / ...)
        |
        v
C renderer entries (mgl_renderer_entries.c / mgl_draw_entry.c)
        |
        v
C++ issue / encode (mgl_draw_issue.cpp / mgl_draw_encode.cpp)
        | value-state / opaque handle
        v
mgl_renderer_backend.cpp (owners, caches, transactions)
        | C ABI
        v
mgl_render.cpp (Metal-cpp) ---> Metal queue / encoder / resource
        ^
        | device, layer, drawable
mgl_platform_shell.cpp (C++ / libobjc -> AppKit / CAMetalLayer)
```

## Debugging and Repro Cases

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
- [openglonmetal](https://github.com/openglonmetal/MGL) - Original MGL framework
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)
