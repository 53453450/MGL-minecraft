# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-LGPL--3.0--only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6-green.svg)]()
[![Metal](https://img.shields.io/badge/Metal-3.0-orange.svg)]()

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

- macOS 14 or newer

- Xcode Command Line Tools  

- Homebrew  

- CMake  

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

`clone_external.sh` only fetches OpenGL-Registry, ezxml, and Apple's official
[metal-cpp](https://github.com/apple/metal-cpp) when those directories are missing.
`external/glfw` is the repository's locally modified checkout; it is never cloned
or pulled from upstream and is always used for the build.

### 3. Build MGL

```bash
cd MGL-minecraft
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
├── MGL/                         # Core library source
│   ├── include/                 # OpenGL/MGL headers
│   └── src/                     # C/Objective-C implementation
│       ├── MGLRenderer.m        # Metal renderer and draw paths
│       ├── MGLTextures.m        # Metal texture bridge
│       ├── buffers.c            # Buffer/UBO/SSBO state
│       ├── framebuffers.c       # FBO/RBO and completeness
│       ├── get.c                # glGet/internalformat queries
│       ├── pixel_utils.c        # Pixel format and layout helpers
│       ├── rendering.c          # Render state and draw dispatch
│       ├── shaders.c            # GLSL frontend entry
│       ├── tex_param.c          # Texture parameters and internalformat queries
│       └── textures.c           # Texture upload, clear, and compressed paths
├── external/                    # SPIRV-Cross, SPIRV-Tools, glslang, GLFW, etc.
├── benchmark/                   # Performance test tools
├── test_mgl/                    # Local smoke/functional tests
├── MGL_Golden_Images/           # Image regression baselines
├── TestImages/                  # Test texture assets
├── enum_parser/                 # OpenGL enum generation helper
├── spec_parser/                 # Specification parsing helper
├── build/                       # Local build output
├── Makefile                     # Sole build entry point
├── README.md                    # Chinese README
├── README_EN.md                 # English README
├── LICENSE                      # LGPL 3.0 text for repository changes after the baseline
├── LICENSE-APACHE-2.0           # Apache 2.0 text for original MGL code
├── LICENSE-GPL-3.0-only         # GPL 3.0 text incorporated by LGPL 3.0
└── LICENSING.md                 # License scope and commit boundary notice
```

## Core Modules

### Shader Translation (shaders.c)

Shader translation is the core of MGL, converting GLSL into Metal Shading Language (MSL):

```c
GLSL (330/420/450)

    │

    ▼

glslang compilation

    │

    ▼

SPIR-V intermediate

    │

    ▼

SPIRV-Cross

    │

    ▼

Metal Shading Language
```

### State Management

OpenGL state is synchronized to Metal using a dirty-flag system:

```c
// Status change mark
STATE(dirty_bits) |= DIRTY_RENDER_STATE;

// Deal with the dirty state when drawing
processGLState(ctx, true);
```

### Metal Renderer (MGLRenderer.m)

Implemented in Objective-C, responsible for:
- RenderCommandEncoder management
- State mapping (OpenGL → Metal)
- Draw call execution

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

## Acknowledgements

- [Khronos Group](https://www.khronos.org/) - SPIRV-Cross, glslang, SPIRV-Tools,VK-GL-CTS
- [GLFW](https://www.glfw.org/) - Window management library
- [openglonmetal](https://github.com/openglonmetal/MGL) - Original MGL framework
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)
