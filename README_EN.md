# MGL - Metal-GL

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
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

---

## Requirements

**Prerequisites:**

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

- Known bug: 1.21.8's physical texture infring, the player's item bar icon is not rendered, 1.21.11, 1.21.10's world brightness problem, 1.21.11's UI selection box disappears (there may be other unknown bugs in other versions, but I haven't tested it)
- Forge modloader's startup rendering animation is not loaded, neoforge modloader will crash, and fabric modloader will not have an impact (tested in 1.21.7) 
- sodium is supported when loading with fabric modloader (in all the versions mentioned above)

## Project Structure

```
MGL-minecraft/
├── MGL/                          # Core library source
│   ├── src/                      # C/Objective-C source files
│   │   ├── gl_core.c            # OpenGL Core API entry
│   │   ├── gl_es.c              # OpenGL ES API entry
│   │   ├── shaders.c            # Shader management
│   │   ├── textures.c           # Texture management
│   │   ├── buffers.c            # Buffer management
│   │   ├── programs.c           # Shader program management
│   │   ├── rendering.c          # Rendering state management
│   │   ├── MGLRenderer.m        # Metal renderer implementation
│   │   └── MGLTextures.m        # Metal texture implementation
│   ├── include/                  # Headers
│   │   ├── GL/                  # OpenGL headers
│   │   └── glm/                 # GLM math library
│   └── spirv_cross_c.cpp        # SPIRV-Cross bridge
├── external/                     # External dependencies
│   ├── SPIRV-Cross/             # SPIR-V to MSL translator
│   ├── SPIRV-Tools/             # SPIR-V toolchain
│   ├── SPIRV-Headers/           # SPIR-V headers
│   ├── glslang/                 # GLSL compiler
│   ├── OpenGL-Registry/         # OpenGL specifications
│   ├── glfw/                    # Modified GLFW
│   └── ezxml/                   # XML parser
│
├── test_mgl_glfw/               # Test cases
├── MGL.xcodeproj/               # Xcode project
├── Makefile                     # Build script
└── LICENSE                      # Apache 2.0 License
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

## Acknowledgements

- [Khronos Group](https://www.khronos.org/) - SPIRV-Cross, glslang, SPIRV-Tools
- [GLFW](https://www.glfw.org/) - Window management library
- [openglonmetal](https://github.com/openglonmetal/MGL) - Original MGL framework
- [Hexeption/MCP-Reborn](https://github.com/Hexeption/MCP-Reborn) 
- [apitrace](https://github.com/apitrace/apitrace)

## License

This project is licensed under the Apache License 2.0 – see the LICENSE file for details.
