-include config.mk

SHELL := /bin/bash
.DEFAULT_GOAL := lib

# Resolve host/toolchain probes once. Values supplied by config.mk or the
# command line remain overrideable.
SDK_ROOT ?= $(shell xcrun --sdk macosx --show-sdk-path)
SDK_ROOT := $(strip $(SDK_ROOT))
APPLE_CLANG ?= $(shell xcrun --find clang)
APPLE_CLANG := $(strip $(APPLE_CLANG))
HOST_ARCH ?= $(shell uname -m)
HOST_ARCH := $(strip $(HOST_ARCH))

# build dirs
build_dir ?= build
build_core_dir := $(build_dir)/core
build_es_dir := $(build_dir)/es

CFLAGS += -Wall #-Wunused-parameter #-Wextra
CFLAGS += -gfull
CFLAGS += -O2
#CFLAGS += -00
# Sanitizer builds: `make SANITIZE=address lib` (or =thread).  Production
# builds stay unsanitized; ASan-loaded dylibs are known to crash under
# dlopen() so sanitized runs use the standalone regression binary.
ifdef SANITIZE
CFLAGS += -fsanitize=$(SANITIZE) -fno-omit-frame-pointer
CXXFLAGS += -fsanitize=$(SANITIZE) -fno-omit-frame-pointer
LIBS += -fsanitize=$(SANITIZE)
endif
CFLAGS += -arch $(HOST_ARCH)
LIBS += -arch $(HOST_ARCH)

LIBS += -F$(SDK_ROOT)/System/Library/Frameworks
LIBS += -framework Metal -framework OpenGL -framework Foundation

CFLAGS += -IMGL/include
CFLAGS += -IMGL/include/GL # "glcorearb.h"
CFLAGS += -IMGL/src        # "mgl_safety.h" lives in MGL/src/, used by MGLRenderer_Private.h

# GLFW configuration for shared library build
CFLAGS += -I./external/glfw/include -I./external/glfw/src
CXXFLAGS += -I./external/glfw/include -I./external/glfw/src

# macOS specific compile definitions for GLFW
CFLAGS += -D_COCOA -D_GLFW_COCOA
CXXFLAGS += -D_COCOA -D_GLFW_COCOA

# GL_CORE SPECIFIC FLAGS
CFLAGS_GL_CORE := $(CFLAGS) -DMGL_GL_CORE

# GL_ES SPECIFIC FLAGS
CFLAGS_GL_ES := $(CFLAGS) -DMGL_GL_ES

# Add CoreFoundation framework headers for GLFW Objective-C compilation
GLFW_FRAMEWORKS = -framework Cocoa -framework CoreFoundation -framework CoreGraphics \
                  -framework IOKit -framework Foundation -framework QuartzCore \
                  -framework Metal -framework OpenGL

# GLFW sources for shared library build - macOS specific configuration
GLFW_SRC_DIR = external/glfw/src
GLFW_C_SOURCES = $(GLFW_SRC_DIR)/context.c \
                $(GLFW_SRC_DIR)/init.c \
                $(GLFW_SRC_DIR)/input.c \
                $(GLFW_SRC_DIR)/monitor.c \
                $(GLFW_SRC_DIR)/vulkan.c \
                $(GLFW_SRC_DIR)/window.c \
                $(GLFW_SRC_DIR)/osmesa_context.c \
                $(GLFW_SRC_DIR)/egl_context.c \
                $(GLFW_SRC_DIR)/posix_thread.c \
                $(GLFW_SRC_DIR)/posix_module.c \
                $(GLFW_SRC_DIR)/cocoa_time.c \
                $(GLFW_SRC_DIR)/platform.c

GLFW_M_SOURCES = $(GLFW_SRC_DIR)/cocoa_init.m \
                $(GLFW_SRC_DIR)/cocoa_joystick.m \
                $(GLFW_SRC_DIR)/cocoa_monitor.m \
                $(GLFW_SRC_DIR)/cocoa_window.m \
                $(GLFW_SRC_DIR)/mgl_context.m

# Simplified GLFW object paths - use a flat structure for easier building
GLFW_BUILD_DIR = $(build_dir)/glfw
GLFW_C_OBJS = $(GLFW_C_SOURCES:$(GLFW_SRC_DIR)/%.c=$(GLFW_BUILD_DIR)/%.o)

GLFW_M_OBJS = $(GLFW_M_SOURCES:$(GLFW_SRC_DIR)/%.m=$(GLFW_BUILD_DIR)/%.o)
glfw_objs = $(GLFW_C_OBJS) $(GLFW_M_OBJS)

ifneq ($(SDK_ROOT),)
CFLAGS_GL_CORE += -isysroot $(SDK_ROOT)
CFLAGS_GL_ES += -isysroot $(SDK_ROOT)
endif

LIBS += -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
LIBS += -lc++


# --
# no need to tweak after this line, hopefully

default: lib

help:
	@printf '%s\n' \
		'Build targets:' \
		'  make                  Build libmgl.dylib, libmgl_es.dylib, and libglfw.dylib.' \
		'  make lib              Build the runtime dylibs.' \
		'  make core             Build only Core MGL and GLFW (Minecraft path).' \
		'  make es               Build only the OpenGL ES MGL dylib.' \
		'  make bench            Build the MGL benchmark.' \
		'  make test-benchmark   Run the benchmark smoke gate.' \
		'  make test-all         Run the complete non-interactive local test gate.' \
		'  make test-regression  Build and run the headless regression suite.' \
		'  make test-dirty-hash  Run the minimal dirty-hash batch regression.' \
		'  make test             Run the interactive GLFW test application.' \
		'  make check-air-only   Fail if production paths reference the legacy GLSL->SPIR-V->MSL chain.' \
		'  make check-p5-metalcpp Fail if the single-path Metal-cpp renderer regresses.' \
		'  make clean            Remove local build outputs.'

# P3 硬闸：生产路径不得残留旧 source-compile 链（详见 scripts/check_air_only.sh）。
check-air-only:
	@bash scripts/check_air_only.sh

check-p4-metalcpp:
	@bash scripts/check_p4_metalcpp.sh

check-p5-metalcpp:
	@bash scripts/check_p5_metalcpp.sh

# mgl
#mgl_srcs_c := $(wildcard MGL/src/*.c)
mgl_srcs_c := $(filter-out %/gl_core.c  %/gl_es.c, $(wildcard MGL/src/*.c))

# Aux shader assets (P3): the precompiled metallib table embeds all helper
# shaders; the runtime never compiles .metal source.  The table is regenerated
# when a *.metal, the MANIFEST, or the generator changes; the committed
# mgl_aux_assets.* files keep clean clones buildable without the metal tools.
MGL_METAL ?= $(shell xcrun --sdk macosx --find metal 2>/dev/null)
MGL_METALLIB ?= $(shell xcrun --sdk macosx --find metallib 2>/dev/null)
AUX_METAL_SRCS := $(wildcard MGL/aux_shaders/*.metal)
AUX_BUILD_DIR := $(build_dir)/aux
AUX_METALLIBS := $(patsubst MGL/aux_shaders/%.metal,$(AUX_BUILD_DIR)/%.metallib,$(AUX_METAL_SRCS))
AUX_ASSET_STAMP := $(AUX_BUILD_DIR)/aux_assets.stamp

$(AUX_BUILD_DIR)/%.air: MGL/aux_shaders/%.metal
	@mkdir -p $(dir $@)
	$(MGL_METAL) -c $< -o $@

$(AUX_BUILD_DIR)/%.metallib: $(AUX_BUILD_DIR)/%.air
	$(MGL_METALLIB) $< -o $@

$(AUX_ASSET_STAMP): MGL/aux_shaders/MANIFEST $(AUX_METALLIBS) scripts/gen_aux_assets.py
	@mkdir -p $(dir $@)
	python3 scripts/gen_aux_assets.py MGL/aux_shaders/MANIFEST \
		$(AUX_BUILD_DIR) MGL/include/mgl_aux_assets.h MGL/src/mgl_aux_assets.c
	@touch $@

# The generated table is part of both dylibs; regenerate it before compiling.
$(build_core_dir)/MGL/src/mgl_aux_assets.o: $(AUX_ASSET_STAMP)
$(build_es_dir)/MGL/src/mgl_aux_assets.o: $(AUX_ASSET_STAMP)

# MGL/src currently has no C++ sources, but the wildcard must be defined so
# the .cpp rules below are not silently dropped if one is added later.
mgl_srcs_cpp := $(wildcard MGL/src/*.cpp)

mgl_srcs_objc := $(wildcard MGL/src/*.m)

mgl_core_c := MGL/src/gl_core.c
mgl_es_c := MGL/src/gl_es.c

mgl_core_obj := $(mgl_core_c:.c=.o)
mgl_core_obj := $(addprefix $(build_core_dir)/,$(mgl_core_obj))

mgl_es_obj := $(mgl_es_c:.c=.o)
mgl_es_obj := $(addprefix $(build_es_dir)/,$(mgl_es_obj))

# core objs
mgl_core_objs := $(mgl_srcs_c:.c=.o) $(mgl_srcs_cpp:.cpp=.o)
mgl_core_objs := $(addprefix $(build_core_dir)/,$(mgl_core_objs))

mgl_core_arc_objs := $(mgl_srcs_objc:.m=.o)
mgl_core_arc_objs := $(addprefix $(build_core_dir)/arc/,$(mgl_core_arc_objs))

# es objs
mgl_es_objs := $(mgl_srcs_c:.c=.o) $(mgl_srcs_cpp:.cpp=.o)
mgl_es_objs := $(addprefix $(build_es_dir)/,$(mgl_es_objs))

mgl_es_arc_objs := $(mgl_srcs_objc:.m=.o)
mgl_es_arc_objs := $(addprefix $(build_es_dir)/arc/,$(mgl_es_arc_objs))


# Define the directories and repositories
EXT_DIRS = ./external/OpenGL-Registry \
           ./external/ezxml

# Simplified index_of function - find position of directory in EXT_DIRS
define index_of
$(strip $(1))
endef

# Function to get the corresponding repository URL for a directory
# Simplified mapping for common directories
define get_repo_url
$(if $(filter $(1),./external/OpenGL-Registry),https://github.com/KhronosGroup/OpenGL-Registry.git, \
https://github.com/lxfontes/ezxml.git)
endef

# Function to check if a directory exists, and if not, clone it
define check_and_clone
	@echo "Resolving directory $(1)..."; \
	INDEX=$(call index_of,$(1)); \
	REPO=$(call get_repo_url,$(1)); \
	echo "INDEX calculated: $$INDEX"; \
	echo "REPO resolved: $$REPO"; \
	if [ ! -d $(1) ]; then \
		echo "Cloning from $$REPO into $(1)..."; \
		git clone $$REPO $(1) --depth 1; \
	else \
		echo "$(1) already exists, skipping."; \
	fi
endef

# Use the `check_and_clone` function for each directory
$(EXT_DIRS):
	$(call check_and_clone,$@)


deps += $(mgl_core_objs:.o=.d)
deps += $(mgl_es_objs:.o=.d)
deps += $(mgl_core_obj:.o=.d)
deps += $(mgl_es_obj:.o=.d)
deps += $(mgl_core_arc_objs:.o=.d)
deps += $(mgl_es_arc_objs:.o=.d)
deps += $(glfw_objs:.o=.d)


mgl_lib := $(build_dir)/libmgl.dylib
mgl_es_lib := $(build_dir)/libmgl_es.dylib

mgl_core_link_objs := $(mgl_core_objs) $(mgl_core_arc_objs) $(mgl_core_obj)
mgl_es_link_objs := $(mgl_es_objs) $(mgl_es_arc_objs) $(mgl_es_obj)

# M1 AIR backend: GLSL -> metallib -> PSO gate (C++20 + LLVM, Metal runtime).
# Define these before the compile/link configuration hashes below so changes
# to C++ and LLVM flags invalidate existing objects and libraries.
LLVM_ROOT ?= /opt/homebrew/opt/llvm@15
LLVM_CXX ?= $(APPLE_CLANG)
LLVM_CXXFLAGS := -std=c++20 -isysroot $(SDK_ROOT) -I$(LLVM_ROOT)/include -IMGL/include \
	-IMGL/src \
	-IMGL/include/GL \
	-Iexternal/metal-cpp
LLVM_LDFLAGS := -L$(LLVM_ROOT)/lib -lLLVM-15 -lc++
# The *.cpp sources (GLSL->metallib compiler + Metal-cpp renderer/loader) build
# with LLVM headers and metal-cpp (header-only).
M1_AIR_CXXFLAGS := -std=c++20 -I$(LLVM_ROOT)/include -IMGL/include \
	-IMGL/src \
	-IMGL/include/GL \
	-Iexternal/metal-cpp
CXXFLAGS_GL_CORE := $(CXXFLAGS) -DMGL_GL_CORE $(M1_AIR_CXXFLAGS)
CXXFLAGS_GL_ES := $(CXXFLAGS) -DMGL_GL_ES $(M1_AIR_CXXFLAGS)
# Product libs carry the M1 AIR backend, so they depend on the LLVM runtime.
# -lobjc: mgl_render_cpp.cpp 等纯 C++ TU 内联调用 objc_msgSend，clang++ 不会
# 像 ObjC 目标文件那样自动补链 ObjC runtime。
LIBS += $(LLVM_LDFLAGS) -lobjc

CC_ID := $(shell $(CC) --version 2>/dev/null | sed -n '1p')
CXX_ID := $(shell $(CXX) --version 2>/dev/null | sed -n '1p')
APPLE_CLANG_ID := $(shell $(APPLE_CLANG) --version 2>/dev/null | sed -n '1p')

core_compile_key := $(shell printf '%s\n' "$(CC)" "$(CC_ID)" "$(CXX)" "$(CXX_ID)" "$(APPLE_CLANG)" "$(APPLE_CLANG_ID)" "$(SDK_ROOT)" "$(CFLAGS_GL_CORE)" "$(CXXFLAGS_GL_CORE)" | shasum -a 256 | awk '{print $$1}')
es_compile_key := $(shell printf '%s\n' "$(CC)" "$(CC_ID)" "$(CXX)" "$(CXX_ID)" "$(APPLE_CLANG)" "$(APPLE_CLANG_ID)" "$(SDK_ROOT)" "$(CFLAGS_GL_ES)" "$(CXXFLAGS_GL_ES)" | shasum -a 256 | awk '{print $$1}')
core_link_key := $(shell printf '%s\n' "$(CC)" "$(CC_ID)" "$(SDK_ROOT)" "$(LDFLAGS)" "$(LIBS)" | shasum -a 256 | awk '{print $$1}')
es_link_key := $(core_link_key)

core_compile_stamp := $(build_core_dir)/.compile-config-$(core_compile_key)
es_compile_stamp := $(build_es_dir)/.compile-config-$(es_compile_key)
core_link_stamp := $(build_core_dir)/.link-config-$(core_link_key)
es_link_stamp := $(build_es_dir)/.link-config-$(es_link_key)

$(core_compile_stamp):
	@mkdir -p $(dir $@)
	@rm -f $(build_core_dir)/.compile-config-*
	@sleep 1
	@touch $@

$(es_compile_stamp):
	@mkdir -p $(dir $@)
	@rm -f $(build_es_dir)/.compile-config-*
	@sleep 1
	@touch $@

$(core_link_stamp):
	@mkdir -p $(dir $@)
	@rm -f $(build_core_dir)/.link-config-*
	@sleep 1
	@touch $@

$(es_link_stamp):
	@mkdir -p $(dir $@)
	@rm -f $(build_es_dir)/.link-config-*
	@sleep 1
	@touch $@

$(mgl_core_link_objs): $(core_compile_stamp)
$(mgl_es_link_objs): $(es_compile_stamp)

$(mgl_lib): $(mgl_core_link_objs) $(core_link_stamp)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -dynamiclib -o $@ $(mgl_core_link_objs) $(LIBS)
	# loading dynamic library requires this
	ln -fs $(mgl_lib) .

$(mgl_es_lib): $(mgl_es_link_objs) $(es_link_stamp)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -dynamiclib -o $@ $(mgl_es_link_objs) $(LIBS)
	# loading dynamic library requires this
	ln -fs $(mgl_es_lib) .


# Configure + build GLFW on demand so a clean clone builds with plain
# `make` (no glslang/SPIRV-* trees involved; see external/build_external.sh).
external/glfw/build/src/libglfw3.a:
	@bash external/build_external.sh

# Build GLFW shared library from pre-built static library
$(build_dir)/libglfw.dylib: external/glfw/build/src/libglfw3.a $(mgl_lib)
	@echo "Creating GLFW shared library from static library..."
	@mkdir -p $(dir $@)
	$(CC) -shared -fPIC -dynamiclib \
		-Wl,-force_load,$(word 1,$^) \
		-L$(build_dir) -lmgl \
		-o $@ \
		$(GLFW_FRAMEWORKS) \
		-install_name @rpath/libglfw.dylib
	@install_name_tool -change build/libmgl.dylib @loader_path/libmgl.dylib $@ 2>/dev/null || true
	@install_name_tool -change @rpath/libmgl.dylib @loader_path/libmgl.dylib $@ 2>/dev/null || true
	@echo "✅ GLFW shared library built: $@"
	@echo "This enables compatibility with Minecraft mods and Prism Launcher"


# specific rules

core: $(mgl_lib) $(build_dir)/libglfw.dylib

es: $(mgl_es_lib)

lib: core es

test_exe := $(build_dir)/test_mgl

test: $(test_exe)
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(test_exe)

dbg: $(test_exe)
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) lldb -o run $(test_exe)

$(build_dir)/test_mgl: test_mgl/main.cpp $(mgl_lib) $(build_dir)/libglfw.dylib
	$(CXX) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glfw/include \
		-IMGL/include -IMGL/include/GL \
		-DMGL_GL_CORE \
		-isysroot $(SDK_ROOT) \
		test_mgl/main.cpp \
		-L$(build_dir) -lmgl -lglfw \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@


# generic rules

#
# core build
#
$(build_core_dir)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -MMD $(CFLAGS_GL_CORE) -c $< -o $@

#-std=gnu17 
$(build_core_dir)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MMD $(CXXFLAGS_GL_CORE) -c $< -o $@

#-std=c++14
$(build_core_dir)/arc/%.o: %.m
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -fobjc-arc -fmodules -MMD $(CFLAGS_GL_CORE) \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-c $< -o $@

$(build_core_dir)/%.o: %.m
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -fmodules -MMD $(CFLAGS_GL_CORE) -c $< -o $@


#
# es build
#
$(build_es_dir)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -MMD $(CFLAGS_GL_ES) -c $< -o $@

#-std=gnu17
$(build_es_dir)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MMD $(CXXFLAGS_GL_ES) -c $< -o $@

#-std=c++14
$(build_es_dir)/arc/%.o: %.m
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -fobjc-arc -fmodules -MMD $(CFLAGS_GL_ES) \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-c $< -o $@

$(build_dir)/%.o: %.m
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -fmodules -MMD $(CXXFLAGS_GL_ES) -c $< -o $@




# GLFW-specific build rules with simplified flat directory structure
$(GLFW_BUILD_DIR)/%.o: $(GLFW_SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -MMD $(CFLAGS) -c $< -o $@

$(GLFW_BUILD_DIR)/%.o: $(GLFW_SRC_DIR)/%.m
	@mkdir -p $(dir $@)
	clang -fno-objc-arc -fmodules -MMD $(CFLAGS) $(GLFW_FRAMEWORKS) -c $< -o $@

clean:
	rm -rf $(build_dir)
	rm -f libmgl.dylib
	rm -f libmgl_es.dylib
	rm -f libglfw.dylib

install-pkgdeps: download-pkgdeps compile-pkgdeps

download-pkgdeps:

	brew install glm glfw

compile-pkgdeps:

	@echo "use /external/.sh"

# Benchmark target — builds the comprehensive MGL translation-overhead benchmark.
# Depends on libmgl.dylib and libglfw.dylib being built first (run `make lib`).
BENCHMARK_GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
SYSTEM_GLFW_PREFIX ?= $(if $(wildcard /opt/homebrew/opt/glfw/include/GLFW/glfw3.h),/opt/homebrew/opt/glfw,$(shell brew --prefix glfw 2>/dev/null))

bench: $(build_dir)/libmgl.dylib $(build_dir)/libglfw.dylib
	$(APPLE_CLANG) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		-I./external/glfw/include \
		-IMGL/include -IMGL/include/GL \
		-DMGL_GL_CORE \
		-DMGL_BENCHMARK_GIT_COMMIT=\"$(BENCHMARK_GIT_COMMIT)\" \
		-isysroot $(SDK_ROOT) \
		benchmark/mgl_benchmark.c \
		-L$(build_dir) -lmgl -lglfw \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $(build_dir)/mgl_benchmark
	@echo "✅ Benchmark built: $(build_dir)/mgl_benchmark"

# System Apple OpenGL benchmark target — compiles the same benchmark source
# with -D__MGL_BENCHMARK_SYSTEM_GL__ and links against the system OpenGL
# framework via brew's GLFW (no MGL dependency).  Requires `brew install glfw`.
bench-system: benchmark/mgl_benchmark.c
	$(APPLE_CLANG) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		-I$(SYSTEM_GLFW_PREFIX)/include \
		-IMGL/include -IMGL/include/GL \
		-D__MGL_BENCHMARK_SYSTEM_GL__ \
		-DMGL_BENCHMARK_GIT_COMMIT=\"$(BENCHMARK_GIT_COMMIT)\" \
		-isysroot $(SDK_ROOT) \
		benchmark/mgl_benchmark.c \
		-L$(SYSTEM_GLFW_PREFIX)/lib -lglfw \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework OpenGL \
		-Wl,-rpath,$(SYSTEM_GLFW_PREFIX)/lib \
		-o $(build_dir)/mgl_benchmark_system
	@echo "✅ System OpenGL benchmark built: $(build_dir)/mgl_benchmark_system"

# Draw-pipeline regression suite (Stage 0.1 of RENDERER_EVOLUTION_TODO.md).
# Non-interactive, headless, FBO-offscreen. Covers array/element/instanced/
# multidraw/indirect + FBO switch + XFB + conditional render. Produces TGA
# snapshots compared against MGL_Golden_Images/Reg_*.tga.
$(build_dir)/test_regression: test_regression/main.c $(build_dir)/libmgl.dylib $(build_dir)/libglfw.dylib
	$(APPLE_CLANG) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glfw/include \
		-IMGL/include -IMGL/include/GL \
		-DMGL_GL_CORE \
		-isysroot $(SDK_ROOT) \
		test_regression/main.c \
		-L$(build_dir) -lmgl -lglfw \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@
	@echo "✅ Regression suite built: $@"

build-test-regression: $(build_dir)/test_regression

test-regression: build-test-regression
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_regression \
		--golden-dir $(abspath MGL_Golden_Images)

$(build_dir)/test_dirty_hash: test_dirty_hash/main.c $(build_dir)/libmgl.dylib
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL \
		-DMGL_GL_CORE \
		-isysroot $(SDK_ROOT) \
		test_dirty_hash/main.c \
		-L$(build_dir) -lmgl \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@

test-dirty-hash: $(build_dir)/test_dirty_hash
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_dirty_hash

test-benchmark: bench
	scripts/run_benchmark_smoke.sh --no-build

$(build_dir)/test_legacy_compat: test_legacy_compat/main.c \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include -IMGL/include/GL \
		test_legacy_compat/main.c MGL/src/mgl_legacy_compat.c \
		-o $@

test-legacy-compat: $(build_dir)/test_legacy_compat
	$(build_dir)/test_legacy_compat

$(build_dir)/test_mglir: test_legacy_compat/test_mglir.c MGL/src/mgl_ir.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include \
		test_legacy_compat/test_mglir.c MGL/src/mgl_ir.c \
		-o $@

test-mglir: $(build_dir)/test_mglir
	$(build_dir)/test_mglir

$(build_dir)/test_mgllex: test_legacy_compat/test_mgllex.c MGL/src/mgl_glsl_lexer.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include \
		test_legacy_compat/test_mgllex.c MGL/src/mgl_glsl_lexer.c \
		-o $@

test-mgllex: $(build_dir)/test_mgllex
	$(build_dir)/test_mgllex

$(build_dir)/test_mglparse: test_legacy_compat/test_mglparse.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include \
		test_legacy_compat/test_mglparse.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		-o $@

test-mglparse: $(build_dir)/test_mglparse
	$(build_dir)/test_mglparse

$(build_dir)/test_mglsema: test_legacy_compat/test_mglsema.c MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include -IMGL/include/GL \
		test_legacy_compat/test_mglsema.c MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
		-o $@

test-mglsema: $(build_dir)/test_mglsema
	$(build_dir)/test_mglsema

$(build_dir)/test_mglair: test_legacy_compat/test_mglair.mm \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h \
	MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
	MGL/src/mgl_ir.c
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -gfull -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Cocoa -framework Foundation -framework Metal \
		test_legacy_compat/test_mglair.mm \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
		MGL/src/mgl_legacy_compat.c \
		MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
		MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		MGL/src/mgl_ir.c \
		-o $@

test-mglair: $(build_dir)/test_mglair
	$(build_dir)/test_mglair

# MC-style shader repro: anonymous std140 UBO blocks + samplers through the
# AIR backend.  C sources build as C (they are not valid C++).
MCREPRO_CSRC := MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
	MGL/src/mgl_legacy_compat.c
MCREPRO_COBJ := $(patsubst MGL/src/%.c,$(build_dir)/mcrepro_%.o,$(MCREPRO_CSRC))

$(build_dir)/mcrepro_%.o: MGL/src/%.c
	$(LLVM_CXX) -x c -std=c11 -g -O0 -isysroot $(SDK_ROOT) -IMGL/include \
		-IMGL/include/GL -c $< -o $@

$(build_dir)/test_mcrepro: test_legacy_compat/test_mcrepro.mm \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
	$(MCREPRO_COBJ)
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -g -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Foundation \
		test_legacy_compat/test_mcrepro.mm \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
		-x none $(MCREPRO_COBJ) \
		-o $@

test-mcrepro: $(build_dir)/test_mcrepro
	$(build_dir)/test_mcrepro

# Phase 0 (METALCPP_RENDERER_PLAN): Metal-cpp 基础接入 smoke gate.
# 桥接现有 id<MTLDevice> -> MTL::Device*，init/shutdown 幂等无崩溃。
$(build_dir)/test_metalcpp_smoke: test_legacy_compat/test_metalcpp_smoke.mm \
	MGL/src/mgl_render_cpp.cpp MGL/src/mgl_render_cpp.h \
	MGL/src/mgl_renderer_backend.cpp MGL/src/mgl_renderer_backend.h \
	MGL/src/MGLPlatformRendererShell.m MGL/include/MGLPlatformRendererShell.h \
	MGL/src/mgl_aux_assets.c \
	MGL/src/mgl_buffer_slots.c \
	MGL/src/mgl_sync.m
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -g -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Cocoa -framework Foundation -framework QuartzCore -framework Metal \
		test_legacy_compat/test_metalcpp_smoke.mm \
		MGL/src/mgl_render_cpp.cpp \
		MGL/src/mgl_renderer_backend.cpp \
		MGL/src/MGLPlatformRendererShell.m \
		MGL/src/mgl_aux_assets.c \
		MGL/src/mgl_buffer_slots.c \
		MGL/src/mgl_sync.m \
		-o $@

test-metalcpp: $(build_dir)/test_metalcpp_smoke
	$(build_dir)/test_metalcpp_smoke

# AIR backend unit tests with GoogleTest (pure compile-time, no GPU).
GTEST_ROOT ?= $(HOME)/googletest
GTEST_CXXFLAGS := -I$(GTEST_ROOT)/googletest/include -I$(GTEST_ROOT)/googlemock/include \
	-IMGL/include/GL
GTEST_LIBS := $(GTEST_ROOT)/build-mgl/lib/libgtest.a \
	$(GTEST_ROOT)/build-mgl/lib/libgtest_main.a

$(build_dir)/test_mglair_gtest: test_legacy_compat/test_mglair_gtest.cpp \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h \
	MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
	MGL/src/mgl_ir.c
	$(LLVM_CXX) -x c++ $(LLVM_CXXFLAGS) $(GTEST_CXXFLAGS) $(LLVM_LDFLAGS) \
		test_legacy_compat/test_mglair_gtest.cpp \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
		MGL/src/mgl_legacy_compat.c \
		MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
		MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		MGL/src/mgl_ir.c \
		-x none $(GTEST_LIBS) -o $@

test-mglair-gtest: $(build_dir)/test_mglair_gtest
	$(build_dir)/test_mglair_gtest

# Standalone test targets may be the first target invoked after `make clean`;
# keep their output directory an explicit prerequisite instead of relying on a
# prior library build to create it.
$(build_dir)/test_regression \
$(build_dir)/test_dirty_hash \
$(build_dir)/test_legacy_compat \
$(build_dir)/test_mglir \
$(build_dir)/test_mgllex \
$(build_dir)/test_mglparse \
$(build_dir)/test_mglsema \
$(build_dir)/test_mglair \
$(build_dir)/test_mcrepro \
$(build_dir)/test_metalcpp_smoke \
$(build_dir)/test_mglair_gtest: | $(build_dir)

$(build_dir):
	@mkdir -p $@

test-frontends:
	$(MAKE) test-legacy-compat
	$(MAKE) test-mglir
	$(MAKE) test-mgllex
	$(MAKE) test-mglparse
	$(MAKE) test-mglsema

test-air:
	$(MAKE) test-mglair
	$(MAKE) test-mglair-gtest
	$(MAKE) test-mcrepro
	$(MAKE) test-metalcpp

# Keep the local gate serial: the GPU suites share Metal compiler/archive state.
# The interactive GLFW application and performance benchmark remain explicit.
test-all:
	$(MAKE) check-air-only
	$(MAKE) check-p5-metalcpp
	$(MAKE) test-frontends
	$(MAKE) test-air
	$(MAKE) test-dirty-hash
	$(MAKE) test-regression

.PHONY: default help test dbg core es lib clean install-pkgdeps test-make bench bench-system \
	build-test-regression test-regression test-dirty-hash test-benchmark \
	test-legacy-compat test-mglir test-mgllex test-mglparse test-mglsema \
	test-mglair test-mglair-gtest test-mcrepro test-metalcpp test-frontends \
	test-air test-all check-air-only

.PHONY: check-p4-metalcpp check-p5-metalcpp

-include $(deps)
