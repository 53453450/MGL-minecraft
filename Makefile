-include config.mk

SHELL := /bin/bash
.DEFAULT_GOAL := lib

# Resolve host/toolchain probes once. Values supplied by config.mk or the
# command line remain overrideable.
SDK_ROOT ?= $(shell xcrun --sdk macosx --show-sdk-path)
SDK_ROOT := $(strip $(SDK_ROOT))
APPLE_CLANG ?= $(shell xcrun --find clang)
APPLE_CLANG := $(strip $(APPLE_CLANG))
APPLE_CLANGXX ?= $(shell xcrun --find clang++)
APPLE_CLANGXX := $(strip $(APPLE_CLANGXX))
HOST_ARCH ?= $(shell uname -m)
HOST_ARCH := $(strip $(HOST_ARCH))
# Default C/C++ TU compilers.  make would otherwise fall back to bare `cc` /
# `c++` resolved through $PATH, which silently drifts when a CI job prepends a
# Homebrew LLVM bin dir: brew clang++ carries its own libc++ and fails against
# the macOS SDK headers (16 "reference to unresolved using declaration" errors
# in <cmath>/<compare>).  Pinning to the selected Xcode keeps every TU - C,
# ObjC ($(APPLE_CLANG)), C++ and the LLVM-linked test binaries ($(LLVM_CXX)) -
# on one toolchain.  config.mk / environment overrides still win.
# NOTE: `?=` cannot be used here: make's built-in CC/CXX have origin "default",
# which `?=` treats as already defined, so the pin would be silently ignored.
# Only take over when nobody (environment, config.mk, command line) set them.
ifeq ($(origin CC),default)
CC := $(APPLE_CLANG)
endif
ifeq ($(origin CXX),default)
CXX := $(APPLE_CLANGXX)
endif

# Metal toolchain floor.  The injected aux metallibs (see MGL/aux_shaders) are
# built by the Metal 4 frontend that ships with the macOS 26+ SDK; Xcode 15.x
# (Metal 3) rejects them, e.g. "lambda expressions are not supported in Metal"
# in gs_xfb_scatter.metal.  `make verify-toolchain` fails fast and names the
# fix instead of surfacing an opaque MSL error from deep inside a -j build.
MACOS_SDK_VERSION ?= $(shell xcrun --sdk macosx --show-sdk-version 2>/dev/null)
MACOS_SDK_VERSION := $(strip $(MACOS_SDK_VERSION))
MACOS_SDK_MAJOR := $(firstword $(subst ., ,$(MACOS_SDK_VERSION)))
MGL_MIN_MACOS_SDK_MAJOR ?= 26

# build dirs
build_dir ?= build
build_core_dir := $(build_dir)/core
build_es_dir := $(build_dir)/es

CFLAGS += -Wall #-Wunused-parameter #-Wextra
CFLAGS += -gfull
CFLAGS += -O2
# Keep default C++ TUs aligned with C/ObjC: same opt, debug, arch, and SDK.
# CXXFLAGS_GL_* previously omitted these and compiled mgl_render.cpp /
# mgl_air_backend.cpp without -O*/-arch/-isysroot (ARCHITECTURE_AUDIT A11).
CXXFLAGS += -gfull
CXXFLAGS += -O2
CXXFLAGS += -arch $(HOST_ARCH)
CXXFLAGS += -isysroot $(SDK_ROOT)
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

# Source edges for the fork static library (not a parallel object tree).
GLFW_STATIC_DEPS = $(GLFW_C_SOURCES) $(GLFW_M_SOURCES) \
                external/glfw/CMakeLists.txt \
                external/build_external.sh

ifneq ($(SDK_ROOT),)
CFLAGS_GL_CORE += -isysroot $(SDK_ROOT)
CFLAGS_GL_ES += -isysroot $(SDK_ROOT)
# Link against the same SDK the translation units were compiled with.
# Without this the linker falls back to the CommandLineTools default SDK,
# which drifts independently of $(SDK_ROOT): it makes the dylib's minos
# host-dependent, and a CLT SDK newer than the Xcode linker fails outright
# ("ld: library 'System' not found" on a macOS 27 CLT SDK + Xcode 26 ld).
LDFLAGS += -isysroot $(SDK_ROOT)
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
		'  make test-regression-update  Rebuild missing/changed golden TGA images.' \
		'  make gtest            Clone and build the pinned GoogleTest used by AIR unit tests.' \
		'  make test-dirty-hash  Run the minimal dirty-hash batch regression.' \
		'  make test             Run the interactive GLFW test application.' \
		'  make clean            Remove local build outputs.'

# mgl
#mgl_srcs_c := $(wildcard MGL/src/*.c)
mgl_srcs_c := $(filter-out %/gl_core.c  %/gl_es.c, $(wildcard MGL/src/*.c))

# Aux shader assets: the precompiled metallib table embeds all helper
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
	@test -n "$(MGL_METAL)" || { echo "ERROR: metal compiler not found (xcrun --find metal); refusing to emit an empty $@"; exit 1; }
	$(MGL_METAL) -c $< -o $@
	@test -s $@ && test $$(stat -f%z $@) -gt 1024 || { echo "ERROR: $@ is empty or truncated (metal toolchain broken?)"; exit 1; }

$(AUX_BUILD_DIR)/%.metallib: $(AUX_BUILD_DIR)/%.air
	@test -n "$(MGL_METALLIB)" || { echo "ERROR: metallib tool not found (xcrun --find metallib); refusing to emit an empty $@"; exit 1; }
	$(MGL_METALLIB) $< -o $@
	@test -s $@ && test $$(stat -f%z $@) -gt 1024 || { echo "ERROR: $@ is empty or truncated"; exit 1; }

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

# metal-cpp is a header-only external dependency.  Keep it out of the source
# tree's object recipes while making a missing checkout an explicit fetch step;
# this also prevents `make -j` from compiling C++ TUs before the headers arrive.
MGL_METAL_CPP_HEADER := external/metal-cpp/Metal/Metal.hpp
$(MGL_METAL_CPP_HEADER):
	@bash external/clone_external.sh

$(mgl_core_objs) $(mgl_es_objs): $(MGL_METAL_CPP_HEADER)


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


mgl_lib := $(build_dir)/libmgl.dylib
mgl_es_lib := $(build_dir)/libmgl_es.dylib

mgl_core_link_objs := $(mgl_core_objs) $(mgl_core_arc_objs) $(mgl_core_obj)
mgl_es_link_objs := $(mgl_es_objs) $(mgl_es_arc_objs) $(mgl_es_obj)

# M1 AIR backend: GLSL -> metallib -> PSO gate (C++20 + LLVM, Metal runtime).
# Define these before the compile/link configuration hashes below so changes
# to C++ and LLVM flags invalidate existing objects and libraries.
BREW_LLVM15 := $(shell brew --prefix llvm@15 2>/dev/null)
LLVM_ROOT ?= $(if $(strip $(BREW_LLVM15)),$(BREW_LLVM15),/opt/homebrew/opt/llvm@15)
LLVM_CXX ?= $(APPLE_CLANG)
LLVM_CXXFLAGS := -std=c++20 -isysroot $(SDK_ROOT) -I$(LLVM_ROOT)/include -IMGL/include \
	-IMGL/src \
	-IMGL/include/GL \
	-Iexternal/metal-cpp
LLVM_LDFLAGS := -L$(LLVM_ROOT)/lib -lLLVM-15 -lc++
# The *.cpp sources (GLSL->metallib compiler + Metal-cpp renderer/loader) build
# with LLVM headers and metal-cpp (header-only).
# NOTE: keep C++20 — LLVM 15 headers (APFloat.h unique_ptr<APFloat[]>) do not
# survive C++23.  The <stdatomic.h>/<atomic> clash is instead resolved inside
# mgl_types_sync.h / mgl_frame_activity.h (C++ branches use <atomic>).
M1_AIR_CXXFLAGS := -std=c++20 -I$(LLVM_ROOT)/include -IMGL/include \
	-IMGL/src \
	-IMGL/include/GL \
	-Iexternal/metal-cpp
CXXFLAGS_GL_CORE := $(CXXFLAGS) -DMGL_GL_CORE $(M1_AIR_CXXFLAGS)
CXXFLAGS_GL_ES := $(CXXFLAGS) -DMGL_GL_ES $(M1_AIR_CXXFLAGS)
# Product libs carry the M1 AIR backend, so they depend on the LLVM runtime.
# Pure C++ translation units call objc_msgSend through metal-cpp, so the
# Objective-C runtime must be linked explicitly.
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
# Depend on fork sources so edits (e.g. mgl_context.m) invalidate the archive
# and re-enter the cmake incremental build (ARCHITECTURE_AUDIT A07).
external/glfw/build/src/libglfw3.a: $(GLFW_STATIC_DEPS)
	@bash external/build_external.sh

# Build GLFW shared library from pre-built static library
$(build_dir)/libglfw.dylib: external/glfw/build/src/libglfw3.a $(mgl_lib)
	@echo "Creating GLFW shared library from static library..."
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -shared -fPIC -dynamiclib \
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

# Toolchain floor: the injected aux metallibs need the Metal 4 frontend that
# ships with Xcode 26 / macOS 26 SDKs.  Checked on every `core` / `es` build so
# an unsupported toolchain fails here with an actionable message instead of
# deep inside a parallel build (e.g. "lambda expressions are not supported in
# Metal" from MGL/aux_shaders/gs_xfb_scatter.metal).
verify-toolchain:
	@printf 'xcode:    '; xcodebuild -version 2>/dev/null | tr '\n' ' '; echo
	@echo "sdk:      $(SDK_ROOT) ($(MACOS_SDK_VERSION))"
	@echo "c/cxx:    $(CC) / $(CXX)"
	@echo "metal:    $(MGL_METAL)"
	@test -n "$(MACOS_SDK_MAJOR)" || { echo "ERROR: cannot read the macOS SDK version (xcrun --sdk macosx --show-sdk-version); is Xcode installed?"; exit 1; }
	@case "$(MACOS_SDK_MAJOR)" in *[!0-9]*) echo "ERROR: unparseable macOS SDK version '$(MACOS_SDK_VERSION)'."; exit 1;; esac
	@test -n "$(MGL_METAL)" || { echo "ERROR: metal compiler not found (xcrun --sdk macosx --find metal)."; exit 1; }
	@test "$(MACOS_SDK_MAJOR)" -ge "$(MGL_MIN_MACOS_SDK_MAJOR)" || { \
		echo "ERROR: macOS SDK $(MACOS_SDK_VERSION) is older than the required $(MGL_MIN_MACOS_SDK_MAJOR)."; \
		echo "       MGL aux shaders are Metal 4 and need Xcode 26+."; \
		echo "       Fix:    sudo xcode-select -s /Applications/Xcode_26.app"; \
		echo "       Bypass: make MGL_MIN_MACOS_SDK_MAJOR=0 <target>  (unsupported)"; \
		exit 1; }

core: verify-toolchain $(mgl_lib) $(build_dir)/libglfw.dylib

es: verify-toolchain $(mgl_es_lib)

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

clean:
	rm -rf $(build_dir)
	rm -f libmgl.dylib
	rm -f libmgl_es.dylib
	rm -f libglfw.dylib

install-pkgdeps: download-pkgdeps compile-pkgdeps

download-pkgdeps:
	# llvm@15 is required to link the AIR backend. cmake is required by
	# GLFW and by `make gtest`. glm remains a convenience header for
	# optional host-side tools; bench-system still needs a separately
	# installed Homebrew GLFW.
	brew install llvm@15 cmake glm

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
# framework via a separately installed system GLFW (no MGL dependency).  This
# optional target is outside the normal build and never replaces local GLFW.
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

test-regression-update: build-test-regression
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_regression \
		--golden-dir $(abspath MGL_Golden_Images) --update

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

$(build_dir)/test_arch_correctness: test_legacy_compat/test_arch_correctness.c $(build_dir)/libmgl.dylib
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-DMGL_GL_CORE \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_arch_correctness.c \
		-L$(build_dir) -lmgl \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@

$(build_dir)/test_tess_domain: test_legacy_compat/test_tess_domain.c \
	MGL/src/mgl_tess_factor_normalize.c MGL/src/mgl_tess_domain_gen.c \
	MGL/include/mgl_tess_domain.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_tess_domain.c \
		MGL/src/mgl_tess_factor_normalize.c \
		MGL/src/mgl_tess_domain_gen.c \
		-o $@

test-tess-domain: $(build_dir)/test_tess_domain
	$(build_dir)/test_tess_domain

$(build_dir)/test_xfb_plan: test_legacy_compat/test_xfb_plan.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_xfb_plan.c \
		-o $@

test-xfb-plan: $(build_dir)/test_xfb_plan
	$(build_dir)/test_xfb_plan

$(build_dir)/test_batch_path: test_legacy_compat/test_batch_path.c \
	MGL/src/mgl_batch_path.c MGL/include/mgl_batch_path.h \
	MGL/include/mgl_env_flag.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_batch_path.c \
		MGL/src/mgl_batch_path.c \
		-o $@

test-batch-path: $(build_dir)/test_batch_path
	$(build_dir)/test_batch_path

$(build_dir)/test_batch_hazard: test_legacy_compat/test_batch_hazard.c \
	MGL/src/mgl_batch_hazard.c MGL/include/mgl_batch_hazard.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_batch_hazard.c \
		MGL/src/mgl_batch_hazard.c \
		-o $@

test-batch-hazard: $(build_dir)/test_batch_hazard
	$(build_dir)/test_batch_hazard

$(build_dir)/test_batch_icb: test_legacy_compat/test_batch_icb.c \
	MGL/src/mgl_batch_path.c MGL/include/mgl_batch_path.h \
	MGL/include/mgl_env_flag.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_batch_icb.c \
		MGL/src/mgl_batch_path.c \
		-o $@

test-batch-icb: $(build_dir)/test_batch_icb
	$(build_dir)/test_batch_icb

$(build_dir)/test_batch_restore: test_legacy_compat/test_batch_restore.c \
	MGL/src/mgl_batch_restore.c MGL/include/mgl_batch_restore.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_batch_restore.c \
		MGL/src/mgl_batch_restore.c \
		-o $@

test-batch-restore: $(build_dir)/test_batch_restore
	$(build_dir)/test_batch_restore

$(build_dir)/test_batch_issue: test_legacy_compat/test_batch_issue.c \
	MGL/src/mgl_batch_issue.c MGL/include/mgl_batch_issue.h \
	MGL/src/mgl_batch_rt_mark.c MGL/include/mgl_batch_rt_mark.h \
	MGL/src/mgl_batch_restore.c MGL/src/mgl_batch_path.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_batch_issue.c \
		MGL/src/mgl_batch_issue.c MGL/src/mgl_batch_rt_mark.c \
		MGL/src/mgl_batch_restore.c MGL/src/mgl_batch_path.c \
		-o $@

test-batch-issue: $(build_dir)/test_batch_issue
	$(build_dir)/test_batch_issue



$(build_dir)/test_process_gl_state_plan: test_legacy_compat/test_process_gl_state_plan.c \
	MGL/src/mgl_render_pass_plan.c MGL/include/mgl_render_pass_plan.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_process_gl_state_plan.c \
		MGL/src/mgl_render_pass_plan.c \
		-o $@

test-process-gl-state-plan: $(build_dir)/test_process_gl_state_plan
	$(build_dir)/test_process_gl_state_plan

$(build_dir)/test_render_pass_clear_plan: test_legacy_compat/test_render_pass_clear_plan.c \
	MGL/src/mgl_render_pass_plan.c MGL/include/mgl_render_pass_plan.h \
	MGL/include/mgl_render_pass_clear.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_render_pass_clear_plan.c \
		MGL/src/mgl_render_pass_plan.c \
		-o $@

test-render-pass-clear-plan: $(build_dir)/test_render_pass_clear_plan
	$(build_dir)/test_render_pass_clear_plan

$(build_dir)/test_buffer_plan: test_legacy_compat/test_buffer_plan.c \
	MGL/src/mgl_vertex_attrib_plan.c MGL/include/mgl_vertex_attrib_plan.h \
	MGL/include/mgl_vertex_attrib_binding.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_buffer_plan.c \
		MGL/src/mgl_vertex_attrib_plan.c \
		-o $@

test-buffer-plan: $(build_dir)/test_buffer_plan
	$(build_dir)/test_buffer_plan

$(build_dir)/test_reference_query: test_legacy_compat/test_reference_query.c \
	MGL/src/mgl_frontend_session.c MGL/include/mgl_frontend_session.h \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h \
	MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_cpp.c \
	MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_reference_query.c \
		MGL/src/mgl_frontend_session.c MGL/src/mgl_legacy_compat.c \
		MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_cpp.c \
		MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
		-o $@

test-reference-query: $(build_dir)/test_reference_query
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_reference_query

$(build_dir)/test_per_vertex_signature: test_legacy_compat/test_per_vertex_signature.c \
	MGL/src/mgl_program_reflection.c MGL/include/mgl_program_reflection.h \
	MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_glsl_cpp.c \
	MGL/src/mgl_metal_ref.c MGL/include/mgl_metal_ref.h \
	MGL/src/mgl_uniform_reflection.c MGL/src/mgl_binding_policy.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_per_vertex_signature.c \
		MGL/src/mgl_program_reflection.c MGL/src/mgl_metal_ref.c \
		MGL/src/mgl_uniform_reflection.c MGL/src/mgl_binding_policy.c \
		MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_glsl_cpp.c \
		-framework CoreFoundation -framework Foundation -framework Metal \
		-o $@

test-per-vertex-signature: $(build_dir)/test_per_vertex_signature
	$(build_dir)/test_per_vertex_signature

$(build_dir)/test_render_pass_load_store_plan: test_legacy_compat/test_render_pass_load_store_plan.c \
	MGL/src/mgl_render_pass_plan.c MGL/include/mgl_render_pass_plan.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_render_pass_load_store_plan.c \
		MGL/src/mgl_render_pass_plan.c \
		-o $@

test-render-pass-load-store: $(build_dir)/test_render_pass_load_store_plan
	$(build_dir)/test_render_pass_load_store_plan

$(build_dir)/test_blit_plan: test_legacy_compat/test_blit_plan.c \
	MGL/src/mgl_blit_plan.c MGL/include/mgl_blit_plan.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL -IMGL/src \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_blit_plan.c \
		MGL/src/mgl_blit_plan.c \
		-o $@

test-blit-plan: $(build_dir)/test_blit_plan
	$(build_dir)/test_blit_plan

$(build_dir)/test_binding_stage: test_legacy_compat/test_binding_stage.c \
	MGL/src/mgl_binding_stage.c MGL/include/mgl_binding_stage.h
	@mkdir -p $(dir $@)
	$(CC) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -g -O0 -std=c11 \
		-IMGL/include -IMGL/src \
		test_legacy_compat/test_binding_stage.c \
		MGL/src/mgl_binding_stage.c \
		-o $@

$(build_dir)/test_binding_texture: test_legacy_compat/test_binding_texture.c \
	MGL/src/mgl_binding_texture.c MGL/include/mgl_binding_texture.h \
	MGL/src/mgl_binding_stage.c MGL/include/mgl_binding_stage.h \
	MGL/src/mgl_binding_policy.c MGL/include/mgl_binding_policy.h
	@mkdir -p $(dir $@)
	$(CC) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -g -O0 -std=c11 \
		-IMGL/include -IMGL/src \
		test_legacy_compat/test_binding_texture.c \
		MGL/src/mgl_binding_texture.c \
		MGL/src/mgl_binding_stage.c \
		MGL/src/mgl_binding_policy.c \
		-o $@

test-binding-stage: $(build_dir)/test_binding_stage $(build_dir)/test_binding_texture
	$(build_dir)/test_binding_stage
	$(build_dir)/test_binding_texture

$(build_dir)/test_geometry_gather: test_legacy_compat/test_geometry_gather.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_geometry_gather.c \
		-o $@

test-geometry-gather: $(build_dir)/test_geometry_gather
	$(build_dir)/test_geometry_gather

$(build_dir)/test_validate_arrays_early: test_legacy_compat/test_validate_arrays_early.c \
	MGL/src/mgl_draw_validate.c MGL/include/mgl_draw_validate.h
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-IMGL/include -IMGL/include/GL \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_validate_arrays_early.c \
		MGL/src/mgl_draw_validate.c \
		-o $@

test-validate-arrays-early: $(build_dir)/test_validate_arrays_early
	$(build_dir)/test_validate_arrays_early



$(build_dir)/test_tess_air: test_legacy_compat/test_tess_air.mm $(build_dir)/libmgl.dylib \
	MGL/include/mgl_tess_domain.h MGL/include/mgl_air_tess_abi.h
	$(LLVM_CXX) -x objective-c++ -fobjc-arc $(LLVM_CXXFLAGS) \
		test_legacy_compat/test_tess_air.mm -L$(build_dir) -lmgl -lc++ \
		-framework Foundation -framework Metal -o $@

test-tess-air: $(build_dir)/test_tess_air
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_tess_air

$(build_dir)/test_es_smoke: test_legacy_compat/test_es_smoke.c $(build_dir)/libmgl_es.dylib
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O0 -arch $(HOST_ARCH) \
		$(CFLAGS_GL_ES) \
		-isysroot $(SDK_ROOT) \
		test_legacy_compat/test_es_smoke.c \
		-L$(build_dir) -lmgl_es \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@

test-es-smoke: $(build_dir)/test_es_smoke
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_es_smoke

verify-gl-api:
	bash scripts/fetch_opengl_registry.sh
	python3 scripts/verify_gl_api.py

test-arch-correctness: $(build_dir)/test_arch_correctness
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_arch_correctness

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

# Non-Metal C1b type golden: carrier / mangle / typeFromIR (LLVM only).
$(build_dir)/test_mgl_air_type: test_legacy_compat/test_mgl_air_type.cpp \
	MGL/src/mgl_air_type.cpp MGL/include/mgl_air_type.h \
	MGL/include/mgl_air_codegen.h MGL/src/mgl_ir.c
	@mkdir -p $(build_dir)
	# clang rejects -std=c++* for C inputs, so compile mgl_ir.c separately.
	$(CC) -isysroot $(SDK_ROOT) -IMGL/include -IMGL/src -c MGL/src/mgl_ir.c \
		-o $(build_dir)/test_mgl_air_type_ir.o
	$(LLVM_CXX) -x c++ $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		test_legacy_compat/test_mgl_air_type.cpp \
		MGL/src/mgl_air_type.cpp -x none \
		$(build_dir)/test_mgl_air_type_ir.o \
		-o $@

test-mgl-air-type: $(build_dir)/test_mgl_air_type
	$(build_dir)/test_mgl_air_type

$(build_dir)/test_mgllex: test_legacy_compat/test_mgllex.c MGL/src/mgl_glsl_lexer.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include \
		test_legacy_compat/test_mgllex.c MGL/src/mgl_glsl_lexer.c \
		-o $@

test-mgllex: $(build_dir)/test_mgllex
	$(build_dir)/test_mgllex

$(build_dir)/test_mglparse: test_legacy_compat/test_mglparse.c MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include \
		test_legacy_compat/test_mglparse.c MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		-o $@

test-mglparse: $(build_dir)/test_mglparse
	$(build_dir)/test_mglparse

$(build_dir)/test_mglsema: test_legacy_compat/test_mglsema.c MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c
	$(APPLE_CLANG) -isysroot $(SDK_ROOT) -Wall -Wextra -Werror -gfull -O0 \
		-IMGL/include -IMGL/include/GL \
		test_legacy_compat/test_mglsema.c MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
		-o $@

test-mglsema: $(build_dir)/test_mglsema
	$(build_dir)/test_mglsema

$(build_dir)/test_mglair: test_legacy_compat/test_mglair.mm \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h \
	MGL/src/mgl_frontend_session.c MGL/include/mgl_frontend_session.h \
	MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
	MGL/src/mgl_ir.c
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -gfull -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Cocoa -framework Foundation -framework Metal \
		test_legacy_compat/test_mglair.mm \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
		MGL/src/mgl_legacy_compat.c \
		MGL/src/mgl_frontend_session.c \
		MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
		MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		MGL/src/mgl_ir.c \
		-o $@

test-mglair: $(build_dir)/test_mglair
	$(build_dir)/test_mglair

# MC-style shader repro: anonymous std140 UBO blocks + samplers through the
# AIR backend.  C sources build as C (they are not valid C++).
MCREPRO_CSRC := MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
	MGL/src/mgl_legacy_compat.c MGL/src/mgl_frontend_session.c
MCREPRO_COBJ := $(patsubst MGL/src/%.c,$(build_dir)/mcrepro_%.o,$(MCREPRO_CSRC))

$(build_dir)/mcrepro_%.o: MGL/src/%.c
	$(LLVM_CXX) -x c -std=c11 -g -O0 -isysroot $(SDK_ROOT) -IMGL/include \
		-IMGL/include/GL -IMGL/src -c $< -o $@

$(build_dir)/test_mcrepro: test_legacy_compat/test_mcrepro.mm \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
	$(MCREPRO_COBJ)
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -g -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Foundation \
		test_legacy_compat/test_mcrepro.mm \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
		-x none $(MCREPRO_COBJ) \
		-o $@

test-mcrepro: $(build_dir)/test_mcrepro
	$(build_dir)/test_mcrepro

# The binding/trace diagnostics TUs are compiled by the library's C rule and
# linked into the smoke target with -x none (they used to be Objective-C, which
# the smoke gate compiled itself; ObjC-zeroing moved them to C).
METALCPP_C_SRC := MGL/src/mgl_binding_texture_log.c MGL/src/mgl_trace_log.c
METALCPP_C_OBJ := $(patsubst MGL/src/%.c,$(build_dir)/metalcpp_%.o,$(METALCPP_C_SRC))

$(build_dir)/metalcpp_%.o: MGL/src/%.c
	@mkdir -p $(dir $@)
	$(APPLE_CLANG) -MMD $(CFLAGS_GL_CORE) -c $< -o $@

# Metal-cpp initialization smoke gate. Device bridging and repeated
# initialization/shutdown must remain stable.
$(build_dir)/test_metalcpp_smoke: test_legacy_compat/test_metalcpp_smoke.mm \
	MGL/src/mgl_render.cpp MGL/src/mgl_render.h \
	MGL/src/mgl_readback_policy.c MGL/include/mgl_readback_policy.h \
	MGL/src/mgl_binding_policy.c MGL/include/mgl_binding_policy.h \
	MGL/src/mgl_binding_stage.c MGL/include/mgl_binding_stage.h \
	MGL/src/mgl_binding_texture.c MGL/include/mgl_binding_texture.h \
	MGL/src/mgl_program_resource.c MGL/include/mgl_program_resource.h \
	MGL/include/mgl_trace_log.h \
	MGL/src/mgl_pso_format_class.c MGL/include/mgl_pso_format_class.h \
	MGL/src/mgl_tess_factor_normalize.c MGL/src/mgl_tess_domain_gen.c \
	MGL/include/mgl_tess_domain.h \
	MGL/src/mgl_renderer_backend.cpp MGL/src/mgl_renderer_backend.h \
	MGL/include/mgl_backend_handles.h \
	MGL/src/MGLPlatformRendererShell.m MGL/include/MGLPlatformRendererShell.h \
	MGL/src/mgl_aux_assets.c \
	MGL/src/mgl_buffer_slots.c \
	MGL/src/mgl_sync.c \
	$(METALCPP_C_OBJ)
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -g -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Cocoa -framework Foundation -framework QuartzCore -framework Metal \
		test_legacy_compat/test_metalcpp_smoke.mm \
		MGL/src/mgl_render.cpp \
		MGL/src/mgl_readback_policy.c \
		MGL/src/mgl_binding_policy.c \
		MGL/src/mgl_binding_stage.c \
		MGL/src/mgl_binding_texture.c \
		MGL/src/mgl_program_resource.c \
		MGL/src/mgl_pso_format_class.c \
		MGL/src/mgl_tess_factor_normalize.c \
		MGL/src/mgl_tess_domain_gen.c \
		MGL/src/mgl_renderer_backend.cpp \
		MGL/src/MGLPlatformRendererShell.m \
		MGL/src/mgl_aux_assets.c \
		MGL/src/mgl_buffer_slots.c \
		MGL/src/mgl_sync.c \
		-x none $(METALCPP_C_OBJ) \
		-o $@

test-metalcpp: $(build_dir)/test_metalcpp_smoke
	$(build_dir)/test_metalcpp_smoke

# AIR backend unit tests with GoogleTest (pure compile-time, no GPU).
GTEST_TAG ?= v1.18.0
GTEST_ROOT ?= $(HOME)/googletest
GTEST_CXXFLAGS := -I$(GTEST_ROOT)/googletest/include -I$(GTEST_ROOT)/googlemock/include \
	-IMGL/include/GL
GTEST_LIBS := $(GTEST_ROOT)/build-mgl/lib/libgtest.a \
	$(GTEST_ROOT)/build-mgl/lib/libgtest_main.a
GTEST_STAMP := $(GTEST_ROOT)/build-mgl/.mgl-built

gtest: $(GTEST_STAMP)

$(GTEST_STAMP):
	@if [ ! -f "$(GTEST_ROOT)/CMakeLists.txt" ]; then \
		rm -rf "$(GTEST_ROOT)"; \
		git clone --depth 1 --branch $(GTEST_TAG) \
			https://github.com/google/googletest.git "$(GTEST_ROOT)"; \
	fi
	cmake -S "$(GTEST_ROOT)" -B "$(GTEST_ROOT)/build-mgl" \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build "$(GTEST_ROOT)/build-mgl" --parallel
	@touch $@

$(build_dir)/test_mglair_gtest: test_legacy_compat/test_mglair_gtest.cpp \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
	MGL/src/mgl_legacy_compat.c MGL/include/mgl_legacy_compat.h \
	MGL/src/mgl_frontend_session.c MGL/include/mgl_frontend_session.h \
	MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
	MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
	MGL/src/mgl_ir.c \
	$(GTEST_STAMP)
	$(LLVM_CXX) -x c++ $(LLVM_CXXFLAGS) $(GTEST_CXXFLAGS) $(LLVM_LDFLAGS) \
		test_legacy_compat/test_mglair_gtest.cpp \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_air_type.cpp MGL/src/mgl_air_resource.cpp MGL/src/mgl_air_math.cpp MGL/src/mgl_air_varsym.cpp MGL/src/mgl_air_matrix.cpp MGL/src/mgl_air_stmt.cpp MGL/src/mgl_metallib_writer.cpp \
		MGL/src/mgl_legacy_compat.c MGL/src/mgl_frontend_session.c \
		MGL/src/mgl_air_reflect.c MGL/src/mgl_glsl_sema.c \
		MGL/src/mgl_glsl_cpp.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
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
$(build_dir)/test_mgl_air_type \
$(build_dir)/test_mgllex \
$(build_dir)/test_mglparse \
$(build_dir)/test_mglsema \
$(build_dir)/test_mglair \
$(build_dir)/test_mcrepro \
$(build_dir)/test_metalcpp_smoke \
$(build_dir)/test_mglair_gtest \
$(build_dir)/test_es_smoke: | $(build_dir)

$(build_dir):
	@mkdir -p $@

test-frontends:
	$(MAKE) test-legacy-compat
	$(MAKE) test-mglir
	$(MAKE) test-mgl-air-type
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
	$(MAKE) verify-gl-api
	$(MAKE) test-frontends
	$(MAKE) test-air
	$(MAKE) test-dirty-hash
	$(MAKE) test-arch-correctness
	$(MAKE) test-tess-domain
	$(MAKE) test-xfb-plan
	$(MAKE) test-batch-path
	$(MAKE) test-batch-hazard
	$(MAKE) test-batch-icb
	$(MAKE) test-batch-restore
	$(MAKE) test-batch-issue
	$(MAKE) test-process-gl-state-plan
	$(MAKE) test-render-pass-clear-plan
	$(MAKE) test-buffer-plan
	$(MAKE) test-reference-query
	$(MAKE) test-per-vertex-signature
	$(MAKE) test-render-pass-load-store
	$(MAKE) test-blit-plan
	$(MAKE) test-binding-stage
	$(MAKE) test-geometry-gather
	$(MAKE) test-validate-arrays-early
	$(MAKE) test-tess-air
# The self-contained ObjC / AIR gates below were previously reachable only by
# naming them individually.  A tessellation-factor contract change (3a979a0)
# stayed red for two days because test-mglair -- the only consumer that
# caught it -- was not in any aggregate target.  They need llvm@15, which
# test-tess-air above already requires.
	$(MAKE) test-mglair
	$(MAKE) test-mcrepro
	$(MAKE) test-metalcpp
	$(MAKE) test-legacy-compat
	$(MAKE) test-es-smoke
	$(MAKE) test-regression

.PHONY: default help test dbg core es lib clean install-pkgdeps test-make bench bench-system \
	build-test-regression test-regression test-dirty-hash test-arch-correctness test-tess-domain test-xfb-plan test-batch-path test-batch-hazard test-batch-icb test-batch-restore test-batch-issue test-process-gl-state-plan test-binding-stage test-geometry-gather test-validate-arrays-early test-tess-air test-benchmark \
	test-buffer-plan test-reference-query test-per-vertex-signature test-render-pass-clear-plan \
	test-render-pass-load-store test-blit-plan \
	test-legacy-compat test-mglir test-mgl-air-type test-mgllex test-mglparse test-mglsema \
	test-mglair test-mglair-gtest test-mcrepro test-metalcpp test-frontends \
	test-air test-all gtest test-regression-update verify-gl-api test-es-smoke \
	verify-toolchain

-include $(deps)
