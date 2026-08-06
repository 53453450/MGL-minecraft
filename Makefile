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

# lets only install from external, devs complained about brew and we want the latest build from spirv
spirv_cross_include_path ?= ./external/SPIRV-Cross
spirv_cross_config_include_path ?= ./external/SPIRV-Cross
spirv_cross_lib_path ?= ./external/SPIRV-Cross/build

spirv_tools_include_path ?= ./external/SPIRV-Tools/include
spirv_tools_path ?= ./external/SPIRV-Tools/build

glslang_include_path ?= ./external/glslang/glslang/Include
glslang_lib_path ?= ./external/glslang/build


#glslang_path ?= glslang
#glslang_include_path ?= $(glslang_path)/build/include/glslang $(glslang_path)/glslang/Include
#glslang_lib_path ?= $(glslang_path)/build/glslang $(glslang_path)/build/OGLCompilersDLL $(glslang_path)/build/glslang/OSDependent/Unix $(glslang_path)/build/StandAlone $(glslang_path)/build/SPIRV

# build dirs
build_dir ?= build
build_core_dir := $(build_dir)/core
build_es_dir := $(build_dir)/es

CFLAGS += -Wall #-Wunused-parameter #-Wextra
CFLAGS += -gfull
CFLAGS += -O2
#CFLAGS += -00
# Disable AddressSanitizer for production - causes crashes when loaded via dlopen()
#CFLAGS += -fsanitize=address
#LIBS += -fsanitize=address
CFLAGS += -arch $(HOST_ARCH)
LIBS += -arch $(HOST_ARCH)

LIBS += -F$(SDK_ROOT)/System/Library/Frameworks
LIBS += -framework Metal -framework OpenGL -framework Foundation

CFLAGS += -I$(spirv_cross_include_path)
CFLAGS += -I$(spirv_cross_config_include_path)
CFLAGS += -I$(spirv_tools_include_path)
CFLAGS += -I$(glslang_include_path)

# lets only install from external, devs complained about brew
# CFLAGS += $(shell pkg-config --cflags SPIRV-Tools)
# CFLAGS += $(shell pkg-config --cflags glm)

CFLAGS += -IMGL/include
CFLAGS += -IMGL/include/GL # "glcorearb.h"
CFLAGS += -IMGL/src        # "mgl_safety.h" lives in MGL/src/, used by MGLRenderer_Private.h
CFLAGS += -IMGL/SPIRV/SPIRV-Cross
CFLAGS += -DENABLE_OPT=0 -DSPIRV_CROSS_C_API_MSL=1 -DSPIRV_CROSS_C_API_GLSL=1 -DSPIRV_CROSS_C_API_CPP=1 -DSPIRV_CROSS_C_API_REFLECT=1

# GLFW configuration for shared library build
CFLAGS += -I./external/glfw/include -I./external/glfw/src
CXXFLAGS += -I./external/glfw/include -I./external/glfw/src

# macOS specific compile definitions for GLFW
CFLAGS += -D_COCOA -D_GLFW_COCOA
CXXFLAGS += -D_COCOA -D_GLFW_COCOA

# GL_CORE SPECIFIC FLAGS
CFLAGS_GL_CORE := $(CFLAGS) -DMGL_GL_CORE
CXXFLAGS_GL_CORE := $(CXXFLAGS) -DMGL_GL_CORE

# GL_ES SPECIFIC FLAGS
CFLAGS_GL_ES := $(CFLAGS) -DMGL_GL_ES
CXXFLAGS_GL_ES := $(CXXFLAGS) -DMGL_GL_ES

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

SPIRV_CROSS_ARCHIVES := \
	$(spirv_cross_lib_path)/libspirv-cross-c.a \
	$(spirv_cross_lib_path)/libspirv-cross-msl.a \
	$(spirv_cross_lib_path)/libspirv-cross-glsl.a \
	$(spirv_cross_lib_path)/libspirv-cross-hlsl.a \
	$(spirv_cross_lib_path)/libspirv-cross-cpp.a \
	$(spirv_cross_lib_path)/libspirv-cross-reflect.a \
	$(spirv_cross_lib_path)/libspirv-cross-util.a \
	$(spirv_cross_lib_path)/libspirv-cross-core.a

GLSLANG_ARCHIVES := \
	$(glslang_lib_path)/glslang/libglslang.a \
	$(glslang_lib_path)/glslang/libMachineIndependent.a \
	$(glslang_lib_path)/glslang/libGenericCodeGen.a \
	$(glslang_lib_path)/glslang/OSDependent/Unix/libOSDependent.a \
	$(glslang_lib_path)/glslang/libglslang-default-resource-limits.a \
	$(glslang_lib_path)/SPIRV/libSPIRV.a

SPIRV_TOOLS_ARCHIVES := \
	$(spirv_tools_path)/source/lint/libSPIRV-Tools-lint.a \
	$(spirv_tools_path)/source/reduce/libSPIRV-Tools-reduce.a \
	$(spirv_tools_path)/source/diff/libSPIRV-Tools-diff.a \
	$(spirv_tools_path)/source/link/libSPIRV-Tools-link.a \
	$(spirv_tools_path)/source/opt/libSPIRV-Tools-opt.a \
	$(spirv_tools_path)/source/libSPIRV-Tools.a

THIRD_PARTY_ARCHIVES := $(SPIRV_CROSS_ARCHIVES) $(GLSLANG_ARCHIVES) $(SPIRV_TOOLS_ARCHIVES)
LIBS += $(THIRD_PARTY_ARCHIVES)
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
		'  make test-regression  Build the headless regression suite.' \
		'  make test-dirty-hash  Run the minimal dirty-hash batch regression.' \
		'  make test-msl-bindings Run focused MSL binding reconciliation tests.' \
		'  make clean            Remove local build outputs.'

# mgl
#mgl_srcs_c := $(wildcard MGL/src/*.c)
mgl_srcs_c := $(filter-out %/gl_core.c  %/gl_es.c, $(wildcard MGL/src/*.c))

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
           ./external/SPIRV-Cross \
           ./external/SPIRV-Headers \
           ./external/SPIRV-Tools \
           ./external/glslang \
           ./external/ezxml

# Simplified index_of function - find position of directory in EXT_DIRS
define index_of
$(strip $(1))
endef

# Function to get the corresponding repository URL for a directory
# Simplified mapping for common directories
define get_repo_url
$(if $(filter $(1),./external/OpenGL-Registry),https://github.com/KhronosGroup/OpenGL-Registry.git, \
$(if $(filter $(1),./external/SPIRV-Cross),https://github.com/53453450/SPIRV-Cross.git, \
$(if $(filter $(1),./external/SPIRV-Headers),https://github.com/KhronosGroup/SPIRV-Headers.git, \
$(if $(filter $(1),./external/SPIRV-Tools),https://github.com/KhronosGroup/SPIRV-Tools.git, \
$(if $(filter $(1),./external/glslang),https://github.com/KhronosGroup/glslang.git, \
https://github.com/lxfontes/ezxml.git)))))
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
		if [ "$(1)" = "./external/SPIRV-Cross" ]; then \
			git clone $$REPO -b main $(1) --depth 1; \
			echo "SPIRV-Cross pinned to 53453450 fork @ main (MSL text-patch compatibility)"; \
		else \
			git clone $$REPO $(1) --depth 1; \
		fi; \
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

mgl_toolchain_obj := $(build_dir)/MGL/src/mgl_toolchain.o
mgl_toolchain_lib := $(build_dir)/libmgl_toolchain.a

mgl_core_link_objs := $(mgl_core_objs) $(mgl_core_arc_objs) $(mgl_core_obj)
mgl_es_link_objs := $(mgl_es_objs) $(mgl_es_arc_objs) $(mgl_es_obj)

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

$(mgl_lib): $(mgl_core_link_objs) $(THIRD_PARTY_ARCHIVES) $(core_link_stamp)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -dynamiclib -o $@ $(mgl_core_link_objs) $(LIBS)
	# loading dynamic library requires this
	ln -fs $(mgl_lib) .

$(mgl_es_lib): $(mgl_es_link_objs) $(THIRD_PARTY_ARCHIVES) $(es_link_stamp)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -dynamiclib -o $@ $(mgl_es_link_objs) $(LIBS)
	# loading dynamic library requires this
	ln -fs $(mgl_es_lib) .


$(mgl_toolchain_lib): $(mgl_toolchain_obj)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

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

toolchain: $(mgl_toolchain_lib)

test_exe := $(build_dir)/test_mgl

test: $(test_exe)
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(test_exe)

dbg: $(test_exe)
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) lldb -o run $(test_exe)

$(build_dir)/test_mgl: test_mgl/main.cpp $(mgl_lib) $(build_dir)/libglfw.dylib
	$(CXX) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glfw/include \
		-I./external/glslang/glslang/Include \
		-I./external/SPIRV-Cross \
		-I./external/SPIRV-Tools/include \
		-IMGL/include -IMGL/include/GL -IMGL/SPIRV/SPIRV-Cross \
		-DMGL_GL_CORE -DENABLE_OPT=0 \
		-DSPIRV_CROSS_C_API_MSL=1 -DSPIRV_CROSS_C_API_GLSL=1 \
		-DSPIRV_CROSS_C_API_CPP=1 -DSPIRV_CROSS_C_API_REFLECT=1 \
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

	brew install glm glslang spirv-tools glfw

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
test-regression: $(build_dir)/libmgl.dylib $(build_dir)/libglfw.dylib
	$(APPLE_CLANG) -Wall -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glfw/include \
		-I./external/glslang/glslang/Include \
		-I./external/SPIRV-Cross \
		-I./external/SPIRV-Tools/include \
		-IMGL/include -IMGL/include/GL -IMGL/SPIRV/SPIRV-Cross \
		-DMGL_GL_CORE -DENABLE_OPT=0 \
		-DSPIRV_CROSS_C_API_MSL=1 -DSPIRV_CROSS_C_API_GLSL=1 \
		-DSPIRV_CROSS_C_API_CPP=1 -DSPIRV_CROSS_C_API_REFLECT=1 \
		-isysroot $(SDK_ROOT) \
		test_regression/main.c \
		-L$(build_dir) -lmgl -lglfw \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $(build_dir)/test_regression
	@echo "✅ Regression suite built: $(build_dir)/test_regression"

$(build_dir)/test_dirty_hash: test_dirty_hash/main.c $(build_dir)/libmgl.dylib
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glslang/glslang/Include \
		-I./external/SPIRV-Cross \
		-I./external/SPIRV-Tools/include \
		-IMGL/include -IMGL/include/GL -IMGL/SPIRV/SPIRV-Cross \
		-DMGL_GL_CORE -DENABLE_OPT=0 \
		-DSPIRV_CROSS_C_API_MSL=1 -DSPIRV_CROSS_C_API_GLSL=1 \
		-DSPIRV_CROSS_C_API_CPP=1 -DSPIRV_CROSS_C_API_REFLECT=1 \
		-isysroot $(SDK_ROOT) \
		test_dirty_hash/main.c \
		-L$(build_dir) -lmgl \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@

test-dirty-hash: $(build_dir)/test_dirty_hash
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_dirty_hash

$(build_dir)/test_msl_bindings: test_msl_bindings/main.c $(build_dir)/libmgl.dylib
	$(APPLE_CLANG) -Wall -Wextra -Werror -gfull -O2 -arch $(HOST_ARCH) \
		$(CFLAGS) \
		-I./external/glslang/glslang/Include \
		-I./external/SPIRV-Cross \
		-I./external/SPIRV-Tools/include \
		-IMGL/include -IMGL/include/GL -IMGL/SPIRV/SPIRV-Cross \
		-DMGL_GL_CORE -DENABLE_OPT=0 \
		-DSPIRV_CROSS_C_API_MSL=1 -DSPIRV_CROSS_C_API_GLSL=1 \
		-DSPIRV_CROSS_C_API_CPP=1 -DSPIRV_CROSS_C_API_REFLECT=1 \
		-isysroot $(SDK_ROOT) \
		test_msl_bindings/main.c \
		-L$(build_dir) -lmgl \
		-framework Cocoa -framework CoreFoundation -framework CoreGraphics \
		-framework IOKit -framework Foundation -framework QuartzCore \
		-framework Metal -framework OpenGL \
		-o $@

test-msl-bindings: $(build_dir)/test_msl_bindings
	DYLD_LIBRARY_PATH=$(abspath $(build_dir)) $(build_dir)/test_msl_bindings

test-benchmark: bench
	scripts/run_benchmark_smoke.sh --no-build

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
		-IMGL/include \
		test_legacy_compat/test_mglsema.c MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
		-o $@

test-mglsema: $(build_dir)/test_mglsema
	$(build_dir)/test_mglsema

# M1 AIR backend: GLSL -> metallib -> PSO gate (C++20 + LLVM, Metal runtime).
LLVM_ROOT ?= /opt/homebrew/opt/llvm@15
LLVM_CXX ?= $(APPLE_CLANG)
LLVM_CXXFLAGS := -std=c++20 -isysroot $(SDK_ROOT) -I$(LLVM_ROOT)/include -IMGL/include
LLVM_LDFLAGS := -L$(LLVM_ROOT)/lib -lLLVM-15 -lc++

$(build_dir)/test_mglair: test_legacy_compat/test_mglair.mm \
	MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
	MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
	MGL/src/mgl_ir.c
	$(LLVM_CXX) -x objective-c++ -fobjc-arc -gfull -O0 $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) \
		-framework Cocoa -framework Foundation -framework Metal \
		test_legacy_compat/test_mglair.mm \
		MGL/src/mgl_air_backend.cpp MGL/src/mgl_metallib_writer.cpp \
		MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
		MGL/src/mgl_ir.c \
		-o $@

test-mglair: $(build_dir)/test_mglair
	$(build_dir)/test_mglair

.PHONY: default help test dbg core es lib clean install-pkgdeps test-make bench bench-system test-regression test-dirty-hash test-msl-bindings test-benchmark test-mglir test-mgllex test-mglparse test-mglsema test-mglair

-include $(deps)
