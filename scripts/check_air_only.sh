#!/bin/bash
# P3 硬闸：生产路径不得残留旧 GLSL -> SPIR-V -> MSL 链。
#
# 检查范围：
#   - 符号扫描：MGL/src、MGL/include、Makefile、config.mk.example、
#     test_legacy_compat（产品测试）。
#   - AIR 白名单：纯后端文件中的历史文案命中不阻断（注释引用旧名），
#     但新增源码不得引用旧链符号。
#   - 旧链文件必须已删除：mgl_msl_compiler.* 与 external/glslang、SPIRV-* 目录。
#
# 迁移记录文档（docs/）允许命中，不在此范围。
set -u

failures=0
warn() { printf 'check-air-only FAIL: %s\n' "$1"; failures=$((failures + 1)); }

# AIR 白名单：这些文件是 AIR backend 本体/测试，历史注释可引用旧链名。
air_whitelist=(
    'MGL/src/mgl_air_backend.cpp'
    'MGL/src/mgl_air_reflect.c'
    'MGL/src/mgl_metallib_writer.cpp'
    'MGL/src/mgl_air_loader.cpp'
    'MGL/src/mgl_air_loader.h'
)

symbol_pattern='newLibraryWithSource|mglCompileMSL|compileShader:|mgl_msl_compiler|mtl4Compiler'

symbol_hits=$(grep -rEn "$symbol_pattern" \
    MGL/src MGL/include Makefile config.mk.example test_legacy_compat 2>/dev/null || true)
if [ -n "$symbol_hits" ]; then
    filtered=""
    while IFS= read -r line; do
        source_file=$(printf '%s' "$line" | cut -d: -f1)
        whitelisted=0
        for allowed in "${air_whitelist[@]}"; do
            if [ "$source_file" = "$allowed" ]; then
                whitelisted=1
                break
            fi
        done
        if [ "$whitelisted" -eq 0 ]; then
            filtered="$filtered
$line"
        fi
    done <<EOF
$symbol_hits
EOF
    if [ -n "$filtered" ]; then
        warn "legacy source-compile symbols in production paths:"
        printf '%s\n' "$filtered" | sed 's/^/    /'
    fi
fi

# 旧外部树 build 引用（构建规则仅限这些文件；checker 自身在 scripts/ 不扫描）
oldtree_hits=$(grep -rEn \
    'external/(glslang|SPIRV-Cross|SPIRV-Tools|SPIRV-Headers)' \
    Makefile config.mk.example .gitignore 2>/dev/null || true)
if [ -n "$oldtree_hits" ]; then
    warn "legacy third-party tree references in build rules:"
    printf '%s\n' "$oldtree_hits" | sed 's/^/    /'
fi

# 旧链文件必须已删除
for path in \
    MGL/src/mgl_msl_compiler.m \
    MGL/include/mgl_msl_compiler.h \
    external/glslang \
    external/SPIRV-Cross \
    external/SPIRV-Tools \
    external/SPIRV-Headers \
    external/SPIRV-Cross.r58bak; do
    if [ -e "$path" ]; then
        warn "legacy chain path still exists: $path"
    fi
done

if [ "$failures" -ne 0 ]; then
    printf 'check-air-only: %d violation(s) — production paths must not use the legacy GLSL->SPIR-V->MSL chain\n' \
        "$failures"
    exit 1
fi
printf 'check-air-only OK\n'