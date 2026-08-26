// Bisect driver: compile FS GLSL variants through the MGL AIR backend and
// dump each .air for the offline PSO reproducer.
//   ./fs_bisect <variant-index>
#include "mgl_shader_abi.h"
#include "mgl_types_program.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* kHeader =
    "#version 460 core\n"
    "precision highp float;\n"
    "layout(location = 0) out vec4 result;\n";

static const char* kVariants[] = {
    // 0: exact CTS body (uses gl_PrimitiveID)
    "void main() {\n"
    "    result.x =       float(gl_PrimitiveID % 64)          / 64.0f;\n"
    "    result.y = floor(float(gl_PrimitiveID)      / 64.0f) / 64.0f;\n"
    "    result.z =       float(gl_PrimitiveID)               / 4096.0f;\n"
    "    result.w = ((gl_PrimitiveID % 2) == 0) ? 1.0f : 0.0f;\n"
    "}\n",
    // 1: declare-only passthrough
    "flat in int test_gl_PrimitiveIDIn;\n"
    "void main() { result = vec4(1,1,1,1); }\n",
    // 2: srem only
    "void main() { result.x = float(gl_PrimitiveID % 64) / 64.0f;\n"
    "              result.yzw = vec3(0,0,1); }\n",
    // 3: plain read only
    "void main() { result = vec4(float(gl_PrimitiveID)) / 4096.0f; }\n",
    // 4: ternary only
    "void main() { result = vec4(vec3(((gl_PrimitiveID % 2) == 0) ? 1.0f : 0.0f), 1.0f); }\n",
    // 5: floor only
    "void main() { result = vec4(floor(float(gl_PrimitiveID) / 64.0f) / 64.0f);\n"
    "              result.a = 1.0f; }\n",
};

int main(int argc, char** argv)
{
    int variant = argc > 1 ? atoi(argv[1]) : 0;
    if (variant < 0 || variant >= (int)(sizeof(kVariants)/sizeof(kVariants[0]))) return 1;

    char src[2048];
    snprintf(src, sizeof(src), "%s%s", kHeader, kVariants[variant]);

    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES];
    memset(lists, 0, sizeof(lists));
    MGLAIRStageInfo stage_info; memset(&stage_info, 0, sizeof(stage_info));
    unsigned char *bytes = NULL; size_t size = 0;
    char err[512] = {0};
    int rc = mglAirCompileGLSLWithReflectInfoEx(
        src, MGL_STAGE_FRAGMENT, NULL, &bytes, &size, lists,
        &stage_info, MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER, err, sizeof(err));
    printf("variant %d rc=%d size=%zu err=%s\n", variant, rc, size, err);
    if (rc == 0 && bytes) {
        char path[64];
        snprintf(path, sizeof(path), "/tmp/fsvar%d.air", variant);
        FILE* f = fopen(path, "wb");
        fwrite(bytes, 1, size, f);
        fclose(f);
        printf("wrote %s\n", path);
        mglShaderFree(bytes);
    }
    return rc;
}
