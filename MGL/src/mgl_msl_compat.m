/*
 * mgl_msl_compat.m
 * MGL
 *
 * Implementation of the MSL Post-Processing Subsystem.
 *
 * See mgl_msl_compat.h for the architectural rationale.  This module owns
 * the pure spec-compliance helpers for translating GLSL/SPIR-V shader
 * semantics to Metal Shading Language (MSL):
 *   - MSL struct size computation.
 *   - MSL texture type / data-kind inference.
 *   - MSL named-argument lookup and stale-resource skip gating.
 *   - Vertex clip-space variant generation.
 *   - Shader stage naming.
 *
 * The helpers here are pure: they do not touch the renderer ivar, the
 * command buffer, or the render encoder.  They operate only on the
 * Program / Shader / SpirvResource structures passed in as arguments.
 *
 * External dependencies:
 *   - Program / Shader / SpirvResource types (glm_context.h).
 *   - MGLTextureDataKind enum (mgl_texture_compat.h).
 *   - mglRendererResourceLooksSamplerLike (mgl_sampler_compat.h) for
 *     sampler-like resource classification in stale-resource gating.
 *   - Metal framework for MTLTextureType.
 *   - Foundation framework for NSString (vertex clip variant helpers).
 */

#import "mgl_msl_compat.h"
#import "mgl_sampler_compat.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "spirv_cross_c.h"
#include <string.h>
#include <stdlib.h>

/* Diagnostic log gate for stale-resource skip messages.  Mirrors the
 * kMGLDiagnosticStateLogs flag in MGLRenderer.m; kept separate to avoid
 * creating a cross-module extern for a diagnostic-only switch. */
static const BOOL kMGLMSLDiagnosticLogs = NO;

/* === Shader stage naming === */

const char *mglShaderStageName(int stage)
{
    switch (stage) {
        case _VERTEX_SHADER: return "vertex";
        case _TESS_CONTROL_SHADER: return "tess_control";
        case _TESS_EVALUATION_SHADER: return "tess_eval";
        case _GEOMETRY_SHADER: return "geometry";
        case _FRAGMENT_SHADER: return "fragment";
        case _COMPUTE_SHADER: return "compute";
        default: return "unknown";
    }
}

/* === MSL struct size computation === */

NSUInteger mglComputeMSLStructSizeBySuffix(const char *msl, const char *suffix, size_t suffixLen)
{
    if (!msl || !suffix || suffixLen == 0) return 0;

    const char *cursor = msl;
    while ((cursor = strstr(cursor, "struct ")) != NULL)
    {
        cursor += 7;
        while (*cursor == ' ' || *cursor == '\t') cursor++;

        const char *nameStart = cursor;
        while (*cursor && *cursor != ' ' && *cursor != '\t' &&
               *cursor != '\n' && *cursor != '\r' && *cursor != '{')
            cursor++;
        size_t nameLen = (size_t)(cursor - nameStart);

        if (nameLen <= suffixLen || strncmp(nameStart + nameLen - suffixLen, suffix, suffixLen) != 0)
            continue;

        const char *braceStart = strchr(cursor, '{');
        if (!braceStart) continue;

        const char *braceEnd = braceStart + 1;
        int depth = 1;
        while (*braceEnd && depth > 0)
        {
            if (*braceEnd == '{') depth++;
            else if (*braceEnd == '}') depth--;
            braceEnd++;
        }
        if (depth != 0) continue;

        NSUInteger running = 0;
        NSUInteger maxAlign = 1;

        const char *p = braceStart + 1;
        while (p < braceEnd - 1)
        {
            const char *semi = p;
            while (semi < braceEnd - 1 && *semi != ';') semi++;
            if (semi >= braceEnd - 1) break;

            const char *mp = p;
            while (mp < semi && (*mp == ' ' || *mp == '\t' ||
                                  *mp == '\n' || *mp == '\r'))
                mp++;
            size_t mlen = (size_t)(semi - mp);
            if (mlen == 0) { p = semi + 1; continue; }

            char member[512];
            if (mlen >= sizeof(member)) mlen = sizeof(member) - 1;
            memcpy(member, mp, mlen);
            member[mlen] = '\0';

            NSUInteger arrayCount = 1;
            char *arr = strchr(member, '[');
            if (arr)
            {
                *arr = '\0';
                unsigned long cnt = strtoul(arr + 1, NULL, 10);
                if (cnt > 0) arrayCount = (NSUInteger)cnt;
            }

            NSUInteger memberSize = 0, memberAlign = 0;

            if      (strstr(member, "float4"))  { memberSize = 16; memberAlign = 16; }
            else if (strstr(member, "float3"))  { memberSize = 16; memberAlign = 16; }
            else if (strstr(member, "float2"))  { memberSize =  8; memberAlign =  8; }
            else if (strstr(member, "float"))   { memberSize =  4; memberAlign =  4; }
            else if (strstr(member, "double4")) { memberSize = 32; memberAlign = 32; }
            else if (strstr(member, "double3")) { memberSize = 32; memberAlign = 32; }
            else if (strstr(member, "double2")) { memberSize = 16; memberAlign = 16; }
            else if (strstr(member, "double"))  { memberSize =  8; memberAlign =  8; }
            else if (strstr(member, "half4"))   { memberSize =  8; memberAlign =  8; }
            else if (strstr(member, "half3"))   { memberSize =  8; memberAlign =  8; }
            else if (strstr(member, "half2"))   { memberSize =  4; memberAlign =  4; }
            else if (strstr(member, "half"))    { memberSize =  2; memberAlign =  2; }
            else if (strstr(member, "4") && (strstr(member, "int") || strstr(member, "char"))) { memberSize = 16; memberAlign = 16; }
            else if (strstr(member, "3") && (strstr(member, "int") || strstr(member, "char"))) { memberSize = 16; memberAlign = 16; }
            else if (strstr(member, "2") && (strstr(member, "int") || strstr(member, "char"))) { memberSize =  8; memberAlign =  8; }
            else if (strstr(member, "int") || strstr(member, "uint") || strstr(member, "char") || strstr(member, "uchar")) { memberSize = 4; memberAlign = 4; }
            else if (strstr(member, "short4") || strstr(member, "ushort4")) { memberSize = 8; memberAlign = 8; }
            else if (strstr(member, "short3") || strstr(member, "ushort3")) { memberSize = 8; memberAlign = 8; }
            else if (strstr(member, "short2") || strstr(member, "ushort2")) { memberSize = 4; memberAlign = 4; }
            else if (strstr(member, "short")  || strstr(member, "ushort"))  { memberSize = 2; memberAlign = 2; }
            else if (strstr(member, "bool"))   { memberSize =  1; memberAlign =  1; }
            else                                { memberSize = 16; memberAlign = 16; }

            memberSize *= arrayCount;
            if (memberAlign > maxAlign) maxAlign = memberAlign;

            running = (running + memberAlign - 1) & ~(memberAlign - 1);
            running += memberSize;

            p = semi + 1;
        }

        running = (running + maxAlign - 1) & ~(maxAlign - 1);
        return running;
    }

    return 0;
}

NSUInteger mglComputeMSLOutputStructSize(const char *msl)
{
    return mglComputeMSLStructSizeBySuffix(msl, "_out", 4);
}

/* === MSL texture type / data-kind inference === */

MTLTextureType mglExpectedTextureTypeFromMSL(const char *msl, GLuint binding)
{
    if (!msl) {
        return 0;
    }

    char needle[32];
    snprintf(needle, sizeof(needle), "[[texture(%u)]]", (unsigned)binding);

    const char *cursor = msl;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        const char *lineStart = cursor;
        while (lineStart > msl && lineStart[-1] != '\n' && lineStart[-1] != '\r' && lineStart[-1] != ',') {
            lineStart--;
        }

        size_t lineLen = (size_t)(cursor - lineStart);
        if (lineLen > 1024u) {
            lineLen = 1024u;
        }

        char line[1025];
        memcpy(line, lineStart, lineLen);
        line[lineLen] = '\0';

        if (strstr(line, "texture_buffer")) {
            return MTLTextureTypeTextureBuffer;
        }
        if (strstr(line, "texturecube_array") || strstr(line, "depthcube_array")) {
            return MTLTextureTypeCubeArray;
        }
        if (strstr(line, "texturecube") || strstr(line, "depthcube")) {
            return MTLTextureTypeCube;
        }
        if (strstr(line, "texture3d")) {
            return MTLTextureType3D;
        }
        if (strstr(line, "texture2d_ms_array")) {
            return MTLTextureType2DMultisampleArray;
        }
        if (strstr(line, "texture2d_ms")) {
            return MTLTextureType2DMultisample;
        }
        if (strstr(line, "texture2d_array") || strstr(line, "depth2d_array")) {
            return MTLTextureType2DArray;
        }
        if (strstr(line, "texture2d") || strstr(line, "depth2d")) {
            return MTLTextureType2D;
        }
        if (strstr(line, "texture1d_array")) {
            return MTLTextureType1DArray;
        }
        if (strstr(line, "texture1d")) {
            return MTLTextureType1D;
        }

        cursor += strlen(needle);
    }

    return 0;
}

MGLTextureDataKind mglExpectedTextureDataKindFromMSL(const char *msl, GLuint binding)
{
    if (!msl) {
        return MGLTextureDataKindUnknown;
    }

    char needle[32];
    snprintf(needle, sizeof(needle), "[[texture(%u)]]", (unsigned)binding);

    const char *cursor = msl;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        const char *lineStart = cursor;
        while (lineStart > msl && lineStart[-1] != '\n' && lineStart[-1] != '\r' && lineStart[-1] != ',') {
            lineStart--;
        }

        size_t lineLen = (size_t)(cursor - lineStart);
        if (lineLen > 1024u) {
            lineLen = 1024u;
        }

        char line[1025];
        memcpy(line, lineStart, lineLen);
        line[lineLen] = '\0';

        if (strstr(line, "depth1d") ||
            strstr(line, "depth2d") ||
            strstr(line, "depthcube") ||
            strstr(line, "unknown_depth_texture_type")) {
            return MGLTextureDataKindDepth;
        }
        if (strstr(line, "<int") || strstr(line, "<short") || strstr(line, "<char")) {
            return MGLTextureDataKindSint;
        }
        if (strstr(line, "<uint") || strstr(line, "<ushort") || strstr(line, "<uchar")) {
            return MGLTextureDataKindUint;
        }
        if (strstr(line, "<float") || strstr(line, "<half")) {
            return MGLTextureDataKindFloat;
        }

        cursor += strlen(needle);
    }

    return MGLTextureDataKindUnknown;
}

/* === MSL named-argument lookup === */

bool mglMSLArgumentIdentifierChar(char c)
{
    return c == '_' ||
           (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

static uint8_t mglMSLArgumentCacheKind(const char *attributeKind)
{
    if (strcmp(attributeKind, "buffer") == 0) {
        return 1u;
    }
    if (strcmp(attributeKind, "texture") == 0) {
        return 2u;
    }
    if (strcmp(attributeKind, "sampler") == 0) {
        return 3u;
    }
    return 0u;
}

static bool mglMSLArgumentCacheLookup(Program *program,
                                      int stage,
                                      const char *name,
                                      uint8_t attributeKind,
                                      GLuint metalBinding,
                                      bool *result)
{
    if (!program || !result || attributeKind == 0u) {
        return false;
    }

    for (size_t index = 0u; index < MGL_MSL_NAMED_ARGUMENT_CACHE_CAPACITY; index++) {
        MGLMSLNamedArgumentCacheEntry *entry = &program->msl_named_argument_cache[index];
        if (entry->name == name &&
            entry->binding == metalBinding &&
            entry->stage == (uint8_t)stage &&
            entry->attribute_kind == attributeKind) {
            *result = entry->result == GL_TRUE;
            return true;
        }
    }

    return false;
}

static void mglMSLArgumentCacheStore(Program *program,
                                     int stage,
                                     const char *name,
                                     uint8_t attributeKind,
                                     GLuint metalBinding,
                                     bool result)
{
    if (!program || attributeKind == 0u) {
        return;
    }

    size_t index = program->msl_named_argument_cache_next %
                   MGL_MSL_NAMED_ARGUMENT_CACHE_CAPACITY;
    MGLMSLNamedArgumentCacheEntry *entry = &program->msl_named_argument_cache[index];
    entry->name = name;
    entry->binding = metalBinding;
    entry->stage = (uint8_t)stage;
    entry->attribute_kind = attributeKind;
    entry->result = result ? GL_TRUE : GL_FALSE;
    program->msl_named_argument_cache_next =
        (uint8_t)((index + 1u) % MGL_MSL_NAMED_ARGUMENT_CACHE_CAPACITY);
}

bool mglStageMSLHasNamedArgument(Program *program,
                                 int stage,
                                 const char *name,
                                 const char *attributeKind,
                                 GLuint metalBinding)
{
    if (!program || !name || !attributeKind || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return true;
    }

    uint8_t cacheKind = mglMSLArgumentCacheKind(attributeKind);
    bool cachedResult = false;
    if (mglMSLArgumentCacheLookup(program,
                                  stage,
                                  name,
                                  cacheKind,
                                  metalBinding,
                                  &cachedResult)) {
        return cachedResult;
    }

    const char *msl = program->spirv[stage].msl_str;
    if (!msl || !*msl) {
        return true;
    }

    char attribute[32];
    snprintf(attribute, sizeof(attribute), "[[%s(%u)]]", attributeKind, (unsigned)metalBinding);

    const char *cursor = msl;
    size_t nameLen = strlen(name);
    if (nameLen == 0) {
        return true;
    }

    bool result = false;
    while ((cursor = strstr(cursor, name)) != NULL) {
        char before = (cursor == msl) ? '\0' : cursor[-1];
        char after = cursor[nameLen];
        if (mglMSLArgumentIdentifierChar(before) ||
            mglMSLArgumentIdentifierChar(after)) {
            cursor += nameLen;
            continue;
        }

        const char *lineStart = cursor;
        while (lineStart > msl && lineStart[-1] != '\n' && lineStart[-1] != '\r') {
            lineStart--;
        }

        const char *lineEnd = cursor;
        while (*lineEnd && *lineEnd != '\n' && *lineEnd != '\r') {
            lineEnd++;
        }

        size_t lineLen = (size_t)(lineEnd - lineStart);
        if (lineLen > 2048u) {
            lineLen = 2048u;
        }

        char line[2049];
        memcpy(line, lineStart, lineLen);
        line[lineLen] = '\0';

        if (strstr(line, attribute)) {
            result = true;
            break;
        }

        cursor += nameLen;
    }

    mglMSLArgumentCacheStore(program, stage, name, cacheKind, metalBinding, result);
    return result;
}

bool mglStageMSLHasNamedBufferArgument(Program *program,
                                       int stage,
                                       const char *name,
                                       GLuint metalBinding)
{
    return mglStageMSLHasNamedArgument(program, stage, name, "buffer", metalBinding);
}

bool mglStageMSLHasNamedTextureArgument(Program *program,
                                        int stage,
                                        const char *name,
                                        GLuint metalBinding)
{
    return mglStageMSLHasNamedArgument(program, stage, name, "texture", metalBinding);
}

bool mglStageMSLHasNamedSamplerArgument(Program *program,
                                        int stage,
                                        const char *name,
                                        GLuint metalBinding)
{
    return mglStageMSLHasNamedArgument(program, stage, name, "sampler", metalBinding);
}

bool mglStageMSLHasArgumentAtBinding(Program *program,
                                     int stage,
                                     const char *attributeKind,
                                     GLuint metalBinding)
{
    if (!program || !attributeKind || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return false;
    }

    uint8_t cacheKind = mglMSLArgumentCacheKind(attributeKind);
    bool cachedResult = false;
    if (mglMSLArgumentCacheLookup(program,
                                  stage,
                                  NULL,
                                  cacheKind,
                                  metalBinding,
                                  &cachedResult)) {
        return cachedResult;
    }

    const char *msl = program->spirv[stage].msl_str;
    if (!msl || !*msl) {
        return false;
    }

    char attribute[32];
    snprintf(attribute, sizeof(attribute), "[[%s(%u)]]", attributeKind, (unsigned)metalBinding);
    bool result = strstr(msl, attribute) != NULL;
    mglMSLArgumentCacheStore(program, stage, NULL, cacheKind, metalBinding, result);
    return result;
}

/* === Stale resource skip gating === */

bool mglShouldSkipStageBufferResource(Program *program,
                                      int stage,
                                      int resourceType,
                                      const SpirvResource *resource)
{
    if (!program || !resource) {
        return false;
    }

    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
        mglRendererResourceLooksSamplerLike(resource, resourceType)) {
        return true;
    }

    if (!resource->name) {
        return false;
    }

    if (resourceType != SPVC_RESOURCE_TYPE_UNIFORM_BUFFER) {
        return false;
    }

    if (mglStageMSLHasNamedBufferArgument(program, stage, resource->name, resource->binding)) {
        return false;
    }

    if (mglStageMSLHasArgumentAtBinding(program, stage, "buffer", resource->binding)) {
        static uint64_t s_staleBufferResourceSkipLogs = 0;
        uint64_t hit = ++s_staleBufferResourceSkipLogs;
        if (kMGLMSLDiagnosticLogs &&
            (hit <= 64ull || (hit % 512ull) == 0ull)) {
            NSLog(@"MGL RESOURCE SKIP stale buffer program=%u stage=%s name=%s glBinding=%u metalBinding=%u hit=%llu",
                  (unsigned)program->name,
                  mglShaderStageName(stage),
                  resource->name,
                  (unsigned)resource->gl_binding,
                  (unsigned)resource->binding,
                  (unsigned long long)hit);
        }
        return true;
    }

    return false;
}

bool mglShouldSkipStageTextureResource(Program *program,
                                       int stage,
                                       int resourceType,
                                       const SpirvResource *resource)
{
    if (!program || !resource || !resource->name) {
        return false;
    }

    switch (resourceType) {
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            break;
        default:
            return false;
    }

    /* AIR path: there is no MSL text to scan; the reflected resource list
     * is authoritative, so nothing is stale. */
    if (!program->spirv[stage].msl_str) {
        return false;
    }

    if (mglStageMSLHasNamedTextureArgument(program, stage, resource->name, resource->binding)) {
        return false;
    }

    /* MGL renames GLSL uniforms named "sampler" to "mgl_sampler_tex" in MSL
     * (see program.c) to avoid Metal type-name conflicts.  Reflection keeps
     * the original name, so the lookup above fails.  Retry with the renamed
     * identifier before treating the resource as stale. */
    if (resource->name && strcmp(resource->name, "sampler") == 0 &&
        mglStageMSLHasNamedTextureArgument(program, stage, "mgl_sampler_tex", resource->binding)) {
        return false;
    }

    static uint64_t s_staleTextureResourceSkipLogs = 0;
    uint64_t hit = ++s_staleTextureResourceSkipLogs;
    if (hit <= 2ull || (hit % 512ull) == 0ull) {
        NSLog(@"MGL RESOURCE SKIP stale texture program=%u stage=%s type=%d name=%s glBinding=%u metalBinding=%u hit=%llu",
              (unsigned)program->name,
              mglShaderStageName(stage),
              resourceType,
              resource->name,
              (unsigned)resource->gl_binding,
              (unsigned)resource->binding,
              (unsigned long long)hit);
    }
    return true;
}

bool mglShouldSkipStageSamplerResource(Program *program,
                                       int stage,
                                       int resourceType,
                                       const SpirvResource *resource)
{
    if (!program || !resource || !resource->name) {
        return false;
    }

    if (resourceType != SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS) {
        return false;
    }

    /* AIR path: reflected resources are authoritative. */
    if (!program->spirv[stage].msl_str) {
        return false;
    }

    if (mglStageMSLHasNamedSamplerArgument(program, stage, resource->name, resource->binding)) {
        return false;
    }

    static uint64_t s_staleSamplerResourceSkipLogs = 0;
    uint64_t hit = ++s_staleSamplerResourceSkipLogs;
    if (hit <= 64ull || (hit % 512ull) == 0ull) {
        NSLog(@"MGL RESOURCE SKIP stale sampler program=%u stage=%s type=%d name=%s glBinding=%u metalBinding=%u hit=%llu",
              (unsigned)program->name,
              mglShaderStageName(stage),
              resourceType,
              resource->name,
              (unsigned)resource->gl_binding,
              (unsigned)resource->binding,
              (unsigned long long)hit);
    }
    return true;
}

/* === Vertex clip-space variant generation === */

NSString *mglVertexClipVariantMSLSource(Program *program,
                                        Shader *shader,
                                        BOOL keepDepthFixup,
                                        BOOL flipY,
                                        NSString *entrySuffix)
{
    if (!program || !shader || !program->spirv[_VERTEX_SHADER].msl_str ||
        !shader->entry_point || !entrySuffix) {
        return nil;
    }

    NSString *source = [NSString stringWithUTF8String:program->spirv[_VERTEX_SHADER].msl_str];
    if (!source) {
        return nil;
    }

    NSString *entry = [NSString stringWithUTF8String:shader->entry_point];
    if (!entry) {
        return nil;
    }

    NSMutableArray<NSString *> *lines = [[source componentsSeparatedByString:@"\n"] mutableCopy];
    for (NSInteger i = (NSInteger)lines.count - 1; i >= 0; i--) {
        if (!keepDepthFixup && [lines[(NSUInteger)i] containsString:@"Adjust clip-space for Metal"]) {
            [lines removeObjectAtIndex:(NSUInteger)i];
        }
    }

    NSString *patched = [lines componentsJoinedByString:@"\n"];
    if (flipY) {
        patched = [patched stringByReplacingOccurrencesOfString:@"return out;"
                                                     withString:@"out.gl_Position.y = -out.gl_Position.y;       // Adjust clip origin for GL_UPPER_LEFT\n    return out;"];
    }

    NSString *variantEntry = [entry stringByAppendingString:entrySuffix];
    NSRange entryRange = [patched rangeOfString:entry];
    if (entryRange.location == NSNotFound) {
        return nil;
    }

    return [patched stringByReplacingOccurrencesOfString:entry withString:variantEntry];
}

NSString *mglVertexClipVariantEntryName(Shader *shader, NSString *entrySuffix)
{
    if (!shader || !shader->entry_point || !entrySuffix) {
        return nil;
    }

    NSString *entry = [NSString stringWithUTF8String:shader->entry_point];
    return entry ? [entry stringByAppendingString:entrySuffix] : nil;
}

NSString *mglZeroToOneVertexMSLSource(Program *program, Shader *shader)
{
    return mglVertexClipVariantMSLSource(program, shader, NO, NO, @"_zero_to_one");
}

NSString *mglZeroToOneVertexEntryName(Shader *shader)
{
    return mglVertexClipVariantEntryName(shader, @"_zero_to_one");
}

NSString *mglUpperLeftVertexMSLSource(Program *program, Shader *shader, BOOL zeroToOneDepth)
{
    return mglVertexClipVariantMSLSource(program,
                                         shader,
                                         !zeroToOneDepth,
                                         YES,
                                         zeroToOneDepth ? @"_upper_left_zero_to_one" : @"_upper_left");
}

NSString *mglUpperLeftVertexEntryName(Shader *shader, BOOL zeroToOneDepth)
{
    return mglVertexClipVariantEntryName(shader,
                                         zeroToOneDepth ? @"_upper_left_zero_to_one" : @"_upper_left");
}
