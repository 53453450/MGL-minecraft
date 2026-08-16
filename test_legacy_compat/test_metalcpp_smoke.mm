// test_metalcpp_smoke.mm — Phase 0 验收：mglRenderCppInit 桥接现有 id<MTLDevice>
// 拿到非空 MTL::Device*（void* 形式）无崩溃；shutdown 幂等；可重建。
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <simd/simd.h>
#include <math.h>
#include <stdio.h>
#include <mach/mach.h>
#include <atomic>

#include "mgl_render_cpp_objc.h"
#include "mgl_air_loader.h"
#include "mgl_aux_assets.h"
#include "mgl_buffer_slots.h"
#include "mgl_types_texture.h"
#include "mgl_types_buffer.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "mgl_sync.h"

/* This target exercises the renderer facade without constructing AIR loader
 * objects. Product builds link the real implementation. */
extern "C" void mglAirLoaderShutdown(void) {}
extern "C" int mglAirLoadLibrary(const void *, const unsigned char *, size_t,
                                  void **, char *, size_t) { return -1; }
extern "C" int mglAirCreateRenderPipelineWithArchive(
    const void *, void *, void *, const MGLRenderCppPipelineDescriptorState *,
    void *, void **, char *, size_t) { return -1; }
static uint64_t s_metalReleaseCount = 0;
static uint64_t s_metalCreateCount = 0;
extern "C" void mglMetalCountRelease(int) { ++s_metalReleaseCount; }
extern "C" void mglMetalCountCreate(int) { ++s_metalCreateCount; }
extern "C" void mglRecordBufferCowSnapshot(uint64_t) {}

/* Load a precompiled aux metallib with the plain Metal load-from-data API.
 * Smoke fixtures must not compile MSL source. */
static id<MTLLibrary> smokeLoadAssetLibrary(id<MTLDevice> device,
                                            const char *assetName)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset) {
        fprintf(stderr, "FAIL: asset library %s missing\n", assetName);
        return nil;
    }
    dispatch_data_t data = dispatch_data_create(
        asset->data, asset->size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library) {
        fprintf(stderr, "FAIL: asset library %s: %s\n", assetName,
                error.localizedDescription.UTF8String ?: "unknown");
    }
    return library;
}
static int s_legacyBufferBindCount = 0;
extern "C" void mtlBindBuffer(GLMContext, Buffer *) {
    ++s_legacyBufferBindCount;
}
static int s_legacyBufferSubDataCount = 0;
extern "C" void mtlBufferSubData(GLMContext, Buffer *, size_t, size_t,
                                  const void *) {
    ++s_legacyBufferSubDataCount;
}
static int s_legacyBufferMapCount = 0;
extern "C" void *mtlMapUnmapBuffer(GLMContext, Buffer *, size_t, size_t,
                                    unsigned int, bool) {
    ++s_legacyBufferMapCount;
    return reinterpret_cast<void *>(0x1234u);
}
static int s_legacyBufferFlushRangeCount = 0;
extern "C" void mtlFlushBufferRange(GLMContext, Buffer *, intptr_t,
                                     intptr_t) {
    ++s_legacyBufferFlushRangeCount;
}
static int s_legacyProgramBindCount = 0;
extern "C" void mtlBindProgram(GLMContext, Program *) {
    ++s_legacyProgramBindCount;
}

static std::atomic<int> s_commandBufferCompletionCount{0};
static std::atomic<int> s_commandBufferContextDestroyCount{0};
static std::atomic<uint32_t> s_commandBufferCompletionStatus{0};

static MGLRenderCppRenderPassState renderPassStateWithColorTarget(
    id<MTLTexture> texture,
    MTLLoadAction loadAction,
    MTLStoreAction storeAction) {
    MGLRenderCppRenderPassState state;
    mglRenderCppInitDefaultRenderPassState(&state);
    state.color[0].attachment.texture = (__bridge void *)texture;
    state.color[0].attachment.load_action = (uint32_t)loadAction;
    state.color[0].attachment.store_action = (uint32_t)storeAction;
    return state;
}

static void commandBufferCompletion(
    void *, const MGLRenderCppCommandBufferState *state) {
    s_commandBufferCompletionStatus.store(
        state ? state->status : UINT32_MAX, std::memory_order_relaxed);
    s_commandBufferCompletionCount.fetch_add(1, std::memory_order_relaxed);
}

static void destroyCommandBufferCompletionContext(void *context) {
    delete static_cast<int *>(context);
    s_commandBufferContextDestroyCount.fetch_add(
        1, std::memory_order_relaxed);
}

static int verifyAttachmentSubresource(void) {
    FBOAttachment attachment = {};
    attachment.level = 3u;
    attachment.textarget = GL_TEXTURE_CUBE_MAP;
    for (GLuint layer = 0u; layer < _CUBE_MAP_MAX_FACE; ++layer) {
        attachment.layer = layer;
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(&attachment);
        if (subresource.level != 3u || subresource.slice != layer ||
            subresource.depthPlane != 0u) {
            fprintf(stderr,
                    "FAIL: cube attachment layer=%u level=%lu slice=%lu depth=%lu\n",
                    layer, (unsigned long)subresource.level,
                    (unsigned long)subresource.slice,
                    (unsigned long)subresource.depthPlane);
            return 1;
        }
    }

    const GLuint invalidCubeLayers[] = {6u, UINT32_MAX};
    for (GLuint layer : invalidCubeLayers) {
        attachment.layer = layer;
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(&attachment);
        if (subresource.level != 3u || subresource.slice != 0u ||
            subresource.depthPlane != 0u) {
            fprintf(stderr,
                    "FAIL: invalid cube attachment layer=%u slice=%lu\n",
                    layer, (unsigned long)subresource.slice);
            return 1;
        }
    }

    attachment.textarget = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
    attachment.layer = UINT32_MAX;
    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(&attachment);
    if (subresource.level != 3u || subresource.slice != 5u ||
        subresource.depthPlane != 0u) {
        fprintf(stderr, "FAIL: cube negative-Z attachment slice=%lu\n",
                (unsigned long)subresource.slice);
        return 1;
    }

    attachment.textarget = GL_TEXTURE_CUBE_MAP_ARRAY;
    attachment.layer = 11u;
    subresource = mglMetalAttachmentSubresourceForAttachment(&attachment);
    if (subresource.level != 3u || subresource.slice != 11u ||
        subresource.depthPlane != 0u) {
        fprintf(stderr, "FAIL: cube-array attachment slice=%lu\n",
                (unsigned long)subresource.slice);
        return 1;
    }

    attachment.textarget = GL_TEXTURE_3D;
    attachment.layer = 4u;
    subresource = mglMetalAttachmentSubresourceForAttachment(&attachment);
    if (subresource.level != 3u || subresource.slice != 0u ||
        subresource.depthPlane != 4u) {
        fprintf(stderr, "FAIL: 3D attachment depth=%lu\n",
                (unsigned long)subresource.depthPlane);
        return 1;
    }

    printf("ATTACHMENT_SUBRESOURCE_OK\n");
    return 0;
}

static int verifyBufferBinding(void) {
    uint32_t source[4] = {11u, 22u, 33u, 44u};
    Buffer buffer = {};
    buffer.name = 101u;
    buffer.size = sizeof(source);
    buffer.data.buffer_size = sizeof(source);
    buffer.data.buffer_data = (vm_address_t)(uintptr_t)source;
    buffer.storage_flags = GL_MAP_READ_BIT;
    char message[128] = {0};
    if (mglRenderCppBindBufferStorage(&buffer, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_BOUND ||
        !buffer.data.mtl_data || buffer.data.mtl_owns_buffer_data) {
        fprintf(stderr, "FAIL: C++ buffer binder: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLBuffer> metalBuffer =
        (__bridge id<MTLBuffer>)buffer.data.mtl_data;
    if (metalBuffer.length != sizeof(source) || !metalBuffer.contents ||
        memcmp(metalBuffer.contents, source, sizeof(source)) != 0) {
        fprintf(stderr, "FAIL: C++ buffer binder contents\n");
        return 1;
    }

    uint32_t gpuValues[4] = {51u, 52u, 53u, 54u};
    memcpy(metalBuffer.contents, gpuValues, sizeof(gpuValues));
    void *mapped = NULL;
    if (mglRenderCppMapBufferStorage(
            &buffer, 0, sizeof(gpuValues), GL_READ_ONLY, true,
            &mapped, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        mapped != source || memcmp(source, gpuValues, sizeof(gpuValues)) != 0) {
        fprintf(stderr, "FAIL: C++ buffer map/read synchronization\n");
        return 1;
    }
    uint32_t readbackValues[4] = {61u, 62u, 63u, 64u};
    memcpy(metalBuffer.contents, readbackValues, sizeof(readbackValues));
    mglRenderCppReadBackBuffer(NULL, &buffer, 0, sizeof(readbackValues));
    if (memcmp(source, readbackValues, sizeof(readbackValues)) != 0) {
        fprintf(stderr, "FAIL: C++ buffer readback synchronization\n");
        return 1;
    }
    uint32_t cowValue = 71u;
    void *initialBacking = buffer.data.mtl_data;
    mglRenderCppBufferSubData(NULL, &buffer, 0, sizeof(cowValue), &cowValue);
    id<MTLBuffer> firstCowBuffer =
        (__bridge id<MTLBuffer>)buffer.data.mtl_data;
    if (s_legacyBufferSubDataCount != 0 ||
        buffer.data.mtl_data == initialBacking || !buffer.mtl_cpp_cow_pool ||
        source[0] != cowValue || !firstCowBuffer.contents ||
        memcmp(firstCowBuffer.contents, source, sizeof(source)) != 0) {
        fprintf(stderr, "FAIL: Metal-cpp split-shadow COW snapshot\n");
        return 1;
    }

    void *firstCowBacking = buffer.data.mtl_data;
    uint64_t generation1 = mglRenderCppAdvanceBufferGeneration();
    mglRenderCppNoteBufferEncoded(&buffer);
    cowValue = 72u;
    mglRenderCppBufferSubData(NULL, &buffer, 0, sizeof(cowValue), &cowValue);
    void *secondCowBacking = buffer.data.mtl_data;
    if (secondCowBacking == firstCowBacking || source[0] != cowValue) {
        fprintf(stderr, "FAIL: in-flight COW slot was reused\n");
        return 1;
    }

    uint64_t generation2 = mglRenderCppAdvanceBufferGeneration();
    mglRenderCppNoteBufferEncoded(&buffer);
    mglRenderCppRecordBufferGenerationCompleted(generation1);
    cowValue = 73u;
    mglRenderCppBufferSubData(NULL, &buffer, 0, sizeof(cowValue), &cowValue);
    if (buffer.data.mtl_data != firstCowBacking ||
        mglRenderCppCompletedBufferGeneration() < generation1 ||
        generation2 <= generation1) {
        fprintf(stderr, "FAIL: completed COW slot was not reused\n");
        return 1;
    }
    mglRenderCppReleaseBufferMetalData(NULL, &buffer);
    mglRenderCppReleaseBufferCowPool(&buffer);
    if (buffer.data.mtl_data != NULL) {
        fprintf(stderr, "FAIL: C++ buffer release did not clear backing\n");
        return 1;
    }
    if (buffer.mtl_cpp_cow_pool != NULL) {
        fprintf(stderr, "FAIL: C++ buffer COW pool release did not clear state\n");
        return 1;
    }

    Buffer direct = {};
    direct.name = 103u;
    direct.size = sizeof(source);
    direct.data.buffer_size = sizeof(source);
    uint32_t directValues[4] = {81u, 82u, 83u, 84u};
    if (mglRenderCppBufferSubDataStorage(
            &direct, 0, sizeof(directValues), directValues,
            message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        !direct.data.mtl_data) {
        fprintf(stderr, "FAIL: direct Metal buffer subdata: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLBuffer> directMetal =
        (__bridge id<MTLBuffer>)direct.data.mtl_data;
    if (!directMetal.contents ||
        memcmp(directMetal.contents, directValues, sizeof(directValues)) != 0) {
        fprintf(stderr, "FAIL: direct Metal buffer subdata contents\n");
        return 1;
    }
    mglRenderCppReleaseBufferMetalData(NULL, &direct);

    uint32_t flushSource[4] = {91u, 92u, 93u, 94u};
    Buffer flushBuffer = {};
    flushBuffer.name = 104u;
    flushBuffer.size = sizeof(flushSource);
    flushBuffer.data.buffer_size = sizeof(flushSource);
    flushBuffer.data.buffer_data =
        (vm_address_t)(uintptr_t)flushSource;
    if (mglRenderCppBindBufferStorage(
            &flushBuffer, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_BOUND) {
        fprintf(stderr, "FAIL: range-flush buffer bind: %s\n", message);
        return 1;
    }
    void *flushInitialBacking = flushBuffer.data.mtl_data;
    flushSource[1] = 95u;
    if (mglRenderCppFlushBufferRangeStorage(
            &flushBuffer, sizeof(uint32_t), sizeof(uint32_t),
            message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        flushBuffer.data.mtl_data == flushInitialBacking ||
        !flushBuffer.mtl_cpp_cow_pool) {
        fprintf(stderr, "FAIL: C++ mapped range COW flush: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLBuffer> flushedMetal =
        (__bridge id<MTLBuffer>)flushBuffer.data.mtl_data;
    if (!flushedMetal.contents ||
        memcmp(flushedMetal.contents, flushSource, sizeof(flushSource)) != 0) {
        fprintf(stderr, "FAIL: C++ mapped range COW contents\n");
        return 1;
    }
    mglRenderCppReleaseBufferMetalData(NULL, &flushBuffer);
    mglRenderCppReleaseBufferCowPool(&flushBuffer);

    Buffer plainUniform = {};
    uint32_t uniformValue = 101u;
    plainUniform.name = 105u;
    plainUniform.size = sizeof(uniformValue);
    plainUniform.data.buffer_size = sizeof(uniformValue);
    plainUniform.data.buffer_data =
        (vm_address_t)(uintptr_t)&uniformValue;
    plainUniform.data.dirty_bits = DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR;
    plainUniform.plain_uniform_slot = GL_TRUE;
    if (mglRenderCppUpdateDirtyBuffer(
            &plainUniform, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        plainUniform.data.dirty_bits != 0 ||
        plainUniform.data.mtl_data != NULL) {
        fprintf(stderr, "FAIL: C++ plain-uniform dirty update\n");
        return 1;
    }

    uint32_t smallSource[4] = {111u, 112u, 113u, 114u};
    Buffer smallDirty = {};
    smallDirty.name = 106u;
    smallDirty.size = sizeof(smallSource);
    smallDirty.data.buffer_size = sizeof(smallSource);
    smallDirty.data.buffer_data =
        (vm_address_t)(uintptr_t)smallSource;
    smallDirty.data.dirty_bits = DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR;
    smallDirty.cpu_shadow_pending = GL_TRUE;
    if (mglRenderCppUpdateDirtyBuffer(
            &smallDirty, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        smallDirty.data.dirty_bits != 0 || smallDirty.cpu_shadow_pending ||
        !smallDirty.data.mtl_data) {
        fprintf(stderr, "FAIL: C++ small dirty update: %s\n", message);
        return 1;
    }
    id<MTLBuffer> smallMetal =
        (__bridge id<MTLBuffer>)smallDirty.data.mtl_data;
    if (!smallMetal.contents ||
        memcmp(smallMetal.contents, smallSource, sizeof(smallSource)) != 0) {
        fprintf(stderr, "FAIL: C++ small dirty contents\n");
        return 1;
    }
    smallDirty.access_flags = GL_MAP_COHERENT_BIT;
    smallDirty.data.dirty_bits = DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR;
    smallSource[0] = 115u;
    if (mglRenderCppUpdateDirtyBuffer(
            &smallDirty, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        smallDirty.data.dirty_bits != DIRTY_BUFFER_DATA) {
        fprintf(stderr, "FAIL: C++ coherent small dirty state\n");
        return 1;
    }
    mglRenderCppReleaseBufferMetalData(NULL, &smallDirty);
    mglRenderCppReleaseBufferCowPool(&smallDirty);

    uint32_t largeSource[1024] = {};
    largeSource[0] = 121u;
    Buffer largeDirty = {};
    largeDirty.name = 107u;
    largeDirty.size = sizeof(largeSource);
    largeDirty.data.buffer_size = sizeof(largeSource);
    largeDirty.data.buffer_data =
        (vm_address_t)(uintptr_t)largeSource;
    largeDirty.data.dirty_bits = DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR;
    largeDirty.cpu_shadow_pending = GL_TRUE;
    if (mglRenderCppUpdateDirtyBuffer(
            &largeDirty, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        largeDirty.data.dirty_bits != 0 || largeDirty.cpu_shadow_pending ||
        !largeDirty.data.mtl_data) {
        fprintf(stderr, "FAIL: C++ large dirty update: %s\n", message);
        return 1;
    }
    id<MTLBuffer> largeMetal =
        (__bridge id<MTLBuffer>)largeDirty.data.mtl_data;
    if (!largeMetal.contents ||
        memcmp(largeMetal.contents, largeSource, sizeof(largeSource)) != 0) {
        fprintf(stderr, "FAIL: C++ large dirty contents\n");
        return 1;
    }
    largeDirty.data.dirty_bits = DIRTY_BUFFER_ADDR;
    if (mglRenderCppUpdateDirtyBuffer(
            &largeDirty, message, sizeof(message)) !=
            MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED ||
        largeDirty.data.dirty_bits != 0) {
        fprintf(stderr, "FAIL: C++ address-only dirty update\n");
        return 1;
    }
    mglRenderCppReleaseBufferMetalData(NULL, &largeDirty);
    mglRenderCppReleaseBufferCowPool(&largeDirty);

    Buffer noCopy = {};
    noCopy.name = 102u;
    noCopy.size = sizeof(source);
    noCopy.data.buffer_size = sizeof(source);
    noCopy.data.buffer_data = (vm_address_t)(uintptr_t)source;
    noCopy.storage_flags = GL_CLIENT_STORAGE_BIT;
    if (mglRenderCppBindBufferStorage(&noCopy, message, sizeof(message)) !=
        MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
        fprintf(stderr, "FAIL: no-copy buffer was claimed by C++ binder\n");
        return 1;
    }
    mglRenderCppBindBuffer(NULL, &noCopy);
    if (s_legacyBufferBindCount != 1 || noCopy.data.mtl_data) {
        fprintf(stderr, "FAIL: no-copy buffer callback fallback\n");
        return 1;
    }
    void *fallbackMap = mglRenderCppMapUnmapBuffer(
        NULL, &noCopy, 0, sizeof(source), GL_READ_ONLY, true);
    if (fallbackMap != reinterpret_cast<void *>(0x1234u) ||
        s_legacyBufferMapCount != 1) {
        fprintf(stderr, "FAIL: no-copy buffer map fallback\n");
        return 1;
    }
    mglRenderCppFlushBufferRange(
        NULL, &noCopy, 0, (intptr_t)sizeof(source));
    if (s_legacyBufferFlushRangeCount != 1) {
        fprintf(stderr, "FAIL: no-copy buffer range flush fallback\n");
        return 1;
    }
    return 0;
}

static int verifyPackedStructBufferRing(void) {
    const uint8_t firstBytes[] = {0x12, 0x34, 0x56, 0x78, 0x9a};
    char message[256] = {0};
    Buffer *slots[128] = {};
    const uint64_t createCountBefore = s_metalCreateCount;
    const uint64_t releaseCountBefore = s_metalReleaseCount;
    slots[0] = mglRenderCppAcquirePackedStructBuffer(
        firstBytes, sizeof(firstBytes), message, sizeof(message));
    if (!slots[0] || !slots[0]->data.mtl_data ||
        slots[0]->size != 256 || slots[0]->data.buffer_size != 256 ||
        slots[0]->target != GL_UNIFORM_BUFFER ||
        slots[0]->usage != GL_STATIC_DRAW ||
        !slots[0]->transient_batch_buffer) {
        fprintf(stderr, "FAIL: packed struct ring initialization: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }

    id<MTLBuffer> firstMetal =
        (__bridge id<MTLBuffer>)slots[0]->data.mtl_data;
    const uint8_t *firstContents =
        static_cast<const uint8_t *>(firstMetal.contents);
    if (firstMetal.length != 256 || !firstContents ||
        memcmp(firstContents, firstBytes, sizeof(firstBytes)) != 0) {
        fprintf(stderr, "FAIL: packed struct contents or padding size\n");
        return 1;
    }
    for (NSUInteger i = sizeof(firstBytes); i < firstMetal.length; ++i) {
        if (firstContents[i] != 0) {
            fprintf(stderr, "FAIL: packed struct padding was not zeroed\n");
            return 1;
        }
    }

    void *firstBackingAddress = slots[0]->data.mtl_data;
    firstMetal = nil;
    for (size_t i = 1; i < 128; ++i) {
        uint32_t value = static_cast<uint32_t>(i);
        slots[i] = mglRenderCppAcquirePackedStructBuffer(
            &value, sizeof(value), message, sizeof(message));
        if (!slots[i] || slots[i] == slots[0] ||
            slots[i]->name != (0xF0000000u | static_cast<GLuint>(i))) {
            fprintf(stderr, "FAIL: packed struct ring slot %zu: %s\n", i,
                    message[0] ? message : "unknown");
            return 1;
        }
    }

    const uint32_t wrappedValue = 0xdecafbadU;
    Buffer *wrapped = mglRenderCppAcquirePackedStructBuffer(
        &wrappedValue, sizeof(wrappedValue), message, sizeof(message));
    if (wrapped != slots[0] ||
        wrapped->data.mtl_data == firstBackingAddress ||
        s_metalCreateCount != createCountBefore + 129 ||
        s_metalReleaseCount != releaseCountBefore + 1) {
        fprintf(stderr, "FAIL: packed struct ring did not replace slot 0\n");
        return 1;
    }
    id<MTLBuffer> wrappedMetal =
        (__bridge id<MTLBuffer>)wrapped->data.mtl_data;
    if (!wrappedMetal.contents ||
        memcmp(wrappedMetal.contents, &wrappedValue, sizeof(wrappedValue)) != 0) {
        fprintf(stderr, "FAIL: wrapped packed struct contents\n");
        return 1;
    }

    printf("PACKED_STRUCT_RING_OK\n");
    return 0;
}

static int verifyVertexConversions(void) {
    char message[256] = {0};
    uint64_t stride = 0;
    void *rawBuffer = NULL;

    double doubleSource[4] = {1.5, -2.25, 3.0, 4.5};
    Buffer doubleBuffer = {};
    doubleBuffer.name = 201u;
    doubleBuffer.size = sizeof(doubleSource);
    doubleBuffer.data.buffer_size = sizeof(doubleSource);
    doubleBuffer.data.buffer_data =
        (vm_address_t)(uintptr_t)doubleSource;
    MGLRenderCppVertexConversion conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT;
    conversion.component_count = 2;
    conversion.source_type = GL_DOUBLE;
    conversion.stride = 2 * sizeof(double);
    if (mglRenderCppConvertVertexBuffer(
            &doubleBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 16) {
        fprintf(stderr, "FAIL: double vertex conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> doubleMetal =
        (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const float *doubleValues = (const float *)doubleMetal.contents;
    if (!doubleValues || fabsf(doubleValues[0] - 1.5f) > 0.0001f ||
        fabsf(doubleValues[1] + 2.25f) > 0.0001f ||
        fabsf(doubleValues[4] - 3.0f) > 0.0001f ||
        fabsf(doubleValues[5] - 4.5f) > 0.0001f) {
        fprintf(stderr, "FAIL: double vertex conversion values\n");
        return 1;
    }
    rawBuffer = NULL;
    uint64_t cachedStride = 0;
    if (mglRenderCppConvertVertexBuffer(
            &doubleBuffer, &conversion, &cachedStride, &rawBuffer,
            message, sizeof(message)) != 0 ||
        rawBuffer != (__bridge void *)doubleMetal || cachedStride != stride) {
        fprintf(stderr, "FAIL: converted vertex cache hit\n");
        return 1;
    }
    id<MTLBuffer> cachedDoubleMetal =
        (__bridge_transfer id<MTLBuffer>)rawBuffer;
    if (cachedDoubleMetal != doubleMetal) {
        fprintf(stderr, "FAIL: converted vertex cache identity\n");
        return 1;
    }

    int32_t intSource[2] = {INT32_MIN, INT32_MAX};
    Buffer intBuffer = {};
    intBuffer.name = 202u;
    intBuffer.size = sizeof(intSource);
    intBuffer.data.buffer_size = sizeof(intSource);
    intBuffer.data.buffer_data = (vm_address_t)(uintptr_t)intSource;
    conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_INT_TO_FLOAT;
    conversion.component_count = 2;
    conversion.source_type = GL_INT;
    conversion.normalized = 1;
    rawBuffer = NULL;
    if (mglRenderCppConvertVertexBuffer(
            &intBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 8) {
        fprintf(stderr, "FAIL: int vertex conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> intMetal = (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const float *intValues = (const float *)intMetal.contents;
    if (!intValues || intValues[0] != -1.0f || intValues[1] != 1.0f) {
        fprintf(stderr, "FAIL: normalized int vertex conversion values\n");
        return 1;
    }

    int32_t fixedSource[2] = {65536, -32768};
    Buffer fixedBuffer = {};
    fixedBuffer.name = 203u;
    fixedBuffer.size = sizeof(fixedSource);
    fixedBuffer.data.buffer_size = sizeof(fixedSource);
    fixedBuffer.data.buffer_data = (vm_address_t)(uintptr_t)fixedSource;
    conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT;
    conversion.component_count = 2;
    conversion.source_type = GL_FIXED;
    rawBuffer = NULL;
    if (mglRenderCppConvertVertexBuffer(
            &fixedBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 8) {
        fprintf(stderr, "FAIL: fixed vertex conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> fixedMetal = (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const float *fixedValues = (const float *)fixedMetal.contents;
    if (!fixedValues || fixedValues[0] != 1.0f ||
        fixedValues[1] != -0.5f) {
        fprintf(stderr, "FAIL: fixed vertex conversion values\n");
        return 1;
    }

    uint32_t packed1010102 =
        (1023u << 22) | (512u << 12) | (0u << 2) | 3u;
    Buffer packedBuffer = {};
    packedBuffer.name = 204u;
    packedBuffer.size = sizeof(packed1010102);
    packedBuffer.data.buffer_size = sizeof(packed1010102);
    packedBuffer.data.buffer_data =
        (vm_address_t)(uintptr_t)&packed1010102;
    conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_PACKED_1010102_TO_FLOAT;
    conversion.component_count = 4;
    conversion.source_type = GL_UNSIGNED_INT_10_10_10_2;
    rawBuffer = NULL;
    if (mglRenderCppConvertVertexBuffer(
            &packedBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 16) {
        fprintf(stderr, "FAIL: packed 1010102 conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> packedMetal = (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const float *packedValues = (const float *)packedMetal.contents;
    if (!packedValues || packedValues[0] != 1.0f ||
        fabsf(packedValues[1] - (512.0f / 1023.0f)) > 0.0001f ||
        packedValues[2] != 0.0f || packedValues[3] != 1.0f) {
        fprintf(stderr, "FAIL: packed 1010102 conversion values\n");
        return 1;
    }

    const uint32_t float11One = 15u << 6;
    const uint32_t float10One = 15u << 5;
    uint32_t packed10f11f11f =
        float11One | (float11One << 11) | (float10One << 22);
    Buffer packedFloatBuffer = {};
    packedFloatBuffer.name = 205u;
    packedFloatBuffer.size = sizeof(packed10f11f11f);
    packedFloatBuffer.data.buffer_size = sizeof(packed10f11f11f);
    packedFloatBuffer.data.buffer_data =
        (vm_address_t)(uintptr_t)&packed10f11f11f;
    conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_PACKED_10F11F11F_TO_FLOAT;
    conversion.component_count = 3;
    conversion.source_type = GL_UNSIGNED_INT_10F_11F_11F_REV;
    rawBuffer = NULL;
    if (mglRenderCppConvertVertexBuffer(
            &packedFloatBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 12) {
        fprintf(stderr, "FAIL: packed 10f11f11f conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> packedFloatMetal =
        (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const float *packedFloatValues =
        (const float *)packedFloatMetal.contents;
    if (!packedFloatValues || packedFloatValues[0] != 1.0f ||
        packedFloatValues[1] != 1.0f || packedFloatValues[2] != 1.0f) {
        fprintf(stderr, "FAIL: packed 10f11f11f conversion values\n");
        return 1;
    }

    uint8_t byteSource[4] = {255u, 2u, 3u, 4u};
    Buffer byteBuffer = {};
    byteBuffer.name = 206u;
    byteBuffer.size = sizeof(byteSource);
    byteBuffer.data.buffer_size = sizeof(byteSource);
    byteBuffer.data.buffer_data = (vm_address_t)(uintptr_t)byteSource;
    conversion = {};
    conversion.kind = MGL_RENDER_CPP_VERTEX_INTEGER_TO_32;
    conversion.component_count = 2;
    conversion.source_type = GL_UNSIGNED_BYTE;
    conversion.destination_signed = 1;
    conversion.stride = 2;
    rawBuffer = NULL;
    if (mglRenderCppConvertVertexBuffer(
            &byteBuffer, &conversion, &stride, &rawBuffer,
            message, sizeof(message)) != 0 || !rawBuffer || stride != 8) {
        fprintf(stderr, "FAIL: integer widening conversion: %s\n", message);
        return 1;
    }
    id<MTLBuffer> byteMetal = (__bridge_transfer id<MTLBuffer>)rawBuffer;
    const int32_t *byteValues = (const int32_t *)byteMetal.contents;
    if (!byteValues || byteValues[0] != 255 || byteValues[1] != 2 ||
        byteValues[2] != 3 || byteValues[3] != 4) {
        fprintf(stderr, "FAIL: integer widening conversion values\n");
        return 1;
    }

    printf("VERTEX_CONVERSION_OK\n");
    return 0;
}

static int verifySamplerConversion(void) {
    TextureParameter params = {};
    params.min_filter = GL_LINEAR_MIPMAP_LINEAR;
    params.mag_filter = GL_NEAREST;
    params.wrap_s = GL_REPEAT;
    params.wrap_t = GL_CLAMP_TO_EDGE;
    params.wrap_r = GL_CLAMP_TO_BORDER;
    params.border_color[0] = 1.0f;
    params.border_color[1] = 1.0f;
    params.border_color[2] = 1.0f;
    params.border_color[3] = 1.0f;
    params.max_anisotropy = 4.5f;
    params.compare_mode = GL_COMPARE_REF_TO_TEXTURE;
    params.compare_func = GL_GEQUAL;
    params.min_lod = -4.0f;
    params.max_lod = 1000.0f;

    char message[256] = {0};
    void *samplerPtr = NULL;
    if (mglRenderCppCreateSamplerForGL(
            &params, GL_TEXTURE_2D, &samplerPtr,
            message, sizeof(message)) != 0 || !samplerPtr) {
        fprintf(stderr, "FAIL: GL sampler conversion: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLSamplerState> sampler =
        (__bridge_transfer id<MTLSamplerState>)samplerPtr;
    if (!sampler) {
        fprintf(stderr, "FAIL: GL sampler conversion returned nil\n");
        return 1;
    }

    TextureParameter rectangle = params;
    rectangle.wrap_s = GL_REPEAT;
    rectangle.wrap_t = GL_MIRRORED_REPEAT;
    rectangle.wrap_r = GL_CLAMP_TO_BORDER;
    samplerPtr = NULL;
    if (mglRenderCppCreateSamplerForGL(
            &rectangle, GL_TEXTURE_RECTANGLE, &samplerPtr,
            message, sizeof(message)) != 0 || !samplerPtr) {
        fprintf(stderr, "FAIL: rectangle sampler conversion: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLSamplerState> rectangleSampler =
        (__bridge_transfer id<MTLSamplerState>)samplerPtr;
    if (!rectangleSampler) return 1;

    TextureParameter invalid = params;
    invalid.min_filter = 0xdead;
    samplerPtr = NULL;
    if (mglRenderCppCreateSamplerForGL(
            &invalid, GL_TEXTURE_2D, &samplerPtr,
            message, sizeof(message)) == 0 || samplerPtr) {
        fprintf(stderr, "FAIL: invalid GL sampler filter was accepted\n");
        if (samplerPtr) {
            id<MTLSamplerState> invalidSampler =
                (__bridge_transfer id<MTLSamplerState>)samplerPtr;
            invalidSampler = nil;
        }
        return 1;
    }

    printf("SAMPLER_CONVERSION_OK\n");
    return 0;
}

static int verifyAIRProgramClassification(void) {
    Program program = {};
    Shader shader = {};
    char message[128] = {0};
    int failedStage = -1;

    program.shader_slots[_VERTEX_SHADER] = &shader;
    if (mglRenderCppBindAIRProgram(&program, &failedStage, message,
                                   sizeof(message)) !=
        MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE) {
        fprintf(stderr, "FAIL: legacy program was claimed by AIR binder\n");
        return 1;
    }
    program.dirty_bits = DIRTY_PROGRAM;
    mglRenderCppBindProgram(NULL, &program);
    if (s_legacyProgramBindCount != 1 ||
        (program.dirty_bits & DIRTY_PROGRAM) != 0u) {
        fprintf(stderr, "FAIL: legacy program callback fallback\n");
        return 1;
    }

    program.shader_slots[_VERTEX_SHADER] = NULL;
    program.shader_slots[_GEOMETRY_SHADER] = &shader;
    program.modules[_GEOMETRY_SHADER].metallib_bytes =
        reinterpret_cast<unsigned char *>(&program);
    program.modules[_GEOMETRY_SHADER].metallib_size = 1;
    if (mglRenderCppBindAIRProgram(&program, &failedStage, message,
                                   sizeof(message)) !=
            MGL_RENDER_CPP_AIR_PROGRAM_ERROR ||
        failedStage != _GEOMETRY_SHADER) {
        fprintf(stderr, "FAIL: unsupported GS route was not rejected\n");
        return 1;
    }
    return 0;
}

static int verifyResourceCreation(void) {
    MTLTextureDescriptor *textureDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:8
                                                          height:4
                                                       mipmapped:NO];
    textureDesc.usage = MTLTextureUsageShaderRead;
    void *texturePtr = NULL;
    MGLRenderCppTextureDescriptorState textureState =
        mglRenderCppTextureDescriptorStateFromObjC(textureDesc);
    if (mglRenderCppCreateTextureFromState(
            &textureState, "MGL Smoke Texture", &texturePtr) != 0 ||
        !texturePtr) {
        fprintf(stderr, "FAIL: texture creation facade\n");
        return 1;
    }
    id<MTLTexture> texture =
        (__bridge_transfer id<MTLTexture>)texturePtr;
    if (texture.width != 8 || texture.height != 4 ||
        texture.pixelFormat != MTLPixelFormatRGBA8Unorm ||
        ![texture.label isEqualToString:@"MGL Smoke Texture"]) {
        fprintf(stderr, "FAIL: texture creation properties\n");
        return 1;
    }

    void *simpleViewPtr = NULL;
    if (mglRenderCppCreateTextureView(
            (__bridge void *)texture, (uint32_t)MTLPixelFormatRGBA8Unorm,
            &simpleViewPtr) != 0 || !simpleViewPtr) {
        fprintf(stderr, "FAIL: simple texture view facade\n");
        return 1;
    }
    id<MTLTexture> simpleView =
        (__bridge_transfer id<MTLTexture>)simpleViewPtr;
    if (simpleView.width != texture.width ||
        simpleView.height != texture.height) {
        fprintf(stderr, "FAIL: simple texture view properties\n");
        return 1;
    }

    MTLTextureDescriptor *mipDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:8
                                                          height:8
                                                       mipmapped:YES];
    mipDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    void *mipTexturePtr = NULL;
    MGLRenderCppTextureDescriptorState mipState =
        mglRenderCppTextureDescriptorStateFromObjC(mipDesc);
    if (mglRenderCppCreateTextureFromState(
            &mipState, "MGL Smoke Mip Texture", &mipTexturePtr) != 0 ||
        !mipTexturePtr) {
        fprintf(stderr, "FAIL: mip texture creation facade\n");
        return 1;
    }
    id<MTLTexture> mipTexture =
        (__bridge_transfer id<MTLTexture>)mipTexturePtr;
    void *rangeViewPtr = NULL;
    if (mglRenderCppCreateTextureViewRange(
            (__bridge void *)mipTexture,
            (uint32_t)MTLPixelFormatRGBA8Unorm,
            (uint32_t)MTLTextureType2D,
            1, 1, 0, 1, 1,
            (uint32_t)MTLTextureSwizzleBlue,
            (uint32_t)MTLTextureSwizzleGreen,
            (uint32_t)MTLTextureSwizzleRed,
            (uint32_t)MTLTextureSwizzleAlpha,
            &rangeViewPtr) != 0 || !rangeViewPtr) {
        fprintf(stderr, "FAIL: ranged texture view facade\n");
        return 1;
    }
    id<MTLTexture> rangeView =
        (__bridge_transfer id<MTLTexture>)rangeViewPtr;
    MTLTextureSwizzleChannels rangeSwizzle = rangeView.swizzle;
    if (rangeView.width != 4 || rangeView.height != 4 ||
        rangeView.mipmapLevelCount != 1 ||
        rangeSwizzle.red != MTLTextureSwizzleBlue ||
        rangeSwizzle.blue != MTLTextureSwizzleRed) {
        fprintf(stderr, "FAIL: ranged texture view properties\n");
        return 1;
    }

    void *textureBufferStoragePtr = NULL;
    if (mglRenderCppCreateBuffer(
            4096, MTLResourceStorageModeShared,
            "MGL Smoke Texture Buffer", &textureBufferStoragePtr) != 0 ||
        !textureBufferStoragePtr) {
        fprintf(stderr, "FAIL: texture buffer storage facade\n");
        return 1;
    }
    id<MTLBuffer> textureBufferStorage =
        (__bridge_transfer id<MTLBuffer>)textureBufferStoragePtr;
    MTLTextureDescriptor *textureBufferDesc = [MTLTextureDescriptor new];
    textureBufferDesc.textureType = MTLTextureTypeTextureBuffer;
    textureBufferDesc.pixelFormat = MTLPixelFormatRGBA8Uint;
    textureBufferDesc.width = 64;
    textureBufferDesc.usage = MTLTextureUsageShaderRead;
    textureBufferDesc.storageMode = MTLStorageModeShared;
    void *bufferTexturePtr = NULL;
    MGLRenderCppTextureDescriptorState textureBufferState =
        mglRenderCppTextureDescriptorStateFromObjC(textureBufferDesc);
    if (mglRenderCppCreateBufferTextureFromState(
            (__bridge void *)textureBufferStorage, &textureBufferState, 0, 256,
            &bufferTexturePtr) != 0 || !bufferTexturePtr) {
        fprintf(stderr, "FAIL: buffer-backed texture facade\n");
        return 1;
    }
    id<MTLTexture> bufferTexture =
        (__bridge_transfer id<MTLTexture>)bufferTexturePtr;
    if (bufferTexture.textureType != MTLTextureTypeTextureBuffer ||
        bufferTexture.width != 64 ||
        bufferTexture.pixelFormat != MTLPixelFormatRGBA8Uint) {
        fprintf(stderr, "FAIL: buffer-backed texture properties\n");
        return 1;
    }

    MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
    samplerDesc.label = @"MGL Smoke Sampler";
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    void *samplerPtr = NULL;
    if (mglRenderCppCreateSampler((__bridge void *)samplerDesc,
                                  &samplerPtr) != 0 || !samplerPtr) {
        fprintf(stderr, "FAIL: sampler creation facade\n");
        return 1;
    }
    id<MTLSamplerState> sampler =
        (__bridge_transfer id<MTLSamplerState>)samplerPtr;
    if (![sampler.label isEqualToString:@"MGL Smoke Sampler"]) {
        fprintf(stderr, "FAIL: sampler creation properties\n");
        return 1;
    }

    MTLDepthStencilDescriptor *depthDesc = [MTLDepthStencilDescriptor new];
    depthDesc.label = @"MGL Smoke Depth State";
    depthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    depthDesc.depthWriteEnabled = YES;
    void *depthStatePtr = NULL;
    if (mglRenderCppCreateDepthStencilState((__bridge void *)depthDesc,
                                             &depthStatePtr) != 0 ||
        !depthStatePtr) {
        fprintf(stderr, "FAIL: depth-stencil creation facade\n");
        return 1;
    }
    id<MTLDepthStencilState> depthState =
        (__bridge_transfer id<MTLDepthStencilState>)depthStatePtr;
    if (![depthState.label isEqualToString:@"MGL Smoke Depth State"]) {
        fprintf(stderr, "FAIL: depth-stencil creation properties\n");
        return 1;
    }

    void *eventPtr = NULL;
    if (mglRenderCppCreateEvent(&eventPtr) != 0 || !eventPtr) {
        fprintf(stderr, "FAIL: event creation facade\n");
        return 1;
    }
    id<MTLEvent> event = (__bridge_transfer id<MTLEvent>)eventPtr;
    event.label = @"MGL Smoke Event";
    if (![event.label isEqualToString:@"MGL Smoke Event"]) {
        fprintf(stderr, "FAIL: event creation properties\n");
        return 1;
    }

    printf("RESOURCE_CREATION_OK\n");
    return 0;
}

static int verifyTextureTransferFacade(void) {
    const NSUInteger width = 4;
    const NSUInteger height = 3;
    const NSUInteger bytesPerRow = width * 4;
    const NSUInteger bytesPerImage = bytesPerRow * height;
    uint8_t source[bytesPerImage];
    uint8_t readback[bytesPerImage];
    for (NSUInteger i = 0; i < bytesPerImage; ++i) {
        source[i] = (uint8_t)(i * 7u + 3u);
    }
    memset(readback, 0, sizeof(readback));

    MTLTextureDescriptor *textureDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    textureDesc.storageMode = MTLStorageModeShared;
    textureDesc.usage = MTLTextureUsageShaderRead;
    void *texturePtr = NULL;
    MGLRenderCppTextureDescriptorState textureState =
        mglRenderCppTextureDescriptorStateFromObjC(textureDesc);
    if (mglRenderCppCreateTextureFromState(
            &textureState, "MGL Smoke Texture Transfer", &texturePtr) != 0 ||
        !texturePtr) {
        fprintf(stderr, "FAIL: texture transfer texture creation\n");
        return 1;
    }
    id<MTLTexture> texture =
        (__bridge_transfer id<MTLTexture>)texturePtr;
    if (mglRenderCppTextureReplaceRegion(
            (__bridge void *)texture, 0, 0, 0, width, height, 1,
            0, 0, source, bytesPerRow, 0, 0) != 0 ||
        mglRenderCppTextureGetBytes(
            (__bridge void *)texture, readback, bytesPerRow, 0,
            0, 0, 0, width, height, 1, 0, 0, 0) != 0 ||
        memcmp(source, readback, sizeof(source)) != 0) {
        fprintf(stderr, "FAIL: 2D texture transfer facade\n");
        return 1;
    }

    MTLTextureDescriptor *arrayDesc = [MTLTextureDescriptor new];
    arrayDesc.textureType = MTLTextureType2DArray;
    arrayDesc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    arrayDesc.width = width;
    arrayDesc.height = height;
    arrayDesc.depth = 1;
    arrayDesc.arrayLength = 2;
    arrayDesc.mipmapLevelCount = 1;
    arrayDesc.sampleCount = 1;
    arrayDesc.storageMode = MTLStorageModeShared;
    arrayDesc.usage = MTLTextureUsageShaderRead;
    void *arrayTexturePtr = NULL;
    MGLRenderCppTextureDescriptorState arrayState =
        mglRenderCppTextureDescriptorStateFromObjC(arrayDesc);
    if (mglRenderCppCreateTextureFromState(
            &arrayState, "MGL Smoke Array Texture Transfer",
            &arrayTexturePtr) != 0 ||
        !arrayTexturePtr) {
        fprintf(stderr, "FAIL: array texture transfer texture creation\n");
        return 1;
    }
    id<MTLTexture> arrayTexture =
        (__bridge_transfer id<MTLTexture>)arrayTexturePtr;
    memset(readback, 0, sizeof(readback));
    if (mglRenderCppTextureReplaceRegion(
            (__bridge void *)arrayTexture, 0, 0, 0, width, height, 1,
            0, 1, source, bytesPerRow, bytesPerImage, 1) != 0 ||
        mglRenderCppTextureGetBytes(
            (__bridge void *)arrayTexture, readback, bytesPerRow,
            bytesPerImage, 0, 0, 0, width, height, 1, 0, 1, 1) != 0 ||
        memcmp(source, readback, sizeof(source)) != 0) {
        fprintf(stderr, "FAIL: array-slice texture transfer facade\n");
        return 1;
    }

    printf("TEXTURE_TRANSFER_OK\n");
    return 0;
}

static int verifyNoCopyBufferFacade(void) {
    const size_t length = 16384u;
    vm_address_t address = 0;
    kern_return_t allocation = vm_allocate(
        (vm_map_t)mach_task_self(), &address, (vm_size_t)length,
        VM_FLAGS_ANYWHERE);
    if (allocation != KERN_SUCCESS || address == 0) {
        fprintf(stderr, "FAIL: no-copy VM allocation err=%d\n", allocation);
        return 1;
    }
    uint8_t *bytes = reinterpret_cast<uint8_t *>(address);
    for (size_t i = 0; i < length; i++) bytes[i] = (uint8_t)(i ^ 0x5a);

    void *bufferPtr = NULL;
    if (mglRenderCppCreateBufferWithBytesNoCopy(
            bytes, length, MTLResourceStorageModeShared, "MGL Smoke NoCopy",
            1, &bufferPtr) != 0 || !bufferPtr) {
        vm_deallocate((vm_map_t)mach_task_self(), address, length);
        fprintf(stderr, "FAIL: no-copy buffer facade\n");
        return 1;
    }
    id<MTLBuffer> buffer =
        (__bridge_transfer id<MTLBuffer>)bufferPtr;
    const uint8_t *contents = (const uint8_t *)buffer.contents;
    if (!contents || buffer.length != length ||
        contents[0] != (uint8_t)0x5a ||
        contents[length - 1] != (uint8_t)((length - 1u) ^ 0x5a)) {
        fprintf(stderr, "FAIL: no-copy buffer contents\n");
        return 1;
    }
    buffer = nil; /* Metal invokes the C++-owned VM deallocator. */
    printf("NO_COPY_BUFFER_OK\n");
    return 0;
}

/* P3.1: precompiled aux shader assets.  Verifies the embedded table (entries
 * + hash), render/compute PSO creation from metallib bytes, cache hits, the
 * function resolver, and every error path (bad hash / unknown entry / empty
 * row).  All pipelines must come from the precompiled bytes — no source is
 * compiled anywhere on this path. */
static int verifyAuxShaderAssets(void) {
    static const struct {
        const char *name;
        const char *firstEntry;
        size_t expectedEntries;
    } kAssets[] = {
        {"scaled_blit", "mgl_scaled_blit_vs", 2},
        {"scaled_blit_cs", "mgl_scaled_blit_cs", 1},
        {"scaled_depth_blit", "mgl_scaled_depth_blit_vs", 2},
        {"msaa_integer_resolve", "mgl_msaa_resolve_uint", 2},
        {"clear_rect", "mgl_clear_rect_vs", 2},
        {"safe_fallback", "mgl_safe_fallback_vs", 2},
    };
    char message[1024] = {0};

    for (size_t i = 0; i < sizeof(kAssets) / sizeof(kAssets[0]); ++i) {
        const MGLAuxShaderAsset *asset =
            mglAuxShaderAssetFind(kAssets[i].name);
        if (!asset || !asset->data || asset->size == 0 || asset->hash == 0) {
            fprintf(stderr, "FAIL: aux asset %s missing from table\n",
                    kAssets[i].name);
            return 1;
        }
        if (mglAuxAssetHash(asset->data, asset->size) != asset->hash) {
            fprintf(stderr, "FAIL: aux asset %s hash mismatch\n",
                    kAssets[i].name);
            return 1;
        }
        if (strcmp(asset->functions[0], kAssets[i].firstEntry) != 0) {
            fprintf(stderr, "FAIL: aux asset %s first entry mismatch\n",
                    kAssets[i].name);
            return 1;
        }
        size_t count = 0;
        while (count < 4 && asset->functions[count]) ++count;
        if (count != kAssets[i].expectedEntries) {
            fprintf(stderr, "FAIL: aux asset %s entry count %zu != %zu\n",
                    kAssets[i].name, count, kAssets[i].expectedEntries);
            return 1;
        }
    }
    if (mglAuxShaderAssetFind("no_such_asset") != NULL ||
        mglAuxShaderAssetCount !=
            sizeof(kAssets) / sizeof(kAssets[0])) {
        fprintf(stderr, "FAIL: aux asset table integrity\n");
        return 1;
    }

    /* Render PSO create + cache hit + variant split. */
    const MGLAuxShaderAsset *blit = mglAuxShaderAssetFind("scaled_blit");
    void *p1 = NULL, *p2 = NULL, *p3 = NULL;
#define MGL_AUX_CREATE_RENDER(_asset, _fmt, _out)                          \
    do {                                                                   \
        if (mglRenderCppGetOrCreateAuxRenderPipelineFromMetallib(          \
                (_asset)->data, (_asset)->size, (_asset)->hash,            \
                "mgl_scaled_blit_vs", "mgl_scaled_blit_fs",                \
                MGL_RENDER_CPP_AUX_RENDER_SCALED_BLIT,                     \
                (uint64_t)(uint32_t)(_fmt), (uint32_t)(_fmt), 0, 0,        \
                MTLColorWriteMaskAll, 0, 1u, &(_out), message,             \
                sizeof(message)) != 0 || !(_out)) {                        \
            fprintf(stderr, "FAIL: aux render PSO create: %s\n",           \
                    message[0] ? message : "?");                           \
            return 1;                                                      \
        }                                                                  \
    } while (0)
    MGL_AUX_CREATE_RENDER(blit, MTLPixelFormatBGRA8Unorm, p1);
    MGL_AUX_CREATE_RENDER(blit, MTLPixelFormatBGRA8Unorm, p2);
    if (p1 != p2) {
        fprintf(stderr, "FAIL: aux render PSO cache miss (same key)\n");
        return 1;
    }
    MGL_AUX_CREATE_RENDER(blit, MTLPixelFormatRGBA8Unorm, p3);
    if (p1 == p3) {
        fprintf(stderr, "FAIL: aux render PSO variants collapsed\n");
        return 1;
    }
    CFRelease(p1);
    CFRelease(p2);
    CFRelease(p3);

    /* Compute PSO create + cache hit, plus variant split for msaa. */
    const MGLAuxShaderAsset *cs = mglAuxShaderAssetFind("scaled_blit_cs");
    void *cp1 = NULL, *cp2 = NULL, *mp0 = NULL, *mp1 = NULL;
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            cs->data, cs->size, cs->hash, "mgl_scaled_blit_cs",
            MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 1u,
            &cp1, message, sizeof(message)) != 0 || !cp1) {
        fprintf(stderr, "FAIL: aux compute PSO create: %s\n",
                message[0] ? message : "?");
        return 1;
    }
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            cs->data, cs->size, cs->hash, "mgl_scaled_blit_cs",
            MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 1u,
            &cp2, message, sizeof(message)) != 0 || !cp2 || cp1 != cp2) {
        fprintf(stderr, "FAIL: aux compute PSO cache miss (same key)\n");
        return 1;
    }
    const MGLAuxShaderAsset *msaa =
        mglAuxShaderAssetFind("msaa_integer_resolve");
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            msaa->data, msaa->size, msaa->hash, "mgl_msaa_resolve_uint",
            MGL_RENDER_CPP_AUX_COMPUTE_MSAA_INTEGER_RESOLVE, 0u,
            &mp0, message, sizeof(message)) != 0 || !mp0) {
        fprintf(stderr, "FAIL: msaa uint PSO create: %s\n",
                message[0] ? message : "?");
        return 1;
    }
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            msaa->data, msaa->size, msaa->hash, "mgl_msaa_resolve_int",
            MGL_RENDER_CPP_AUX_COMPUTE_MSAA_INTEGER_RESOLVE, 1u,
            &mp1, message, sizeof(message)) != 0 || !mp1 || mp0 == mp1) {
        fprintf(stderr, "FAIL: msaa int PSO create/variant split\n");
        return 1;
    }
    CFRelease(cp1);
    CFRelease(cp2);
    CFRelease(mp0);
    CFRelease(mp1);

    /* Fragment-less clear_rect: fsEntry NULL is a valid config. */
    const MGLAuxShaderAsset *clear = mglAuxShaderAssetFind("clear_rect");
    void *cr = NULL;
    if (mglRenderCppGetOrCreateAuxRenderPipelineFromMetallib(
            clear->data, clear->size, clear->hash,
            "mgl_clear_rect_vs", NULL,
            MGL_RENDER_CPP_AUX_RENDER_CLEAR_RECT, 0u,
            MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth32Float, 0,
            MTLColorWriteMaskNone, 0, 1u,
            &cr, message, sizeof(message)) != 0 || !cr) {
        fprintf(stderr, "FAIL: fragment-less clear_rect PSO create: %s\n",
                message[0] ? message : "?");
        return 1;
    }
    CFRelease(cr);

    /* Function resolver for descriptor-assembled paths (safe fallback). */
    const MGLAuxShaderAsset *safe = mglAuxShaderAssetFind("safe_fallback");
    void *vs = NULL, *fs = NULL;
    if (mglRenderCppCreateAuxFunctions(
            safe->data, safe->size, safe->hash,
            "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
            &vs, &fs, message, sizeof(message)) != 0 || !vs || !fs) {
        fprintf(stderr, "FAIL: aux function resolver: %s\n",
                message[0] ? message : "?");
        return 1;
    }
    id<MTLFunction> vsFn = (__bridge_transfer id<MTLFunction>)vs;
    id<MTLFunction> fsFn = (__bridge_transfer id<MTLFunction>)fs;
    if (![vsFn.name isEqualToString:@"mgl_safe_fallback_vs"] ||
        ![fsFn.name isEqualToString:@"mgl_safe_fallback_fs"]) {
        fprintf(stderr, "FAIL: aux function names\n");
        return 1;
    }
    void *vsOnly = NULL, *fsOnly = (void *)0x1;
    if (mglRenderCppCreateAuxFunctions(
            blit->data, blit->size, blit->hash,
            "mgl_scaled_blit_vs", NULL,
            &vsOnly, &fsOnly, message, sizeof(message)) != 0 ||
        !vsOnly || fsOnly != NULL) {
        fprintf(stderr, "FAIL: fragment-less aux function resolver\n");
        return 1;
    }
    CFRelease(vsOnly);

    /* Error paths: bad hash, unknown entry, empty row. */
    void *bad = NULL;
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            cs->data, cs->size, cs->hash + 1, "mgl_scaled_blit_cs",
            MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 0u, &bad,
            message, sizeof(message)) == 0 || bad != NULL ||
        !strstr(message, "hash")) {
        fprintf(stderr, "FAIL: bad aux hash accepted\n");
        return 1;
    }
    bad = NULL;
    message[0] = 0;
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            cs->data, cs->size, cs->hash, "mgl_no_such_kernel",
            MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 0u, &bad,
            message, sizeof(message)) == 0 || bad != NULL) {
        fprintf(stderr, "FAIL: unknown aux entry accepted\n");
        return 1;
    }
    bad = NULL;
    message[0] = 0;
    if (mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            NULL, 0, 0, "mgl_scaled_blit_cs",
            MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 0u, &bad,
            message, sizeof(message)) == 0 || bad != NULL) {
        fprintf(stderr, "FAIL: empty aux row accepted\n");
        return 1;
    }

    printf("AUX_ASSETS_OK\n");
    return 0;
}

static int verifyCompilerAndBinaryArchive(void) {
    /* P3.3: no source compiler exists anymore.  The smoke now exercises the
     * precompiled path end to end: embedded metallib bytes -> library ->
     * functions -> render PSO (descriptor-assembled) -> compute PSO -> binary
     * archive lifecycle. */
    char message[1024] = {0};

    const MGLAuxShaderAsset *safe =
        mglAuxShaderAssetFind("safe_fallback");
    if (!safe) {
        fprintf(stderr, "FAIL: safe_fallback asset missing\n");
        return 1;
    }
    void *vertexPtr = NULL;
    void *fragmentPtr = NULL;
    if (mglRenderCppCreateAuxFunctions(
            safe->data, safe->size, safe->hash,
            "mgl_safe_fallback_vs", "mgl_safe_fallback_fs",
            &vertexPtr, &fragmentPtr,
            message, sizeof(message)) != 0 || !vertexPtr || !fragmentPtr) {
        fprintf(stderr, "FAIL: precompiled library function resolve: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLFunction> vertex =
        (__bridge_transfer id<MTLFunction>)vertexPtr;
    id<MTLFunction> fragment =
        (__bridge_transfer id<MTLFunction>)fragmentPtr;
    if (![vertex.name isEqualToString:@"mgl_safe_fallback_vs"] ||
        ![fragment.name isEqualToString:@"mgl_safe_fallback_fs"]) {
        fprintf(stderr, "FAIL: precompiled function names\n");
        return 1;
    }

    MTLRenderPipelineDescriptor *pipelineDesc =
        [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertex;
    pipelineDesc.fragmentFunction = fragment;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    void *renderPipelinePtr = NULL;
    if (mglRenderCppCreateRenderPipelineState(
            (__bridge void *)pipelineDesc, &renderPipelinePtr,
            message, sizeof(message)) != 0 || !renderPipelinePtr) {
        fprintf(stderr, "FAIL: precompiled render PSO: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLRenderPipelineState> renderPipeline =
        (__bridge_transfer id<MTLRenderPipelineState>)renderPipelinePtr;

    const MGLAuxShaderAsset *msaa =
        mglAuxShaderAssetFind("msaa_integer_resolve");
    void *computePipelinePtr = NULL;
    if (!msaa ||
        mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
            msaa->data, msaa->size, msaa->hash, "mgl_msaa_resolve_uint",
            MGL_RENDER_CPP_AUX_COMPUTE_MSAA_INTEGER_RESOLVE, 0u,
            &computePipelinePtr,
            message, sizeof(message)) != 0 || !computePipelinePtr) {
        fprintf(stderr, "FAIL: precompiled compute PSO: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLComputePipelineState> computePipeline =
        (__bridge_transfer id<MTLComputePipelineState>)computePipelinePtr;
    if (!renderPipeline || !computePipeline) return 1;

    MTLBinaryArchiveDescriptor *archiveDesc =
        [[MTLBinaryArchiveDescriptor alloc] init];
    void *archivePtr = NULL;
    if (mglRenderCppCreateBinaryArchive(
            (__bridge void *)archiveDesc, "MGL Smoke Binary Archive",
            &archivePtr, message, sizeof(message)) != 0 || !archivePtr) {
        fprintf(stderr, "FAIL: binary archive creation facade: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    id<MTLBinaryArchive> archive =
        (__bridge_transfer id<MTLBinaryArchive>)archivePtr;

    if (mglRenderCppSetRenderPipelineBinaryArchive(
            (__bridge void *)pipelineDesc,
            (__bridge void *)archive) != 0 ||
        pipelineDesc.binaryArchives.count != 1 ||
        pipelineDesc.binaryArchives.firstObject != archive) {
        fprintf(stderr, "FAIL: binary archive descriptor binding facade\n");
        return 1;
    }
    if (mglRenderCppAddRenderPipelineFunctionsToBinaryArchive(
            (__bridge void *)archive, (__bridge void *)pipelineDesc,
            message, sizeof(message)) != 0) {
        fprintf(stderr, "FAIL: binary archive add pipeline facade: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }

    NSString *archivePath = [NSTemporaryDirectory()
        stringByAppendingPathComponent:
            [NSString stringWithFormat:@"mgl-metalcpp-smoke-%@.binaryarchive",
                                       NSUUID.UUID.UUIDString]];
    NSURL *archiveURL = [NSURL fileURLWithPath:archivePath];
    if (mglRenderCppSerializeBinaryArchive(
            (__bridge void *)archive, (__bridge void *)archiveURL,
            message, sizeof(message)) != 0 ||
        ![NSFileManager.defaultManager fileExistsAtPath:archivePath]) {
        fprintf(stderr, "FAIL: binary archive serialize facade: %s\n",
                message[0] ? message : "unknown");
        return 1;
    }
    NSError *removeError = nil;
    if (![NSFileManager.defaultManager removeItemAtURL:archiveURL
                                                 error:&removeError]) {
        fprintf(stderr, "FAIL: binary archive smoke cleanup: %s\n",
                removeError.localizedDescription.UTF8String ?: "unknown");
        return 1;
    }

    printf("PRECOMPILED_PSO_OK\n");
    return 0;
}

static int verifyPipelineCacheOwner(id<MTLDevice> device) {
    id<MTLLibrary> library = smokeLoadAssetLibrary(device, "scaled_blit");
    NSError *error = nil;
    id<MTLFunction> vertex = [library newFunctionWithName:@"mgl_scaled_blit_vs"];
    id<MTLFunction> fragment = [library newFunctionWithName:@"mgl_scaled_blit_fs"];
    MTLRenderPipelineDescriptor *descriptor =
        [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = vertex;
    descriptor.fragmentFunction = fragment;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!library || !vertex || !fragment || !pipeline) {
        fprintf(stderr, "FAIL: pipeline cache fixture: %s\n",
                error.localizedDescription.UTF8String ?: "unknown");
        return 1;
    }

    void *owner = NULL;
    if (mglRenderCppCreatePipelineCacheOwner(1, 1, 1, &owner) != 0 ||
        !owner) {
        fprintf(stderr, "FAIL: pipeline cache owner create\n");
        return 1;
    }
    int psoDedup = 0;
    int dsCache = 0;
    int binaryArchive = 0;
    if (mglRenderCppGetPipelineCacheFlags(
            owner, &psoDedup, &dsCache, &binaryArchive) != 0 ||
        !psoDedup || !dsCache || !binaryArchive) {
        fprintf(stderr, "FAIL: pipeline cache owner flags\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    MGLRenderCppPipelineBlendState blend = {
        .source_rgb_factor = (uint32_t)MTLBlendFactorSourceAlpha,
        .destination_rgb_factor =
            (uint32_t)MTLBlendFactorOneMinusSourceAlpha,
        .source_alpha_factor = (uint32_t)MTLBlendFactorOne,
        .destination_alpha_factor = (uint32_t)MTLBlendFactorZero,
        .rgb_operation = (uint32_t)MTLBlendOperationAdd,
        .alpha_operation = (uint32_t)MTLBlendOperationMax,
        .color_write_mask = (uint32_t)MTLColorWriteMaskAll,
    };
    MGLRenderCppPipelineBlendState blendSnapshot = {};
    if (mglRenderCppSetPipelineBlendState(owner, 3, &blend) != 0 ||
        mglRenderCppGetPipelineBlendState(
            owner, 3, &blendSnapshot) != 0 ||
        memcmp(&blend, &blendSnapshot, sizeof(blend)) != 0) {
        fprintf(stderr, "FAIL: pipeline cache blend state\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    MGLRenderCppPipelineActiveState active = {
        .pipeline_state = (__bridge void *)pipeline,
        .vertex_function = (__bridge void *)vertex,
        .fragment_function = (__bridge void *)fragment,
        .color0_format = (uint32_t)MTLPixelFormatRGBA8Unorm,
        .depth_format = (uint32_t)MTLPixelFormatDepth32Float,
        .stencil_format = (uint32_t)MTLPixelFormatInvalid,
        .program_name = 77,
    };
    uint64_t key[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS] =
        {77, 1, 2, 3, 4, 5, 6};
    uint32_t evicted = UINT32_MAX;
    /* P4.2: descriptor cache 已改为 value-state 版。 */
    MGLRenderCppPipelineDescriptorState descriptorState = {};
    descriptorState.color_count = 8;
    descriptorState.color_format[0] = (uint32_t)MTLPixelFormatRGBA8Unorm;
    descriptorState.depth_format = (uint32_t)MTLPixelFormatDepth32Float;
    descriptorState.raster_sample_count = 1;
    if (mglRenderCppActivatePipelineState(owner, &active) != 0 ||
        mglRenderCppStorePipeline(owner, key, &active, &evicted) != 0 ||
        evicted != 0 ||
        mglRenderCppStorePipelineDescriptorState(
            owner, key, &descriptorState) != 0) {
        fprintf(stderr, "FAIL: pipeline cache owner store\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    MGLRenderCppPipelineActiveState activeSnapshot = {};
    MGLRenderCppPipelineActiveState cached = {};
    MGLRenderCppPipelineDescriptorState cachedState = {};
    if (mglRenderCppGetPipelineActiveState(owner, &activeSnapshot) != 0 ||
        activeSnapshot.pipeline_state != (__bridge void *)pipeline ||
        activeSnapshot.vertex_function != (__bridge void *)vertex ||
        activeSnapshot.fragment_function != (__bridge void *)fragment ||
        activeSnapshot.program_name != 77 ||
        mglRenderCppLookupPipeline(owner, key, &cached) != 1 ||
        cached.pipeline_state != (__bridge void *)pipeline ||
        cached.vertex_function != (__bridge void *)vertex ||
        cached.fragment_function != (__bridge void *)fragment ||
        mglRenderCppLookupPipelineDescriptorState(
            owner, key, &cachedState) != 1 ||
        memcmp(&cachedState, &descriptorState, sizeof(cachedState)) != 0) {
        fprintf(stderr, "FAIL: pipeline cache owner lookup\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    MGLRenderCppDepthStencilDescriptorState depth = {
        .depth_compare_function = (uint32_t)MTLCompareFunctionLessEqual,
        .depth_write_enabled = 1,
        .front = {
            .present = 1,
            .compare_function = (uint32_t)MTLCompareFunctionAlways,
            .read_mask = 0xff,
            .write_mask = 0x7f,
            .stencil_failure_operation = (uint32_t)MTLStencilOperationKeep,
            .depth_failure_operation = (uint32_t)MTLStencilOperationZero,
            .depth_stencil_pass_operation =
                (uint32_t)MTLStencilOperationIncrementClamp,
        },
    };
    void *depthState1 = NULL;
    void *depthState2 = NULL;
    int created1 = 0;
    int created2 = 0;
    if (mglRenderCppGetOrCreateDepthStencilState(
            owner, &depth, &depthState1, &created1) != 0 ||
        mglRenderCppGetOrCreateDepthStencilState(
            owner, &depth, &depthState2, &created2) != 0 ||
        !depthState1 || depthState1 != depthState2 || created1 != 1 ||
        created2 != 0) {
        fprintf(stderr, "FAIL: pipeline depth-stencil cache\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    mglRenderCppDisablePipelineBinaryArchive(owner);
    if (mglRenderCppGetPipelineCacheFlags(
            owner, NULL, NULL, &binaryArchive) != 0 || binaryArchive != 0 ||
        mglRenderCppInvalidatePipelineActiveState(owner) != 0 ||
        mglRenderCppGetPipelineActiveState(owner, &activeSnapshot) != 0 ||
        activeSnapshot.pipeline_state != NULL ||
        activeSnapshot.color0_format != (uint32_t)MTLPixelFormatInvalid) {
        fprintf(stderr, "FAIL: pipeline cache active invalidation\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }

    mglRenderCppResetPipelineCacheOwner(owner);
    cached = {};
    cachedState = {};
    if (mglRenderCppLookupPipeline(owner, key, &cached) != 0 ||
        mglRenderCppLookupPipelineDescriptorState(
            owner, key, &cachedState) != 0 ||
        cachedState.color_format[0] != 0u) {
        fprintf(stderr, "FAIL: pipeline cache owner reset\n");
        mglRenderCppDestroyPipelineCacheOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyPipelineCacheOwner(&owner);
    if (owner) {
        fprintf(stderr, "FAIL: pipeline cache owner destroy\n");
        return 1;
    }
    printf("PIPELINE_CACHE_OWNER_OK\n");
    return 0;
}

static int verifyBindingDedup(id<MTLDevice> device) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return 1;

    MTLTextureDescriptor *renderTargetDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    renderTargetDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> renderTarget = [device newTextureWithDescriptor:renderTargetDesc];

    MTLTextureDescriptor *shaderTextureDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    shaderTextureDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> shaderTexture = [device newTextureWithDescriptor:shaderTextureDesc];
    MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
    MTLDepthStencilDescriptor *depthDesc = [MTLDepthStencilDescriptor new];
    id<MTLDepthStencilState> depthState =
        [device newDepthStencilStateWithDescriptor:depthDesc];
    id<MTLBuffer> bindingBuffer =
        [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    if (!renderTarget || !shaderTexture || !sampler || !depthState ||
        !bindingBuffer) return 1;

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = renderTarget;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:pass];
    void *state = mglRenderCppBindingCreate(8);
    if (!encoder || !state) return 1;
    mglRenderCppBindingSetDepthStencilState(
        state, (__bridge void *)depthState);
    void *ownedDepthState = NULL;
    void *ownedPipelineState = reinterpret_cast<void *>(1u);
    if (mglRenderCppBindingGetDepthStencilState(
            state, &ownedDepthState) != 0 ||
        ownedDepthState != (__bridge void *)depthState ||
        mglRenderCppBindingGetPipelineState(
            state, &ownedPipelineState) != 0 || ownedPipelineState) {
        fprintf(stderr, "FAIL: C++ pipeline/depth binding ownership\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }

    MTLViewport viewport = {0.0, 0.0, 4.0, 4.0, 0.0, 1.0};
    MTLScissorRect scissor = {0, 0, 4, 4};
#define EXPECT_EMIT(call) do { \
    if ((call) != 1) { \
        fprintf(stderr, "FAIL: expected binding setter emit at line %d\n", __LINE__); \
        mglRenderCppBindingDestroy(state); \
        return 1; \
    } \
} while (0)
#define EXPECT_SKIP(call) do { \
    if ((call) != 0) { \
        fprintf(stderr, "FAIL: expected binding setter skip at line %d\n", __LINE__); \
        mglRenderCppBindingDestroy(state); \
        return 1; \
    } \
} while (0)

    if (mglRenderCppBindingRecordVertexBuffer(
            state, (__bridge void *)bindingBuffer, 16, 0) != 0 ||
        mglRenderCppBindingRecordFragmentBuffer(
            state, (__bridge void *)bindingBuffer, 32, 1) != 0 ||
        mglRenderCppBindingUpdateVertexBuffer(
            state, (__bridge void *)bindingBuffer, 64, 0) != 0 ||
        mglRenderCppBindingUpdateFragmentBuffer(
            state, (__bridge void *)bindingBuffer, 96, 1) != 0 ||
        mglRenderCppBindingInvalidateVertexBuffer(state, 0) != 0 ||
        mglRenderCppBindingInvalidateFragmentBuffer(state, 1) != 0 ||
        mglRenderCppBindingClearVertexBuffer(state, 0) != 0 ||
        mglRenderCppBindingClearFragmentBuffer(state, 1) != 0) {
        fprintf(stderr, "FAIL: buffer binding state API\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    mglRenderCppBindingOrVertexBufferMask(state, 1U << 2);
    mglRenderCppBindingOrFragmentBufferMask(state, 1U << 3);
    void *ownedVertexBuffer = NULL;
    void *ownedFragmentBuffer = NULL;
    uint64_t ownedVertexOffset = 0;
    uint64_t ownedFragmentOffset = 0;
    if (mglRenderCppBindingRecordVertexBuffer(
            state, (__bridge void *)bindingBuffer, 48, 2) != 0 ||
        mglRenderCppBindingRecordFragmentBuffer(
            state, (__bridge void *)bindingBuffer, 80, 3) != 0 ||
        mglRenderCppBindingGetBuffer(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 2,
            &ownedVertexBuffer, &ownedVertexOffset) != 0 ||
        mglRenderCppBindingGetBuffer(
            state, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 3,
            &ownedFragmentBuffer, &ownedFragmentOffset) != 0 ||
        ownedVertexBuffer != (__bridge void *)bindingBuffer ||
        ownedFragmentBuffer != (__bridge void *)bindingBuffer ||
        ownedVertexOffset != 48 || ownedFragmentOffset != 80) {
        fprintf(stderr, "FAIL: C++ buffer binding ownership\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }

    EXPECT_EMIT(mglRenderCppBindingSetTexture(
        state, (__bridge void *)encoder, (__bridge void *)shaderTexture,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0));
    EXPECT_EMIT(mglRenderCppBindingSetTexture(
        state, (__bridge void *)encoder, (__bridge void *)shaderTexture,
        MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0));
    EXPECT_EMIT(mglRenderCppBindingSetSampler(
        state, (__bridge void *)encoder, (__bridge void *)sampler,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0));
    EXPECT_EMIT(mglRenderCppBindingSetSampler(
        state, (__bridge void *)encoder, (__bridge void *)sampler,
        MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0));
    void *vertexTexture = NULL;
    void *fragmentTexture = NULL;
    void *vertexSampler = NULL;
    void *fragmentSampler = NULL;
    if (mglRenderCppBindingGetTexture(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0,
            &vertexTexture) != 0 ||
        mglRenderCppBindingGetTexture(
            state, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0,
            &fragmentTexture) != 0 ||
        mglRenderCppBindingGetSampler(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0,
            &vertexSampler) != 0 ||
        mglRenderCppBindingGetSampler(
            state, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0,
            &fragmentSampler) != 0 ||
        vertexTexture != (__bridge void *)shaderTexture ||
        fragmentTexture != (__bridge void *)shaderTexture ||
        vertexSampler != (__bridge void *)sampler ||
        fragmentSampler != (__bridge void *)sampler) {
        fprintf(stderr, "FAIL: C++ texture/sampler binding ownership\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    EXPECT_EMIT(mglRenderCppBindingSetViewport(
        state, (__bridge void *)encoder, viewport.originX, viewport.originY,
        viewport.width, viewport.height, viewport.znear, viewport.zfar));
    EXPECT_EMIT(mglRenderCppBindingSetScissor(
        state, (__bridge void *)encoder, scissor.x, scissor.y,
        scissor.width, scissor.height));
    EXPECT_EMIT(mglRenderCppBindingSetTriangleFill(
        state, (__bridge void *)encoder, (uint32_t)MTLTriangleFillModeLines));

    mglRenderCppBindingSetValid(state, 1);
    EXPECT_SKIP(mglRenderCppBindingSetTexture(
        state, (__bridge void *)encoder, (__bridge void *)shaderTexture,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0));
    EXPECT_SKIP(mglRenderCppBindingSetTexture(
        state, (__bridge void *)encoder, (__bridge void *)shaderTexture,
        MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0));
    EXPECT_SKIP(mglRenderCppBindingSetSampler(
        state, (__bridge void *)encoder, (__bridge void *)sampler,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0));
    EXPECT_SKIP(mglRenderCppBindingSetSampler(
        state, (__bridge void *)encoder, (__bridge void *)sampler,
        MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0));
    EXPECT_SKIP(mglRenderCppBindingSetViewport(
        state, (__bridge void *)encoder, viewport.originX, viewport.originY,
        viewport.width, viewport.height, viewport.znear, viewport.zfar));
    EXPECT_SKIP(mglRenderCppBindingSetScissor(
        state, (__bridge void *)encoder, scissor.x, scissor.y,
        scissor.width, scissor.height));
    EXPECT_SKIP(mglRenderCppBindingSetTriangleFill(
        state, (__bridge void *)encoder, (uint32_t)MTLTriangleFillModeLines));

    mglRenderCppBindingInvalidate(state);
    ownedVertexBuffer = reinterpret_cast<void *>(1u);
    ownedVertexOffset = UINT64_MAX;
    if (mglRenderCppBindingGetBuffer(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 2,
            &ownedVertexBuffer, &ownedVertexOffset) != 0 ||
        ownedVertexBuffer || ownedVertexOffset != 0) {
        fprintf(stderr, "FAIL: C++ buffer binding invalidation\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    ownedDepthState = reinterpret_cast<void *>(1u);
    if (mglRenderCppBindingGetDepthStencilState(
            state, &ownedDepthState) != 0 || ownedDepthState) {
        fprintf(stderr, "FAIL: C++ depth binding invalidation\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    vertexTexture = reinterpret_cast<void *>(1u);
    vertexSampler = reinterpret_cast<void *>(1u);
    if (mglRenderCppBindingGetTexture(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0,
            &vertexTexture) != 0 || vertexTexture ||
        mglRenderCppBindingGetSampler(
            state, MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0,
            &vertexSampler) != 0 || vertexSampler) {
        fprintf(stderr, "FAIL: C++ texture/sampler binding invalidation\n");
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    EXPECT_EMIT(mglRenderCppBindingSetViewport(
        state, (__bridge void *)encoder, viewport.originX, viewport.originY,
        viewport.width, viewport.height, viewport.znear, viewport.zfar));

    MGLRenderCppBindingStats stats = {};
    if (mglRenderCppBindingGetStats(state, &stats) != 0) return 1;
    const uint64_t expectedEmitted[MGL_RENDER_CPP_BINDING_SETTER_COUNT] =
        {1, 1, 1, 1, 2, 1, 1};
    for (uint32_t i = 0; i < MGL_RENDER_CPP_BINDING_SETTER_COUNT; i++) {
        if (stats.emitted[i] != expectedEmitted[i] || stats.skipped[i] != 1) {
            fprintf(stderr,
                    "FAIL: binding stats index=%u emitted=%llu skipped=%llu\n",
                    i, (unsigned long long)stats.emitted[i],
                    (unsigned long long)stats.skipped[i]);
            mglRenderCppBindingDestroy(state);
            return 1;
        }
    }

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: binding command buffer: %s\n",
                commandBuffer.error.localizedDescription.UTF8String);
        mglRenderCppBindingDestroy(state);
        return 1;
    }
    mglRenderCppBindingDestroy(state);
    printf("BINDING_DEDUP_OK emitted=8 skipped=7\n");
    printf("BINDING_TEXTURE_OWNER_OK\n");
    printf("BINDING_PIPELINE_OWNER_OK\n");
    printf("BINDING_BUFFER_OWNER_OK\n");
    return 0;
}

static int verifyComputeSetters(id<MTLDevice> device) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buffer =
        [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *textureDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    textureDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:textureDesc];
    id<MTLSamplerState> sampler =
        [device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
    if (!queue || !buffer || !texture || !sampler) return 1;

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    void *encoderPtr = NULL;
    if (mglRenderCppCreateComputeEncoder((__bridge void *)commandBuffer,
                                         &encoderPtr) != 0 || !encoderPtr) {
        fprintf(stderr, "FAIL: compute encoder facade\n");
        return 1;
    }
    id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)encoderPtr;

    /* P4.5: compute binding snapshot replay.  Valid buffer+bytes ops encode;
     * NULL encoder / count overflow / NULL bytes / bad kind are rejected;
     * NULL buffer ops are legal slot clears.  No compute pipeline is needed
     * for setBuffer/setBytes encoding (only dispatch requires one). */
    {
        MGLRenderCppComputeBindingSnapshot cbsnap = {};
        cbsnap.ops[cbsnap.op_count++] =
            (MGLRenderCppComputeBindingOp){/* kind */ 0u, /* index */ 0,
                /* offset */ 16, /* buffer */ (__bridge void *)buffer,
                /* bytes */ NULL, /* length */ 0u};
        cbsnap.ops[cbsnap.op_count++] =
            (MGLRenderCppComputeBindingOp){/* kind */ 0u, /* index */ 1,
                /* offset */ 0, /* buffer */ NULL, /* bytes */ NULL,
                /* length */ 0u}; /* NULL buffer = slot clear */
        cbsnap.ops[cbsnap.op_count++] =
            (MGLRenderCppComputeBindingOp){/* kind */ 1u, /* index */ 2,
                /* offset */ 0, /* buffer */ NULL,
                /* bytes */ "ABCD", /* length */ 4u};
        cbsnap.ops[cbsnap.op_count++] =
            (MGLRenderCppComputeBindingOp){/* kind */ 2u, /* index */ 3,
                /* offset */ 0, /* buffer */ (__bridge void *)texture,
                /* bytes */ NULL, /* length */ 0u};
        cbsnap.ops[cbsnap.op_count++] =
            (MGLRenderCppComputeBindingOp){/* kind */ 3u, /* index */ 4,
                /* offset */ 0, /* buffer */ (__bridge void *)sampler,
                /* bytes */ NULL, /* length */ 0u};
        MGLRenderCppComputeBindingSnapshot cboverflow = cbsnap;
        cboverflow.op_count =
            MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS + 1;
        MGLRenderCppComputeBindingSnapshot cbnullBytes = cbsnap;
        cbnullBytes.ops[2].bytes = NULL;
        MGLRenderCppComputeBindingSnapshot cbbadKind = cbsnap;
        cbbadKind.ops[0].kind = 0xdead;
        MGLRenderCppComputeBindingSnapshot cbnullTex = cbsnap;
        cbnullTex.ops[3].buffer = NULL; /* NULL texture = slot clear */
        char cbError[128] = {0};
        if (mglRenderCppEncodeComputeBindingSnapshot(
                encoderPtr, &cbsnap, cbError, sizeof(cbError)) != 0 ||
            mglRenderCppEncodeComputeBindingSnapshot(
                encoderPtr, &cbnullTex, cbError, sizeof(cbError)) != 0 ||
            mglRenderCppEncodeComputeBindingSnapshot(
                NULL, &cbsnap, NULL, 0) != -1 ||
            mglRenderCppEncodeComputeBindingSnapshot(
                encoderPtr, &cboverflow, cbError, sizeof(cbError)) != -1 ||
            mglRenderCppEncodeComputeBindingSnapshot(
                encoderPtr, &cbnullBytes, cbError, sizeof(cbError)) != -1 ||
            mglRenderCppEncodeComputeBindingSnapshot(
                encoderPtr, &cbbadKind, cbError, sizeof(cbError)) != -1) {
            fprintf(stderr, "FAIL: compute binding snapshot err='%s'\n",
                    cbError);
            mglRenderCppEndComputeEncoder(encoderPtr);
            return 1;
        }
        printf("COMPUTE_BINDING_SNAPSHOT_OK\n");
    }

    if (mglRenderCppSetComputeBuffer(encoderPtr, (__bridge void *)buffer,
                                     16, 0) != 0 ||
        mglRenderCppSetComputeTexture(encoderPtr, (__bridge void *)texture,
                                      0) != 0 ||
        mglRenderCppSetComputeSampler(encoderPtr, (__bridge void *)sampler,
                                      0) != 0 ||
        mglRenderCppSetComputeThreadgroupMemoryLength(encoderPtr, 64, 0) != 0) {
        fprintf(stderr, "FAIL: compute setter facade\n");
        mglRenderCppEndComputeEncoder(encoderPtr);
        return 1;
    }
    if (mglRenderCppEndComputeEncoder(encoderPtr) != 0) {
        fprintf(stderr, "FAIL: compute encoder end facade\n");
        return 1;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: compute setter command buffer: %s\n",
                commandBuffer.error.localizedDescription.UTF8String);
        return 1;
    }
    printf("COMPUTE_SETTERS_OK\n");

    /* P4.5: compute dispatch plan argument validation.  Encoding a real
     * dispatch requires a compute pipeline (AGX crashes at encode without
     * one), so the success-path encode is exercised by test_regression's
     * compute_dispatch_ssbo on both gates; here only the rejection paths are
     * asserted (all fail before touching the encoder). */
    {
        MGLRenderCppComputePlan directPlan = {
            .dispatch_kind = MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT,
            .groups_x = 2, .groups_y = 1, .groups_z = 1,
            .local_x = 4, .local_y = 1, .local_z = 1,
            .indirect_buffer = NULL, .indirect_offset = 0,
        };
        MGLRenderCppComputePlan indirectPlan = {
            .dispatch_kind = MGL_RENDER_CPP_COMPUTE_DISPATCH_INDIRECT,
            .groups_x = 0, .groups_y = 0, .groups_z = 0,
            .local_x = 8, .local_y = 1, .local_z = 1,
            .indirect_buffer = (__bridge void *)buffer,
            .indirect_offset = 0,
        };
        MGLRenderCppComputePlan badKind = directPlan;
        badKind.dispatch_kind = 0xdead;
        MGLRenderCppComputePlan badIndirect = indirectPlan;
        badIndirect.indirect_buffer = NULL;
        char planErr[64] = {0};
        if (mglRenderCppDispatchComputePlan(
                NULL, &directPlan, planErr, sizeof(planErr)) != -1 ||
            mglRenderCppDispatchComputePlan(
                NULL, NULL, planErr, sizeof(planErr)) != -1 ||
            mglRenderCppDispatchComputePlan(
                NULL, &badKind, planErr, sizeof(planErr)) != -1 ||
            mglRenderCppDispatchComputePlan(
                NULL, &badIndirect, planErr, sizeof(planErr)) != -1) {
            fprintf(stderr, "FAIL: compute dispatch plan rejection\n");
            return 1;
        }
        printf("COMPUTE_DISPATCH_PLAN_ERR_OK\n");
    }
    return 0;
}

static int verifySyncCallbacks(id<MTLDevice> device) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!queue || !commandBuffer) return 1;

    Sync sync = {};
    sync.mtl_command_buffer = (void *)CFBridgingRetain(commandBuffer);
    [commandBuffer commit];
    unsigned int status = mglRenderCppGetSyncStatus(NULL, &sync);
    if (status != GL_SIGNALED && status != GL_UNSIGNALED) {
        fprintf(stderr, "FAIL: invalid pre-wait sync status=0x%x\n", status);
        mglRenderCppReleaseSync(NULL, &sync);
        return 1;
    }
    mglRenderCppWaitForSync(NULL, &sync);
    if (sync.mtl_command_buffer || sync.mtl_event ||
        mglRenderCppGetSyncStatus(NULL, &sync) != GL_SIGNALED) {
        fprintf(stderr, "FAIL: sync wait did not release/signaled state\n");
        mglRenderCppReleaseSync(NULL, &sync);
        return 1;
    }

    Sync releaseOnly = {};
    id<MTLEvent> event = [device newEvent];
    if (event) releaseOnly.mtl_event = (void *)CFBridgingRetain(event);
    mglRenderCppReleaseSync(NULL, &releaseOnly);
    if (releaseOnly.mtl_event || releaseOnly.mtl_command_buffer) {
        fprintf(stderr, "FAIL: sync release did not clear resources\n");
        return 1;
    }
    printf("SYNC_CALLBACKS_OK\n");
    return 0;
}

static int verifyCommandQueueOwner(void) {
    void *owner = NULL;
    void *queuePtr = NULL;
    if (mglRenderCppCreateCommandQueueOwner(2, &owner, &queuePtr) != 0 ||
        !owner || !queuePtr) {
        fprintf(stderr, "FAIL: command queue owner create\n");
        mglRenderCppDestroyCommandQueueOwner(&owner);
        return 1;
    }

    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)queuePtr;
    void *commandBuffer = NULL;
    if (!queue || mglRenderCppCreateCommandBuffer(
            queuePtr, &commandBuffer) != 0 || !commandBuffer) {
        fprintf(stderr, "FAIL: command queue owner command buffer\n");
        mglRenderCppDestroyCommandQueueOwner(&owner);
        return 1;
    }

    void *resetQueuePtr = NULL;
    if (mglRenderCppResetCommandQueueOwner(
            owner, 0, &resetQueuePtr) != 0 || !resetQueuePtr) {
        fprintf(stderr, "FAIL: command queue owner reset\n");
        mglRenderCppDestroyCommandQueueOwner(&owner);
        return 1;
    }
    queue = (__bridge id<MTLCommandQueue>)resetQueuePtr;
    commandBuffer = NULL;
    if (!queue || mglRenderCppCreateCommandBuffer(
            resetQueuePtr, &commandBuffer) != 0 || !commandBuffer) {
        fprintf(stderr, "FAIL: reset command queue command buffer\n");
        mglRenderCppDestroyCommandQueueOwner(&owner);
        return 1;
    }

    mglRenderCppDestroyCommandQueueOwner(&owner);
    if (owner) {
        fprintf(stderr, "FAIL: command queue owner destroy\n");
        return 1;
    }
    printf("COMMAND_QUEUE_OWNER_OK\n");
    return 0;
}

static int verifyCommandBufferOwner(void) {
    void *queueOwner = NULL;
    void *queue = NULL;
    void *owner = NULL;
    void *commandBuffer = NULL;
    if (mglRenderCppCreateCommandQueueOwner(
            0, &queueOwner, &queue) != 0 || !queueOwner || !queue ||
        mglRenderCppCreateCommandBufferOwner(
            queue, &owner, &commandBuffer) != 0 || !owner ||
        !commandBuffer) {
        fprintf(stderr, "FAIL: command buffer owner create\n");
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }

    id<MTLCommandBuffer> submitted =
        (__bridge id<MTLCommandBuffer>)commandBuffer;
    MGLRenderCppCommandBufferState initialState = {};
    int *completionContext = new int(7);
    if (mglRenderCppGetCommandBufferState(
            commandBuffer, &initialState) != 0 ||
        initialState.status != MTLCommandBufferStatusNotEnqueued ||
        initialState.has_error ||
        mglRenderCppAddCommandBufferCompletion(
            commandBuffer, commandBufferCompletion, completionContext,
            destroyCommandBufferCompletionContext) != 0) {
        fprintf(stderr, "FAIL: command buffer state/completion setup\n");
        delete completionContext;
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }
    void *submission = NULL;
    void *detached = NULL;
    if (mglRenderCppTakeCommandBufferSubmission(
            owner, &submission, &detached) != 0 || !submission ||
        detached != commandBuffer ||
        mglRenderCppCommandBufferSubmissionMatchesBuffer(
            submission, detached) != 1 ||
        mglRenderCppCommandBufferSubmissionMatchesBuffer(
            submission, NULL) != -1 ||
        mglRenderCppCommandBufferSubmissionMatchesBuffer(
            submission, (void *)(uintptr_t)0xdeadbeef) != 0 ||
        mglRenderCppCommitCommandBufferSubmission(&submission) != 0 ||
        submission || mglRenderCppWaitCommandBuffer(detached) != 0 ||
        submitted.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: command buffer submission commit\n");
        mglRenderCppDestroyCommandBufferSubmission(&submission);
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }
    MGLRenderCppCommandBufferState completedState = {};
    if (mglRenderCppGetCommandBufferState(
            detached, &completedState) != 0 ||
        completedState.status != MTLCommandBufferStatusCompleted ||
        completedState.has_error ||
        s_commandBufferCompletionCount.load(std::memory_order_relaxed) != 1 ||
        s_commandBufferContextDestroyCount.load(
            std::memory_order_relaxed) != 1 ||
        s_commandBufferCompletionStatus.load(std::memory_order_relaxed) !=
            MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "FAIL: command buffer state/completion result\n");
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }

    commandBuffer = NULL;
    if (mglRenderCppResetCommandBufferOwner(
            owner, queue, &commandBuffer) != 0 || !commandBuffer) {
        fprintf(stderr, "FAIL: command buffer owner reset\n");
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }
    submission = NULL;
    detached = NULL;
    if (mglRenderCppTakeCommandBufferSubmission(
            owner, &submission, &detached) != 0 || !submission ||
        !detached ||
        mglRenderCppCommandBufferSubmissionMatchesBuffer(
            submission, detached) != 1) {
        fprintf(stderr, "FAIL: command buffer submission detach\n");
        mglRenderCppDestroyCommandBufferSubmission(&submission);
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }
    mglRenderCppDestroyCommandBufferSubmission(&submission);
    if (submission) {
        fprintf(stderr, "FAIL: command buffer submission destroy\n");
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }

    commandBuffer = NULL;
    if (mglRenderCppResetCommandBufferOwner(
            owner, queue, &commandBuffer) != 0 || !commandBuffer) {
        fprintf(stderr, "FAIL: command buffer owner discard fixture\n");
        mglRenderCppDestroyCommandBufferOwner(&owner);
        mglRenderCppDestroyCommandQueueOwner(&queueOwner);
        return 1;
    }
    mglRenderCppDiscardCommandBufferOwnerCurrent(owner);
    mglRenderCppDestroyCommandBufferOwner(&owner);
    mglRenderCppDestroyCommandQueueOwner(&queueOwner);
    if (owner || queueOwner) {
        fprintf(stderr, "FAIL: command buffer owner destroy\n");
        return 1;
    }
    printf("COMMAND_BUFFER_OWNER_OK\n");
    printf("COMMAND_BUFFER_STATE_OK\n");
    return 0;
}

/* Stub definitions of the compat-subsystem helpers the upload-prep path
 * calls (the standalone smoke binary does not link mgl_texture_compat.m).
 * Only the cases exercised by LEVEL_UPLOAD_PREP_OK are implemented; the
 * production A/B parity is covered by the regression suite. */
extern "C" {
GLuint sizeForInternalFormat(GLenum internalformat, GLenum, GLenum) {
    switch (internalformat) {
        case GL_RGB16: return 6; /* 3 x 16-bit */
        case GL_RGB8: return 3;
        default: return 0;
    }
}
bool mglTextureInternalFormatNeedsRGBA8Expansion(
    GLenum internalformat, uint32_t pixelFormat) {
    bool isRGBA8 = (pixelFormat == (uint32_t)MTLPixelFormatRGBA8Unorm ||
                    pixelFormat == (uint32_t)MTLPixelFormatRGBA8Unorm_sRGB);
    if (!isRGBA8) return false;
    switch (internalformat) {
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB565:
            return true;
        default:
            return false;
    }
}
bool mglTextureNeedsChannelExpansion(
    GLenum internalformat, uint32_t pixelFormat) {
    bool isRGBA16 =
        (pixelFormat == (uint32_t)MTLPixelFormatRGBA16Unorm ||
         pixelFormat == (uint32_t)MTLPixelFormatRGBA16Snorm ||
         pixelFormat == (uint32_t)MTLPixelFormatRGBA16Float);
    return isRGBA16 && internalformat == GL_RGB16;
}
}

static int verifyCommandBufferGetterAndAdopt(void) {
    /* P4.5 (item 1141): owner getter + adopt — the gate-off fallback keeps
     * the owner as the single source on both gates. */
    if (mglRenderCppCommandBufferOwnerGetCurrent(NULL) != NULL) {
        fprintf(stderr, "FAIL: cb getter null owner\n");
        return 1;
    }
    if (mglRenderCppCreateCommandBufferOwnerAdopt(NULL, NULL) != -1) {
        fprintf(stderr, "FAIL: cb adopt bad args\n");
        return 1;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return 0; /* covered by main's guard */
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> objcCB = [queue commandBuffer];
    if (!objcCB) {
        fprintf(stderr, "FAIL: cb adopt setup\n");
        return 1;
    }
    void *owner = NULL;
    if (mglRenderCppCreateCommandBufferOwnerAdopt(
            (__bridge void *)objcCB, &owner) != 0 || !owner) {
        fprintf(stderr, "FAIL: cb adopt create\n");
        return 1;
    }
    id<MTLCommandBuffer> readBack =
        (__bridge id<MTLCommandBuffer>)mglRenderCppCommandBufferOwnerGetCurrent(
            owner);
    if (readBack != objcCB) {
        fprintf(stderr, "FAIL: cb adopt identity\n");
        return 1;
    }
    mglRenderCppDiscardCommandBufferOwnerCurrent(owner);
    if (mglRenderCppCommandBufferOwnerGetCurrent(owner) != NULL) {
        fprintf(stderr, "FAIL: cb getter after discard\n");
        return 1;
    }
    mglRenderCppDestroyCommandBufferOwner(&owner);
    if (owner != NULL) {
        fprintf(stderr, "FAIL: cb owner not cleared\n");
        return 1;
    }
    printf("CB_GETTER_ADOPT_OK\n");
    return 0;
}

static int verifyRenderEncoderGetter(void) {
    /* P4.5 (item 1141): render-encoder owner getter — the ObjC mirror is
     * gone and reads borrow through the C++ owner on both gates. */
    if (mglRenderCppRenderEncoderOwnerGetCurrent(NULL) != NULL) {
        fprintf(stderr, "FAIL: re getter null owner\n");
        return 1;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return 0; /* covered by main's guard */
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4 height:4 mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [dev newTextureWithDescriptor:desc];
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor new];
    rpd.colorAttachments[0].texture = texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:rpd];
    if (!cb || !texture || !encoder) {
        fprintf(stderr, "FAIL: re getter setup\n");
        return 1;
    }
    void *owner = NULL;
    if (mglRenderCppCreateRenderEncoderOwner(
            (__bridge void *)encoder, &owner) != 0 || !owner) {
        fprintf(stderr, "FAIL: re owner create\n");
        return 1;
    }
    id<MTLRenderCommandEncoder> readBack =
        (__bridge id<MTLRenderCommandEncoder>)
            mglRenderCppRenderEncoderOwnerGetCurrent(owner);
    if (readBack != encoder) {
        fprintf(stderr, "FAIL: re getter identity\n");
        return 1;
    }
    if (mglRenderCppEndRenderEncoderOwner(owner) != 0) {
        fprintf(stderr, "FAIL: re owner end\n");
        return 1;
    }
    /* The owner keeps the (ended) encoder pointer until destroy — matches
     * the mirror's end semantics (clear is a separate step). */
    if (mglRenderCppRenderEncoderOwnerGetCurrent(owner) !=
        (__bridge void *)encoder) {
        fprintf(stderr, "FAIL: re getter after end\n");
        return 1;
    }
    mglRenderCppDestroyRenderEncoderOwner(&owner);
    if (owner != NULL) {
        fprintf(stderr, "FAIL: re owner not cleared\n");
        return 1;
    }
    /* The owner already called endEncoding via EndRenderEncoderOwner. */
    printf("RE_GETTER_OK\n");
    return 0;
}

static int verifyMDIScratchOwner(void) {
    /* P4.5 (item 1155): MDI scratch allocator — the ObjC gate-off allocator
     * now delegates to the same C++ owner. */
    void *owner = NULL;
    if (mglRenderCppCreateMDIScratchOwner(&owner) != 0 || !owner) {
        fprintf(stderr, "FAIL: mdi owner create\n");
        return 1;
    }
    if (mglRenderCppAllocateMDIScratch(NULL, 16, 256, NULL, NULL, NULL) != -1 ||
        mglRenderCppAllocateMDIScratch(owner, 0, 256, NULL, NULL, NULL) != -1 ||
        mglRenderCppAllocateMDIScratch(owner, 16, 3, NULL, NULL, NULL) != -1) {
        fprintf(stderr, "FAIL: mdi bad args\n");
        return 1;
    }
    void *buffer = NULL;
    uint64_t offset = 0, capacity = 0;
    if (mglRenderCppAllocateMDIScratch(
            owner, 128, 256, &buffer, &offset, &capacity) != 0 ||
        !buffer || offset != 0 || capacity < 65536) {
        fprintf(stderr, "FAIL: mdi first alloc (off=%llu cap=%llu)\n",
                (unsigned long long)offset, (unsigned long long)capacity);
        return 1;
    }
    void *sameBuffer = NULL;
    uint64_t offset2 = 0;
    if (mglRenderCppAllocateMDIScratch(
            owner, 128, 256, &sameBuffer, &offset2, &capacity) != 0 ||
        sameBuffer != buffer || offset2 != 256) {
        fprintf(stderr, "FAIL: mdi second alloc (off=%llu)\n",
                (unsigned long long)offset2);
        return 1;
    }
    void *grownBuffer = NULL;
    uint64_t offset3 = 0;
    if (mglRenderCppAllocateMDIScratch(
            owner, 200000, 256, &grownBuffer, &offset3, &capacity) != 0 ||
        !grownBuffer || offset3 != 0 || capacity < 200000) {
        fprintf(stderr, "FAIL: mdi grow (off=%llu cap=%llu)\n",
                (unsigned long long)offset3, (unsigned long long)capacity);
        return 1;
    }
    mglRenderCppDestroyMDIScratchOwner(&owner);
    if (owner != NULL) {
        fprintf(stderr, "FAIL: mdi owner not cleared\n");
        return 1;
    }
    printf("MDI_SCRATCH_OK\n");
    return 0;
}

static int verifyUInt8ToUInt16(void) {
    /* P.5 (item 1141): GL_UNSIGNED_BYTE -> UInt16 element expansion. */
    const uint8_t b[] = {0, 1, 0xff, 250, 5};
    uint16_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandUInt8ToUInt16(NULL, 3, &idx, &n) != -1 ||
        mglRenderCppExpandUInt8ToUInt16(b, 0, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: u16 bad args\n");
        return 1;
    }
    if (mglRenderCppExpandUInt8ToUInt16(b, 5, &idx, &n) != 0 ||
        n != 5 || idx[0]!=0 || idx[1]!=1 || idx[2]!=0xff || idx[3]!=250 || idx[4]!=5) {
        fprintf(stderr, "FAIL: u16 content\n");
        return 1;
    }
    free(idx);
    printf("EXPAND_U16_OK\n");
    return 0;
}

static int verifyArrayVariants(void) {
    /* P4.5 (item 1141/887): fan/strip/line-loop ARRAY emulations. */
    uint32_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandTriangleFanArrayIndices(5, &idx, &n) != 0 ||
        n != 9 || idx[0]!=0 || idx[1]!=1 || idx[2]!=2 ||
        idx[3]!=0 || idx[4]!=2 || idx[5]!=3 ||
        idx[6]!=0 || idx[7]!=3 || idx[8]!=4) {
        fprintf(stderr, "FAIL: fan array\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandTriangleStripArrayIndices(5, &idx, &n) != 0 ||
        n != 9 ||
        idx[0]!=0||idx[1]!=1||idx[2]!=2 ||
        idx[3]!=2||idx[4]!=1||idx[5]!=3 ||
        idx[6]!=2||idx[7]!=3||idx[8]!=4) {
        fprintf(stderr, "FAIL: strip array\n");
        return 1;
    }
    free(idx);
    /* Line loop array firstVertex-relative. */
    if (mglRenderCppExpandLineLoopArrayIndices(100, 3, &idx, &n) != 0 ||
        n != 4 || idx[0]!=100||idx[1]!=101||idx[2]!=102||idx[3]!=100) {
        fprintf(stderr, "FAIL: line loop array\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandTriangleFanArrayIndices(2, &idx, &n) != -1 ||
        mglRenderCppExpandTriangleStripArrayIndices(2, &idx, &n) != -1 ||
        mglRenderCppExpandLineLoopArrayIndices(0, 1, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: array bad args\n");
        return 1;
    }
    printf("EXPAND_ARRAY_VARIANTS_OK\n");
    return 0;
}

static int verifyQuadLine(void) {
    /* P4.5 (item 1141/887): quad-array/element line-loop emulation (8/quad). */
    uint32_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandQuadArrayLineIndices(1, &idx, &n) != 0 ||
        n != 8 ||
        idx[0]!=0||idx[1]!=1||idx[2]!=1||idx[3]!=2||idx[4]!=2||idx[5]!=3||idx[6]!=3||idx[7]!=0) {
        fprintf(stderr, "FAIL: quad array line\n");
        return 1;
    }
    free(idx);
    const uint16_t q[] = {10,20,30,40};
    if (mglRenderCppExpandQuadElementLineIndices((const uint8_t*)q, 2, 1, &idx, &n) != 0 ||
        n != 8 ||
        idx[0]!=10||idx[1]!=20||idx[2]!=20||idx[3]!=30||idx[4]!=30||idx[5]!=40||idx[6]!=40||idx[7]!=10) {
        fprintf(stderr, "FAIL: quad elem line\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandQuadArrayLineIndices(0, &idx, &n) != -1 ||
        mglRenderCppExpandQuadElementLineIndices(NULL, 2, 1, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: quad line bad args\n");
        return 1;
    }
    printf("EXPAND_QUAD_LINE_OK\n");
    return 0;
}

static int verifyExpandQuad(void) {
    /* P4.5 (item 1141/887): quad-array/quad-element emulation. */
    uint32_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandQuadArrayIndices(2, &idx, &n) != 0 ||
        n != 12 ||
        idx[0]!=0||idx[1]!=1||idx[2]!=2||idx[3]!=0||idx[4]!=2||idx[5]!=3 ||
        idx[6]!=4||idx[7]!=5||idx[8]!=6||idx[9]!=4||idx[10]!=6||idx[11]!=7) {
        fprintf(stderr, "FAIL: quad array\n");
        return 1;
    }
    free(idx);
    const uint16_t q[] = {10,11,12,13, 20,21,22,23};
    if (mglRenderCppExpandQuadElementIndices((const uint8_t*)q, 2, 2, &idx, &n) != 0 ||
        n != 12 ||
        idx[0]!=10||idx[1]!=11||idx[2]!=12||idx[3]!=10||idx[4]!=12||idx[5]!=13 ||
        idx[6]!=20||idx[7]!=21||idx[8]!=22||idx[9]!=20||idx[10]!=22||idx[11]!=23) {
        fprintf(stderr, "FAIL: quad element\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandQuadArrayIndices(0, &idx, &n) != -1 ||
        mglRenderCppExpandQuadElementIndices(NULL, 2, 1, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: quad bad args\n");
        return 1;
    }
    printf("EXPAND_QUAD_OK\n");
    return 0;
}

static int verifyExpandStripAndLineLoop(void) {
    /* P4.5 (item 1141/887): triangle-strip + LINE_LOOP element expansion. */
    const uint16_t s16[] = {0,1,2,3,4};  /* count-2 = 3 strips */
    uint32_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandTriangleStripIndices((const uint8_t*)s16, 2, 5, &idx, &n) != 0 ||
        n != 9 ||
        idx[0]!=0||idx[1]!=1||idx[2]!=2 ||
        idx[3]!=2||idx[4]!=1||idx[5]!=3 ||
        idx[6]!=2||idx[7]!=3||idx[8]!=4) {
        fprintf(stderr, "FAIL: strip\n");
        return 1;
    }
    free(idx);
    const uint8_t l8[] = {7,8,9};
    if (mglRenderCppExpandLineLoopIndices(l8, 1, 3, &idx, &n) != 0 ||
        n != 4 || idx[0]!=7||idx[1]!=8||idx[2]!=9||idx[3]!=7) {
        fprintf(stderr, "FAIL: loop\n");
        return 1;
    }
    free(idx);
    const uint8_t l8b[] = {5,6};
    if (mglRenderCppExpandLineLoopIndices(l8b, 1, 2, &idx, &n) != 0 ||
        n != 3 || idx[0]!=5||idx[1]!=6||idx[2]!=5) {
        fprintf(stderr, "FAIL: loop2\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandTriangleStripIndices(NULL, 2, 5, &idx, &n) != -1 ||
        mglRenderCppExpandLineLoopIndices(l8, 1, 1, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: strip/loop bad args\n");
        return 1;
    }
    printf("EXPAND_STRIP_AND_LINE_LOOP_OK\n");
    return 0;
}


static int verifyExpandTriangleFan(void) {
    /* P4.5 (item 1141/887): triangle-fan element emulation expansion. */
    uint32_t *idx = NULL; uint64_t n = 0;
    if (mglRenderCppExpandTriangleFanIndices(NULL, 2, 4, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: fan bad args\n");
        return 1;
    }
    /* center=7; triangles (7,10,11),(7,11,12),(7,12,13) -> 9 indices. */
    const uint16_t u16[] = {7, 10, 11, 12, 13};
    if (mglRenderCppExpandTriangleFanIndices((const uint8_t*)u16, 2, 5, &idx, &n) != 0 ||
        n != 9 || idx[0]!=7 || idx[1]!=10 || idx[2]!=11 || idx[3]!=7 ||
        idx[4]!=11 || idx[5]!=12 || idx[6]!=7 || idx[7]!=12 || idx[8]!=13) {
        fprintf(stderr, "FAIL: fan content n=%llu\n", (unsigned long long)n);
        return 1;
    }
    free(idx);
    const uint8_t u8[] = {0,1,2,3};
    if (mglRenderCppExpandTriangleFanIndices(u8, 1, 4, &idx, &n) != 0 ||
        n != 6 || idx[1]!=1 || idx[2]!=2 || idx[4]!=2) {
        fprintf(stderr, "FAIL: fan u8\n");
        return 1;
    }
    free(idx);
    if (mglRenderCppExpandTriangleFanIndices(u8, 1, 2, &idx, &n) != -1) {
        fprintf(stderr, "FAIL: fan too short\n");
        return 1;
    }
    printf("EXPAND_TRIANGLE_FAN_OK\n");
    return 0;
}

static int verifyGLIndexValueRead(void) {
    /* P4.5 (item 1141): GL index element size + index-value read. */
    if (mglRenderCppGLIndexElementSize(GL_UNSIGNED_BYTE) != 1 ||
        mglRenderCppGLIndexElementSize(GL_UNSIGNED_SHORT) != 2 ||
        mglRenderCppGLIndexElementSize(GL_UNSIGNED_INT) != 4 ||
        mglRenderCppGLIndexElementSize(0xbad) != 0) {
        fprintf(stderr, "FAIL: index elem size\\n");
        return 1;
    }
    const uint8_t b[] = {3, 0, 1, 0}; /* UInt16 values: 3 then 1 */
    if (mglRenderCppReadGLIndexValue(b, 1, 0) != 3 ||
        mglRenderCppReadGLIndexValue(b, 2, 1) != 1 ||
        mglRenderCppReadGLIndexValue(NULL, 1, 0) != 0 ||
        mglRenderCppReadGLIndexValue(b, 0, 0) != 0) {
        fprintf(stderr, "FAIL: read index value\\n");
        return 1;
    }
    printf("GL_INDEX_VALUE_READ_OK\\n");
    return 0;
}

static int verifyGLTypeElementByteSize(void) {
    /* P4.5 (item 1144): GL type -> element byte size. */
    if (mglRenderCppGLTypeElementByteSize(GL_FLOAT) != 4 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_VEC2) != 8 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_VEC3) != 12 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_VEC4) != 16 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_MAT2) != 8 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_MAT3) != 12 ||
        mglRenderCppGLTypeElementByteSize(GL_FLOAT_MAT4) != 16 ||
        mglRenderCppGLTypeElementByteSize(GL_DOUBLE) != 8 ||
        mglRenderCppGLTypeElementByteSize(0xfeed) != 4) {
        fprintf(stderr, "FAIL: gl type element byte size\\n");
        return 1;
    }
    printf("GL_TYPE_ELEM_SIZE_OK\\n");
    return 0;
}

static int verifyPrimitiveRestartFixedIndex(void) {
    /* P4.5 (item 1141): fixed restart index by GL type. */
    uint32_t idx = 0;
    if (mglRenderCppPrimitiveRestartFixedIndex(GL_UNSIGNED_BYTE, &idx) != 1 || idx != 0xffu ||
        mglRenderCppPrimitiveRestartFixedIndex(GL_UNSIGNED_SHORT, &idx) != 1 || idx != 0xffffu ||
        mglRenderCppPrimitiveRestartFixedIndex(GL_UNSIGNED_INT, &idx) != 1 || idx != 0xffffffffu ||
        mglRenderCppPrimitiveRestartFixedIndex(0xdead, &idx) != 0 ||
        mglRenderCppPrimitiveRestartFixedIndex(GL_UNSIGNED_BYTE, NULL) != -1) {
        fprintf(stderr, "FAIL: fixed restart index\\n");
        return 1;
    }
    printf("RESTART_FIXED_INDEX_OK\\n");
    return 0;
}

static int verifyHashStepU64(void) {
    /* P4.5 (item 1144): FNV-1a single step = (h^v)*fF. */
    /* mglHashStepU64(0, 0) = 0*const = 0. */
    uint64_t h = mglRenderCppHashStepU64(0, 0);
    if (h != 0) { fprintf(stderr, "FAIL: hash 0,0\\n"); return 1; }
    h = mglRenderCppHashStepU64(0, 1);
    if (h != 1099511628211ull) { fprintf(stderr, "FAIL: hash 0,1\\n"); return 1; }
    h = mglRenderCppHashStepU64(1099511628211ull, 2);  /* (const^1^2) */
    printf("HASH_STEP_U64_OK\\n");
    return 0;
}

static int verifyDoubleAttribFormat(void) {
    /* P4.5 (item 1141): double-attrib size -> MTL Fmt value. */
    if (mglRenderCppDoubleVertexAttribFloatFormat(1) != 28 ||
        mglRenderCppDoubleVertexAttribFloatFormat(2) != 29 ||
        mglRenderCppDoubleVertexAttribFloatFormat(3) != 30 ||
        mglRenderCppDoubleVertexAttribFloatFormat(4) != 31 ||
        mglRenderCppDoubleVertexAttribFloatFormat(5) != 0) {
        fprintf(stderr, "FAIL: double attrib fmt\\n");
        return 1;
    }
    printf("DOUBLE_ATTRIB_FORMAT_OK\\n");
    return 0;
}

static int verifyIntegerAttribConversionFormat(void) {
    struct IntegerAttribFormatCase {
        const char *label;
        uint64_t source_type;
        uint64_t shader_type;
        uint32_t size;
        MTLVertexFormat expected;
    } cases[] = {
        {"ubyte-int-1", GL_UNSIGNED_BYTE, GL_INT, 1u,
         MTLVertexFormatInt},
        {"ushort-ivec2-2", GL_UNSIGNED_SHORT, GL_INT_VEC2, 2u,
         MTLVertexFormatInt2},
        {"uint-ivec3-3", GL_UNSIGNED_INT, GL_INT_VEC3, 3u,
         MTLVertexFormatInt3},
        {"ubyte-ivec4-4", GL_UNSIGNED_BYTE, GL_INT_VEC4, 4u,
         MTLVertexFormatInt4},
        {"byte-uint-1", GL_BYTE, GL_UNSIGNED_INT, 1u,
         MTLVertexFormatUInt},
        {"short-uvec2-2", GL_SHORT, GL_UNSIGNED_INT_VEC2, 2u,
         MTLVertexFormatUInt2},
        {"int-uvec3-3", GL_INT, GL_UNSIGNED_INT_VEC3, 3u,
         MTLVertexFormatUInt3},
        {"byte-uvec4-4", GL_BYTE, GL_UNSIGNED_INT_VEC4, 4u,
         MTLVertexFormatUInt4},
        {"signed-compatible", GL_SHORT, GL_INT_VEC4, 4u,
         MTLVertexFormatInvalid},
        {"unsigned-compatible", GL_UNSIGNED_SHORT, GL_UNSIGNED_INT_VEC4, 4u,
         MTLVertexFormatInvalid},
        {"float-shader", GL_UNSIGNED_BYTE, GL_FLOAT_VEC4, 4u,
         MTLVertexFormatInvalid},
        {"unknown-source", 0xfeedu, GL_INT_VEC4, 4u,
         MTLVertexFormatInvalid},
        {"unknown-shader", GL_UNSIGNED_BYTE, 0xfeedu, 4u,
         MTLVertexFormatInvalid},
        {"zero-size", GL_UNSIGNED_BYTE, GL_INT, 0u,
         MTLVertexFormatInvalid},
        {"oversized", GL_BYTE, GL_UNSIGNED_INT_VEC4, 5u,
         MTLVertexFormatInvalid},
    };
    for (const IntegerAttribFormatCase &test : cases) {
        uint32_t actual = mglRenderCppIntegerAttribConversionFormat(
            test.source_type, test.shader_type, test.size);
        if (actual != (uint32_t)test.expected) {
            fprintf(stderr,
                    "FAIL: integer attrib format %s expected=%u actual=%u\n",
                    test.label, (uint32_t)test.expected, actual);
            return 1;
        }
    }
    printf("INTEGER_ATTRIB_FORMAT_OK\n");
    return 0;
}

static int verifyAlignStride(void) {
    /* P4.5 (item 1141): vertex stride aligned to 4. */
    if (mglRenderCppAlignVertexStrideForMetal(0) != 0 ||
        mglRenderCppAlignVertexStrideForMetal(4) != 4 ||
        mglRenderCppAlignVertexStrideForMetal(2) != 4 ||
        mglRenderCppAlignVertexStrideForMetal(9) != 12) {
        fprintf(stderr, "FAIL: align stride\\n");
        return 1;
    }
    printf("ALIGN_STRIDE_OK\\n");
    return 0;
}

static int verifyQuadTriangleCount(void) {
    /* P4.5 (item 1141): quad -> triangle index count arithmetic. */
    if (mglRenderCppQuadTriangleIndexCount(0) != 0 ||
        mglRenderCppQuadTriangleIndexCount(4) != 6 ||
        mglRenderCppQuadTriangleIndexCount(8) != 12 ||
        mglRenderCppQuadTriangleIndexCount(3) != 0 ||
        mglRenderCppQuadTriangleIndexCount(1) != 0) {
        fprintf(stderr, "FAIL: quad tri count\\n");
        return 1;
    }
    printf("QUAD_TRIANGLE_COUNT_OK\\n");
    return 0;
}

static int verifyDrawModePredicates(void) {
    /* P4.5 (item 1141): draw-mode classification predicates. */
    if (mglRenderCppDrawModeProducesPolygons(GL_TRIANGLES) != 1 ||
        mglRenderCppDrawModeProducesPolygons(GL_QUADS) != 1 ||
        mglRenderCppDrawModeProducesPolygons(GL_LINES) != 0 ||
        mglRenderCppDrawModeProducesPolygons(GL_POINTS) != 0) {
        fprintf(stderr, "FAIL: draws polygons\\n");
        return 1;
    }
    if (mglRenderCppPrimitiveModeHasDrawableSegment(GL_LINES, 1) != 0 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_LINES, 2) != 1 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_TRIANGLES, 2) != 0 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_TRIANGLES, 3) != 1 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_QUADS, 4) != 1 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_POINTS, 0) != 0 ||
        mglRenderCppPrimitiveModeHasDrawableSegment(GL_POINTS, 1) != 1) {
        fprintf(stderr, "FAIL: has drawable segment\\n");
        return 1;
    }
    printf("DRAW_MODE_PREDICATES_OK\\n");
    return 0;
}

static int verifyVertexAttribBytes(void) {
    /* P4.5 (item 1141): vertex-attribute component size + element bytes. */
    uint32_t comp = mglRenderCppVertexAttribComponentSize(GL_UNSIGNED_BYTE);
    uint32_t compf = mglRenderCppVertexAttribComponentSize(GL_FLOAT);
    uint32_t compd = mglRenderCppVertexAttribComponentSize(GL_DOUBLE);
    if (comp != 1 || compf != 4 || compd != 8 ||
        mglRenderCppVertexAttribComponentSize(0xbad) != 0) {
        fprintf(stderr, "FAIL: attrib comp size\\n");
        return 1;
    }
    if (mglRenderCppVertexAttribElementBytes(GL_FLOAT, 3) != 12 ||
        mglRenderCppVertexAttribElementBytes(GL_UNSIGNED_INT_2_10_10_10_REV, 4) != 4 ||
        mglRenderCppVertexAttribElementBytes(GL_FLOAT, 0) != 0 ||
        mglRenderCppVertexAttribElementBytes(0xbad, 3) != 0) {
        fprintf(stderr, "FAIL: attrib element bytes\\n");
        return 1;
    }
    printf("VERTEX_ATTRIB_BYTES_OK\\n");
    return 0;
}

static int verifyComputeIndexByteOffset(void) {
    /* P.5 (item 1141): base + first*stride with overflow checks. */
    uint64_t out = 0u;
    if (mglRenderCppComputeIndexByteOffset(10, 3, 4, &out) != 0 || out != 22) {
        fprintf(stderr, "FAIL: idx offset\\n");
        return 1;
    }
    if (mglRenderCppComputeIndexByteOffset(0, 0, 7, &out) != 0 || out != 0) {
        fprintf(stderr, "FAIL: idx offset zero\\n");
        return 1;
    }
    if (mglRenderCppComputeIndexByteOffset(0, 100, 0, &out) != -1) {
        fprintf(stderr, "FAIL: idx stride zero\\n");
        return 1;
    }
    if (mglRenderCppComputeIndexByteOffset(0, 3, 4, NULL) != -1) {
        fprintf(stderr, "FAIL: idx null out\\n");
        return 1;
    }
    printf("COMPUTE_INDEX_BYTE_OFFSET_OK\\n");
    return 0;
}

static int verifyComputePreparedByteOffset(void) {
    /* P4.5 (item 1141/887): prepared (Metal-side) byte-offset math.
     * GL_UNSIGNED_BYTE doubles; other types pass through; overflow caught. */
    uint64_t out = 0u;
    if (mglRenderCppComputePreparedIndexByteOffset(GL_UNSIGNED_SHORT, 100, &out) != 0 ||
        out != 100) {
        fprintf(stderr, "FAIL: prepared short pass-through\\n");
        return 1;
    }
    if (mglRenderCppComputePreparedIndexByteOffset(GL_UNSIGNED_BYTE, 100, &out) != 0 ||
        out != 200) {
        fprintf(stderr, "FAIL: prepared byte doubled\\n");
        return 1;
    }
    if (mglRenderCppComputePreparedIndexByteOffset(GL_UNSIGNED_BYTE, 0, &out) != 0 ||
        out != 0) {
        fprintf(stderr, "FAIL: prepared byte zero\\n");
        return 1;
    }
    if (mglRenderCppComputePreparedIndexByteOffset(GL_UNSIGNED_SHORT, 100, NULL) != -1) {
        fprintf(stderr, "FAIL: prepared null out\\n");
        return 1;
    }
    printf("COMPUTE_PREPARED_BYTE_OFFSET_OK\\n");
    return 0;
}

static int verifyScanIndexRange(void) {
    /* P4.5 (item 1141/887): index-range scan ignoring restart (scalar outs). */
    const uint8_t b[] = {3, 1, 0xff, 4, 9};   /* elem width 1; no restart */
    uint32_t lo = 0, hi = 0; int valid = 0;
    if (mglRenderCppScanIndexRangeIgnoringRestart(b, 1, 5, 0, 0, &lo, &hi, &valid) != 0 ||
        !valid || lo != 1 || hi != 0xff) {
        fprintf(stderr, "FAIL: scan no-restart\\n");
        return 1;
    }
    /* Restart 0xff skipped -> hi=9. */
    if (mglRenderCppScanIndexRangeIgnoringRestart(b, 1, 5, 1, 0xff, &lo, &hi, &valid) != 0 ||
        !valid || lo != 1 || hi != 9) {
        fprintf(stderr, "FAIL: scan restart\\n");
        return 1;
    }
    /* All-restart -> invalid. */
    const uint8_t all[] = {0xff, 0xff};
    if (mglRenderCppScanIndexRangeIgnoringRestart(all, 1, 2, 1, 0xff, &lo, &hi, &valid) != 0 ||
        valid != 0) {
        fprintf(stderr, "FAIL: scan all-restart\\n");
        return 1;
    }
    /* Bad args. */
    if (mglRenderCppScanIndexRangeIgnoringRestart(NULL, 1, 5, 0, 0, &lo, &hi, &valid) != -1 ||
        mglRenderCppScanIndexRangeIgnoringRestart(b, 1, 5, 0, 0, NULL, &hi, &valid) != -1) {
        fprintf(stderr, "FAIL: scan bad args\\n");
        return 1;
    }
    printf("SCAN_INDEX_RANGE_OK\\n");
    return 0;
}

static int verifyGeometryGather(void) {
    /* P4.4 (item 1141/887): geometry gather indices. */
    MGLRenderCppGeometryGatherResult r = {0};
    if (mglRenderCppGeometryGatherIndices(NULL, 2, 4, 0, 0, 3, &r) != -1) {
        fprintf(stderr, "FAIL: gather bad args\n");
        return 1;
    }

    auto verifyCase = [](
        const char *label, const void *indices, uint32_t elemWidth,
        uint32_t count, int restartEnabled, uint32_t restartIndex,
        const uint32_t *expected, uint32_t expectedCount,
        uint32_t expectedPrimitives, uint32_t expectedMax) -> int {
        MGLRenderCppGeometryGatherResult result = {0};
        int rc = mglRenderCppGeometryGatherIndices(
            (const uint8_t *)indices, elemWidth, count, restartEnabled,
            restartIndex, 3u, &result);
        bool matches = rc == 0 && result.gather &&
            result.gather_count == expectedCount &&
            result.primitive_count == expectedPrimitives &&
            result.max_index == expectedMax;
        if (matches) {
            for (uint32_t i = 0u; i < expectedCount; ++i) {
                if (result.gather[i] != expected[i]) {
                    matches = false;
                    break;
                }
            }
        }
        if (!matches) {
            fprintf(stderr,
                    "FAIL: gather %s rc=%d gc=%u prim=%u max=%u\n",
                    label, rc, (unsigned)result.gather_count,
                    (unsigned)result.primitive_count,
                    (unsigned)result.max_index);
        }
        free(result.gather);
        return matches ? 0 : 1;
    };

    const uint16_t noRestart[] = {0, 1, 2, 3, 4, 5};
    const uint32_t noRestartExpected[] = {0, 1, 2, 3, 4, 5};
    if (verifyCase("no restart", noRestart, 2u, 6u, 0, 0u,
                   noRestartExpected, 6u, 2u, 5u) != 0) {
        return 1;
    }

    const uint16_t midPrimitiveRestart[] = {
        0, 1, 0xFFFF, 2, 3, 4,
    };
    const uint32_t midPrimitiveExpected[] = {2, 3, 4};
    if (verifyCase("mid-primitive restart", midPrimitiveRestart, 2u, 6u,
                   1, 0xFFFFu, midPrimitiveExpected, 3u, 1u, 4u) != 0) {
        return 1;
    }

    const uint16_t boundaryRestarts[] = {
        0xFFFF, 0xFFFF, 0, 1, 2, 0xFFFF, 0xFFFF,
    };
    const uint32_t boundaryExpected[] = {0, 1, 2};
    if (verifyCase("leading/consecutive/trailing restart", boundaryRestarts,
                   2u, 7u, 1, 0xFFFFu, boundaryExpected, 3u, 1u, 2u) != 0) {
        return 1;
    }

    const uint16_t completeThenRestart[] = {
        0, 1, 2, 0xFFFF, 3, 4, 5,
    };
    const uint32_t completeExpected[] = {0, 1, 2, 3, 4, 5};
    if (verifyCase("complete primitive then restart", completeThenRestart,
                   2u, 7u, 1, 0xFFFFu, completeExpected, 6u, 2u, 5u) != 0) {
        return 1;
    }

    const uint32_t u32[] = {10,11,12, 13,14}; /* trailing partial dropped */
    const uint32_t u32Expected[] = {10, 11, 12};
    if (verifyCase("trailing partial", u32, 4u, 5u, 0, 0u,
                   u32Expected, 3u, 1u, 14u) != 0) {
        return 1;
    }
    const uint8_t u8[] = {0,1}; /* too short for a patch -> no primitives replace */
    if (mglRenderCppGeometryGatherIndices(u8, 1, 2, 0, 0, 3, &r) != -1) {
        fprintf(stderr, "FAIL: gather incomplete reject\n");
        return 1;
    }
    printf("GEOMETRY_GATHER_INDICES_OK\n");
    return 0;
}

static int verifyReadTextureRegionClip(void) {
    /* P4.5 (item 1141/887): readPixels region-vs-level clip. */
    MGLRenderCppReadTextureRegionClip c = {0};
    if (mglRenderCppReadTextureRegionClip(0, 0, 10, 10, 100, 100, NULL) != -1) {
        fprintf(stderr, "FAIL: clip bad args\n");
        return 1;
    }
    /* Fully inside. */
    if (mglRenderCppReadTextureRegionClip(2, 3, 10, 10, 100, 100, &c) != 0 ||
        c.copy_w != 10 || c.copy_h != 10 || c.dst_x != 0 || c.dst_y != 0 ||
        c.metal_src_x != 2 || c.metal_src_y != 87 || c.empty) {
        fprintf(stderr, "FAIL: clip inside\n");
        return 1;
    }
    /* Partially outside right/top. */
    if (mglRenderCppReadTextureRegionClip(90, 90, 20, 20, 100, 100, &c) != 0 ||
        c.copy_w != 10 || c.copy_h != 10 || c.dst_x != 0 || c.dst_y != 0 ||
        c.metal_src_x != 90 || c.metal_src_y != 0 || c.empty) {
        fprintf(stderr, "FAIL: clip partial\n");
        return 1;
    }
    /* Negative origin clips to zero. */
    if (mglRenderCppReadTextureRegionClip(-5, -5, 10, 10, 100, 100, &c) != 0 ||
        c.copy_w != 5 || c.copy_h != 5 || c.dst_x != 5 || c.dst_y != 5 ||
        c.metal_src_x != 0 || c.metal_src_y != 95 || c.empty) {
        fprintf(stderr, "FAIL: clip negative\n");
        return 1;
    }
    /* Fully outside -> empty. */
    if (mglRenderCppReadTextureRegionClip(200, 200, 10, 10, 100, 100, &c) != 0 ||
        !c.empty) {
        fprintf(stderr, "FAIL: clip empty\n");
        return 1;
    }
    printf("READ_TEXTURE_REGION_CLIP_OK\n");
    return 0;
}

static int verifyLevelDimension(void) {
    /* P4.5 (item 1141/887): mip level -> level dimension halving. */
    if (mglRenderCppMetalTextureLevelDimension(1024, 0) != 1024 ||
        mglRenderCppMetalTextureLevelDimension(1024, 1) != 512 ||
        mglRenderCppMetalTextureLevelDimension(1024, 10) != 1 ||
        mglRenderCppMetalTextureLevelDimension(1024, 99) != 1 ||
        mglRenderCppMetalTextureLevelDimension(1, 0) != 1 ||
        mglRenderCppMetalTextureLevelDimension(0, 0) != 1 ||
        mglRenderCppMetalTextureLevelDimension(64, 2) != 16 ||
        mglRenderCppMetalTextureLevelDimension(65, 1) != 32) {
        fprintf(stderr, "FAIL: level dimension\n");
        return 1;
    }
    printf("LEVEL_DIMENSION_OK\n");
    return 0;
}

static int verifyLayerPixelFormat(void) {
    if (!mglRenderCppMetalLayerPixelFormatIsSupported(
            (uint32_t)MTLPixelFormatBGRA8Unorm) ||
        !mglRenderCppMetalLayerPixelFormatIsSupported(
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB) ||
        mglRenderCppMetalLayerPixelFormatIsSupported(
            (uint32_t)MTLPixelFormatRGBA8Unorm) ||
        mglRenderCppMetalLayerPixelFormatIsSupported(
            (uint32_t)MTLPixelFormatInvalid)) {
        fprintf(stderr, "FAIL: layer pixel format support\n");
        return 1;
    }
    if (mglRenderCppSRGBPixelFormat((uint32_t)MTLPixelFormatRGBA8Unorm) !=
            (uint32_t)MTLPixelFormatRGBA8Unorm_sRGB ||
        mglRenderCppSRGBPixelFormat((uint32_t)MTLPixelFormatBGRA8Unorm) !=
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB ||
        mglRenderCppSRGBPixelFormat((uint32_t)MTLPixelFormatRGBA8Unorm_sRGB) !=
            (uint32_t)MTLPixelFormatRGBA8Unorm_sRGB ||
        mglRenderCppSRGBPixelFormat((uint32_t)MTLPixelFormatR8Unorm) !=
            (uint32_t)MTLPixelFormatR8Unorm) {
        fprintf(stderr, "FAIL: sRGB pixel format map\n");
        return 1;
    }
    if (mglRenderCppLinearPixelFormat((uint32_t)MTLPixelFormatRGBA8Unorm_sRGB) !=
            (uint32_t)MTLPixelFormatRGBA8Unorm ||
        mglRenderCppLinearPixelFormat((uint32_t)MTLPixelFormatBGRA8Unorm_sRGB) !=
            (uint32_t)MTLPixelFormatBGRA8Unorm ||
        mglRenderCppLinearPixelFormat((uint32_t)MTLPixelFormatRGBA8Unorm) !=
            (uint32_t)MTLPixelFormatRGBA8Unorm) {
        fprintf(stderr, "FAIL: linear pixel format map\n");
        return 1;
    }
    if (mglRenderCppEffectiveMTLPixelFormat(
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB, GL_SKIP_DECODE_EXT) !=
            (uint32_t)MTLPixelFormatBGRA8Unorm ||
        mglRenderCppEffectiveMTLPixelFormat(
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB, GL_DECODE_EXT) !=
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB ||
        mglRenderCppEffectiveMTLPixelFormat(
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB, 0u) !=
            (uint32_t)MTLPixelFormatBGRA8Unorm_sRGB) {
        fprintf(stderr, "FAIL: effective sRGB decode\n");
        return 1;
    }
    printf("LAYER_PIXEL_FORMAT_OK\n");
    return 0;
}

static int verifyComputeThreadgroupSize(void) {
    /* P4.5 (item 1147/887): compute threadgroup 0->1 fallback. */
    MGLRenderCppThreadgroupSize t = {0};
    if (mglRenderCppThreadgroupSize(16, 8, 1, NULL) != -1) {
        fprintf(stderr, "FAIL: tg bad args\n");
        return 1;
    }
    if (mglRenderCppThreadgroupSize(16, 8, 1, &t) != 0 ||
        t.x != 16 || t.y != 8 || t.z != 1) {
        fprintf(stderr, "FAIL: tg passthrough\n");
        return 1;
    }
    /* Zero components resolve to 1. */
    if (mglRenderCppThreadgroupSize(0, 0, 0, &t) != 0 ||
        t.x != 1 || t.y != 1 || t.z != 1) {
        fprintf(stderr, "FAIL: tg zeros\n");
        return 1;
    }
    /* Mixed. */
    if (mglRenderCppThreadgroupSize(32, 0, 4, &t) != 0 ||
        t.x != 32 || t.y != 1 || t.z != 4) {
        fprintf(stderr, "FAIL: tg mixed\n");
        return 1;
    }
    printf("COMPUTE_THREADGROUP_SIZE_OK\n");
    return 0;
}

static int verifyMetalTypeTables(void) {
    /* P4.5 (item 1141/887): GL mode/index -> Metal type numbering. */
    if (mglRenderCppMTLPrimitiveTypeForGLMode(GL_POINTS) != 0 ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_LINES) != 1 ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_LINE_STRIP) != 2 ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_TRIANGLES) != 3 ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_TRIANGLE_STRIP) != 4 ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_LINE_LOOP) != 0xFFFFFFFFu ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_QUADS) != 0xFFFFFFFFu ||
        mglRenderCppMTLPrimitiveTypeForGLMode(GL_PATCHES) != 0xFFFFFFFFu ||
        mglRenderCppMTLPrimitiveTypeForGLMode(0x1234) != 0xFFFFFFFFu) {
        fprintf(stderr, "FAIL: prim type table\n");
        return 1;
    }
    if (mglRenderCppMTLIndexTypeForGLType(GL_UNSIGNED_BYTE) != 0 ||
        mglRenderCppMTLIndexTypeForGLType(GL_UNSIGNED_SHORT) != 0 ||
        mglRenderCppMTLIndexTypeForGLType(GL_UNSIGNED_INT) != 1 ||
        mglRenderCppMTLIndexTypeForGLType(0x1234) != 0xFFFFFFFFu) {
        fprintf(stderr, "FAIL: index type table\n");
        return 1;
    }
    printf("METAL_TYPE_TABLES_OK\n");
    return 0;
}

static int verifyShaderResourceTextureTypes(void) {
    struct TextureTypeCase {
        const char *label;
        uint32_t present;
        uint32_t dimension;
        uint32_t arrayed;
        uint32_t multisampled;
        uint32_t expected;
    } cases[] = {
        {"1D", 1u, MGL_IMAGE_DIM_1D, 0u, 0u, (uint32_t)MTLTextureType1D},
        {"1DArray", 1u, MGL_IMAGE_DIM_1D, 1u, 0u, (uint32_t)MTLTextureType1DArray},
        {"2D", 1u, MGL_IMAGE_DIM_2D, 0u, 0u, (uint32_t)MTLTextureType2D},
        {"2DArray", 1u, MGL_IMAGE_DIM_2D, 1u, 0u, (uint32_t)MTLTextureType2DArray},
        {"2DMS", 1u, MGL_IMAGE_DIM_2D, 0u, 1u, (uint32_t)MTLTextureType2DMultisample},
        {"2DMSArray", 1u, MGL_IMAGE_DIM_2D, 1u, 1u, (uint32_t)MTLTextureType2DMultisampleArray},
        {"3D", 1u, MGL_IMAGE_DIM_3D, 0u, 0u, (uint32_t)MTLTextureType3D},
        {"Cube", 1u, MGL_IMAGE_DIM_CUBE, 0u, 0u, (uint32_t)MTLTextureTypeCube},
        {"CubeArray", 1u, MGL_IMAGE_DIM_CUBE, 1u, 0u, (uint32_t)MTLTextureTypeCubeArray},
        {"Buffer", 1u, MGL_IMAGE_DIM_BUFFER, 0u, 0u, (uint32_t)MTLTextureTypeTextureBuffer},
        {"invalid", 1u, UINT32_MAX, 0u, 0u, 0u},
        {"null", 0u, MGL_IMAGE_DIM_2D, 1u, 1u, 0u},
    };
    for (const TextureTypeCase &test : cases) {
        uint32_t actual = mglRenderCppTextureTypeForShaderResource(
            test.present, test.dimension, test.arrayed, test.multisampled);
        if (actual != test.expected) {
            fprintf(stderr,
                    "FAIL: shader resource texture type %s expected=%u actual=%u\n",
                    test.label, test.expected, actual);
            return 1;
        }
    }
    printf("SHADER_RESOURCE_TEXTURE_TYPE_OK\n");
    return 0;
}

static int verifyTextureCreationTargetPlans(void) {
    struct TextureTargetPlanCase {
        const char *label;
        GLenum target;
        uint32_t samples;
        MTLTextureType expected_type;
        uint32_t expected_faces;
        uint32_t expected_array;
        uint32_t expected_1d_2d;
        uint32_t expected_1d_array_2d_array;
    } cases[] = {
        {"1D", GL_TEXTURE_1D, 1u, MTLTextureType2D, 1u, 0u, 1u, 0u},
        {"renderbuffer", GL_RENDERBUFFER, 1u, MTLTextureType2D, 1u, 0u, 0u, 0u},
        {"renderbuffer MS", GL_RENDERBUFFER, 4u, MTLTextureType2DMultisample, 1u, 0u, 0u, 0u},
        {"1D array", GL_TEXTURE_1D_ARRAY, 1u, MTLTextureType2DArray, 1u, 1u, 0u, 1u},
        {"2D", GL_TEXTURE_2D, 1u, MTLTextureType2D, 1u, 0u, 0u, 0u},
        {"rectangle", GL_TEXTURE_RECTANGLE, 1u, MTLTextureType2D, 1u, 0u, 0u, 0u},
        {"2D array", GL_TEXTURE_2D_ARRAY, 1u, MTLTextureType2DArray, 1u, 1u, 0u, 0u},
        {"2D MS", GL_TEXTURE_2D_MULTISAMPLE, 4u, MTLTextureType2DMultisample, 1u, 0u, 0u, 0u},
        {"cube", GL_TEXTURE_CUBE_MAP, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube +X", GL_TEXTURE_CUBE_MAP_POSITIVE_X, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube -X", GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube +Y", GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube -Y", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube +Z", GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube -Z", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 1u, MTLTextureTypeCube, 6u, 0u, 0u, 0u},
        {"cube array", GL_TEXTURE_CUBE_MAP_ARRAY, 1u, MTLTextureTypeCubeArray, 6u, 1u, 0u, 0u},
        {"3D", GL_TEXTURE_3D, 1u, MTLTextureType3D, 1u, 0u, 0u, 0u},
        {"2D MS array", GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 4u, MTLTextureType2DMultisampleArray, 1u, 1u, 0u, 0u},
    };

    for (const TextureTargetPlanCase &test : cases) {
        MGLRenderCppTextureTargetPlan plan = {};
        if (mglRenderCppTextureTargetPlan(
                (uint32_t)test.target, test.samples, &plan) != 0 ||
            plan.texture_type != (uint32_t)test.expected_type ||
            plan.num_faces != test.expected_faces ||
            plan.is_array != test.expected_array ||
            plan.texture_1d_backed_by_2d != test.expected_1d_2d ||
            plan.texture_1d_array_backed_by_2d_array !=
                test.expected_1d_array_2d_array) {
            fprintf(stderr,
                    "FAIL: texture target plan %s type=%u faces=%u array=%u 1d2d=%u 1da2da=%u\n",
                    test.label, plan.texture_type, plan.num_faces,
                    plan.is_array, plan.texture_1d_backed_by_2d,
                    plan.texture_1d_array_backed_by_2d_array);
            return 1;
        }
    }

    MGLRenderCppTextureTargetPlan invalid = {
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    if (mglRenderCppTextureTargetPlan(UINT32_MAX, 1u, &invalid) != -1 ||
        invalid.texture_type != 0u || invalid.num_faces != 0u ||
        invalid.is_array != 0u || invalid.texture_1d_backed_by_2d != 0u ||
        invalid.texture_1d_array_backed_by_2d_array != 0u ||
        mglRenderCppTextureTargetPlan(GL_TEXTURE_2D, 1u, NULL) != -1) {
        fprintf(stderr, "FAIL: texture target plan invalid arguments\n");
        return 1;
    }

    MGLRenderCppTextureSubUploadPlan sub = {};
    if (mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_1D_ARRAY, (uint32_t)MTLTextureType2DArray, 9u,
            1u, 3u, 7u, 4u, 2u, 1u, 16u, 16u, &sub) != 0 ||
        sub.destination_base_slice != 3u || sub.destination_x != 1u ||
        sub.destination_y != 0u || sub.destination_z != 0u ||
        sub.copy_width != 4u || sub.copy_height != 1u ||
        sub.copy_depth != 1u || sub.layer_count != 2u ||
        sub.source_layer_stride != 16u) {
        fprintf(stderr, "FAIL: 1D-array texture sub-upload plan\n");
        return 1;
    }
    if (mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_3D, (uint32_t)MTLTextureType3D, 9u,
            1u, 2u, 3u, 4u, 5u, 6u, 16u, 80u, &sub) != 0 ||
        sub.destination_base_slice != 0u || sub.destination_x != 1u ||
        sub.destination_y != 2u || sub.destination_z != 3u ||
        sub.copy_width != 4u || sub.copy_height != 5u ||
        sub.copy_depth != 6u || sub.layer_count != 1u ||
        sub.source_layer_stride != 0u) {
        fprintf(stderr, "FAIL: 3D texture sub-upload plan\n");
        return 1;
    }
    if (mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D_ARRAY, (uint32_t)MTLTextureType2DArray, 9u,
            1u, 2u, 3u, 4u, 5u, 2u, 16u, 80u, &sub) != 0 ||
        sub.destination_base_slice != 3u || sub.destination_x != 1u ||
        sub.destination_y != 2u || sub.destination_z != 0u ||
        sub.copy_width != 4u || sub.copy_height != 5u ||
        sub.copy_depth != 1u || sub.layer_count != 2u ||
        sub.source_layer_stride != 80u) {
        fprintf(stderr, "FAIL: 2D-array texture sub-upload plan\n");
        return 1;
    }
    if (mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_CUBE_MAP, (uint32_t)MTLTextureTypeCube, 5u,
            1u, 2u, 0u, 4u, 5u, 1u, 16u, 80u, &sub) != 0 ||
        sub.destination_base_slice != 5u || sub.destination_y != 2u ||
        sub.layer_count != 1u || sub.source_layer_stride != 0u) {
        fprintf(stderr, "FAIL: cube texture sub-upload plan\n");
        return 1;
    }
    MGLRenderCppTextureSubUploadPlan rejected = {
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    if (mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 1u, 1u, 1u, 4u, 4u, NULL) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 0u, 1u, 1u, 4u, 4u, &rejected) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 1u, 0u, 1u, 4u, 4u, &rejected) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 1u, 1u, 0u, 4u, 4u, &rejected) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 1u, 1u, 1u, 0u, 4u, &rejected) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_2D, (uint32_t)MTLTextureType2D, 0u,
            0u, 0u, 0u, 1u, 1u, 1u, 4u, 0u, &rejected) != -1 ||
        mglRenderCppTextureSubUploadPlan(
            GL_TEXTURE_1D_ARRAY, (uint32_t)MTLTextureType2DArray, 0u,
            0u, UINT64_MAX, 0u, 1u, 2u, 1u, 4u, 4u,
            &rejected) != -1 ||
        rejected.destination_base_slice != 0u || rejected.copy_width != 0u ||
        rejected.layer_count != 0u) {
        fprintf(stderr, "FAIL: texture sub-upload plan invalid arguments\n");
        return 1;
    }

    printf("TEXTURE_CREATION_TARGET_PLAN_OK\n");
    printf("TEXTURE_SUB_UPLOAD_PLAN_OK\n");
    return 0;
}

static int verifyTextureTargetIndices(void) {
    struct TextureTargetIndexCase {
        const char *label;
        MTLTextureType type;
        int32_t expected;
    } cases[] = {
        {"1D", MTLTextureType1D, _TEXTURE_1D},
        {"1DArray", MTLTextureType1DArray, _TEXTURE_1D_ARRAY},
        {"2D", MTLTextureType2D, _TEXTURE_2D},
        {"2DMS", MTLTextureType2DMultisample, _TEXTURE_2D_MULTISAMPLE},
        {"2DArray", MTLTextureType2DArray, _TEXTURE_2D_ARRAY},
        {"2DMSArray", MTLTextureType2DMultisampleArray,
         _TEXTURE_2D_MULTISAMPLE_ARRAY},
        {"3D", MTLTextureType3D, _TEXTURE_3D},
        {"Cube", MTLTextureTypeCube, _TEXTURE_CUBE_MAP},
        {"CubeArray", MTLTextureTypeCubeArray, _TEXTURE_CUBE_MAP_ARRAY},
        {"Buffer", MTLTextureTypeTextureBuffer, _TEXTURE_BUFFER},
    };
    for (const TextureTargetIndexCase &test : cases) {
        int32_t actual = mglRenderCppTextureIndexForMetalType(
            (uint32_t)test.type);
        if (actual != test.expected) {
            fprintf(stderr,
                    "FAIL: texture target index %s expected=%d actual=%d\n",
                    test.label, test.expected, actual);
            return 1;
        }
    }
    if (mglRenderCppTextureIndexForMetalType(UINT32_MAX) != -1) {
        fprintf(stderr, "FAIL: texture target index invalid type\n");
        return 1;
    }
    printf("TEXTURE_TARGET_INDEX_OK\n");
    return 0;
}

static int verifyTextureDataKinds(void) {
    struct TextureDataKindCase {
        const char *label;
        MTLPixelFormat format;
        uint32_t expected;
    } cases[] = {
        {"R8Sint", MTLPixelFormatR8Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RG8Sint", MTLPixelFormatRG8Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RGBA8Sint", MTLPixelFormatRGBA8Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"R16Sint", MTLPixelFormatR16Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RG16Sint", MTLPixelFormatRG16Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RGBA16Sint", MTLPixelFormatRGBA16Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"R32Sint", MTLPixelFormatR32Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RG32Sint", MTLPixelFormatRG32Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"RGBA32Sint", MTLPixelFormatRGBA32Sint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT},
        {"R8Uint", MTLPixelFormatR8Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RG8Uint", MTLPixelFormatRG8Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RGBA8Uint", MTLPixelFormatRGBA8Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"R16Uint", MTLPixelFormatR16Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RG16Uint", MTLPixelFormatRG16Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RGBA16Uint", MTLPixelFormatRGBA16Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"R32Uint", MTLPixelFormatR32Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RG32Uint", MTLPixelFormatRG32Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RGBA32Uint", MTLPixelFormatRGBA32Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"RGB10A2Uint", MTLPixelFormatRGB10A2Uint, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT},
        {"Invalid", MTLPixelFormatInvalid, MGL_RENDER_CPP_TEXTURE_DATA_KIND_UNKNOWN},
        {"Depth16", MTLPixelFormatDepth16Unorm, MGL_RENDER_CPP_TEXTURE_DATA_KIND_DEPTH},
        {"Depth32", MTLPixelFormatDepth32Float, MGL_RENDER_CPP_TEXTURE_DATA_KIND_DEPTH},
        {"Depth24Stencil8", MTLPixelFormatDepth24Unorm_Stencil8,
         MGL_RENDER_CPP_TEXTURE_DATA_KIND_DEPTH},
        {"Depth32Stencil8", MTLPixelFormatDepth32Float_Stencil8,
         MGL_RENDER_CPP_TEXTURE_DATA_KIND_DEPTH},
        {"Stencil8", MTLPixelFormatStencil8, MGL_RENDER_CPP_TEXTURE_DATA_KIND_FLOAT},
        {"RGBA8Unorm", MTLPixelFormatRGBA8Unorm,
         MGL_RENDER_CPP_TEXTURE_DATA_KIND_FLOAT},
        {"RGBA16Float", MTLPixelFormatRGBA16Float,
         MGL_RENDER_CPP_TEXTURE_DATA_KIND_FLOAT},
    };
    for (const TextureDataKindCase &test : cases) {
        uint32_t actual = mglRenderCppTextureDataKindForPixelFormat(
            (uint32_t)test.format);
        if (actual != test.expected) {
            fprintf(stderr,
                    "FAIL: texture data kind %s expected=%u actual=%u\n",
                    test.label, test.expected, actual);
            return 1;
        }
    }
    if (mglRenderCppTextureDataKindForPixelFormat(UINT32_MAX) !=
        MGL_RENDER_CPP_TEXTURE_DATA_KIND_FLOAT) {
        fprintf(stderr, "FAIL: texture data kind unknown default\n");
        return 1;
    }
    printf("TEXTURE_DATA_KIND_OK\n");
    return 0;
}

static int verifyVertexAttribResolve(void) {
    /* P4.5 (item 1141/887): ARB_vertex_attrib_binding resolve. */
    MGLRenderCppVertexAttribResolve r = {0};
    if (mglRenderCppResolveVertexAttribBinding(0, 0, 0, 0, 0, 0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: vattrib bad args\n");
        return 1;
    }
    /* No table buffer -> legacy attrib values (including -1 offset). */
    if (mglRenderCppResolveVertexAttribBinding(0, 0, 0, 0, -1, 12, 0, 1, &r) != 0 ||
        r.use_binding_table || r.binding_offset != -1 || r.stride != 12 ||
        r.divisor != 1) {
        fprintf(stderr, "FAIL: vattrib legacy\n");
        return 1;
    }
    /* Table buffer with stride -> table values. */
    if (mglRenderCppResolveVertexAttribBinding(3, 1, 512, 32, -1, 12, 2, 1, &r) != 0 ||
        !r.use_binding_table || r.binding_offset != 512 || r.stride != 32 ||
        r.divisor != 2) {
        fprintf(stderr, "FAIL: vattrib table\n");
        return 1;
    }
    /* Table buffer with zero stride -> falls back to the attrib stride. */
    if (mglRenderCppResolveVertexAttribBinding(3, 1, 512, 0, -1, 12, 2, 1, &r) != 0 ||
        r.stride != 12 || r.divisor != 2) {
        fprintf(stderr, "FAIL: vattrib zero stride\n");
        return 1;
    }
    /* Out-of-range binding index -> legacy. */
    if (mglRenderCppResolveVertexAttribBinding(64, 1, 512, 32, -1, 12, 2, 1, &r) != 0 ||
        r.use_binding_table || r.binding_offset != -1 || r.stride != 12) {
        fprintf(stderr, "FAIL: vattrib oob\n");
        return 1;
    }
    printf("VERTEX_ATTRIB_RESOLVE_OK\n");
    return 0;
}

static int verifyBufferShadowUploadRange(void) {
    /* P4.5 (item 1141/887): shadow-upload range math. */
    uint64_t off = 0, len = 0;
    /* Not a GPU write target -> whole limit. */
    if (mglRenderCppBufferShadowUploadRange(0, 0, 0, 4096, &off, &len) != 0 ||
        off != 0 || len != 4096) {
        fprintf(stderr, "FAIL: shadow whole\n");
        return 1;
    }
    /* GPU write target, span inside the limit. */
    if (mglRenderCppBufferShadowUploadRange(1, 128, 512, 4096, &off, &len) != 0 ||
        off != 128 || len != 384) {
        fprintf(stderr, "FAIL: shadow span\n");
        return 1;
    }
    /* Span exceeding the limit clamps. */
    if (mglRenderCppBufferShadowUploadRange(1, 128, 8192, 4096, &off, &len) != 0 ||
        off != 128 || len != 3968) {
        fprintf(stderr, "FAIL: shadow clamp\n");
        return 1;
    }
    /* Empty / invalid spans reject. */
    if (mglRenderCppBufferShadowUploadRange(1, -1, 512, 4096, &off, &len) != -1 ||
        mglRenderCppBufferShadowUploadRange(1, 512, 512, 4096, &off, &len) != -1 ||
        mglRenderCppBufferShadowUploadRange(1, 128, 64, 4096, &off, &len) != -1 ||
        mglRenderCppBufferShadowUploadRange(0, 0, 0, 0, &off, &len) != -1 ||
        mglRenderCppBufferShadowUploadRange(0, 0, 0, 4096, NULL, NULL) != -1) {
        fprintf(stderr, "FAIL: shadow reject\n");
        return 1;
    }
    printf("BUFFER_SHADOW_UPLOAD_RANGE_OK\n");
    return 0;
}

static int verifyPolygonOffsetAndPrimCount(void) {
    /* P4.5 (item 1141/887): polygon-offset decision + prim vertex counts. */
    MGLRenderCppPolygonOffsetDecision d = {0};
    if (mglRenderCppPolygonOffsetDecision(0, 0, 0, 0, 0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: poloff bad args\n");
        return 1;
    }
    /* No ctx -> all off. */
    if (mglRenderCppPolygonOffsetDecision(0, 0, 1, GL_FILL, 1, 1, 1, &d) != 0 ||
        d.triangle_fill_mode || d.needs_polygon_mode_repair ||
        d.enable_depth_bias) {
        fprintf(stderr, "FAIL: poloff no ctx\n");
        return 1;
    }
    /* FILL + cap_fill -> bias on, no repair, fill mode. */
    if (mglRenderCppPolygonOffsetDecision(0, 1, 1, GL_FILL, 0, 0, 1, &d) != 0 ||
        d.triangle_fill_mode || d.needs_polygon_mode_repair ||
        !d.enable_depth_bias) {
        fprintf(stderr, "FAIL: poloff fill\n");
        return 1;
    }
    /* LINE -> lines fill mode + cap_line bias. */
    if (mglRenderCppPolygonOffsetDecision(0, 1, 1, GL_LINE, 0, 1, 0, &d) != 0 ||
        !d.triangle_fill_mode || d.needs_polygon_mode_repair ||
        !d.enable_depth_bias) {
        fprintf(stderr, "FAIL: poloff line\n");
        return 1;
    }
    /* POINT -> cap_point bias, no repair. */
    if (mglRenderCppPolygonOffsetDecision(0, 1, 1, GL_POINT, 1, 0, 0, &d) != 0 ||
        d.triangle_fill_mode || d.needs_polygon_mode_repair ||
        !d.enable_depth_bias) {
        fprintf(stderr, "FAIL: poloff point\n");
        return 1;
    }
    /* Invalid mode 0x9999 -> repair; the bias switch's default falls
     * through to cap_fill (the original repairs to GL_FILL first, so the
     * enable result is the same). */
    if (mglRenderCppPolygonOffsetDecision(0, 1, 1, 0x9999, 0, 0, 1, &d) != 0 ||
        !d.needs_polygon_mode_repair || !d.enable_depth_bias) {
        fprintf(stderr, "FAIL: poloff repair\n");
        return 1;
    }
    /* Non-polygon mode (GL_POINTS) -> all off even with caps. */
    if (mglRenderCppPolygonOffsetDecision(GL_POINTS, 1, 0, GL_FILL, 0, 0, 1, &d) != 0 ||
        d.enable_depth_bias || d.triangle_fill_mode) {
        fprintf(stderr, "FAIL: poloff non-polygon\n");
        return 1;
    }

    if (mglRenderCppPrimitiveVertexCountForMode(GL_TRIANGLES) != 3 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_TRIANGLE_STRIP) != 3 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_TRIANGLE_FAN) != 3 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_LINES) != 2 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_LINE_STRIP) != 2 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_LINE_LOOP) != 2 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_QUADS) != 4 ||
        mglRenderCppPrimitiveVertexCountForMode(GL_POINTS) != 1 ||
        mglRenderCppPrimitiveVertexCountForMode(0x1234) != 1) {
        fprintf(stderr, "FAIL: prim count table\n");
        return 1;
    }
    printf("POLYGON_OFFSET_AND_PRIM_COUNT_OK\n");
    return 0;
}

static int verifyScaledBlitUVsAndScissor(void) {
    /* P4.5 (item 1069/1141): scaled-blit UVs + destination scissor base. */
    MGLRenderCppScaledBlitUVs u = {0};
    if (mglRenderCppScaledBlitUVs(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, NULL) != -1) {
        fprintf(stderr, "FAIL: uvs bad args\n");
        return 1;
    }
    /* 100x100 tex, src 10..30 x 10..30, all forward -> uv 0.1..0.3,
     * uvTop (metal) = (100-30)/100 = 0.7, bottom = 0.9. */
    if (mglRenderCppScaledBlitUVs(100, 100, 10, 30, 10, 30, 1, 1, 1, 1, &u) != 0 ||
        fabs(u.uv_left - 0.1f) > 1e-6 || fabs(u.uv_right - 0.3f) > 1e-6 ||
        fabs(u.uv_top - 0.7f) > 1e-6 || fabs(u.uv_bottom - 0.9f) > 1e-6) {
        fprintf(stderr, "FAIL: uvs basic\n");
        return 1;
    }
    /* Source X flipped vs destination -> uvLeft/uvRight swap. */
    if (mglRenderCppScaledBlitUVs(100, 100, 10, 30, 10, 30, 0, 1, 1, 1, &u) != 0 ||
        fabs(u.uv_left - 0.3f) > 1e-6 || fabs(u.uv_right - 0.1f) > 1e-6) {
        fprintf(stderr, "FAIL: uvs x swap\n");
        return 1;
    }
    /* Source Y flipped -> uvTop/uvBottom swap. */
    if (mglRenderCppScaledBlitUVs(100, 100, 10, 30, 10, 30, 1, 0, 1, 1, &u) != 0 ||
        fabs(u.uv_top - 0.9f) > 1e-6 || fabs(u.uv_bottom - 0.7f) > 1e-6) {
        fprintf(stderr, "FAIL: uvs y swap\n");
        return 1;
    }
    /* Out-of-range source clamps to [0,1]. */
    if (mglRenderCppScaledBlitUVs(100, 100, -50, 150, 0, 10, 1, 1, 1, 1, &u) != 0 ||
        fabs(u.uv_left - 0.0f) > 1e-6 || fabs(u.uv_right - 1.0f) > 1e-6) {
        fprintf(stderr, "FAIL: uvs clamp\n");
        return 1;
    }

    MGLRenderCppBlitScissorRect s = {0};
    if (mglRenderCppBlitScissorRect(0, 0, 0, 0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: scissor bad args\n");
        return 1;
    }
    /* dst 10..30, metalY 70, h 20 -> 10..30 x 70..90 on 100x100. */
    if (mglRenderCppBlitScissorRect(10, 30, 70, 20, 100, 100, &s) != 0 ||
        s.x0 != 10 || s.x1 != 30 || s.y0 != 70 || s.y1 != 90) {
        fprintf(stderr, "FAIL: scissor basic\n");
        return 1;
    }
    /* Out-of-range clamps: dst -5..105 on 100 wide -> 0..100. */
    if (mglRenderCppBlitScissorRect(-5, 105, -10, 120, 100, 100, &s) != 0 ||
        s.x0 != 0 || s.x1 != 100 || s.y0 != 0 || s.y1 != 100) {
        fprintf(stderr, "FAIL: scissor clamp\n");
        return 1;
    }
    printf("SCALED_BLIT_UVS_AND_SCISSOR_OK\n");
    return 0;
}

static int verifyBlitFramebufferPlan(void) {
    /* P4.5 (item 1069/1141): glBlitFramebuffer region math + decisions. */
    MGLRenderCppBlitFramebufferPlan p = {0};
    if (mglRenderCppBlitFramebufferPlan(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: blit plan bad args\n");
        return 1;
    }
    /* Identity blit 0..10 -> 0..10 on 100x100 tex, no conversions. */
    if (mglRenderCppBlitFramebufferPlan(
            0, 10, 0, 10, 0, 10, 0, 10, 100, 100, 100, 100,
            0, 0, 0, &p) != 0 ||
        p.src_x_forward != 1 || p.dst_x_forward != 1 ||
        p.blit_needs_flip != 0 || p.needs_scaled_blit != 0 ||
        p.copy_src_x != 0 || p.copy_src_y != 0 ||
        p.copy_w != 10 || p.copy_h != 10 ||
        p.src_metal_y != 90 || p.dst_metal_y != 90 ||
        p.scaled_dst_metal_y != 90.0) {
        fprintf(stderr, "FAIL: blit plan identity\n");
        return 1;
    }
    /* Y-flipped source (src 10..0) -> needs scaled. */
    if (mglRenderCppBlitFramebufferPlan(
            0, 10, 10, 0, 0, 10, 0, 10, 100, 100, 100, 100,
            0, 0, 0, &p) != 0 ||
        p.src_y_forward != 0 || p.blit_needs_flip != 1 ||
        p.needs_scaled_blit != 1 || p.src_min_y != 0.0 ||
        p.src_max_y != 10.0) {
        fprintf(stderr, "FAIL: blit plan flip\n");
        return 1;
    }
    /* Size mismatch 10 -> 20 (scaled) with epsilon edge: 10 vs 10.000005
     * stays direct. */
    if (mglRenderCppBlitFramebufferPlan(
            0, 10, 0, 10, 0, 20, 0, 20, 100, 100, 100, 100,
            0, 0, 0, &p) != 0 ||
        p.needs_scaled_blit != 1 || p.copy_w != 10) {
        fprintf(stderr, "FAIL: blit plan scaled\n");
        return 1;
    }
    if (mglRenderCppBlitFramebufferPlan(
            0, 10, 0, 10, 0, 10.000005, 0, 10, 100, 100, 100, 100,
            0, 0, 0, &p) != 0 ||
        p.needs_scaled_blit != 0) {
        fprintf(stderr, "FAIL: blit plan epsilon\n");
        return 1;
    }
    /* Scissor forces scaled. */
    if (mglRenderCppBlitFramebufferPlan(
            0, 10, 0, 10, 0, 10, 0, 10, 100, 100, 100, 100,
            0, 0, 1, &p) != 0 || p.needs_scaled_blit != 1) {
        fprintf(stderr, "FAIL: blit plan scissor\n");
        return 1;
    }
    /* Zero-extent clipped region -> -1. */
    if (mglRenderCppBlitFramebufferPlan(
            5, 5, 0, 10, 0, 10, 0, 10, 100, 100, 100, 100,
            0, 0, 0, &p) != -1) {
        fprintf(stderr, "FAIL: blit plan empty\n");
        return 1;
    }
    printf("BLIT_FRAMEBUFFER_PLAN_OK\n");
    return 0;
}

static int verifyPackedTypeClassify(void) {
    /* P4.5 (item 1171/1116): packed-type classification. */
    MGLRenderCppIntegerPackedType p = {0};
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0, NULL) != -1) {
        fprintf(stderr, "FAIL: packed bad args\n");
        return 1;
    }
    /* GL_UNSIGNED_BYTE_3_3_2 (0x8032) -> 1B, 3 comps, widths 3/3/2,
     * shifts 5/2/0. */
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0x8032, &p) != 0 ||
        !p.is_packed || p.output_bytes != 1u || p.output_components != 3u ||
        p.bit_widths[0] != 3 || p.bit_widths[2] != 2 ||
        p.shifts[0] != 5 || p.shifts[2] != 0) {
        fprintf(stderr, "FAIL: packed 3_3_2\n");
        return 1;
    }
    /* GL_UNSIGNED_INT_2_10_10_10_REV (0x8368) -> 4B, 4 comps, rev order. */
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0x8368, &p) != 0 ||
        p.output_bytes != 4u || p.output_components != 4u ||
        p.shifts[0] != 0 || p.shifts[1] != 10 ||
        p.shifts[2] != 20 || p.shifts[3] != 30) {
        fprintf(stderr, "FAIL: packed 2_10_10_10_rev\n");
        return 1;
    }
    /* GL_UNSIGNED_SHORT_5_6_5 (0x8363) -> 2B, 3 comps. */
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0x8363, &p) != 0 ||
        p.output_bytes != 2u || p.output_components != 3u ||
        p.bit_widths[1] != 6 || p.shifts[0] != 11) {
        fprintf(stderr, "FAIL: packed 5_6_5\n");
        return 1;
    }
    /* GL_UNSIGNED_INT_8_8_8_8_REV (0x8367) -> 4B, little-endian order. */
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0x8367, &p) != 0 ||
        p.output_bytes != 4u || p.shifts[0] != 0 || p.shifts[3] != 24) {
        fprintf(stderr, "FAIL: packed 8_8_8_8_rev\n");
        return 1;
    }
    /* Unknown type -> not packed. */
    if (mglRenderCppIntegerReadbackPackedTypeClassify(0x1234, &p) != 0 ||
        p.is_packed) {
        fprintf(stderr, "FAIL: packed unknown\n");
        return 1;
    }
    printf("PACKED_TYPE_CLASSIFY_OK\n");
    return 0;
}

static int verifyIntegerReadbackSourceClassify(void) {
    /* P4.5 (item 1171/1116): integer-readback source classification. */
    MGLRenderCppIntegerReadbackSource s = {0};
    if (mglRenderCppIntegerReadbackSourceClassify(0, NULL) != -1) {
        fprintf(stderr, "FAIL: src classify bad args\n");
        return 1;
    }
    /* R8Uint = 13 -> 1 comp, 1B, unsigned. */
    if (mglRenderCppIntegerReadbackSourceClassify(13u, &s) != 0 ||
        !s.recognized || s.component_count != 1u || s.component_bytes != 1u ||
        s.source_signed) {
        fprintf(stderr, "FAIL: src r8uint\n");
        return 1;
    }
    /* RG8Sint = 34 -> 2 comps, 1B, signed. */
    if (mglRenderCppIntegerReadbackSourceClassify(34u, &s) != 0 ||
        s.component_count != 2u || s.component_bytes != 1u ||
        !s.source_signed) {
        fprintf(stderr, "FAIL: src rg8sint\n");
        return 1;
    }
    /* RGBA32Uint = 123 -> 4 comps, 4B, unsigned. */
    if (mglRenderCppIntegerReadbackSourceClassify(123u, &s) != 0 ||
        s.component_count != 4u || s.component_bytes != 4u ||
        s.source_signed || s.source_rgb10a2_uint) {
        fprintf(stderr, "FAIL: src rgba32uint\n");
        return 1;
    }
    /* RGBA16Sint = 114 -> 4 comps, 2B, signed. */
    if (mglRenderCppIntegerReadbackSourceClassify(114u, &s) != 0 ||
        s.component_count != 4u || s.component_bytes != 2u ||
        !s.source_signed) {
        fprintf(stderr, "FAIL: src rgba16sint\n");
        return 1;
    }
    /* RGB10A2Uint = 91 -> 4 comps, 4B, rgb10a2. */
    if (mglRenderCppIntegerReadbackSourceClassify(91u, &s) != 0 ||
        s.component_count != 4u || s.component_bytes != 4u ||
        !s.source_rgb10a2_uint) {
        fprintf(stderr, "FAIL: src rgb10a2\n");
        return 1;
    }
    /* R32Float = 55 -> not recognized. */
    if (mglRenderCppIntegerReadbackSourceClassify(55u, &s) != 0 ||
        s.recognized) {
        fprintf(stderr, "FAIL: src unknown\n");
        return 1;
    }
    printf("INTEGER_READBACK_SOURCE_OK\n");
    return 0;
}

static int verifyGetTexImagePlan(void) {
    /* P4.5 (item 1171/1116): mtlGetTexImage staging plan. */
    MGLRenderCppGetTexImagePlan p = {0};
    if (mglRenderCppGetTexImagePlan(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: plan bad args\n");
        return 1;
    }
    /* R32Float (55) + RED + FLOAT -> direct read; row = width*max(dst,1). */
    if (mglRenderCppGetTexImagePlan(55u, GL_RED, GL_FLOAT, 8, 4, 1, 4, 4,
                                    1, 0, 0, 0, &p) != 0 ||
        !p.direct_r32_float_read || p.use_bgra8_conversion ||
        p.row_bytes != 32 || p.image_bytes != 128 || p.total_bytes != 128) {
        fprintf(stderr, "FAIL: plan direct r32f\n");
        return 1;
    }
    /* RGBA8Unorm (70) + RGBA + UNSIGNED_BYTE -> bgra8 conv, source is
     * bgra8 family -> row = width*4. */
    if (mglRenderCppGetTexImagePlan(70u, GL_RGBA, GL_UNSIGNED_BYTE, 8, 4, 1,
                                    4, 4, 1, 0, 0, 0, &p) != 0 ||
        !p.use_bgra8_conversion || !p.source_is_bgra8 ||
        p.row_bytes != 32) {
        fprintf(stderr, "FAIL: plan rgba8unorm\n");
        return 1;
    }
    /* RGBA32Float (125) + RGBA + FLOAT -> bgra8 conv but NOT bgra8 family:
     * row = width*sourceBpp(16). */
    if (mglRenderCppGetTexImagePlan(125u, GL_RGBA, GL_FLOAT, 8, 4, 1, 16, 16,
                                    1, 0, 0, 0, &p) != 0 ||
        !p.use_bgra8_conversion || p.source_is_bgra8 ||
        p.row_bytes != 128 || p.image_bytes != 512) {
        fprintf(stderr, "FAIL: plan rgba32f pitch\n");
        return 1;
    }
    /* bytesPerRow fallback when not converting. */
    if (mglRenderCppGetTexImagePlan(125u, GL_RGBA, GL_FLOAT, 8, 4, 1, 16, 16,
                                    0, 64, 0, 0, &p) != 0 ||
        p.use_bgra8_conversion || p.row_bytes != 64) {
        fprintf(stderr, "FAIL: plan bpr fallback\n");
        return 1;
    }
    /* Private storage + depth>1 + bytesPerImage -> total = bpi*depth. */
    if (mglRenderCppGetTexImagePlan(70u, GL_RGBA, GL_UNSIGNED_BYTE, 8, 4, 6,
                                    4, 4, 0, 64, 1024, 1, &p) != 0 ||
        p.use_bgra8_conversion || p.total_bytes != 6144) {
        fprintf(stderr, "FAIL: plan private depth\n");
        return 1;
    }
    printf("GET_TEX_IMAGE_PLAN_OK\n");
    return 0;
}

static int verifyIntegerReadbackClassify(void) {
    /* P4.5 (item 1171/1116): integer-readback classification. */
    MGLRenderCppIntegerReadbackClassify c = {0};
    if (mglRenderCppIntegerReadbackClassify(0, 0, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: classify bad args\n");
        return 1;
    }
    /* RGBA8Uint (MTLPixelFormatRGBA8Uint = 73) + RGBA_INTEGER + UNSIGNED_INT
     * -> 4 comps, identity map, 4B. */
    c.output_components = 0;
    if (mglRenderCppIntegerReadbackClassify(
            73u, GL_RGBA_INTEGER, GL_UNSIGNED_INT, &c) != 0 ||
        !c.source_is_integer_texture || !c.output_is_integer_format ||
        c.output_components != 4u || c.output_component_bytes != 4u ||
        c.component_map[0] != 0 || c.component_map[1] != 1 ||
        c.component_map[2] != 2 || c.component_map[3] != 3) {
        fprintf(stderr, "FAIL: classify rgba8uint\n");
        return 1;
    }
    /* RG8Sint (34) + RG_INTEGER + BYTE -> 2 comps, 1B, map {0,1,-1,-1}. */
    if (mglRenderCppIntegerReadbackClassify(
            34u, GL_RG_INTEGER, GL_BYTE, &c) != 0 ||
        c.output_components != 2u || c.output_component_bytes != 1u ||
        c.component_map[1] != 1 || c.component_map[2] != -1) {
        fprintf(stderr, "FAIL: classify rg8sint\n");
        return 1;
    }
    /* BGRA_INTEGER -> {2,1,0,3}.  RGBA8Sint = 74. */
    if (mglRenderCppIntegerReadbackClassify(
            74u, GL_BGRA_INTEGER, GL_UNSIGNED_SHORT, &c) != 0 ||
        c.output_components != 4u || c.output_component_bytes != 2u ||
        c.component_map[0] != 2 || c.component_map[1] != 1 ||
        c.component_map[2] != 0 || c.component_map[3] != 3) {
        fprintf(stderr, "FAIL: classify bgra\n");
        return 1;
    }
    /* GREEN_INTEGER compat enum (0x8d95) -> 1 comp from channel 1.
     * RGBA16Sint = 114. */
    if (mglRenderCppIntegerReadbackClassify(
            114u, 0x8d95, GL_UNSIGNED_BYTE, &c) != 0 ||
        c.output_components != 1u || c.output_component_bytes != 1u ||
        c.component_map[0] != 1) {
        fprintf(stderr, "FAIL: classify green\n");
        return 1;
    }
    /* R32Float (55) + RED + FLOAT -> not a source integer texture. */
    if (mglRenderCppIntegerReadbackClassify(
            55u, GL_RED, GL_FLOAT, &c) != 0 || c.source_is_integer_texture) {
        fprintf(stderr, "FAIL: classify r32float\n");
        return 1;
    }
    /* RGBA8Uint + RGBA (non-integer output) -> output flag clear. */
    if (mglRenderCppIntegerReadbackClassify(
            73u, GL_RGBA, GL_UNSIGNED_BYTE, &c) != 0 ||
        c.output_is_integer_format) {
        fprintf(stderr, "FAIL: classify non-integer output\n");
        return 1;
    }
    printf("INTEGER_READBACK_CLASSIFY_OK\n");
    return 0;
}

static int verifyRasterizationIsEmpty(void) {
    /* P4.5 (item 1141/887): viewport/scissor/framebuffer intersection. */
    if (mglRenderCppRasterizationIsEmpty(0, 0, 0, 10, 100, 100, 0, 0, 0, 0, 0) != 1) {
        fprintf(stderr, "FAIL: raster empty zero viewport\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0) != 0) {
        fprintf(stderr, "FAIL: raster empty zero pass\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(200, 0, 10, 10, 100, 100, 0, 0, 0, 0, 0) != 1) {
        fprintf(stderr, "FAIL: raster empty viewport outside\n");
        return 1;
    }
    /* Partially outside: [-5,5) x [-5,5) does intersect [0,100)^2 -> not empty. */
    if (mglRenderCppRasterizationIsEmpty(-5, -5, 10, 10, 100, 100, 0, 0, 0, 0, 0) != 0) {
        fprintf(stderr, "FAIL: raster empty viewport partial\n");
        return 1;
    }
    /* Fully negative: [-15,-5) -> vx1 <= 0 -> empty. */
    if (mglRenderCppRasterizationIsEmpty(-15, -5, 10, 10, 100, 100, 0, 0, 0, 0, 0) != 1) {
        fprintf(stderr, "FAIL: raster empty viewport negative\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(10, 10, 50, 50, 100, 100, 0, 0, 0, 0, 0) != 0) {
        fprintf(stderr, "FAIL: raster empty viewport inside\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(10, 10, 50, 50, 100, 100, 1, 0, 0, 0, 0) != 1) {
        fprintf(stderr, "FAIL: raster empty zero scissor\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(10, 10, 50, 50, 100, 100, 1, 200, 200, 10, 10) != 1) {
        fprintf(stderr, "FAIL: raster empty scissor outside\n");
        return 1;
    }
    if (mglRenderCppRasterizationIsEmpty(10, 10, 50, 50, 100, 100, 1, 5, 5, 20, 20) != 0) {
        fprintf(stderr, "FAIL: raster empty scissor inside\n");
        return 1;
    }
    printf("RASTERIZATION_EMPTY_OK\n");
    return 0;
}

static int verifyNativeTESInterfaceGuards(void) {
    /* P4.5 (item 1141/887): the native-TES support decision's guard paths
     * (all return before the MTL::Function patchType read, which cannot be
     * constructed in the smoke). */
    void *fn = (void *)(uintptr_t)0x1;
    if (mglRenderCppNativeTESInterfaceSupported(
            NULL, 64, 0, 0, GL_TRIANGLES, NULL, 0, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface null function\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 0, 0, 0, GL_TRIANGLES, NULL, 0, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface no metallib\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 1, 0, GL_TRIANGLES, NULL, 0, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface point mode\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 0, 1, GL_TRIANGLES, NULL, 0, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface xfb\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 0, 0, GL_ISOLINES, NULL, 0, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface gen mode\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 0, 0, GL_TRIANGLES, fn, 0, 4) != 0) {
        fprintf(stderr, "FAIL: TES iface tcs no metallib\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 0, 0, GL_TRIANGLES, fn, 64, 0) != 0) {
        fprintf(stderr, "FAIL: TES iface tcs zero vertices\n");
        return 1;
    }
    if (mglRenderCppNativeTESInterfaceSupported(
            fn, 64, 0, 0, GL_TRIANGLES, fn, 64, 33) != 0) {
        fprintf(stderr, "FAIL: TES iface tcs vertices > 32\n");
        return 1;
    }
    printf("NATIVE_TES_INTERFACE_GUARDS_OK\n");
    return 0;
}

static int verifyTessEvalItemsAndCaptureSize(void) {
    /* P4.5 (item 1141/887): per-patch eval items + checked capture size. */
    /* patch record: edge {1,2,0,0} inside {0.5, 0.5} — 0.5=0x3800, 1.0=0x3C00,
     * 2.0=0x4000, 2.5=0x4100. */
    uint16_t rec[6] = {0x3C00, 0x4000, 0x4200, 0x4400, 0x3800, 0x3800};
    if (mglRenderCppTessEvalItemsPerPatch(rec, GL_ISOLINES, 0, 0) != 4) {
        fprintf(stderr, "FAIL: eval items isolines\n");
        return 1;
    }
    /* quad point-mode: i0=0.5->1, i1=2.5->3, spacing 0 (passthrough) -> 3. */
    rec[5] = 0x4100;
    if (mglRenderCppTessEvalItemsPerPatch(rec, GL_QUADS, 0, 1) != 3) {
        fprintf(stderr, "FAIL: eval items quad point\n");
        return 1;
    }
    /* triangle point-mode: i0=2.5 -> n=3 -> 9. */
    rec[4] = 0x4100;
    if (mglRenderCppTessEvalItemsPerPatch(rec, GL_TRIANGLES, 0, 1) != 9) {
        fprintf(stderr, "FAIL: eval items tri point\n");
        return 1;
    }
    /* non-point quad -> 0. */
    if (mglRenderCppTessEvalItemsPerPatch(rec, GL_QUADS, 0, 0) != 0) {
        fprintf(stderr, "FAIL: eval items non-point\n");
        return 1;
    }
    /* discarded (edge0 = 0) -> 0. */
    rec[0] = 0;
    if (mglRenderCppTessEvalItemsPerPatch(rec, GL_TRIANGLES, 0, 1) != 0 ||
        mglRenderCppTessEvalItemsPerPatch(NULL, GL_TRIANGLES, 0, 1) != 0) {
        fprintf(stderr, "FAIL: eval items discard/null\n");
        return 1;
    }

    uint64_t size = 0, offset = 0;
    if (mglRenderCppCheckedTessCaptureSize(3, 5, 32, 16, &size, &offset) != 0 ||
        size != 480 || offset != 0) {
        fprintf(stderr, "FAIL: capture size basic\n");
        return 1;
    }
    if (mglRenderCppCheckedTessCaptureSize(0, 5, 32, 16, &size, &offset) != -1 ||
        mglRenderCppCheckedTessCaptureSize(3, 5, 8, 16, &size, &offset) != -1 ||
        mglRenderCppCheckedTessCaptureSize(INT64_MAX, 2, 32, 16, &size, &offset) != -1 ||
        mglRenderCppCheckedTessCaptureSize(3, 5, 32, 16, NULL, NULL) != -1) {
        fprintf(stderr, "FAIL: capture size bad args\n");
        return 1;
    }
    printf("TESS_EVAL_ITEMS_AND_SIZE_OK\n");
    return 0;
}

static int verifyTessFactorDiscardPredicate(void) {
    /* P4.5 (item 1141/887): patch discard is a C++ single source shared by
     * native primitive accounting and TES compute eval-item accounting. */
    float edge[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float inside[2] = {1.0f, 2.0f};
    if (mglRenderCppTessFactorsDiscardPatch(
            GL_TRIANGLES, edge, inside)) {
        fprintf(stderr, "FAIL: tess discard triangle valid\n");
        return 1;
    }
    edge[2] = 0.0f;
    if (!mglRenderCppTessFactorsDiscardPatch(
            GL_TRIANGLES, edge, inside)) {
        fprintf(stderr, "FAIL: tess discard triangle edge\n");
        return 1;
    }
    edge[2] = 3.0f;
    inside[0] = NAN;
    if (!mglRenderCppTessFactorsDiscardPatch(
            GL_TRIANGLES, edge, inside)) {
        fprintf(stderr, "FAIL: tess discard triangle nan\n");
        return 1;
    }
    inside[0] = 1.0f;
    inside[1] = 0.0f;
    if (!mglRenderCppTessFactorsDiscardPatch(GL_QUADS, edge, inside)) {
        fprintf(stderr, "FAIL: tess discard quad inside\n");
        return 1;
    }
    /* Isolines only consume edge[0:2]; unrelated levels must not discard. */
    edge[2] = 0.0f;
    edge[3] = NAN;
    inside[0] = 0.0f;
    inside[1] = NAN;
    if (mglRenderCppTessFactorsDiscardPatch(GL_ISOLINES, edge, inside)) {
        fprintf(stderr, "FAIL: tess discard isolines unrelated levels\n");
        return 1;
    }
    edge[1] = -1.0f;
    if (!mglRenderCppTessFactorsDiscardPatch(GL_ISOLINES, edge, inside) ||
        !mglRenderCppTessFactorsDiscardPatch(
            GL_TRIANGLES, NULL, inside) ||
        !mglRenderCppTessFactorsDiscardPatch(
            GL_TRIANGLES, edge, NULL)) {
        fprintf(stderr, "FAIL: tess discard isolines/null\n");
        return 1;
    }
    printf("TESS_FACTOR_DISCARD_OK\n");
    return 0;
}

static int verifyTessRoundLevelForSpacing(void) {
    /* P4.5 (item 1141/887): GL 4.6 §11.2.2.2 subdivision-count rounding.
     * fractional_even -> next even, min 2; fractional_odd -> next odd;
     * integer/other spacing keeps ceil(level). */
    if (mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 1) != 2 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 2) != 2 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 3) != 4 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 4) != 4 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_EVEN, 5) != 6) {
        fprintf(stderr, "FAIL: round level even\n");
        return 1;
    }
    if (mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 1) != 1 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 2) != 3 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 3) != 3 ||
        mglRenderCppTessRoundLevelForSpacing(GL_FRACTIONAL_ODD, 4) != 5) {
        fprintf(stderr, "FAIL: round level odd\n");
        return 1;
    }
    if (mglRenderCppTessRoundLevelForSpacing(GL_EQUAL, 5) != 5 ||
        mglRenderCppTessRoundLevelForSpacing(0xfeed, 7) != 7) {
        fprintf(stderr, "FAIL: round level passthrough\n");
        return 1;
    }
    printf("TESS_ROUND_LEVEL_OK\n");
    return 0;
}

static int verifyCheckedProductAndXFBFieldByteSize(void) {
    /* P4.5 (item 1141/887): overflow-checked product + TES XFB field size. */
    uint64_t out = 0;
    if (mglRenderCppCheckedProduct(0, 5, &out) != 0 || out != 0) {
        fprintf(stderr, "FAIL: checked product zero\n");
        return 1;
    }
    if (mglRenderCppCheckedProduct(3, 5, &out) != 0 || out != 15) {
        fprintf(stderr, "FAIL: checked product basic\n");
        return 1;
    }
    if (mglRenderCppCheckedProduct(UINT64_MAX, 2, &out) != -1 ||
        mglRenderCppCheckedProduct(1, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: checked product overflow/bad args\n");
        return 1;
    }
    if (mglRenderCppTESXFBFieldByteSize(GL_FLOAT) != 4 ||
        mglRenderCppTESXFBFieldByteSize(GL_INT) != 4 ||
        mglRenderCppTESXFBFieldByteSize(GL_UNSIGNED_INT_VEC2) != 8 ||
        mglRenderCppTESXFBFieldByteSize(GL_FLOAT_VEC3) != 12 ||
        mglRenderCppTESXFBFieldByteSize(GL_INT_VEC4) != 16 ||
        mglRenderCppTESXFBFieldByteSize(GL_FLOAT_MAT4) != 0 ||
        mglRenderCppTESXFBFieldByteSize(0xfeed) != 0) {
        fprintf(stderr, "FAIL: xfb field byte size\n");
        return 1;
    }
    printf("CHECKED_PRODUCT_XFB_FIELD_OK\n");
    return 0;
}

static int verifyFloatUnpack(void) {
    /* P4.5 (item 1141/887): 11-bit / 10-bit unsigned float unpacking
     * (GL_UNSIGNED_INT_10F_11F_11F_REV CPU decode). */
    if (mglRenderCppFloat11ToFloat(0u) != 0.0f ||
        mglRenderCppFloat11ToFloat(0x3C0u) != 1.0f ||
        mglRenderCppFloat11ToFloat(0x440u) != 4.0f) {
        fprintf(stderr, "FAIL: float11 normalized\n");
        return 1;
    }
    if (fabsf(mglRenderCppFloat11ToFloat(0x0001u) -
              (float)(1.0 / 64.0) * (float)(1.0 / 16384.0)) > 1e-12f) {
        fprintf(stderr, "FAIL: float11 denormal\n");
        return 1;
    }
    if (!isinf(mglRenderCppFloat11ToFloat(0x7C0u)) ||
        !isnan(mglRenderCppFloat11ToFloat(0x7FFu))) {
        fprintf(stderr, "FAIL: float11 inf/nan\n");
        return 1;
    }
    if (mglRenderCppFloat10ToFloat(0u) != 0.0f ||
        mglRenderCppFloat10ToFloat(0x1E0u) != 1.0f ||
        mglRenderCppFloat10ToFloat(0x260u) != 16.0f) {
        fprintf(stderr, "FAIL: float10 normalized\n");
        return 1;
    }
    if (fabsf(mglRenderCppFloat10ToFloat(0x0001u) -
              (float)(1.0 / 32.0) * (float)(1.0 / 16384.0)) > 1e-12f) {
        fprintf(stderr, "FAIL: float10 denormal\n");
        return 1;
    }
    if (!isinf(mglRenderCppFloat10ToFloat(0x3E0u)) ||
        !isnan(mglRenderCppFloat10ToFloat(0x3FFu))) {
        fprintf(stderr, "FAIL: float10 inf/nan\n");
        return 1;
    }
    printf("FLOAT_UNPACK_OK\n");
    return 0;
}

static int verifyReadbackScalarConvert(void) {
    /* P4.5 (item 1171): CPU readback scalar converters — float->unorm8
     * round-to-nearest (0.5 up), snorm16/8 decode with INT_MIN -> -1.0. */
    if (mglRenderCppFloatToUnorm8(-0.1f) != 0u ||
        mglRenderCppFloatToUnorm8(0.0f) != 0u ||
        mglRenderCppFloatToUnorm8(0.5f) != 128u ||   /* 127.5 + 0.5 = 128 */
        mglRenderCppFloatToUnorm8(0.75f) != 191u ||  /* 191.25 + 0.5 = 191 */
        mglRenderCppFloatToUnorm8(1.0f) != 255u ||
        mglRenderCppFloatToUnorm8(2.0f) != 255u ||
        mglRenderCppFloatToUnorm8(NAN) != 0u) {
        fprintf(stderr, "FAIL: float->unorm8\n");
        return 1;
    }
    if (mglRenderCppSnorm16ToFloat(0) != 0.0f ||
        fabsf(mglRenderCppSnorm16ToFloat(32767) - 1.0f) > 1e-6f ||
        mglRenderCppSnorm16ToFloat(INT16_MIN) != -1.0f ||
        fabsf(mglRenderCppSnorm16ToFloat(-16384) - (-16384.0f / 32767.0f)) > 1e-6f) {
        fprintf(stderr, "FAIL: snorm16\n");
        return 1;
    }
    if (mglRenderCppSnorm8ToFloat(0) != 0.0f ||
        fabsf(mglRenderCppSnorm8ToFloat(127) - 1.0f) > 1e-6f ||
        mglRenderCppSnorm8ToFloat(INT8_MIN) != -1.0f ||
        fabsf(mglRenderCppSnorm8ToFloat(-64) - (-64.0f / 127.0f)) > 1e-6f) {
        fprintf(stderr, "FAIL: snorm8\n");
        return 1;
    }
    /* Readback bytes-per-pixel table (MTLPixelFormat ABI values). */
    if (mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRGBA32Float) != 16u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatR8Unorm) != 1u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatR16Unorm) != 2u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRG8Unorm) != 2u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatABGR4Unorm) != 2u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRG32Float) != 8u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRGBA16Unorm) != 8u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatR8Sint) != 1u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRG8Uint) != 2u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRGBA8Snorm) != 4u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)MTLPixelFormatRGBA8Unorm) != 4u ||
        mglRenderCppReadbackBytesPerPixel((uint32_t)0x7FFFFFFFu) != 4u) { /* unknown -> 4 */
        fprintf(stderr, "FAIL: readback bpp table\n");
        return 1;
    }
    /* Format classification tables. */
    if (!mglRenderCppReadbackFormatIsBGRA8Compatible((uint32_t)MTLPixelFormatRGBA8Unorm) ||
        !mglRenderCppReadbackFormatIsBGRA8Compatible((uint32_t)MTLPixelFormatRGBA32Float) ||
        !mglRenderCppReadbackFormatIsBGRA8Compatible((uint32_t)MTLPixelFormatRGBA8Uint) ||
        mglRenderCppReadbackFormatIsBGRA8Compatible((uint32_t)MTLPixelFormatDepth32Float) ||
        mglRenderCppReadbackFormatIsBGRA8Compatible((uint32_t)0x7FFFFFFFu)) {
        fprintf(stderr, "FAIL: bgra8-compatible table\n");
        return 1;
    }
    if (!mglRenderCppPixelFormatIsIntegerColor((uint32_t)MTLPixelFormatRGBA8Uint) ||
        !mglRenderCppPixelFormatIsIntegerColor((uint32_t)MTLPixelFormatR32Sint) ||
        !mglRenderCppPixelFormatIsIntegerColor((uint32_t)MTLPixelFormatRGB10A2Uint) ||
        mglRenderCppPixelFormatIsIntegerColor((uint32_t)MTLPixelFormatRGBA8Unorm) ||
        mglRenderCppPixelFormatIsIntegerColor((uint32_t)MTLPixelFormatDepth32Float)) {
        fprintf(stderr, "FAIL: integer-color table\n");
        return 1;
    }
    if (!mglRenderCppPixelFormatIsSignedIntegerColor((uint32_t)MTLPixelFormatRGBA32Sint) ||
        !mglRenderCppPixelFormatIsSignedIntegerColor((uint32_t)MTLPixelFormatR8Sint) ||
        mglRenderCppPixelFormatIsSignedIntegerColor((uint32_t)MTLPixelFormatRGBA8Uint) ||
        mglRenderCppPixelFormatIsSignedIntegerColor((uint32_t)MTLPixelFormatRGBA8Unorm)) {
        fprintf(stderr, "FAIL: signed-integer-color table\n");
        return 1;
    }
    /* GL BGRA8 -> BGRA8-compatible format row copy (incl. flipY). */
    {
        uint8_t src[2][4] = {{10, 20, 30, 40}, {50, 60, 70, 80}};
        uint8_t dst[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        /* RGBA8Unorm: BGRA8 source maps to RGBA order (B,G,R,A -> R,G,B,A). */
        if (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 4u, dst, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatRGBA8Unorm, 0) != 1 ||
            dst[0][0] != 30 || dst[0][1] != 20 || dst[0][2] != 10 ||
            dst[0][3] != 40 ||
            dst[1][0] != 70 || dst[1][1] != 60 || dst[1][2] != 50 ||
            dst[1][3] != 80) {
            fprintf(stderr, "FAIL: bgra8->rgba8 rows\n");
            return 1;
        }
        /* flipY reverses the row order. */
        memset(dst, 0, sizeof(dst));
        if (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 4u, dst, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatRGBA8Unorm, 1) != 1 ||
            dst[0][0] != 70 || dst[1][0] != 30) {
            fprintf(stderr, "FAIL: bgra8->rgba8 flipY\n");
            return 1;
        }
        /* BGRA8Unorm keeps the byte order. */
        memset(dst, 0, sizeof(dst));
        if (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 4u, dst, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatBGRA8Unorm, 0) != 1 ||
            dst[0][0] != 10 || dst[0][1] != 20 || dst[0][2] != 30 ||
            dst[0][3] != 40) {
            fprintf(stderr, "FAIL: bgra8->bgra8 rows\n");
            return 1;
        }
        /* RGB10A2Unorm: 8-bit values expand to 10/2-bit packed little-endian. */
        memset(dst, 0, sizeof(dst));
        if (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 4u, dst, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm, 0) != 1) {
            fprintf(stderr, "FAIL: bgra8->rgb10a2 rc\n");
            return 1;
        }
        /* r=30 -> 30*1023/255 = 120.3 -> 120; g=20 -> 80; b=10 -> 40;
         * a=40 -> 40*3/255 = 0.47 -> 0.  Packed = 120 | 80<<10 | 40<<20. */
        uint32_t packedExpected =
            120u | (80u << 10) | (40u << 20) | (0u << 30);
        uint32_t packedGot =
            (uint32_t)dst[0][0] | ((uint32_t)dst[0][1] << 8) |
            ((uint32_t)dst[0][2] << 16) | ((uint32_t)dst[0][3] << 24);
        if (packedGot != packedExpected) {
            fprintf(stderr, "FAIL: bgra8->rgb10a2 packed 0x%x != 0x%x\n",
                    packedGot, packedExpected);
            return 1;
        }
        /* Bad args / unsupported format. */
        if (mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                NULL, 4u, dst, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm, 0) != 0 ||
            mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 3u, dst, 4u, 1u, 1u,     /* srcBytesPerRow too small */
                (uint32_t)MTLPixelFormatRGBA8Unorm, 0) != 0 ||
            mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
                src, 4u, dst, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatDepth32Float, 0) != 0) {
            fprintf(stderr, "FAIL: bgra8 rows bad args\n");
            return 1;
        }
    }
    /* Metal texture bytes -> GL BGRA8 (decode + optional flipY). */
    {
        uint8_t rgba[2][4] = {{10, 20, 30, 40}, {50, 60, 70, 80}};
        uint8_t dst[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        /* RGBA8Unorm: R,G,B,A -> B,G,R,A. */
        mglRenderCppCopyTextureBytesToBGRA8(
            rgba, 4u, dst, 4u, 1u, 2u,
            (uint32_t)MTLPixelFormatRGBA8Unorm, 0);
        if (dst[0][0] != 30 || dst[0][1] != 20 || dst[0][2] != 10 ||
            dst[0][3] != 40 ||
            dst[1][0] != 70 || dst[1][1] != 60 || dst[1][2] != 50 ||
            dst[1][3] != 80) {
            fprintf(stderr, "FAIL: rgba8->bgra8\n");
            return 1;
        }
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            rgba, 4u, dst, 4u, 1u, 2u,
            (uint32_t)MTLPixelFormatRGBA8Unorm, 1);
        if (dst[0][0] != 70 || dst[1][0] != 30) {
            fprintf(stderr, "FAIL: rgba8->bgra8 flipY\n");
            return 1;
        }
        /* BGRA8Unorm is not a decoded source: memcpy 4 bytes as-is. */
        uint8_t bgra[4] = {11, 22, 33, 44};
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            bgra, 4u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatBGRA8Unorm, 0);
        if (dst[0][0] != 11 || dst[0][1] != 22 || dst[0][2] != 33 ||
            dst[0][3] != 44) {
            fprintf(stderr, "FAIL: bgra8 passthrough\n");
            return 1;
        }
        /* R8Unorm: R -> B channel, A=255. */
        uint8_t r8 = 90;
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            &r8, 1u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatR8Unorm, 0);
        if (dst[0][0] != 0 || dst[0][1] != 0 || dst[0][2] != 90 ||
            dst[0][3] != 255) {
            fprintf(stderr, "FAIL: r8->bgra8\n");
            return 1;
        }
        /* RGB10A2Unorm: R=1023, G=0, B=0, A=3 -> BGRA (0,0,255,255). */
        uint32_t rgb10 = 1023u | (3u << 30);
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            &rgb10, 4u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatRGB10A2Unorm, 0);
        if (dst[0][0] != 0 || dst[0][1] != 0 || dst[0][2] != 255 ||
            dst[0][3] != 255) {
            fprintf(stderr, "FAIL: rgb10a2->bgra8\n");
            return 1;
        }
        /* BGR5A1Unorm: B=31, G=31, R=0, A=1 -> (255,255,0,255). */
        uint16_t bgr5 = 31u | (31u << 5) | (1u << 15);
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            &bgr5, 2u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatBGR5A1Unorm, 0);
        if (dst[0][0] != 255 || dst[0][1] != 255 || dst[0][2] != 0 ||
            dst[0][3] != 255) {
            fprintf(stderr, "FAIL: bgr5a1->bgra8\n");
            return 1;
        }
        /* RGBA32Float: (1,0,0,1) -> BGRA (0,0,255,255). */
        float rgba32[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            rgba32, 16u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatRGBA32Float, 0);
        if (dst[0][0] != 0 || dst[0][1] != 0 || dst[0][2] != 255 ||
            dst[0][3] != 255) {
            fprintf(stderr, "FAIL: rgba32f->bgra8\n");
            return 1;
        }
        /* RGB9E5: exp=15, mant_r=256 -> 256 * 2^(15-24) = 0.5 -> R=128. */
        uint32_t rgb9e5 = 256u | (15u << 27);
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyTextureBytesToBGRA8(
            &rgb9e5, 4u, dst, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatRGB9E5Float, 0);
        if (dst[0][0] != 0 || dst[0][1] != 0 || dst[0][2] != 128 ||
            dst[0][3] != 255) {
            fprintf(stderr, "FAIL: rgb9e5->bgra8 %u %u %u %u\n",
                    dst[0][0], dst[0][1], dst[0][2], dst[0][3]);
            return 1;
        }
        /* Bad args leave dest unchanged. */
        uint8_t sentinel[4] = {1, 2, 3, 4};
        mglRenderCppCopyTextureBytesToBGRA8(
            NULL, 4u, sentinel, 4u, 1u, 1u,
            (uint32_t)MTLPixelFormatRGBA8Unorm, 0);
        mglRenderCppCopyTextureBytesToBGRA8(
            rgba, 4u, sentinel, 4u, 0u, 1u,
            (uint32_t)MTLPixelFormatRGBA8Unorm, 0);
        if (sentinel[0] != 1 || sentinel[1] != 2 || sentinel[2] != 3 ||
            sentinel[3] != 4) {
            fprintf(stderr, "FAIL: texture->bgra8 bad args\n");
            return 1;
        }
    }
    /* GL readback type-accept table + SNORM8 direct path. */
    if (!mglRenderCppReadbackGLTypeAccepted((uint32_t)GL_UNSIGNED_BYTE) ||
        !mglRenderCppReadbackGLTypeAccepted((uint32_t)GL_FLOAT) ||
        !mglRenderCppReadbackGLTypeAccepted(
            (uint32_t)GL_UNSIGNED_INT_2_10_10_10_REV) ||
        mglRenderCppReadbackGLTypeAccepted((uint32_t)GL_DEPTH_COMPONENT) ||
        mglRenderCppReadbackGLTypeAccepted(0u)) {
        fprintf(stderr, "FAIL: readback type-accept\n");
        return 1;
    }
    {
        int8_t r8 = 127;
        float fdst = 0.0f;
        if (mglRenderCppCopySnorm8TextureBytesToGL(
                &r8, 1u, &fdst, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR8Snorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(fdst - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: snorm8 r8 -> float\n");
            return 1;
        }
        int8_t bdst = 0;
        if (mglRenderCppCopySnorm8TextureBytesToGL(
                &r8, 1u, &bdst, 1u, 1u, 1u,
                (uint32_t)MTLPixelFormatR8Snorm,
                (uint32_t)GL_RED, (uint32_t)GL_BYTE, 0) != 1 ||
            bdst != 127) {
            fprintf(stderr, "FAIL: snorm8 r8 -> byte\n");
            return 1;
        }
        int8_t rgba[2][4] = {{127, 0, -128, 127}, {0, 127, 0, 127}};
        float bgra[2][4];
        memset(bgra, 0, sizeof(bgra));
        if (mglRenderCppCopySnorm8TextureBytesToGL(
                rgba, 4u, bgra, 16u, 1u, 2u,
                (uint32_t)MTLPixelFormatRGBA8Snorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(bgra[0][0] + 1.0f) > 1e-6f ||
            fabsf(bgra[0][1]) > 1e-6f ||
            fabsf(bgra[0][2] - 1.0f) > 1e-6f ||
            fabsf(bgra[0][3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: snorm8 rgba -> bgra float\n");
            return 1;
        }
        memset(bgra, 0, sizeof(bgra));
        if (mglRenderCppCopySnorm8TextureBytesToGL(
                rgba, 4u, bgra, 16u, 1u, 2u,
                (uint32_t)MTLPixelFormatRGBA8Snorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 1) != 1 ||
            fabsf(bgra[0][2]) > 1e-6f ||
            fabsf(bgra[1][2] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: snorm8 flipY\n");
            return 1;
        }
        if (mglRenderCppCopySnorm8TextureBytesToGL(
                rgba, 4u, bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopySnorm8TextureBytesToGL(
                NULL, 4u, bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Snorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 0) {
            fprintf(stderr, "FAIL: snorm8 bad args\n");
            return 1;
        }
    }
    /* RGB10A2Unorm direct path (bypass BGRA8). */
    {
        uint32_t src = 1023u | (3u << 30); /* R=1023, G=0, B=0, A=3 */
        float rgba[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, rgba, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(rgba[0] - 1.0f) > 1e-6f || fabsf(rgba[1]) > 1e-6f ||
            fabsf(rgba[2]) > 1e-6f || fabsf(rgba[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rgb10a2 -> rgba float\n");
            return 1;
        }
        float bgra[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(bgra[0]) > 1e-6f || fabsf(bgra[1]) > 1e-6f ||
            fabsf(bgra[2] - 1.0f) > 1e-6f || fabsf(bgra[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rgb10a2 -> bgra float\n");
            return 1;
        }
        uint32_t rev = 0;
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, &rev, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_INT_2_10_10_10_REV,
                0) != 1 ||
            rev != src) {
            fprintf(stderr, "FAIL: rgb10a2 -> 2_10_10_10_REV 0x%x\n", rev);
            return 1;
        }
        uint32_t msb = 0;
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, &msb, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_INT_10_10_10_2,
                0) != 1 ||
            msb != ((1023u << 22) | 3u)) {
            fprintf(stderr, "FAIL: rgb10a2 -> 10_10_10_2 0x%x\n", msb);
            return 1;
        }
        uint8_t ub[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, ub, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            ub[0] != 255 || ub[1] != 0 || ub[2] != 0 || ub[3] != 255) {
            fprintf(stderr, "FAIL: rgb10a2 -> rgba8\n");
            return 1;
        }
        uint32_t rows[2] = {1023u | (3u << 30), 0u};
        float flip[2][4];
        memset(flip, 0, sizeof(flip));
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                rows, 4u, flip, 16u, 1u, 2u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 1) != 1 ||
            fabsf(flip[0][0]) > 1e-6f ||
            fabsf(flip[1][0] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rgb10a2 flipY\n");
            return 1;
        }
        if (mglRenderCppCopyRGB10A2TextureBytesToGL(
                &src, 4u, rgba, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopyRGB10A2TextureBytesToGL(
                NULL, 4u, rgba, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGB10A2Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 0) {
            fprintf(stderr, "FAIL: rgb10a2 bad args\n");
            return 1;
        }
    }
    /* RG11B10Float direct path (bypass BGRA8). */
    {
        uint32_t src = 0x3C0u; /* R=1.0 float11, G=0, B=0 */
        uint32_t ident = 0;
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, &ident, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_INT_10F_11F_11F_REV,
                0) != 1 ||
            ident != src) {
            fprintf(stderr, "FAIL: rg11b10 -> 10F_11F_11F_REV memcpy\n");
            return 1;
        }
        float red = 0.0f;
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rg11b10 -> red float\n");
            return 1;
        }
        float bgra[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(bgra[0]) > 1e-6f || fabsf(bgra[1]) > 1e-6f ||
            fabsf(bgra[2] - 1.0f) > 1e-6f || fabsf(bgra[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rg11b10 -> bgra float\n");
            return 1;
        }
        uint8_t ub[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, ub, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            ub[0] != 255 || ub[1] != 0 || ub[2] != 0 || ub[3] != 255) {
            fprintf(stderr, "FAIL: rg11b10 -> rgba8\n");
            return 1;
        }
        uint32_t bgr_pack = 0xffffffffu;
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, &bgr_pack, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_BGR, (uint32_t)GL_UNSIGNED_INT_10F_11F_11F_REV,
                0) != 1 ||
            bgr_pack != (0x1E0u << 22)) {
            fprintf(stderr, "FAIL: rg11b10 -> bgr 10F_11F_11F_REV 0x%x\n",
                    bgr_pack);
            return 1;
        }
        uint32_t rows[2] = {0x3C0u, 0u};
        float flip[2];
        memset(flip, 0, sizeof(flip));
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                rows, 4u, flip, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 1) != 1 ||
            fabsf(flip[0]) > 1e-6f ||
            fabsf(flip[1] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rg11b10 flipY\n");
            return 1;
        }
        if (mglRenderCppCopyRG11B10TextureBytesToGL(
                &src, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopyRG11B10TextureBytesToGL(
                NULL, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRG11B10Float,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0) {
            fprintf(stderr, "FAIL: rg11b10 bad args\n");
            return 1;
        }
    }
    /* 16/32-bit direct path (bypass BGRA8). */
    {
        uint16_t r16 = 65535;
        float red = 0.0f;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &r16, 2u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r16unorm -> red float\n");
            return 1;
        }
        uint8_t ub = 0;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &r16, 2u, &ub, 1u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            ub != 255) {
            fprintf(stderr, "FAIL: r16unorm -> red u8\n");
            return 1;
        }
        float rgba[4] = {0, 0, 0, 0};
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &r16, 2u, rgba, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(rgba[0] - 1.0f) > 1e-6f || fabsf(rgba[1] - 1.0f) > 1e-6f ||
            fabsf(rgba[2] - 1.0f) > 1e-6f || fabsf(rgba[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r16unorm -> rgba replicate\n");
            return 1;
        }
        uint16_t rgba16[4] = {65535, 0, 0, 65535};
        float bgra[4] = {0, 0, 0, 0};
        if (mglRenderCppCopy16or32TextureBytesToGL(
                rgba16, 8u, bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA16Unorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(bgra[0]) > 1e-6f || fabsf(bgra[1]) > 1e-6f ||
            fabsf(bgra[2] - 1.0f) > 1e-6f || fabsf(bgra[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rgba16unorm -> bgra float\n");
            return 1;
        }
        uint16_t packed565 = 0;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                rgba16, 8u, &packed565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA16Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 1 ||
            packed565 != 0xF800u) {
            fprintf(stderr, "FAIL: rgba16unorm -> 565 0x%x\n", packed565);
            return 1;
        }
        uint16_t h1 = 0x3C00; /* half 1.0 */
        red = 0.0f;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &h1, 2u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Float,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r16float -> red float\n");
            return 1;
        }
        float f32 = 1.0f;
        red = 0.0f;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &f32, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR32Float,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r32float -> red float\n");
            return 1;
        }
        int16_t s16 = 32767;
        red = 0.0f;
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &s16, 2u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Snorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r16snorm -> red float\n");
            return 1;
        }
        uint16_t rows[2] = {65535, 0};
        float flip[2];
        memset(flip, 0, sizeof(flip));
        if (mglRenderCppCopy16or32TextureBytesToGL(
                rows, 2u, flip, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 1) != 1 ||
            fabsf(flip[0]) > 1e-6f ||
            fabsf(flip[1] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: r16unorm flipY\n");
            return 1;
        }
        if (mglRenderCppCopy16or32TextureBytesToGL(
                &r16, 2u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopy16or32TextureBytesToGL(
                NULL, 2u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0) {
            fprintf(stderr, "FAIL: 16or32 bad args\n");
            return 1;
        }
    }
    /* BGRA8/RGBA8 UNORM scalar readback. */
    {
        uint8_t bgra[4] = {0, 0, 255, 255}; /* B,G,R,A → logical R=255 */
        float red = 0.0f;
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: bgra8 -> red float\n");
            return 1;
        }
        float rgba[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, rgba, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(rgba[0] - 1.0f) > 1e-6f || fabsf(rgba[1]) > 1e-6f ||
            fabsf(rgba[2]) > 1e-6f || fabsf(rgba[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: bgra8 -> rgba float\n");
            return 1;
        }
        float out_bgra[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, out_bgra, 16u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(out_bgra[0]) > 1e-6f || fabsf(out_bgra[1]) > 1e-6f ||
            fabsf(out_bgra[2] - 1.0f) > 1e-6f ||
            fabsf(out_bgra[3] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: bgra8 -> bgra float\n");
            return 1;
        }
        uint8_t rgba8[4] = {255, 0, 0, 255};
        red = 0.0f;
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                rgba8, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 1 ||
            fabsf(red - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: rgba8 -> red float\n");
            return 1;
        }
        uint16_t us = 0;
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &us, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_UNSIGNED_SHORT, 0) != 1 ||
            us != 65535) {
            fprintf(stderr, "FAIL: bgra8 -> red u16 %u\n", us);
            return 1;
        }
        int8_t sb = 0;
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &sb, 1u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_BYTE, 0) != 1 ||
            sb != 127) {
            fprintf(stderr, "FAIL: bgra8 -> red snorm8 %d\n", sb);
            return 1;
        }
        uint16_t half = 0;
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &half, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_HALF_FLOAT, 0) != 1 ||
            half != 0x3C00) {
            fprintf(stderr, "FAIL: bgra8 -> red half 0x%x\n", half);
            return 1;
        }
        uint8_t rows[8] = {0, 0, 255, 255, 0, 0, 0, 255};
        float flip[2];
        memset(flip, 0, sizeof(flip));
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                rows, 4u, flip, 4u, 1u, 2u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 1) != 1 ||
            fabsf(flip[0]) > 1e-6f ||
            fabsf(flip[1] - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: unorm8 scalar flipY\n");
            return 1;
        }
        if (mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_UNSIGNED_BYTE, 0) != 0 ||
            mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                bgra, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
                NULL, 4u, &red, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_FLOAT, 0) != 0) {
            fprintf(stderr, "FAIL: unorm8 scalar bad args\n");
            return 1;
        }
    }
    /* BGRA8/RGBA8 UNORM packed readback. */
    {
        uint8_t bgra[4] = {0, 0, 255, 255}; /* B,G,R,A → logical R=255 */
        uint16_t rgb565 = 0;
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &rgb565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 1 ||
            rgb565 != 0xF800u) {
            fprintf(stderr, "FAIL: bgra8 -> rgb 565 0x%x\n", rgb565);
            return 1;
        }
        uint16_t bgr565 = 0;
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &bgr565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_BGR, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 1 ||
            bgr565 != 0x001Fu) {
            fprintf(stderr, "FAIL: bgra8 -> bgr 565 0x%x\n", bgr565);
            return 1;
        }
        uint32_t rev8888 = 0;
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &rev8888, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_INT_8_8_8_8_REV,
                0) != 1 ||
            rev8888 != 0xFF0000FFu) {
            fprintf(stderr, "FAIL: bgra8 -> 8888_REV 0x%x\n", rev8888);
            return 1;
        }
        uint32_t rev210 = 0;
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &rev210, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_INT_2_10_10_10_REV,
                0) != 1 ||
            rev210 != (1023u | (3u << 30))) {
            fprintf(stderr, "FAIL: bgra8 -> 2_10_10_10_REV 0x%x\n", rev210);
            return 1;
        }
        uint8_t rgba8[4] = {255, 0, 0, 255};
        uint16_t rgba565 = 0;
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                rgba8, 4u, &rgba565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 1 ||
            rgba565 != 0xF800u) {
            fprintf(stderr, "FAIL: rgba8 -> rgb 565 0x%x\n", rgba565);
            return 1;
        }
        uint8_t rows[8] = {0, 0, 255, 255, 0, 0, 0, 255};
        uint16_t flip[2] = {0, 0};
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                rows, 4u, flip, 2u, 1u, 2u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                1) != 1 ||
            flip[0] != 0u || flip[1] != 0xF800u) {
            fprintf(stderr, "FAIL: unorm8 packed flipY 0x%x 0x%x\n",
                    flip[0], flip[1]);
            return 1;
        }
        if (mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &rgb565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_FLOAT, 0) != 0 ||
            mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                bgra, 4u, &rgb565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 0 ||
            mglRenderCppCopyUnorm8PackedTextureBytesToGL(
                NULL, 4u, &rgb565, 2u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_SHORT_5_6_5,
                0) != 0) {
            fprintf(stderr, "FAIL: unorm8 packed bad args\n");
            return 1;
        }
    }
    /* BGRA8/RGBA8 UNSIGNED_BYTE channel swizzle. */
    {
        uint8_t bgra[4] = {0, 0, 255, 255}; /* B,G,R,A → logical R=255 */
        uint8_t rgba[4] = {0, 0, 0, 0};
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, rgba, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            rgba[0] != 255 || rgba[1] != 0 || rgba[2] != 0 || rgba[3] != 255) {
            fprintf(stderr, "FAIL: bgra8 -> rgba8 swizzle\n");
            return 1;
        }
        uint8_t out_bgra[4] = {1, 1, 1, 1};
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, out_bgra, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_BGRA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            out_bgra[0] != 0 || out_bgra[1] != 0 ||
            out_bgra[2] != 255 || out_bgra[3] != 255) {
            fprintf(stderr, "FAIL: bgra8 -> bgra8 swizzle\n");
            return 1;
        }
        uint8_t rgb[3] = {0, 0, 0};
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, rgb, 3u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGB, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            rgb[0] != 255 || rgb[1] != 0 || rgb[2] != 0) {
            fprintf(stderr, "FAIL: bgra8 -> rgb8 swizzle\n");
            return 1;
        }
        uint8_t red = 0, blue = 255;
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, &red, 1u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            red != 255 ||
            mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, &blue, 1u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_BLUE, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            blue != 0) {
            fprintf(stderr, "FAIL: bgra8 -> red/blue swizzle\n");
            return 1;
        }
        uint8_t rgba8[4] = {255, 0, 0, 255};
        uint8_t bgr[3] = {0, 0, 0};
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                rgba8, 4u, bgr, 3u, 1u, 1u,
                (uint32_t)MTLPixelFormatRGBA8Unorm,
                (uint32_t)GL_BGR, (uint32_t)GL_UNSIGNED_BYTE, 0) != 1 ||
            bgr[0] != 0 || bgr[1] != 0 || bgr[2] != 255) {
            fprintf(stderr, "FAIL: rgba8 -> bgr8 swizzle\n");
            return 1;
        }
        uint8_t rows[8] = {0, 0, 255, 255, 0, 0, 0, 255};
        uint8_t flip[2] = {1, 1};
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                rows, 4u, flip, 1u, 1u, 2u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RED, (uint32_t)GL_UNSIGNED_BYTE, 1) != 1 ||
            flip[0] != 0 || flip[1] != 255) {
            fprintf(stderr, "FAIL: unorm8 swizzle flipY\n");
            return 1;
        }
        if (mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                bgra, 4u, rgba, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatR16Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 0 ||
            mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
                NULL, 4u, rgba, 4u, 1u, 1u,
                (uint32_t)MTLPixelFormatBGRA8Unorm,
                (uint32_t)GL_RGBA, (uint32_t)GL_UNSIGNED_BYTE, 0) != 0) {
            fprintf(stderr, "FAIL: unorm8 swizzle bad args\n");
            return 1;
        }
    }
    /* Packed row copy + depth16/float convert. */
    {
        uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint8_t dst[8] = {0};
        mglRenderCppCopyRows(src, 4u, dst, 4u, 4u, 2u, 0);
        if (memcmp(dst, src, 8) != 0) {
            fprintf(stderr, "FAIL: copy rows identity\n");
            return 1;
        }
        memset(dst, 0, sizeof(dst));
        mglRenderCppCopyRows(src, 4u, dst, 4u, 4u, 2u, 1);
        if (memcmp(dst, src + 4, 4) != 0 || memcmp(dst + 4, src, 4) != 0) {
            fprintf(stderr, "FAIL: copy rows flipY\n");
            return 1;
        }
        uint8_t sentinel[4] = {9, 9, 9, 9};
        mglRenderCppCopyRows(NULL, 4u, sentinel, 4u, 4u, 1u, 0);
        if (sentinel[0] != 9) {
            fprintf(stderr, "FAIL: copy rows bad args\n");
            return 1;
        }
        uint16_t d16[2] = {0, 65535};
        float df[2] = {1.0f, 1.0f};
        mglRenderCppCopyDepthTextureBytesToFloat(
            d16, 2u, df, 4u, 1u, 2u, 2u, 1, 1);
        if (fabsf(df[0] - 1.0f) > 1e-6f || fabsf(df[1]) > 1e-6f) {
            fprintf(stderr, "FAIL: depth16 -> float flipY\n");
            return 1;
        }
        float srcf[2] = {0.25f, 0.75f};
        float dstf[2] = {0, 0};
        mglRenderCppCopyDepthTextureBytesToFloat(
            srcf, 4u, dstf, 4u, 1u, 2u, 4u, 0, 0);
        if (fabsf(dstf[0] - 0.25f) > 1e-6f ||
            fabsf(dstf[1] - 0.75f) > 1e-6f) {
            fprintf(stderr, "FAIL: depth float rows\n");
            return 1;
        }
    }
    printf("READBACK_SCALAR_CONVERT_OK\n");
    return 0;
}

static int verifyTessControlPointFormat(void) {
    /* P4.5 (item 1141/887): GL type -> MTLVertexFormat for TES control
     * points.  Assert against the ObjC SDK constants (no magic numbers). */
    if (mglRenderCppTessControlPointFormat(GL_FLOAT) !=
            (uint32_t)MTLVertexFormatFloat ||
        mglRenderCppTessControlPointFormat(GL_FLOAT_VEC2) !=
            (uint32_t)MTLVertexFormatFloat2 ||
        mglRenderCppTessControlPointFormat(GL_FLOAT_VEC3) !=
            (uint32_t)MTLVertexFormatFloat3 ||
        mglRenderCppTessControlPointFormat(GL_FLOAT_VEC4) !=
            (uint32_t)MTLVertexFormatFloat4 ||
        mglRenderCppTessControlPointFormat(GL_INT) !=
            (uint32_t)MTLVertexFormatInt ||
        mglRenderCppTessControlPointFormat(GL_INT_VEC2) !=
            (uint32_t)MTLVertexFormatInt2 ||
        mglRenderCppTessControlPointFormat(GL_INT_VEC3) !=
            (uint32_t)MTLVertexFormatInt3 ||
        mglRenderCppTessControlPointFormat(GL_INT_VEC4) !=
            (uint32_t)MTLVertexFormatInt4 ||
        mglRenderCppTessControlPointFormat(GL_UNSIGNED_INT) !=
            (uint32_t)MTLVertexFormatUInt ||
        mglRenderCppTessControlPointFormat(GL_BOOL_VEC2) !=
            (uint32_t)MTLVertexFormatUInt2 ||
        mglRenderCppTessControlPointFormat(GL_UNSIGNED_INT_VEC3) !=
            (uint32_t)MTLVertexFormatUInt3 ||
        mglRenderCppTessControlPointFormat(GL_BOOL_VEC4) !=
            (uint32_t)MTLVertexFormatUInt4 ||
        mglRenderCppTessControlPointFormat(GL_FLOAT_MAT4) !=
            (uint32_t)MTLVertexFormatInvalid ||
        mglRenderCppTessControlPointFormat(0xfeed) !=
            (uint32_t)MTLVertexFormatInvalid) {
        fprintf(stderr, "FAIL: tess control point format\n");
        return 1;
    }
    printf("TESS_CP_FORMAT_OK\n");
    return 0;
}

static int verifyTESXFBVertexStride(void) {
    /* P4.5 (item 1141/887): TES XFB compact vertex stride (varyings resolved
     * by name against the TES stage-output resource list). */
    MGLShaderResource res[2] = {};
    res[0].name = "pos";
    res[0].gl_type = GL_FLOAT_VEC4; /* 16B */
    res[1].name = "col";
    res[1].gl_type = GL_FLOAT_VEC3; /* 12B */
    MGLShaderResourceList outputs;
    outputs.count = 2;
    outputs.list = res;

    Program p = {};
    p.transform_feedback_varying_count = 2;
    p.shader_resources_list[_TESS_EVALUATION_SHADER][_STAGE_OUTPUT_RES] = outputs;
    strcpy(p.transform_feedback_varying_names[0], "pos");
    strcpy(p.transform_feedback_varying_names[1], "col");
    if (mglRenderCppTESXFBVertexStride(&p) != 28) {
        fprintf(stderr, "FAIL: xfb stride basic\n");
        return 1;
    }
    /* Unknown varying name -> cannot prove stride -> 0. */
    strcpy(p.transform_feedback_varying_names[1], "nope");
    if (mglRenderCppTESXFBVertexStride(&p) != 0) {
        fprintf(stderr, "FAIL: xfb stride unknown field\n");
        return 1;
    }
    /* Unsupported field type (matrix) -> 0. */
    strcpy(p.transform_feedback_varying_names[1], "col");
    res[1].gl_type = GL_FLOAT_MAT4;
    if (mglRenderCppTESXFBVertexStride(&p) != 0) {
        fprintf(stderr, "FAIL: xfb stride unsupported type\n");
        return 1;
    }
    /* No varyings / NULL program -> 0. */
    p.transform_feedback_varying_count = 0;
    if (mglRenderCppTESXFBVertexStride(&p) != 0 ||
        mglRenderCppTESXFBVertexStride(NULL) != 0) {
        fprintf(stderr, "FAIL: xfb stride empty/null\n");
        return 1;
    }
    printf("TES_XFB_STRIDE_OK\n");
    return 0;
}

static int verifyTessFactorTransforms(void) {
    /* P4.5 (item 1141/887): tess-factor CPU transforms. */
    float outer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float inner[2] = {5.0f, 6.0f};

    /* Fill: 2 patches x 12B canonical records. */
    uint8_t fill[24];
    memset(fill, 0xAA, sizeof(fill));
    if (mglRenderCppFillDefaultTessFactorBuffer(
            fill, sizeof(fill), outer, inner, 2) != 0) {
        fprintf(stderr, "FAIL: tess fill rc\n");
        return 1;
    }
    const __fp16 *hf = (const __fp16 *)fill;
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < 4; i++) {
            if (hf[p * 6 + i] != (__fp16)outer[i]) {
                fprintf(stderr, "FAIL: tess fill outer p=%d i=%d\n", p, i);
                return 1;
            }
        }
        for (int i = 0; i < 2; i++) {
            if (hf[p * 6 + 4 + i] != (__fp16)inner[i]) {
                fprintf(stderr, "FAIL: tess fill inner p=%d i=%d\n", p, i);
                return 1;
            }
        }
    }
    if (mglRenderCppFillDefaultTessFactorBuffer(
            fill, 11, outer, inner, 2) != -1 ||
        mglRenderCppFillDefaultTessFactorBuffer(
            NULL, sizeof(fill), outer, inner, 2) != -1) {
        fprintf(stderr, "FAIL: tess fill bad args\n");
        return 1;
    }

    /* Repack: canonical -> triangle (out = in0..2 + in4). */
    uint16_t canon[12] = {100, 200, 300, 400, 500, 600,
                          700, 800, 900, 1000, 1100, 1200};
    uint8_t tri[16];
    memset(tri, 0xBB, sizeof(tri));
    if (mglRenderCppRepackTessFactorTriangles(
            canon, sizeof(canon), tri, sizeof(tri), 2) != 0) {
        fprintf(stderr, "FAIL: tess repack rc\n");
        return 1;
    }
    const uint16_t *tr = (const uint16_t *)tri;
    if (tr[0] != 100 || tr[1] != 200 || tr[2] != 300 || tr[3] != 500 ||
        tr[4] != 700 || tr[5] != 800 || tr[6] != 900 || tr[7] != 1100) {
        fprintf(stderr, "FAIL: tess repack values\n");
        return 1;
    }
    if (mglRenderCppRepackTessFactorTriangles(
            canon, sizeof(canon), tri, 7, 2) != -1 ||
        mglRenderCppRepackTessFactorTriangles(
            NULL, sizeof(canon), tri, sizeof(tri), 2) != -1) {
        fprintf(stderr, "FAIL: tess repack bad args\n");
        return 1;
    }

    /* Primitive count: patch0 inside {0.5, 0.5} -> clamp to 1 -> TRI 1x1=1,
     * QUADS 2x1x1=2; patch1 edge0=0 -> discarded.  Instances x3. */
    uint16_t factors[12];
    memset(factors, 0, sizeof(factors));
    /* patch0: edges all 1.0, inside {0.5, 0.5}; patch1: all zero (discarded). */
    for (int i = 0; i < 4; i++) factors[i] = 0x3C00; /* __fp16 1.0 */
    factors[4] = 0x3800; /* __fp16 0.5 */
    factors[5] = 0x3800;
    if (mglRenderCppTessPrimitiveCount(
            factors, sizeof(factors), 2, GL_TRIANGLES, 3) != 3) {
        fprintf(stderr, "FAIL: tess primcount triangles\n");
        return 1;
    }
    if (mglRenderCppTessPrimitiveCount(
            factors, sizeof(factors), 2, GL_QUADS, 1) != 2) {
        fprintf(stderr, "FAIL: tess primcount quads\n");
        return 1;
    }
    if (mglRenderCppTessPrimitiveCount(
            NULL, sizeof(factors), 2, GL_TRIANGLES, 1) != 0 ||
        mglRenderCppTessPrimitiveCount(
            factors, 7, 2, GL_TRIANGLES, 1) != 0) {
        fprintf(stderr, "FAIL: tess primcount bad args\n");
        return 1;
    }
    printf("TESS_FACTOR_TRANSFORMS_OK\n");
    return 0;
}

static int verifyIntegerReadbackConvert(void) {
    /* P4.5 (item 1171/1116): integer readback CPU conversion in C++. */
    const uint32_t packed_bit_widths[4] = {10u, 10u, 10u, 2u};
    const uint32_t packed_shifts[4] = {0u, 10u, 20u, 30u};

    /* Case 1: non-packed RGBA8Uint -> GL_RGBA_INTEGER/GL_UNSIGNED_INT. */
    uint8_t src1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t dst1[8] = {0};
    int map1[4] = {0, 1, 2, 3};
    MGLRenderCppIntegerReadbackConvertParams p1 = {
        .src = src1, .src_bytes_per_row = 8,
        .source_component_count = 4, .source_component_bytes = 1,
        .source_signed = 0, .source_rgb10a2_uint = 0,
        .copy_w = 2, .copy_h = 1,
        .dst = (uint8_t *)dst1, .dst_bytes_per_row = 16,
        .dst_pixel_bytes = 16, .dst_x = 0, .dst_y = 0,
        .output_components = 4, .component_map = map1,
        .output_component_bytes = 4, .packed_type = GL_UNSIGNED_INT,
        .is_packed_type = 0, .packed_bit_widths = packed_bit_widths,
        .packed_shifts = packed_shifts, .packed_output_bytes = 4,
    };
    if (mglRenderCppConvertIntegerReadback(&p1) != 0 ||
        dst1[0] != 1 || dst1[1] != 2 || dst1[2] != 3 || dst1[3] != 4 ||
        dst1[4] != 5 || dst1[5] != 6 || dst1[6] != 7 || dst1[7] != 8) {
        fprintf(stderr, "FAIL: integer readback non-packed\n");
        return 1;
    }

    /* Case 2: packed 2_10_10_10_REV. */
    uint32_t dst2 = 0;
    MGLRenderCppIntegerReadbackConvertParams p2 = p1;
    p2.copy_w = 1;
    p2.dst = (uint8_t *)&dst2;
    p2.dst_bytes_per_row = 4;
    p2.dst_pixel_bytes = 4;
    p2.packed_type = GL_UNSIGNED_INT_2_10_10_10_REV;
    p2.is_packed_type = 1;
    p2.packed_output_bytes = 4;
    if (mglRenderCppConvertIntegerReadback(&p2) != 0 ||
        dst2 != (1u | (2u << 10) | (3u << 20) | (3u << 30))) {
        fprintf(stderr, "FAIL: integer readback packed (got 0x%x)\n", dst2);
        return 1;
    }

    /* Case 3: unsigned-source clamp — R8Uint value 255 -> GL_BYTE clamps to
     * 127 (unsigned source must not wrap via int32_t cast).  A signed source
     * value 200 (= -56 as int8) is in-range and passes through unchanged. */
    uint8_t src3[1] = {255};
    uint8_t dst3 = 0;
    int map3[1] = {0};
    MGLRenderCppIntegerReadbackConvertParams p3 = {
        .src = src3, .src_bytes_per_row = 1,
        .source_component_count = 1, .source_component_bytes = 1,
        .source_signed = 0, .source_rgb10a2_uint = 0,
        .copy_w = 1, .copy_h = 1,
        .dst = &dst3, .dst_bytes_per_row = 1,
        .dst_pixel_bytes = 1, .dst_x = 0, .dst_y = 0,
        .output_components = 1, .component_map = map3,
        .output_component_bytes = 1, .packed_type = GL_BYTE,
        .is_packed_type = 0, .packed_bit_widths = packed_bit_widths,
        .packed_shifts = packed_shifts, .packed_output_bytes = 1,
    };
    if (mglRenderCppConvertIntegerReadback(&p3) != 0 || dst3 != 127) {
        fprintf(stderr, "FAIL: integer readback unsigned clamp (got %u)\n", dst3);
        return 1;
    }
    src3[0] = 200;
    p3.source_signed = 1;
    if (mglRenderCppConvertIntegerReadback(&p3) != 0 || dst3 != 200) {
        fprintf(stderr, "FAIL: integer readback signed in-range (got %u)\n", dst3);
        return 1;
    }

    /* Case 4: bad args. */
    if (mglRenderCppConvertIntegerReadback(NULL) != -1 ||
        mglRenderCppConvertIntegerReadback(&p1) != 0) {
        fprintf(stderr, "FAIL: integer readback bad args\n");
        return 1;
    }
    printf("INTEGER_READBACK_CONVERT_OK\n");
    return 0;
}

static int verifyLevelUploadOps(void) {
    /* P4.5 (item 1116): the dirty-level loop's iteration + classification
     * moved to mglRenderCppBuildLevelUploadOps. */
    uint8_t backingA[64];
    for (size_t i = 0; i < sizeof(backingA); i++) backingA[i] = (uint8_t)(i + 1);
    TextureLevel levels[3] = {0};
    levels[0].complete = GL_TRUE;
    levels[0].width = 4;
    levels[0].height = 4;
    levels[0].depth = 1;
    levels[0].pitch = 16;
    levels[0].data_size = sizeof(backingA);
    levels[0].data = (vm_address_t)(uintptr_t)backingA;
    levels[0].last_init_source = kTexImageCopy;
    levels[0].has_initialized_data = GL_TRUE;
    levels[0].ever_written = GL_TRUE;
    levels[1] = levels[0]; /* stale: no init, never written */
    levels[1].last_init_source = kTexImageNull;
    levels[1].has_initialized_data = GL_FALSE;
    levels[1].ever_written = GL_FALSE;
    levels[2] = levels[0]; /* incomplete: no pitch -> silently skipped */
    levels[2].pitch = 0;

    MGLRenderCppLevelUploadOp ops[3];
    uint32_t opCount = 99, shortCount = 99, badCount = 99;
    if (mglRenderCppBuildLevelUploadOps(
            levels, 3, (uint32_t)MTLTextureType2D,
            GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm,
            ops, 3, &opCount, &shortCount, &badCount) != 0 ||
        opCount != 1 || shortCount != 0 || badCount != 0) {
        fprintf(stderr, "FAIL: upload ops counts (%u/%u/%u)\n",
                opCount, shortCount, badCount);
        return 1;
    }
    if (ops[0].kind != 0u || ops[0].level != 0 ||
        ops[0].data != backingA || ops[0].owns_data != 0 ||
        ops[0].bytes_per_row != 16 || ops[0].bytes_per_image != 64 ||
        ops[0].copy_depth != 1 || ops[0].width != 4 || ops[0].height != 4) {
        fprintf(stderr, "FAIL: upload op fields\n");
        return 1;
    }
    if (mglRenderCppBuildLevelUploadOps(
            levels, 3, (uint32_t)MTLTextureType2D,
            GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm,
            ops, 2, &opCount, &shortCount, &badCount) != -1) {
        fprintf(stderr, "FAIL: upload ops capacity\n");
        return 1;
    }
    if (mglRenderCppBuildLevelUploadOps(
            NULL, 0, (uint32_t)MTLTextureType2D,
            GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm,
            ops, 3, &opCount, &shortCount, &badCount) != -1) {
        fprintf(stderr, "FAIL: upload ops bad args\n");
        return 1;
    }
    printf("LEVEL_UPLOAD_OPS_OK\n");
    return 0;
}

static int verifyCopyBackEncode(void) {
    /* P4.5 (item 1138): stage-binding copy-back validation / encode. */
    /* Validation only (NULL encoder). */
    MGLRenderCppCopyBackEntry bad[1];
    bad[0].temporary = (void *)(uintptr_t)0x1;
    bad[0].destination = (void *)(uintptr_t)0x2;
    bad[0].destination_offset = 0;
    bad[0].length = 0; /* empty -> skipped */
    if (mglRenderCppEncodeStageBindingCopyBacks(bad, 1, NULL) != 0) {
        fprintf(stderr, "FAIL: copy-back empty entry\n");
        return 1;
    }
    if (mglRenderCppEncodeStageBindingCopyBacks(NULL, 1, NULL) != -1 ||
        mglRenderCppEncodeStageBindingCopyBacks(NULL, 0, NULL) != 0) {
        fprintf(stderr, "FAIL: copy-back bad args\n");
        return 1;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return 0; /* covered by main's guard */
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!cb || !blit) {
        fprintf(stderr, "FAIL: copy-back encoder setup\n");
        return 1;
    }
    id<MTLBuffer> temp = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dest = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!temp || !dest) {
        fprintf(stderr, "FAIL: copy-back buffers\n");
        return 1;
    }
    MGLRenderCppCopyBackEntry good[2];
    good[0].temporary = (__bridge void *)temp;
    good[0].destination = (__bridge void *)dest;
    good[0].destination_buffer = NULL;
    good[0].destination_offset = 0;
    good[0].length = 16;
    good[1].temporary = (__bridge void *)temp;
    good[1].destination = (__bridge void *)dest;
    good[1].destination_buffer = NULL;
    good[1].destination_offset = 32;
    good[1].length = 16;
    if (mglRenderCppEncodeStageBindingCopyBacks(good, 2, (__bridge void *)blit) != 0) {
        fprintf(stderr, "FAIL: copy-back encode\n");
        return 1;
    }
    /* Out-of-bounds: length exceeds the destination remaining space. */
    MGLRenderCppCopyBackEntry oob = good[0];
    oob.destination_offset = 48;
    oob.length = 32; /* 48+32 > 64 */
    if (mglRenderCppEncodeStageBindingCopyBacks(&oob, 1, (__bridge void *)blit) != -1) {
        fprintf(stderr, "FAIL: copy-back OOB not rejected\n");
        return 1;
    }
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    /* CPU prefix sync: a fake GL Buffer whose CPU snapshot must receive the
     * Metal destination's contents. */
    uint8_t metalBytes[64];
    memset(metalBytes, 0xAB, sizeof(metalBytes));
    memcpy(dest.contents, metalBytes, sizeof(metalBytes));
    Buffer glBuffer = {0};
    uint8_t cpuShadow[64];
    memset(cpuShadow, 0x11, sizeof(cpuShadow));
    glBuffer.data.buffer_size = 64;
    glBuffer.data.buffer_data = (vm_address_t)(uintptr_t)cpuShadow;
    glBuffer.ever_written = GL_FALSE;
    glBuffer.cpu_shadow_pending = GL_TRUE;
    MGLRenderCppCopyBackEntry sync = good[0];
    sync.destination_buffer = &glBuffer;
    uint32_t failedIndex = 99;
    if (mglRenderCppCopyBackCPUPrefix(&sync, 1, &failedIndex) != 0 ||
        failedIndex != 1) {
        fprintf(stderr, "FAIL: copy-back cpu prefix rc (idx=%u)\n", failedIndex);
        return 1;
    }
    if (glBuffer.ever_written != GL_TRUE ||
        glBuffer.cpu_shadow_pending != GL_FALSE ||
        cpuShadow[0] != 0xAB || cpuShadow[15] != 0xAB ||
        cpuShadow[16] != 0x11) {
        fprintf(stderr, "FAIL: copy-back cpu prefix contents\n");
        return 1;
    }
    /* CPU prefix failure: offset beyond the GL snapshot. */
    MGLRenderCppCopyBackEntry badSync = sync;
    badSync.destination_offset = 100;
    if (mglRenderCppCopyBackCPUPrefix(&badSync, 1, &failedIndex) != -1 ||
        failedIndex != 0) {
        fprintf(stderr, "FAIL: copy-back cpu prefix OOB not rejected\n");
        return 1;
    }
    printf("COPY_BACK_OK\n");
    return 0;
}

static int verifyRuntimeArraySizes(void) {
    /* P4.5 (item 1138): runtime-array-size SSBO sizing constants fill. */
    MGLRenderCppBufferSizeEntry entries[5];
    uint32_t sizes[32];
    memset(sizes, 0xA5, sizeof(sizes));

    /* Bad args. */
    if (mglRenderCppBuildRuntimeArraySizes(NULL, 1, 25u, 31u, sizes, 32) != -1 ||
        mglRenderCppBuildRuntimeArraySizes(entries, 0, 25u, 31u, NULL, 32) != -1 ||
        mglRenderCppBuildRuntimeArraySizes(entries, 0, 25u, 31u, sizes, 30) != -1) {
        fprintf(stderr, "FAIL: runtime-array-size bad args\n");
        return 1;
    }
    /* NULL entries with count 0 is fine. */
    if (mglRenderCppBuildRuntimeArraySizes(NULL, 0, 25u, 31u, sizes, 32) != 0) {
        fprintf(stderr, "FAIL: runtime-array-size null-empty\n");
        return 1;
    }

    memset(sizes, 0, sizeof(sizes));
    entries[0].metal_slot = 3;   entries[0].visible_size = 4096;
    entries[1].metal_slot = 25;  entries[1].visible_size = 999;   /* self-slot -> skip */
    entries[2].metal_slot = 31;  entries[2].visible_size = 777;   /* cap -> skip */
    entries[3].metal_slot = 8;   entries[3].visible_size = 0x100000000ULL; /* truncates to 0 */
    entries[4].metal_slot = 16;  entries[4].visible_size = 64;
    if (mglRenderCppBuildRuntimeArraySizes(entries, 5, 25u, 31u, sizes, 32) != 0 ||
        sizes[3] != 4096 || sizes[8] != 0 || sizes[16] != 64 ||
        sizes[25] != 0 || sizes[31] != 0 || sizes[0] != 0 || sizes[30] != 0) {
        fprintf(stderr, "FAIL: runtime-array-size fill\n");
        return 1;
    }

    printf("RUNTIME_ARRAY_SIZES_OK\n");
    return 0;
}

static int verifyBufferSlotRegistry(void) {
    /* P0 (2026-08-16 audit): the GS reserved-set must cover the real
     * mgl_air_gs_abi.h slots — INPUT=24, GATHER_PARAMS=25, OUTPUT=28,
     * COUNTS=29, GATHER=30, XFB=31, XFB_META=27 (plus the shared
     * tessellation factor slot 26).  Previously {24,28,29,30} missed 27/31
     * and mislabeled 30 as "GS XFB". */
    const GLuint geometryReserved[] = {24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u};
    for (size_t i = 0; i < sizeof(geometryReserved) / sizeof(geometryReserved[0]); ++i) {
        if (!mglBufferSlotIsReservedForGeometry(geometryReserved[i])) {
            fprintf(stderr, "FAIL: buffer-slot geometry reserved %u\n",
                    geometryReserved[i]);
            return 1;
        }
    }
    /* Slots outside the GS reserved domain must NOT be reserved. */
    const GLuint geometryFree[] = {0u, 3u, 14u, 15u, 22u};
    for (size_t i = 0; i < sizeof(geometryFree) / sizeof(geometryFree[0]); ++i) {
        if (mglBufferSlotIsReservedForGeometry(geometryFree[i])) {
            fprintf(stderr, "FAIL: buffer-slot geometry false-positive %u\n",
                    geometryFree[i]);
            return 1;
        }
    }

    /* Tessellation slots 26-30 (factors / patch output / patch info /
     * indirect / TES gl_in) — unchanged. */
    const GLuint tessReserved[] = {26u, 27u, 28u, 29u, 30u};
    for (size_t i = 0; i < sizeof(tessReserved) / sizeof(tessReserved[0]); ++i) {
        if (!mglBufferSlotIsReservedForTessellation(tessReserved[i])) {
            fprintf(stderr, "FAIL: buffer-slot tessellation reserved %u\n",
                    tessReserved[i]);
            return 1;
        }
    }
    if (mglBufferSlotIsReservedForTessellation(31u) ||
        mglBufferSlotIsReservedForTessellation(3u)) {
        fprintf(stderr, "FAIL: buffer-slot tessellation false-positive\n");
        return 1;
    }

    /* Cull-distance 28/29, FragCoord fixup 30. */
    if (!mglBufferSlotIsReservedForCullDistance(28u) ||
        !mglBufferSlotIsReservedForCullDistance(29u) ||
        mglBufferSlotIsReservedForCullDistance(30u)) {
        fprintf(stderr, "FAIL: buffer-slot cull-distance set\n");
        return 1;
    }
    if (!mglBufferSlotIsReservedForFragCoordFixup(30u) ||
        mglBufferSlotIsReservedForFragCoordFixup(29u)) {
        fprintf(stderr, "FAIL: buffer-slot fragcoord set\n");
        return 1;
    }

    /* Stage-specific: slot 15 is point-size (vertex only); slot 24 is
     * TCS stage_in (tess-control only) AND GS input (MGL_AIR_GS_SLOT_INPUT) —
     * the TCS early-return must not shadow the geometry reservation. */
    if (!mglBufferSlotIsReservedForStage(15, 0) ||   /* vertex */
        mglBufferSlotIsReservedForStage(15, 4) ||    /* fragment: no */
        !mglBufferSlotIsReservedForStage(24, 1) ||   /* TCS */
        !mglBufferSlotIsReservedForStage(24, 3) ||   /* GS (shared slot 24) */
        mglBufferSlotIsReservedForStage(24, 4)) {    /* fragment: no */
        fprintf(stderr, "FAIL: buffer-slot stage set\n");
        return 1;
    }

    /* Reserved-name labels: slot 25 must mention GATHER_PARAMS; 27/31 must
     * mention the GS XFB roles. */
    const char *n25 = mglBufferSlotReservedName(25);
    const char *n27 = mglBufferSlotReservedName(27);
    const char *n31 = mglBufferSlotReservedName(31);
    if (!n25 || !strstr(n25, "GATHER_PARAMS") ||
        !n27 || !strstr(n27, "XFB_META") ||
        !n31 || !strstr(n31, "XFB")) {
        fprintf(stderr, "FAIL: buffer-slot reserved-name labels\n");
        return 1;
    }

    printf("BUFFER_SLOT_REGISTRY_OK\n");
    return 0;
}

static int verifyLevelUploadPrep(void) {
    /* P4.5 (item 1111): per-level CPU upload data preparation. */
    /* 2D geometry: 4x4, pitch 16, 64 bytes -> copy_depth 1, bpi 16. */
    TextureLevel level2d = {0};
    level2d.complete = GL_TRUE;
    level2d.width = 4;
    level2d.height = 4;
    level2d.depth = 1;
    level2d.pitch = 16;
    level2d.data_size = 64;
    uint8_t backing2d[64] = {0};
    level2d.data = (vm_address_t)(uintptr_t)backing2d;
    MGLRenderCppLevelUploadPrep prep = {0};
    if (mglRenderCppTexturePrepareLevelUpload(
            &level2d, (uint32_t)MTLTextureType2D,
            GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm, &prep) != 0 ||
        prep.data != backing2d || prep.bytes_per_row != 16 ||
        prep.bytes_per_image != 64 || prep.copy_depth != 1 ||
        prep.available_bytes != 64 || prep.owns_data != 0) {
        fprintf(stderr, "FAIL: prep 2D geometry\n");
        return 1;
    }
    /* 3D geometry: depth 3 -> copy_depth 3, bpi = pitch*height = 64. */
    TextureLevel level3d = level2d;
    level3d.depth = 3;
    level3d.data_size = 192;
    if (mglRenderCppTexturePrepareLevelUpload(
            &level3d, (uint32_t)MTLTextureType3D,
            GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm, &prep) != 0 ||
        prep.copy_depth != 3 || prep.bytes_per_image != 64 ||
        prep.available_bytes != 192) {
        fprintf(stderr, "FAIL: prep 3D geometry\n");
        return 1;
    }
    /* RGBA8 expansion: GL_RGB8 4x4 pitch 12 (3 B/px) -> 16 B/row, owned. */
    uint8_t rgb[4 * 4 * 3];
    for (int i = 0; i < 4 * 4 * 3; i++) rgb[i] = (uint8_t)(i + 1);
    TextureLevel levelRgb = level2d;
    levelRgb.pitch = 12;
    levelRgb.data_size = sizeof(rgb);
    levelRgb.data = (vm_address_t)(uintptr_t)rgb;
    if (mglRenderCppTexturePrepareLevelUpload(
            &levelRgb, (uint32_t)MTLTextureType2D,
            GL_RGB8, (uint32_t)MTLPixelFormatRGBA8Unorm, &prep) != 0 ||
        prep.owns_data != 1 || prep.bytes_per_row != 16 ||
        prep.bytes_per_image != 64) {
        fprintf(stderr, "FAIL: prep RGBA8 expansion (bpr=%llu bpi=%llu owns=%d)\n",
                (unsigned long long)prep.bytes_per_row,
                (unsigned long long)prep.bytes_per_image, prep.owns_data);
        free((void *)prep.data);
        return 1;
    }
    const uint8_t *expanded = (const uint8_t *)prep.data;
    if (expanded[0] != 1 || expanded[1] != 2 || expanded[2] != 3 ||
        expanded[3] != 255) {
        fprintf(stderr, "FAIL: prep RGBA8 expansion bytes\n");
        free((void *)prep.data);
        return 1;
    }
    free((void *)prep.data);
    /* Channel expansion: RGBA16Unorm 2x2 pitch 12 (6 B/px) -> 16 B/row. */
    uint8_t rgb16[2 * 2 * 6];
    for (int i = 0; i < 2 * 2 * 6; i++) rgb16[i] = (uint8_t)(i + 1);
    TextureLevel level16 = level2d;
    level16.width = 2;
    level16.height = 2;
    level16.pitch = 12;
    level16.data_size = sizeof(rgb16);
    level16.data = (vm_address_t)(uintptr_t)rgb16;
    if (mglRenderCppTexturePrepareLevelUpload(
            &level16, (uint32_t)MTLTextureType2D,
            GL_RGB16, (uint32_t)MTLPixelFormatRGBA16Unorm, &prep) != 0 ||
        prep.owns_data != 1 || prep.bytes_per_row != 16) {
        fprintf(stderr, "FAIL: prep channel expansion\n");
        free((void *)prep.data);
        return 1;
    }
    expanded = (const uint8_t *)prep.data;
    /* alpha (bytes 6-7 of the first pixel) = 0xFFFF. */
    if (expanded[6] != 0xFF || expanded[7] != 0xFF) {
        fprintf(stderr, "FAIL: prep channel alpha (%02x %02x)\n",
                expanded[6], expanded[7]);
        free((void *)prep.data);
        return 1;
    }
    free((void *)prep.data);
    /* Small backing: the MIN clamp makes bpi <= available (the -2 short-
     * backing branch is mathematically unreachable in 2D/3D — defensive
     * parity with the ObjC guard), so the level uploads clamped. */
    TextureLevel shortLevel = level2d;
    shortLevel.data_size = 16;
    int rc = mglRenderCppTexturePrepareLevelUpload(
        &shortLevel, (uint32_t)MTLTextureType2D,
        GL_RGBA8, (uint32_t)MTLPixelFormatRGBA8Unorm, &prep);
    if (rc != 0 || prep.bytes_per_image != 16 || prep.copy_depth != 1 ||
        prep.available_bytes != 16) {
        fprintf(stderr, "FAIL: prep small backing (rc=%d)\n", rc);
        return 1;
    }
    /* Bad args. */
    if (mglRenderCppTexturePrepareLevelUpload(
            NULL, 0, GL_RGBA8, 0, &prep) != -1 ||
        mglRenderCppTexturePrepareLevelUpload(
            &level2d, 0, GL_RGBA8, 0, NULL) != -1) {
        fprintf(stderr, "FAIL: prep bad args\n");
        return 1;
    }
    printf("LEVEL_UPLOAD_PREP_OK\n");
    return 0;
}

static int verifyPendingEventOwner(void) {
    /* P4.5 (item 1141): pending shared-event slot inside the C++ owner. */
    void *owner = NULL;
    if (mglRenderCppCreatePendingEventOwner(&owner) != 0 || !owner) {
        fprintf(stderr, "FAIL: create pending-event owner\n");
        return 1;
    }
    void *first = NULL;
    if (mglRenderCppPendingEventPrepare(owner, 7, &first) != 0 || !first) {
        fprintf(stderr, "FAIL: prepare event\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    /* Re-prepare reuses the same event and updates the name. */
    void *second = NULL;
    if (mglRenderCppPendingEventPrepare(owner, 9, &second) != 0 ||
        second != first) {
        fprintf(stderr, "FAIL: prepare reuse\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    /* Detach transfers ownership; the slot empties. */
    int name = 0;
    void *detached = NULL;
    if (mglRenderCppPendingEventDetach(owner, &name, &detached) != 0 ||
        !detached || detached != first || name != 9) {
        fprintf(stderr, "FAIL: detach (name=%d event=%p first=%p)\n",
                name, detached, first);
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    detached = NULL;
    name = 0;
    if (mglRenderCppPendingEventDetach(owner, &name, &detached) != 0 ||
        detached != NULL || name != 0) {
        fprintf(stderr, "FAIL: detach after empty\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    /* After clear, prepare creates a fresh event. */
    if (mglRenderCppPendingEventPrepare(owner, 1, &first) != 0 || !first) {
        fprintf(stderr, "FAIL: prepare after detach\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    mglRenderCppPendingEventClear(owner);
    /* The slot must be empty after clear (detach returns nothing). */
    void *post_clear = NULL;
    int post_name = 123;
    if (mglRenderCppPendingEventDetach(owner, &post_name, &post_clear) != 0 ||
        post_clear != NULL || post_name != 0) {
        fprintf(stderr, "FAIL: slot not empty after clear\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    /* Prepare after clear works and records the new name. */
    if (mglRenderCppPendingEventPrepare(owner, 2, &post_clear) != 0 ||
        !post_clear) {
        fprintf(stderr, "FAIL: fresh event after clear\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    if (mglRenderCppPendingEventDetach(owner, &post_name, &post_clear) != 0 ||
        !post_clear || post_name != 2) {
        fprintf(stderr, "FAIL: detach name after clear\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    /* Bad-arg rejections. */
    if (mglRenderCppPendingEventPrepare(NULL, 1, &post_clear) != -1 ||
        mglRenderCppPendingEventDetach(NULL, &name, &post_clear) != -1 ||
        mglRenderCppCreatePendingEventOwner(NULL) != -1) {
        fprintf(stderr, "FAIL: bad-arg rejections\n");
        mglRenderCppDestroyPendingEventOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyPendingEventOwner(&owner);
    if (owner != NULL) {
        fprintf(stderr, "FAIL: owner not cleared\n");
        return 1;
    }
    printf("PENDING_EVENT_OWNER_OK\n");
    return 0;
}

static int verifyRenderPassIdentityOwner(void) {
    void *owner = NULL;
    if (mglRenderCppCreateRenderPassIdentityOwner(&owner) != 0 || !owner) {
        fprintf(stderr, "FAIL: render-pass identity owner create\n");
        return 1;
    }
    MGLRenderCppRenderPassIdentityState identity = {};
    identity.framebuffer = reinterpret_cast<void *>(0x1234u);
    identity.framebuffer_name = 37u;
    identity.draw_buffer = GL_COLOR_ATTACHMENT0;
    identity.draw_buffer_count = 2u;
    identity.draw_buffers[0] = GL_COLOR_ATTACHMENT0;
    identity.draw_buffers[1] = GL_COLOR_ATTACHMENT1;
    MGLRenderCppRenderPassIdentityState snapshot = {};
    if (mglRenderCppUpdateRenderPassIdentity(owner, &identity) != 0 ||
        mglRenderCppGetRenderPassIdentity(owner, &snapshot) != 0 ||
        snapshot.framebuffer != identity.framebuffer ||
        snapshot.framebuffer_name != identity.framebuffer_name ||
        snapshot.draw_buffer != identity.draw_buffer ||
        snapshot.draw_buffer_count != identity.draw_buffer_count ||
        snapshot.draw_buffers[0] != identity.draw_buffers[0] ||
        snapshot.draw_buffers[1] != identity.draw_buffers[1]) {
        fprintf(stderr, "FAIL: render-pass identity snapshot\n");
        mglRenderCppDestroyRenderPassIdentityOwner(&owner);
        return 1;
    }
    MGLRenderCppFboMatchCacheState cache = {37u, 91u, 1};
    MGLRenderCppFboMatchCacheState cacheSnapshot = {};
    if (mglRenderCppSetFboMatchCache(owner, &cache) != 0 ||
        mglRenderCppGetFboMatchCache(owner, &cacheSnapshot) != 0 ||
        cacheSnapshot.fbo_name != cache.fbo_name ||
        cacheSnapshot.generation != cache.generation ||
        cacheSnapshot.result != 1) {
        fprintf(stderr, "FAIL: render-pass identity cache\n");
        mglRenderCppDestroyRenderPassIdentityOwner(&owner);
        return 1;
    }
    mglRenderCppClearFboMatchCache(owner);
    if (mglRenderCppGetFboMatchCache(owner, &cacheSnapshot) != 1) {
        fprintf(stderr, "FAIL: render-pass identity cache clear\n");
        mglRenderCppDestroyRenderPassIdentityOwner(&owner);
        return 1;
    }
    identity.draw_buffer_count = MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS + 1u;
    if (mglRenderCppUpdateRenderPassIdentity(owner, &identity) == 0) {
        fprintf(stderr, "FAIL: render-pass identity accepted invalid count\n");
        mglRenderCppDestroyRenderPassIdentityOwner(&owner);
        return 1;
    }
    identity = {};
    if (mglRenderCppUpdateRenderPassIdentity(owner, &identity) != 0 ||
        mglRenderCppGetRenderPassIdentity(owner, &snapshot) != 0 ||
        snapshot.framebuffer || snapshot.framebuffer_name ||
        snapshot.draw_buffer_count) {
        fprintf(stderr, "FAIL: render-pass identity reset\n");
        mglRenderCppDestroyRenderPassIdentityOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyRenderPassIdentityOwner(&owner);
    if (owner) {
        fprintf(stderr, "FAIL: render-pass identity owner destroy\n");
        return 1;
    }
    printf("RENDER_PASS_IDENTITY_OWNER_OK\n");
    return 0;
}

static int verifyRenderPassStateOwner(id<MTLDevice> device) {
    void *defaultOwner = NULL;
    MGLRenderCppRenderPassState defaultState = {};
    if (mglRenderCppCreateDefaultRenderPassStateOwner(&defaultOwner) != 0 ||
        !defaultOwner ||
        mglRenderCppGetRenderPassStateOwner(
            defaultOwner, &defaultState) != 0 ||
        defaultState.color[0].attachment.store_action !=
            MTLStoreActionStore ||
        defaultState.color[0].clear_alpha != 1.0 ||
        defaultState.depth.attachment.store_action != MTLStoreActionStore ||
        defaultState.depth.clear_depth != 1.0 ||
        defaultState.stencil.attachment.store_action != MTLStoreActionStore) {
        fprintf(stderr, "FAIL: render-pass state owner defaults\n");
        mglRenderCppDestroyRenderPassStateOwner(&defaultOwner);
        return 1;
    }
    mglRenderCppDestroyRenderPassStateOwner(&defaultOwner);

    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:2 height:2 mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    __weak id<MTLTexture> weakTexture = texture;
    void *textureRaw = (__bridge void *)texture;
    MGLRenderCppRenderPassState state = {};
    state.render_target_width = 17u;
    state.render_target_height = 19u;
    state.default_raster_sample_count = 1u;
    state.color[0].attachment.texture = textureRaw;
    state.color[0].attachment.load_action = MTLLoadActionClear;
    state.color[0].attachment.store_action = MTLStoreActionStore;
    state.color[0].clear_red = 0.25;
    state.sample_position_count = 1u;
    state.sample_positions[0] = {0.5f, 0.5f};
    void *owner = NULL;
    MGLRenderCppRenderPassState snapshot = {};
    if (mglRenderCppCreateRenderPassStateOwner(&state, &owner) != 0 ||
        !owner || mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
        snapshot.render_target_width != 17u ||
        snapshot.render_target_height != 19u ||
        snapshot.color[0].attachment.texture !=
            textureRaw ||
        snapshot.color[0].clear_red != 0.25 ||
        snapshot.sample_position_count != 1u ||
        snapshot.sample_positions[0].x != 0.5f) {
        fprintf(stderr, "FAIL: render-pass state owner create/snapshot\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    texture = nil;
    if (!weakTexture) {
        fprintf(stderr, "FAIL: render-pass state owner did not retain texture\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    MGLRenderCppRenderPassAttachmentState attachment =
        snapshot.color[0].attachment;
    id<MTLBuffer> visibility =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    __weak id<MTLBuffer> weakVisibility = visibility;
    if (mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            0u, NULL, 3u, 0u, 0u, 0u) != 0 ||
        mglRenderCppSetRenderPassStateAttachmentActions(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0u,
            MTLLoadActionDontCare, MTLStoreActionDontCare, 7u) != 0 ||
        mglRenderCppSetRenderPassStateColorClear(
            owner, 0u, 0.5, 0.25, 0.125, 1.0) != 0 ||
        mglRenderCppSetRenderPassStateDepthClear(owner, 0.75) != 0 ||
        mglRenderCppSetRenderPassStateStencilClear(owner, 37u) != 0 ||
        mglRenderCppSetRenderPassStateVisibility(
            owner, (__bridge void *)visibility, 3u) != 0 ||
        mglRenderCppSetRenderPassStateDimensions(owner, 23u, 29u) != 0 ||
        mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
        snapshot.render_target_width != 23u ||
        snapshot.render_target_height != 29u ||
        snapshot.color[0].attachment.texture ||
        snapshot.color[0].attachment.level != 3u ||
        snapshot.color[0].attachment.load_action != MTLLoadActionDontCare ||
        snapshot.color[0].attachment.store_action != MTLStoreActionDontCare ||
        snapshot.color[0].attachment.store_action_options != 7u ||
        snapshot.color[0].clear_red != 0.5 ||
        snapshot.color[0].clear_green != 0.25 ||
        snapshot.color[0].clear_blue != 0.125 ||
        snapshot.color[0].clear_alpha != 1.0 ||
        snapshot.depth.clear_depth != 0.75 ||
        snapshot.stencil.clear_stencil != 37u ||
        snapshot.visibility_result_buffer != (__bridge void *)visibility ||
        snapshot.visibility_result_type != 3u) {
        fprintf(stderr, "FAIL: render-pass state owner native mutation\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    if (weakTexture) {
        fprintf(stderr, "FAIL: render-pass state owner did not release texture\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }

    auto verifyLayeredAttachment = [&] (
        const char *label,
        MTLTextureDescriptor *textureDescriptor,
        uint64_t level,
        uint64_t expectedArrayLength) -> int {
        textureDescriptor.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> layeredTexture =
            [device newTextureWithDescriptor:textureDescriptor];
        if (!layeredTexture ||
            mglRenderCppSetRenderPassStateAttachmentTexture(
                owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0u,
                (__bridge void *)layeredTexture, level, 5u, 7u, 1u) != 0 ||
            mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
            snapshot.color[0].attachment.texture !=
                (__bridge void *)layeredTexture ||
            snapshot.color[0].attachment.level != level ||
            snapshot.color[0].attachment.slice != 0u ||
            snapshot.color[0].attachment.depth_plane != 0u ||
            snapshot.render_target_array_length != expectedArrayLength) {
            fprintf(stderr,
                    "FAIL: render-pass layered attachment %s expected=%llu actual=%llu\n",
                    label,
                    (unsigned long long)expectedArrayLength,
                    (unsigned long long)snapshot.render_target_array_length);
            return 1;
        }
        if (mglRenderCppSetRenderPassStateAttachmentTexture(
                owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0u,
                (__bridge void *)layeredTexture, level, 2u, 3u, 0u) != 0 ||
            mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
            snapshot.color[0].attachment.slice != 2u ||
            snapshot.color[0].attachment.depth_plane != 3u ||
            snapshot.render_target_array_length != 0u) {
            fprintf(stderr,
                    "FAIL: render-pass non-layered attachment %s did not preserve subresource\n",
                    label);
            return 1;
        }
        return 0;
    };

    MTLTextureDescriptor *arrayDescriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    arrayDescriptor.textureType = MTLTextureType2DArray;
    arrayDescriptor.arrayLength = 3u;
    MTLTextureDescriptor *cubeDescriptor = [MTLTextureDescriptor
        textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        size:4 mipmapped:NO];
    MTLTextureDescriptor *cubeArrayDescriptor = [MTLTextureDescriptor
        textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        size:4 mipmapped:NO];
    cubeArrayDescriptor.textureType = MTLTextureTypeCubeArray;
    cubeArrayDescriptor.arrayLength = 2u;
    MTLTextureDescriptor *volumeDescriptor = [MTLTextureDescriptor new];
    volumeDescriptor.textureType = MTLTextureType3D;
    volumeDescriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    volumeDescriptor.width = 4u;
    volumeDescriptor.height = 4u;
    volumeDescriptor.depth = 4u;
    volumeDescriptor.mipmapLevelCount = 3u;
    if (verifyLayeredAttachment("2d-array", arrayDescriptor, 0u, 3u) != 0 ||
        verifyLayeredAttachment("cube", cubeDescriptor, 0u, 6u) != 0 ||
        verifyLayeredAttachment("cube-array", cubeArrayDescriptor, 0u, 12u) != 0 ||
        verifyLayeredAttachment("3d-mip", volumeDescriptor, 1u, 2u) != 0) {
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }

    MTLTextureDescriptor *shortArrayDescriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    shortArrayDescriptor.textureType = MTLTextureType2DArray;
    shortArrayDescriptor.arrayLength = 2u;
    shortArrayDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> longArrayTexture =
        [device newTextureWithDescriptor:arrayDescriptor];
    id<MTLTexture> shortArrayTexture =
        [device newTextureWithDescriptor:shortArrayDescriptor];
    if (!longArrayTexture || !shortArrayTexture ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0u,
            (__bridge void *)longArrayTexture, 0u, 0u, 0u, 1u) != 0 ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
            (__bridge void *)shortArrayTexture, 0u, 0u, 0u, 1u) != 0 ||
        mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
        snapshot.render_target_array_length != 2u ||
        snapshot.color[0].attachment.slice != 0u ||
        snapshot.depth.attachment.slice != 0u) {
        fprintf(stderr, "FAIL: render-pass layered common array length\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    if (mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH, 0u,
            NULL, 0u, 0u, 0u, 0u) != 0 ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR, 0u,
            NULL, 0u, 0u, 0u, 0u) != 0) {
        fprintf(stderr, "FAIL: render-pass layered attachment cleanup\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }

    visibility = nil;
    if (!weakVisibility ||
        mglRenderCppSetRenderPassStateVisibility(owner, NULL, 0u) != 0 ||
        mglRenderCppGetRenderPassStateOwner(owner, &snapshot) != 0 ||
        snapshot.visibility_result_buffer != NULL ||
        snapshot.visibility_result_type != 0u) {
        fprintf(stderr, "FAIL: render-pass state owner visibility ownership\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    if (mglRenderCppSetRenderPassStateAttachment(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS, &attachment) == 0 ||
        mglRenderCppSetRenderPassStateAttachment(
            owner, 0xffffffffu, 0u, &attachment) == 0 ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS, NULL, 0u, 0u, 0u, 0u) == 0 ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, 0xffffffffu, 0u, NULL, 0u, 0u, 0u, 0u) == 0) {
        fprintf(stderr, "FAIL: render-pass state owner accepted invalid attachment\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    if (mglRenderCppSetRenderPassStateAttachmentActions(
            owner, MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS, MTLLoadActionLoad,
            MTLStoreActionStore, 0u) == 0 ||
        mglRenderCppSetRenderPassStateAttachmentActions(
            owner, 0xffffffffu, 0u, MTLLoadActionLoad,
            MTLStoreActionStore, 0u) == 0 ||
        mglRenderCppSetRenderPassStateColorClear(
            owner, MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS,
            0.0, 0.0, 0.0, 0.0) == 0) {
        fprintf(stderr, "FAIL: render-pass state owner accepted invalid values\n");
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    state.sample_position_count = MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS + 1u;
    void *invalidOwner = NULL;
    if (mglRenderCppCreateRenderPassStateOwner(
            &state, &invalidOwner) == 0 || invalidOwner) {
        fprintf(stderr, "FAIL: render-pass state owner accepted invalid samples\n");
        mglRenderCppDestroyRenderPassStateOwner(&invalidOwner);
        mglRenderCppDestroyRenderPassStateOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyRenderPassStateOwner(&owner);
    if (owner) {
        fprintf(stderr, "FAIL: render-pass state owner destroy\n");
        return 1;
    }
    printf("RENDER_PASS_STATE_OWNER_OK\n");
    return 0;
}

static int verifyTextureStagingOwner(void) {
    const uint32_t values[4] = {
        0x10203040u, 0x50607080u, 0x90a0b0c0u, 0xd0e0f000u
    };
    void *owner = NULL;
    void *bufferRaw = NULL;
    if (mglRenderCppCreateTextureStagingOwner(
            values, sizeof(values), MTLResourceStorageModeShared,
            &owner, &bufferRaw) != 0 || !owner || !bufferRaw) {
        fprintf(stderr, "FAIL: texture staging owner create\n");
        mglRenderCppDestroyTextureStagingOwner(&owner);
        return 1;
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferRaw;
    if (buffer.length < sizeof(values) ||
        memcmp(buffer.contents, values, sizeof(values)) != 0) {
        fprintf(stderr, "FAIL: texture staging owner contents\n");
        mglRenderCppDestroyTextureStagingOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyTextureStagingOwner(&owner);
    if (owner) {
        fprintf(stderr, "FAIL: texture staging owner destroy\n");
        return 1;
    }
    printf("TEXTURE_STAGING_OWNER_OK\n");
    return 0;
}

static int verifyTextureUploadEncoding(id<MTLDevice> device) {
    static const uint8_t pixels[16] = {
        255, 0, 0, 255, 0, 255, 0, 255,
        0, 0, 255, 255, 255, 255, 255, 255,
    };
    void *stagingOwner = NULL;
    void *stagingBuffer = NULL;
    if (mglRenderCppCreateTextureStagingOwner(
            pixels, sizeof(pixels), MTLResourceStorageModeShared,
            &stagingOwner, &stagingBuffer) != 0 ||
        !stagingOwner || !stagingBuffer) {
        fprintf(stderr, "FAIL: texture upload encoding staging\n");
        mglRenderCppDestroyTextureStagingOwner(&stagingOwner);
        return 1;
    }

    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!texture || !queue || !commandBuffer ||
        mglRenderCppEncodeTextureUpload(
            (__bridge void *)commandBuffer, stagingBuffer, 0,
            8, sizeof(pixels), 2, 2, 1,
            (__bridge void *)texture, 0, 0, 0, 0, 0) != 0) {
        fprintf(stderr, "FAIL: texture upload encoding setup\n");
        mglRenderCppDestroyTextureStagingOwner(&stagingOwner);
        return 1;
    }

    /* The encoded command must retain the source after the C++ staging
     * owner releases its +1 reference. */
    mglRenderCppDestroyTextureStagingOwner(&stagingOwner);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    uint8_t readback[sizeof(pixels)] = {0};
    [texture getBytes:readback bytesPerRow:8
           fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0];
    if (commandBuffer.status == MTLCommandBufferStatusError ||
        memcmp(readback, pixels, sizeof(pixels)) != 0) {
        fprintf(stderr, "FAIL: texture upload encoding result: %s\n",
                commandBuffer.error.localizedDescription.UTF8String ?: "mismatch");
        return 1;
    }

    MTLTextureDescriptor *clearDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
    clearDescriptor.storageMode = MTLStorageModeShared;
    clearDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> clearTexture =
        [device newTextureWithDescriptor:clearDescriptor];
    MTLTextureDescriptor *depthDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatDepth32Float width:2 height:2 mipmapped:NO];
    depthDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> depthTexture =
        [device newTextureWithDescriptor:depthDescriptor];
    MTLTextureDescriptor *multisampleDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
    multisampleDescriptor.textureType = MTLTextureType2DMultisample;
    multisampleDescriptor.sampleCount = 4;
    multisampleDescriptor.storageMode = MTLStorageModePrivate;
    multisampleDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> multisampleTexture =
        [device newTextureWithDescriptor:multisampleDescriptor];
    id<MTLTexture> resolveTexture =
        [device newTextureWithDescriptor:clearDescriptor];
    id<MTLCommandBuffer> clearCommandBuffer = [queue commandBuffer];
    if (!clearTexture || !depthTexture || !multisampleTexture ||
        !resolveTexture || !clearCommandBuffer ||
        mglRenderCppEncodeColorClear(
            (__bridge void *)clearCommandBuffer,
            (__bridge void *)clearTexture, 0, 0, 0,
            1.0, 0.0, 0.0, 1.0) != 0 ||
        mglRenderCppEncodeDepthClear(
            (__bridge void *)clearCommandBuffer,
            (__bridge void *)depthTexture, 0, 0, 0, 0.25) != 0 ||
        mglRenderCppEncodeColorClear(
            (__bridge void *)clearCommandBuffer,
            (__bridge void *)multisampleTexture, 0, 0, 0,
            0.0, 1.0, 0.0, 1.0) != 0 ||
        mglRenderCppEncodeMultisampleResolve(
            (__bridge void *)clearCommandBuffer,
            MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR,
            (__bridge void *)multisampleTexture, 0, 0, 0,
            (__bridge void *)resolveTexture, 0, 0, 0, 0) != 0) {
        fprintf(stderr, "FAIL: render-pass clear encoding setup\n");
        return 1;
    }
    [clearCommandBuffer commit];
    [clearCommandBuffer waitUntilCompleted];
    uint8_t clearReadback[16] = {0};
    uint8_t resolveReadback[16] = {0};
    [clearTexture getBytes:clearReadback bytesPerRow:8
                fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0];
    [resolveTexture getBytes:resolveReadback bytesPerRow:8
                  fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0];
    if (clearCommandBuffer.status == MTLCommandBufferStatusError ||
        clearReadback[0] != 255u || clearReadback[1] != 0u ||
        clearReadback[2] != 0u || clearReadback[3] != 255u ||
        resolveReadback[0] != 0u || resolveReadback[1] != 255u ||
        resolveReadback[2] != 0u || resolveReadback[3] != 255u) {
        fprintf(stderr, "FAIL: render-pass clear encoding result: %s\n",
                clearCommandBuffer.error.localizedDescription.UTF8String ?:
                    "mismatch");
        return 1;
    }
    printf("TEXTURE_UPLOAD_ENCODING_OK\n");
    return 0;
}

static int verifyRenderEncoderOwner(id<MTLDevice> device) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    MTLTextureDescriptor *textureDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> texture =
        [device newTextureWithDescriptor:textureDescriptor];
    MGLRenderCppRenderPassState state = renderPassStateWithColorTarget(
        texture, MTLLoadActionClear, MTLStoreActionStore);
    void *stateOwner = NULL;
    void *stateEncoder = NULL;
    void *adoptedStateEncoderOwner = NULL;
    if (!commandBuffer || !texture ||
        mglRenderCppCreateRenderPassStateOwner(&state, &stateOwner) != 0 ||
        !stateOwner || mglRenderCppCreateRenderEncoderFromStateOwner(
            (__bridge void *)commandBuffer, stateOwner, &stateEncoder) != 0 ||
        !stateEncoder || mglRenderCppCreateRenderEncoderOwner(
            stateEncoder, &adoptedStateEncoderOwner) != 0 ||
        !adoptedStateEncoderOwner) {
        fprintf(stderr, "FAIL: render-pass state owner encoder\n");
        mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
        mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
        return 1;
    }

    /* P4.3a: unified draw-plan encode.  Valid array draw encodes; invalid
     * plan kinds and NULL encoder are rejected. */
    {
        id<MTLLibrary> drawLibrary = smokeLoadAssetLibrary(device, "scaled_blit");
        id<MTLFunction> drawVS = drawLibrary
            ? [drawLibrary newFunctionWithName:@"mgl_scaled_blit_vs"] : nil;
        id<MTLFunction> drawFS = drawLibrary
            ? [drawLibrary newFunctionWithName:@"mgl_scaled_blit_fs"] : nil;
        MTLRenderPipelineDescriptor *drawPipelineDesc =
            [MTLRenderPipelineDescriptor new];
        drawPipelineDesc.vertexFunction = drawVS;
        drawPipelineDesc.fragmentFunction = drawFS;
        drawPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        NSError *drawPipelineError = nil;
        id<MTLRenderPipelineState> drawPipeline =
            [device newRenderPipelineStateWithDescriptor:drawPipelineDesc
                                                   error:&drawPipelineError];
        if (!drawVS || !drawFS || !drawPipeline) {
            fprintf(stderr, "FAIL: draw plan pipeline: %s\n",
                    drawPipelineError.localizedDescription.UTF8String
                        ?: "unknown");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        if (mglRenderCppSetRenderPipelineState(
                stateEncoder, (__bridge void *)drawPipeline) != 0) {
            fprintf(stderr, "FAIL: draw plan pipeline bind\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        MGLRenderCppDrawPlan draw = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)MTLPrimitiveTypeTriangleStrip,
            .vertex_start = 0,
            .vertex_count = 4,
            .instance_count = 1,
            .base_instance = 0,
        };
        MGLRenderCppDrawPlan badKind = draw;
        badKind.kind = 0xdead;
        MGLRenderCppDrawPlan noInstance = draw;
        noInstance.instance_count = 0;
        char drawError[128] = {0};
        if (mglRenderCppEncodeDraw(stateEncoder, &draw,
                                   drawError, sizeof(drawError)) != 0 ||
            mglRenderCppEncodeDraw(NULL, &draw, NULL, 0) != -1 ||
            mglRenderCppEncodeDraw(stateEncoder, &badKind,
                                   drawError, sizeof(drawError)) != -1 ||
            mglRenderCppEncodeDraw(stateEncoder, &noInstance, NULL, 0) != -1) {
            fprintf(stderr, "FAIL: draw plan encode\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("DRAW_PLAN_ENCODE_OK\n");
    }

    /* P4.3b: binding snapshot replay.  Valid snapshot encodes; NULL encoder,
     * count overflow, NULL bytes op and bad op kind are rejected.  NULL
     * buffer ops are legal slot clears (P4.3b main-path extension). */
    {
        id<MTLBuffer> snapshotBuffer =
            [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (!snapshotBuffer) {
            fprintf(stderr, "FAIL: binding snapshot buffer\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        MGLRenderCppBindingSnapshot snap = {};
        snap.vertex_ops[snap.vertex_op_count++] =
            (MGLRenderCppBindingOp){
                /* kind */ 0u, /* index */ 0, /* offset */ 0,
                /* buffer */ (__bridge void *)snapshotBuffer,
                /* bytes */ NULL, /* length */ 0u};
        snap.fragment_ops[snap.fragment_op_count++] =
            (MGLRenderCppBindingOp){
                /* kind */ 0u, /* index */ 1, /* offset */ 16,
                /* buffer */ (__bridge void *)snapshotBuffer,
                /* bytes */ NULL, /* length */ 0u};
        snap.fragment_ops[snap.fragment_op_count++] =
            (MGLRenderCppBindingOp){
                /* kind */ 1u, /* index */ 2, /* offset */ 0,
                /* buffer */ NULL,
                /* bytes */ "abcd", /* length */ 4u};
        MGLRenderCppBindingSnapshot overflow = snap;
        overflow.vertex_op_count =
            MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_OPS + 1;
        MGLRenderCppBindingSnapshot nullBytes = snap;
        nullBytes.fragment_ops[1].bytes = NULL;
        MGLRenderCppBindingSnapshot badKind = snap;
        badKind.vertex_ops[0].kind = 0xdead;
        MGLRenderCppBindingSnapshot nullClear = snap;
        nullClear.vertex_ops[0].buffer = NULL;
        char snapError[128] = {0};
        if (mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &snap, snapError, sizeof(snapError)) != 0 ||
            mglRenderCppEncodeBindingSnapshot(NULL, &snap, NULL, 0) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &overflow, snapError, sizeof(snapError)) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &nullBytes, snapError, sizeof(snapError)) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &badKind, snapError, sizeof(snapError)) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &nullClear, snapError, sizeof(snapError)) != 0) {
            fprintf(stderr, "FAIL: binding snapshot encode err='%s'\n",
                    snapError);
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("BINDING_SNAPSHOT_OK\n");
    }

    /* P4.4: texture upload route selection.  Pure decision logic; assert the
     * exact same routing table as uploadTextureSliceViaBlit's inline
     * conditions: 1D/1DArray + non-private → REPLACE_1D; 3D + AGX bug +
     * private → REJECT; 3D + AGX bug + shared → REPLACE_3D; everything else
     * → BLIT. */
    {
        if (mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType1D, (uint32_t)MTLStorageModeShared, 0) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_1D ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType1DArray,
                (uint32_t)MTLStorageModeManaged, 0) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_1D ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType1D, (uint32_t)MTLStorageModePrivate, 0) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_BLIT ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType3D, (uint32_t)MTLStorageModePrivate, 1) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REJECT ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType3D, (uint32_t)MTLStorageModeShared, 1) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_3D ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType3D, (uint32_t)MTLStorageModeShared, 0) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_BLIT ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureType2D, (uint32_t)MTLStorageModeShared, 0) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_BLIT ||
            mglRenderCppTextureUploadRoute(
                (uint32_t)MTLTextureTypeCube, (uint32_t)MTLStorageModePrivate, 1) !=
                MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_BLIT) {
            fprintf(stderr, "FAIL: texture upload route\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("TEXTURE_UPLOAD_ROUTE_OK\n");
    }

    /* P4.4: 3D depth-plane repack — tight-pack a strided multi-plane buffer
     * and verify the output layout byte-for-byte; bad args return NULL. */
    {
        const size_t kBPI = 20;   /* padded plane stride */
        const size_t kTight = 16; /* tight plane stride (bpr*height) */
        const size_t kDepth = 3;
        uint8_t src[kDepth * kBPI];
        for (size_t z = 0; z < kDepth; z++) {
            for (size_t i = 0; i < kBPI; i++) {
                src[z * kBPI + i] = (uint8_t)(z * 40u + i);
            }
        }
        void *packed = mglRenderCppTextureRepackDepthPlanes(
            src, kBPI, kTight, kDepth);
        if (!packed) {
            fprintf(stderr, "FAIL: texture repack returned NULL\n");
            return 1;
        }
        const uint8_t *out = (const uint8_t *)packed;
        for (size_t z = 0; z < kDepth; z++) {
            for (size_t i = 0; i < kTight; i++) {
                if (out[z * kTight + i] != src[z * kBPI + i]) {
                    fprintf(stderr,
                            "FAIL: texture repack plane %zu byte %zu got %u want %u\n",
                            z, i, out[z * kTight + i], src[z * kBPI + i]);
                    free(packed);
                    return 1;
                }
            }
        }
        free(packed);
        if (mglRenderCppTextureRepackDepthPlanes(NULL, kBPI, kTight, 1) != NULL ||
            mglRenderCppTextureRepackDepthPlanes(src, kBPI, kTight, 0) != NULL ||
            mglRenderCppTextureRepackDepthPlanes(src, kTight - 1, kTight, 1) != NULL ||
            mglRenderCppTextureRepackDepthPlanes(src, kBPI, 0, 1) != NULL) {
            fprintf(stderr, "FAIL: texture repack bad-arg rejection\n");
            return 1;
        }
        printf("TEXTURE_REPACK_OK\n");
    }

    /* P4.4: RGB->RGBA channel expansion (CloudFaces texel-buffer 2D
     * fallback).  8-bit case: 3 texels -> 2x2 grid, texel 3 zeroed, alpha
     * 255 injected; 16-bit case exercises the alpha default bytes. */
    {
        const uint8_t src8[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
        uint8_t dst8[4 * 4];
        memset(dst8, 0xAA, sizeof(dst8));
        if (mglRenderCppTextureExpandRGBToRGBA(
                src8, dst8, 3, 2, 2, 1, 1, 255) != 0) {
            fprintf(stderr, "FAIL: texture expand 8-bit returned error\n");
            return 1;
        }
        const uint8_t want8[16] = {
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            0, 0, 0, 0
        };
        if (memcmp(dst8, want8, sizeof(want8)) != 0) {
            fprintf(stderr, "FAIL: texture expand 8-bit layout mismatch\n");
            return 1;
        }
        /* 16-bit case: alpha default 65535 occupies the low 2 bytes. */
        const uint8_t src16[] = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12
        };
        uint8_t dst16[2 * 8];
        if (mglRenderCppTextureExpandRGBToRGBA(
                src16, dst16, 2, 2, 1, 2, 2, 65535) != 0) {
            fprintf(stderr, "FAIL: texture expand 16-bit returned error\n");
            return 1;
        }
        const uint8_t want16[16] = {
            1, 2, 3, 4, 5, 6, 0xFF, 0xFF,
            7, 8, 9, 10, 11, 12, 0xFF, 0xFF
        };
        if (memcmp(dst16, want16, sizeof(want16)) != 0) {
            fprintf(stderr, "FAIL: texture expand 16-bit layout mismatch\n");
            return 1;
        }
        if (mglRenderCppTextureExpandRGBToRGBA(
                NULL, dst8, 3, 2, 2, 1, 1, 255) != -1 ||
            mglRenderCppTextureExpandRGBToRGBA(
                src8, NULL, 3, 2, 2, 1, 1, 255) != -1 ||
            mglRenderCppTextureExpandRGBToRGBA(
                src8, dst8, 3, 0, 2, 1, 1, 255) != -1 ||
            mglRenderCppTextureExpandRGBToRGBA(
                src8, dst8, 3, 2, 2, 0, 1, 255) != -1) {
            fprintf(stderr, "FAIL: texture expand bad-arg rejection\n");
            return 1;
        }
        printf("TEXTURE_EXPAND_OK\n");
    }

    /* P4.4: legacy packed format -> RGBA8 expansion.  RGB565 (2 bytes/texel
     * little-endian), RGB8 (3 bytes), RGBA4 and bad-arg rejections. */
    {
        /* RGB565: R=0x1F -> 255, G=0x3F -> 255, B=0x1F -> 255 (white). */
        const uint8_t src565[] = {0xFF, 0xFF};
        uint8_t dst565[4];
        size_t bpr = 0, bpi = 0;
        uint8_t *out = mglRenderCppCreateRGBA8ExpandedUpload(
            src565, 1, 1, 2, GL_RGB565, &bpr, &bpi);
        if (!out || bpr != 4 || bpi != 4) {
            fprintf(stderr, "FAIL: rgba8 expand RGB565 alloc\n");
            free(out);
            return 1;
        }
        const uint8_t want565[4] = {255, 255, 255, 255};
        if (memcmp(out, want565, 4) != 0) {
            fprintf(stderr,
                    "FAIL: rgba8 expand RGB565 got %u %u %u %u\n",
                    out[0], out[1], out[2], out[3]);
            free(out);
            return 1;
        }
        free(out);

        /* RGB8 3 texels 1x3: alpha 255. */
        const uint8_t src8[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        out = mglRenderCppCreateRGBA8ExpandedUpload(
            src8, 3, 1, 3 * 3, GL_RGB8, &bpr, &bpi);
        if (!out || bpi != 12) {
            fprintf(stderr, "FAIL: rgba8 expand RGB8 alloc\n");
            free(out);
            return 1;
        }
        const uint8_t want8[12] = {
            1, 2, 3, 255,
            4, 5, 6, 255,
            7, 8, 9, 255
        };
        if (memcmp(out, want8, sizeof(want8)) != 0) {
            fprintf(stderr, "FAIL: rgba8 expand RGB8 mismatch\n");
            free(out);
            return 1;
        }
        free(out);

        /* RGBA4: packed 4_4_4_4 (R bits 12-15): 0xFFFF -> all 255. */
        const uint8_t src444[] = {0xFF, 0xFF};
        out = mglRenderCppCreateRGBA8ExpandedUpload(
            src444, 1, 1, 2, GL_RGBA4, &bpr, &bpi);
        if (!out) {
            fprintf(stderr, "FAIL: rgba8 expand RGBA4 alloc\n");
            return 1;
        }
        const uint8_t want444[4] = {255, 255, 255, 255};
        if (memcmp(out, want444, 4) != 0) {
            fprintf(stderr, "FAIL: rgba8 expand RGBA4 got %u %u %u %u\n",
                    out[0], out[1], out[2], out[3]);
            free(out);
            return 1;
        }
        free(out);

        if (mglRenderCppCreateRGBA8ExpandedUpload(
                NULL, 1, 1, 2, GL_RGB565, &bpr, &bpi) != NULL ||
            mglRenderCppCreateRGBA8ExpandedUpload(
                src565, 0, 1, 2, GL_RGB565, &bpr, &bpi) != NULL ||
            mglRenderCppCreateRGBA8ExpandedUpload(
                src565, 1, 1, 1, GL_RGB565, &bpr, &bpi) != NULL ||
            mglRenderCppCreateRGBA8ExpandedUpload(
                src565, 1, 1, 2, 0xdeadbeefu, &bpr, &bpi) != NULL) {
            fprintf(stderr, "FAIL: rgba8 expand bad-arg rejection\n");
            return 1;
        }
        printf("RGBA8_EXPAND_OK\n");
    }

    /* P4.5 (item 1111): R8 swizzle resolve + single-channel expand. */
    {
        if (mglRenderCppResolveR8SwizzledComponent(GL_RED, 0x80) != 0x80 ||
            mglRenderCppResolveR8SwizzledComponent(GL_ONE, 0x10) != 0xff ||
            mglRenderCppResolveR8SwizzledComponent(GL_ALPHA, 0x10) != 0xff ||
            mglRenderCppResolveR8SwizzledComponent(GL_GREEN, 0x80) != 0x00 ||
            mglRenderCppResolveR8SwizzledComponent(GL_BLUE, 0x80) != 0x00 ||
            mglRenderCppResolveR8SwizzledComponent(GL_ZERO, 0x80) != 0x00 ||
            mglRenderCppResolveR8SwizzledComponent(0xdeadbeefu, 0x80) != 0x00) {
            fprintf(stderr, "FAIL: r8 swizzle resolve\n");
            return 1;
        }

        const uint8_t src[] = {0x80, 0xFF};
        size_t bpr = 0, bpi = 0;
        uint8_t *out = mglRenderCppCreateSingleChannelSwizzledUpload(
            GL_R8, GL_RED, GL_ZERO, GL_ZERO, GL_ONE,
            src, 2, 1, 2, &bpr, &bpi);
        if (!out || bpr != 8 || bpi != 8) {
            fprintf(stderr, "FAIL: r8 swizzle expand alloc\n");
            free(out);
            return 1;
        }
        const uint8_t want[8] = {
            0x80, 0x00, 0x00, 0xff,
            0xff, 0x00, 0x00, 0xff
        };
        if (memcmp(out, want, sizeof(want)) != 0) {
            fprintf(stderr, "FAIL: r8 swizzle expand mismatch\n");
            free(out);
            return 1;
        }
        free(out);

        /* src_bytes_per_row > width: second row starts at offset 4. */
        const uint8_t srcPad[] = {0x11, 0x00, 0x00, 0x00, 0x22};
        out = mglRenderCppCreateSingleChannelSwizzledUpload(
            GL_R8, GL_RED, GL_RED, GL_RED, GL_RED,
            srcPad, 1, 2, 4, &bpr, &bpi);
        if (!out || bpr != 4 || bpi != 8 ||
            out[0] != 0x11 || out[1] != 0x11 || out[2] != 0x11 || out[3] != 0x11 ||
            out[4] != 0x22 || out[5] != 0x22 || out[6] != 0x22 || out[7] != 0x22) {
            fprintf(stderr, "FAIL: r8 swizzle padded row\n");
            free(out);
            return 1;
        }
        free(out);

        if (mglRenderCppCreateSingleChannelSwizzledUpload(
                GL_R16, GL_RED, GL_ZERO, GL_ZERO, GL_ONE,
                src, 2, 1, 2, &bpr, &bpi) != NULL ||
            mglRenderCppCreateSingleChannelSwizzledUpload(
                GL_R8, GL_RED, GL_ZERO, GL_ZERO, GL_ONE,
                NULL, 2, 1, 2, &bpr, &bpi) != NULL ||
            mglRenderCppCreateSingleChannelSwizzledUpload(
                GL_R8, GL_RED, GL_ZERO, GL_ZERO, GL_ONE,
                src, 0, 1, 2, &bpr, &bpi) != NULL) {
            fprintf(stderr, "FAIL: r8 swizzle bad-arg rejection\n");
            return 1;
        }
        printf("R8_SWIZZLE_EXPAND_OK\n");
    }

    /* P4.5 (item 1111): R-only upload-swizzle gate. */
    {
        if (mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R8, 0) != 0 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R8, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R8_SNORM, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R16, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R16_SNORM, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R16F, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R32F, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R8I, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R8UI, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R16I, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R16UI, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R32I, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_R32UI, 1) != 1 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_RG8, 1) != 0 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(GL_RGBA8, 1) != 0 ||
            mglRenderCppTextureUploadNeedsSingleChannelSwizzle(0u, 1) != 0) {
            fprintf(stderr, "FAIL: r-only swizzle gate\n");
            return 1;
        }
        printf("R_ONLY_SWIZZLE_GATE_OK\n");
    }

    /* P4.3c: whole-batch simple replay.  Valid batch encodes; unknown command
     * type falls back to NEEDS_OBJC; bad args are rejected. */
    {
        id<MTLBuffer> replayIndexBuffer =
            [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (!replayIndexBuffer) {
            fprintf(stderr, "FAIL: replay batch index buffer\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        MGLRenderCppReplayBatchCommand replayCmds[2] = {
            {
                .cmd_type = 0,   /* MGL_CMD_DRAW_ARRAYS */
                .first = 0,
                .count = 3,
                .instance_count = 1,
            },
            {
                .cmd_type = 1,   /* MGL_CMD_DRAW_ELEMENTS */
                .count = 3,
                .instance_count = 1,
                .index_type = (uint32_t)MTLIndexTypeUInt16,
                .index_buffer = (__bridge void *)replayIndexBuffer,
                .index_buffer_offset = 0,
            },
        };
        MGLRenderCppReplayBatch replayBatch = {
            .primitive_type = (uint32_t)MTLPrimitiveTypeTriangle,
            .command_count = 2,
            .commands = replayCmds,
        };
        MGLRenderCppReplayBatch unknownType = replayBatch;
        unknownType.commands = replayCmds;
        MGLRenderCppReplayBatchCommand unknownCmd = replayCmds[0];
        unknownCmd.cmd_type = 0x7f;
        MGLRenderCppReplayBatchCommand badCmd[1] = {unknownCmd};
        unknownType.commands = badCmd;
        unknownType.command_count = 1;
        MGLRenderCppReplayBatch emptyBatch = replayBatch;
        emptyBatch.command_count = 0;
        char replayError[128] = {0};
        if (mglRenderCppReplayBatchDraws(
                stateEncoder, &replayBatch, replayError,
                sizeof(replayError)) != MGL_RENDER_CPP_REPLAY_BATCH_OK ||
            mglRenderCppReplayBatchDraws(
                stateEncoder, &unknownType, NULL, 0) !=
                MGL_RENDER_CPP_REPLAY_BATCH_NEEDS_OBJC ||
            mglRenderCppReplayBatchDraws(
                stateEncoder, &emptyBatch, NULL, 0) !=
                MGL_RENDER_CPP_REPLAY_BATCH_ERROR ||
            mglRenderCppReplayBatchDraws(NULL, &replayBatch, NULL, 0) !=
                MGL_RENDER_CPP_REPLAY_BATCH_ERROR) {
            fprintf(stderr, "FAIL: replay batch draws\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("REPLAY_BATCH_OK\n");
    }

    /* P4.3e: compute dispatch setup (begin/end).  Uses a private command
     * buffer so the still-active render encoder above is not disturbed. */
    {
        const MGLAuxShaderAsset *cs = mglAuxShaderAssetFind("scaled_blit_cs");
        void *csPipeline = NULL;
        char csMessage[512] = {0};
        if (!cs ||
            mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
                cs->data, cs->size, cs->hash, "mgl_scaled_blit_cs",
                MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT, 1u,
                &csPipeline, csMessage, sizeof(csMessage)) != 0 ||
            !csPipeline) {
            fprintf(stderr, "FAIL: compute dispatch PSO: %s\n",
                    csMessage[0] ? csMessage : "?");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        id<MTLCommandQueue> cdQueue = [device newCommandQueue];
        id<MTLCommandBuffer> cdCommandBuffer = [cdQueue commandBuffer];
        id<MTLBuffer> computeScratch =
            [device newBufferWithLength:256
                                options:MTLResourceStorageModeShared];
        if (!cdQueue || !cdCommandBuffer || !computeScratch) {
            fprintf(stderr, "FAIL: compute dispatch resources\n");
            CFRelease(csPipeline);
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        MGLRenderCppComputeDispatchSetup setup = {
            .pipeline = csPipeline,
            .buffer_count = 1,
            .buffers = { { (__bridge void *)computeScratch, 0, 0 } },
            .bytes_count = 1,
            .bytes = { { (const void *)"ABCD", 4, 1 } },
        };
        void *computeEncoder = NULL;
        char cdError[128] = {0};
        const uint32_t groups[3] = { 1, 1, 1 };
        const uint32_t threads[3] = { 1, 1, 1 };
        if (mglRenderCppBeginComputeDispatch(
                (__bridge void *)cdCommandBuffer, &setup, &computeEncoder,
                cdError, sizeof(cdError)) != 0 || !computeEncoder ||
            mglRenderCppEndComputeDispatch(
                computeEncoder, groups, threads, cdError, sizeof(cdError)) != 0 ||
            mglRenderCppBeginComputeDispatch(
                NULL, &setup, &computeEncoder, NULL, 0) != -1 ||
            mglRenderCppEndComputeDispatch(
                NULL, groups, threads, NULL, 0) != -1) {
            fprintf(stderr, "FAIL: compute dispatch begin/end\n");
            CFRelease(csPipeline);
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        CFRelease(csPipeline);
        [cdCommandBuffer commit];
        [cdCommandBuffer waitUntilCompleted];
        if (cdCommandBuffer.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "FAIL: compute dispatch command buffer\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("COMPUTE_DISPATCH_OK\n");
    }

    if (mglRenderCppEndRenderEncoderOwner(adoptedStateEncoderOwner) != 0) {
        fprintf(stderr, "FAIL: render-pass state owner encoder end\n");
        mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
        mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
        return 1;
    }
    mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
    mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
    void *owner = NULL;
    void *encoder = NULL;
    if (mglRenderCppCreateRenderEncoderOwnerFromState(
            (__bridge void *)commandBuffer, &state,
            &owner, &encoder) != 0 || !owner || !encoder ||
        mglRenderCppEndRenderEncoderOwner(owner) != 0 ||
        mglRenderCppEndRenderEncoderOwner(owner) != 0) {
        fprintf(stderr, "FAIL: render encoder owner create/end\n");
        mglRenderCppDestroyRenderEncoderOwner(&owner);
        return 1;
    }
    void *adoptedOwner = NULL;
    if (mglRenderCppCreateRenderEncoderOwner(
            encoder, &adoptedOwner) != 0 || !adoptedOwner) {
        fprintf(stderr, "FAIL: render encoder adopt owner\n");
        mglRenderCppDestroyRenderEncoderOwner(&owner);
        return 1;
    }
    mglRenderCppDestroyRenderEncoderOwner(&adoptedOwner);
    mglRenderCppDestroyRenderEncoderOwner(&owner);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (owner || commandBuffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: render encoder owner completion\n");
        return 1;
    }
    printf("RENDER_ENCODER_OWNER_OK\n");
    return 0;
}

static int verifyQueryUtilities(id<MTLDevice> device) {
    void *queryOwner = NULL;
    void *visibilityPtr = NULL;
    if (mglRenderCppCreateQueryStateOwner(256u, &queryOwner) != 0 ||
        !queryOwner ||
        mglRenderCppBeginSampleQuery(
            queryOwner, 1u, "MGL Smoke Visibility",
            &visibilityPtr) != 0 || !visibilityPtr) {
        fprintf(stderr, "FAIL: query state owner create/begin\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)visibilityPtr;
    __weak id<MTLBuffer> weakVisibility = visibility;
    uint32_t active = 0;
    uint32_t mode0 = 0;
    uint32_t mode1 = 0;
    uint64_t offset0 = UINT64_MAX;
    uint64_t offset1 = UINT64_MAX;
    if (mglRenderCppIsSampleQueryActive(queryOwner, &active) != 0 ||
        active != 1u ||
        mglRenderCppAcquireSampleQuerySlot(
            queryOwner, &mode0, &offset0) != 0 ||
        mglRenderCppAcquireSampleQuerySlot(
            queryOwner, &mode1, &offset1) != 0 ||
        mode0 != MTLVisibilityResultModeCounting ||
        mode1 != MTLVisibilityResultModeCounting ||
        offset0 != 0u || offset1 != sizeof(uint64_t)) {
        fprintf(stderr, "FAIL: query state owner active/slots\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    visibility = nil;
    if (!weakVisibility) {
        fprintf(stderr, "FAIL: query state owner did not retain buffer\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    visibility = weakVisibility;
    uint64_t *visibilitySlots =
        (uint64_t *)visibility.contents;
    visibilitySlots[0] = 3u;
    visibilitySlots[1] = 5u;
    mglRenderCppEndSampleQuery(queryOwner);
    uint64_t queryResult = 0;
    if (mglRenderCppIsSampleQueryActive(queryOwner, &active) != 0 ||
        active != 0u ||
        mglRenderCppGetSampleQueryResult(
            queryOwner, &queryResult) != 0 || queryResult != 8u) {
        fprintf(stderr, "FAIL: query state owner end/result\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }

    MTLTextureDescriptor *targetDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    targetDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDesc];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    MGLRenderCppRenderPassState renderPassState =
        renderPassStateWithColorTarget(
            target, MTLLoadActionDontCare, MTLStoreActionDontCare);
    renderPassState.visibility_result_buffer =
        (__bridge void *)visibility;
    void *encoder = NULL;
    if (mglRenderCppCreateRenderEncoderFromState(
            (__bridge void *)commandBuffer, &renderPassState, &encoder) != 0 ||
        !encoder || mglRenderCppSetVisibilityResultMode(
            encoder,
            (uint32_t)MTLVisibilityResultModeBoolean, 0) != 0) {
        fprintf(stderr, "FAIL: visibility mode facade\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    if (mglRenderCppEndRenderEncoder(encoder) != 0) {
        fprintf(stderr, "FAIL: visibility render encoder end\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: query utility command buffer: %s\n",
                commandBuffer.error.localizedDescription.UTF8String);
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }

    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderCppSampleTimestamps(&cpuTimestamp, &gpuTimestamp) != 0 ||
        cpuTimestamp == 0 || gpuTimestamp == 0) {
        fprintf(stderr, "FAIL: timestamp facade cpu=%llu gpu=%llu\n",
                (unsigned long long)cpuTimestamp,
                (unsigned long long)gpuTimestamp);
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    uint64_t elapsed = 0;
    if (mglRenderCppBeginTimerQuery(queryOwner) != 0 ||
        mglRenderCppEndTimerQuery(queryOwner, &elapsed) != 0) {
        fprintf(stderr, "FAIL: query state owner timer\n");
        mglRenderCppDestroyQueryStateOwner(&queryOwner);
        return 1;
    }
    visibility = nil;
    mglRenderCppDestroyQueryStateOwner(&queryOwner);
    if (queryOwner) {
        fprintf(stderr, "FAIL: query state owner destroy\n");
        return 1;
    }
    printf("QUERY_UTILITIES_OK\n");
    return 0;
}

static int verifyRawRenderAndBlitFacade(id<MTLDevice> device) {
    /* Precompiled fixture shaders: scaled_blit vs/fs cover a fullscreen
     * quad and sample texture(0) with buffer(0) params — the encoders and
     * readbacks below only require that combination. */
    id<MTLLibrary> library = smokeLoadAssetLibrary(device, "scaled_blit");
    NSError *error = nil;
    id<MTLFunction> vertex = [library newFunctionWithName:@"mgl_scaled_blit_vs"];
    id<MTLFunction> fragment = [library newFunctionWithName:@"mgl_scaled_blit_fs"];
    MTLRenderPipelineDescriptor *pipelineDesc =
        [MTLRenderPipelineDescriptor new];
    pipelineDesc.vertexFunction = vertex;
    pipelineDesc.fragmentFunction = fragment;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!library || !vertex || !fragment || !pipeline) {
        fprintf(stderr, "FAIL: raw render pipeline: %s\n",
                error.localizedDescription.UTF8String ?: "unknown");
        return 1;
    }

    MTLTextureDescriptor *sourceDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:YES];
    sourceDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> sourceTexture = [device newTextureWithDescriptor:sourceDesc];
    const uint8_t sourcePixel[4] = {17, 34, 51, 255};
    uint8_t sourcePixels[4 * 4 * 4] = {};
    for (NSUInteger offset = 0; offset < sizeof(sourcePixels); offset += 4) {
        memcpy(sourcePixels + offset, sourcePixel, sizeof(sourcePixel));
    }
    [sourceTexture replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
                     mipmapLevel:0
                       withBytes:sourcePixels
                     bytesPerRow:16];

    MTLTextureDescriptor *targetDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    targetDesc.usage = MTLTextureUsageRenderTarget;
    targetDesc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDesc];
    MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
    samplerDesc.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
    id<MTLSamplerState> sampler =
        [device newSamplerStateWithDescriptor:samplerDesc];
    id<MTLBuffer> readback =
        [device newBufferWithLength:256 * 4
                            options:MTLResourceStorageModeShared];
    void *uploadPtr = NULL;
    id<MTLBuffer> upload = nil;
    if (mglRenderCppCreateBufferWithBytes(
            sourcePixels, sizeof(sourcePixels),
            MTLResourceStorageModeShared, "smoke upload", &uploadPtr) == 0 &&
        uploadPtr) {
        upload = (__bridge_transfer id<MTLBuffer>)uploadPtr;
    }
    id<MTLBuffer> bufferCopy =
        [device newBufferWithLength:sizeof(sourcePixels)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> uploadReadback =
        [device newBufferWithLength:256 * 4
                            options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!sourceTexture || !target || !sampler || !readback || !upload ||
        !bufferCopy || !uploadReadback ||
        !queue || !commandBuffer) {
        fprintf(stderr, "FAIL: raw render resources\n");
        return 1;
    }

    void *renderEncoder = NULL;
    void *indirectBufferPtr = NULL;
    void *indirectCommand = NULL;
    vector_float4 scale = {1.0f, 1.0f, 0.0f, 0.0f};
    vector_float4 tint = {1.0f, 1.0f, 1.0f, 1.0f};
    MGLRenderCppRenderPassState renderPassState =
        renderPassStateWithColorTarget(
            target, MTLLoadActionDontCare, MTLStoreActionStore);
    if (mglRenderCppCreateRenderEncoderFromState(
            (__bridge void *)commandBuffer, &renderPassState,
            &renderEncoder) != 0 ||
        !renderEncoder ||
        mglRenderCppSetRenderPipelineState(
            renderEncoder, (__bridge void *)pipeline) != 0 ||
        mglRenderCppSetRenderBytes(
            renderEncoder, &scale, sizeof(scale),
            MGL_RENDER_CPP_BINDING_STAGE_VERTEX, 0) != 0 ||
        mglRenderCppSetRenderBytes(
            renderEncoder, &tint, sizeof(tint),
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0) != 0 ||
        mglRenderCppSetRenderTexture(
            renderEncoder, (__bridge void *)sourceTexture,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0) != 0 ||
        mglRenderCppSetRenderSampler(
            renderEncoder, (__bridge void *)sampler,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, 0) != 0 ||
        mglRenderCppSetRenderViewport(
            renderEncoder, 0.0, 0.0, 4.0, 4.0, 0.0, 1.0) != 0 ||
        mglRenderCppSetRenderScissor(renderEncoder, 0, 0, 4, 4) != 0 ||
        mglRenderCppDrawPrimitives(
            renderEncoder, (uint32_t)MTLPrimitiveTypeTriangleStrip,
            0, 4, 1, 0) != 0 ||
        mglRenderCppCreateIndirectCommandBuffer(
            (uint32_t)MTLIndirectCommandTypeDraw, 1, 1, 0, 0, 1,
            MTLResourceStorageModePrivate, &indirectBufferPtr) != 0 ||
        !indirectBufferPtr) {
        fprintf(stderr, "FAIL: raw render facade\n");
        return 1;
    }
    id<MTLIndirectCommandBuffer> indirectBuffer =
        (__bridge_transfer id<MTLIndirectCommandBuffer>)indirectBufferPtr;
    if (mglRenderCppResetIndirectCommandBuffer(
            (__bridge void *)indirectBuffer, 0, 1) != 0 ||
        mglRenderCppGetIndirectRenderCommand(
            (__bridge void *)indirectBuffer, 0, &indirectCommand) != 0 ||
        !indirectCommand ||
        mglRenderCppSetIndirectDraw(
            indirectCommand, (uint32_t)MTLPrimitiveTypeTriangleStrip,
            0, 4, 1, 0) != 0 ||
        mglRenderCppEndRenderEncoder(renderEncoder) != 0) {
        fprintf(stderr, "FAIL: indirect command buffer facade\n");
        return 1;
    }

    void *blitEncoder = NULL;
    if (mglRenderCppCreateBlitEncoder((__bridge void *)commandBuffer,
                                      &blitEncoder) != 0 ||
        !blitEncoder ||
        mglRenderCppBlitCopyBuffer(
            blitEncoder, (__bridge void *)upload, 0,
            (__bridge void *)bufferCopy, 0, sizeof(sourcePixels)) != 0 ||
        mglRenderCppBlitCopyBufferToTexture(
            blitEncoder, (__bridge void *)upload, 0, 16,
            sizeof(sourcePixels), 4, 4, 1,
            (__bridge void *)sourceTexture, 0, 0, 0, 0, 0) != 0 ||
        mglRenderCppBlitGenerateMipmaps(
            blitEncoder, (__bridge void *)sourceTexture) != 0 ||
        mglRenderCppBlitCopyTextureToBuffer(
            blitEncoder, (__bridge void *)sourceTexture, 0, 0, 0, 0, 0,
            4, 4, 1, (__bridge void *)uploadReadback, 0, 256,
            256 * 4) != 0 ||
        mglRenderCppBlitCopyTextureToBuffer(
            blitEncoder, (__bridge void *)target, 0, 0, 0, 0, 0,
            4, 4, 1, (__bridge void *)readback, 0, 256, 256 * 4) != 0 ||
        mglRenderCppEndBlitEncoder(blitEncoder) != 0) {
        fprintf(stderr, "FAIL: texture-to-buffer blit facade\n");
        return 1;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: raw render command buffer: %s\n",
                commandBuffer.error.localizedDescription.UTF8String);
        return 1;
    }
    const uint8_t *result = (const uint8_t *)readback.contents;
    const uint8_t *bufferCopyResult = (const uint8_t *)bufferCopy.contents;
    const uint8_t *uploadResult = (const uint8_t *)uploadReadback.contents;
    if (!result || memcmp(result, sourcePixel, sizeof(sourcePixel)) != 0) {
        fprintf(stderr,
                "FAIL: raw render readback got=%u,%u,%u,%u expected=%u,%u,%u,%u\n",
                result ? result[0] : 0, result ? result[1] : 0,
                result ? result[2] : 0, result ? result[3] : 0,
                sourcePixel[0], sourcePixel[1], sourcePixel[2], sourcePixel[3]);
        return 1;
    }
    if (!bufferCopyResult ||
        memcmp(bufferCopyResult, sourcePixels, sizeof(sourcePixels)) != 0 ||
        !uploadResult ||
        memcmp(uploadResult, sourcePixel, sizeof(sourcePixel)) != 0) {
        fprintf(stderr, "FAIL: raw buffer blit facade\n");
        return 1;
    }
    printf("ICB_ENCODING_OK\n");
    printf("RAW_RENDER_BLIT_OK\n");
    return 0;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device (VM?)\n");
            return 2; // 无 GPU 环境，跳过（非失败）
        }

        int rc = mglRenderCppInit((__bridge void*)device);
        if (rc != 0) {
            fprintf(stderr, "FAIL: mglRenderCppInit rc=%d\n", rc);
            return 1;
        }
        void* dev = mglRenderCppGetDevice();
        if (!dev) {
            fprintf(stderr, "FAIL: device null after init\n");
            return 1;
        }
        printf("SMOKE_OK device=%p\n", dev);

        if (verifyAttachmentSubresource() != 0) return 1;
        if (verifyBufferBinding() != 0) return 1;
        if (verifyPackedStructBufferRing() != 0) return 1;
        if (verifyVertexConversions() != 0) return 1;
        if (verifySamplerConversion() != 0) return 1;
        if (verifyAIRProgramClassification() != 0) return 1;
        if (verifyAuxShaderAssets() != 0) return 1;
        if (verifyResourceCreation() != 0) return 1;
        if (verifyTextureTransferFacade() != 0) return 1;
        if (verifyNoCopyBufferFacade() != 0) return 1;
        if (verifyCompilerAndBinaryArchive() != 0) return 1;
        if (verifyPipelineCacheOwner(device) != 0) return 1;
        if (verifyBindingDedup(device) != 0) return 1;
        if (verifyComputeSetters(device) != 0) return 1;
        if (verifySyncCallbacks(device) != 0) return 1;
        if (verifyCommandQueueOwner() != 0) return 1;
        if (verifyCommandBufferOwner() != 0) return 1;
        if (verifyPendingEventOwner() != 0) return 1;
        if (verifyLevelUploadPrep() != 0) return 1;
        if (verifyCopyBackEncode() != 0) return 1;
        if (verifyRuntimeArraySizes() != 0) return 1;
        if (verifyBufferSlotRegistry() != 0) return 1;
        if (verifyLevelUploadOps() != 0) return 1;
        if (verifyIntegerReadbackConvert() != 0) return 1;
        if (verifyTessFactorDiscardPredicate() != 0) return 1;
        if (verifyTessFactorTransforms() != 0) return 1;
        if (verifyTessEvalItemsAndCaptureSize() != 0) return 1;
        if (verifyTessRoundLevelForSpacing() != 0) return 1;
        if (verifyCheckedProductAndXFBFieldByteSize() != 0) return 1;
        if (verifyFloatUnpack() != 0) return 1;
        if (verifyReadbackScalarConvert() != 0) return 1;
        if (verifyTessControlPointFormat() != 0) return 1;
        if (verifyTESXFBVertexStride() != 0) return 1;
        if (verifyNativeTESInterfaceGuards() != 0) return 1;
        if (verifyRasterizationIsEmpty() != 0) return 1;
        if (verifyIntegerReadbackClassify() != 0) return 1;
        if (verifyGetTexImagePlan() != 0) return 1;
        if (verifyIntegerReadbackSourceClassify() != 0) return 1;
        if (verifyPackedTypeClassify() != 0) return 1;
        if (verifyBlitFramebufferPlan() != 0) return 1;
        if (verifyScaledBlitUVsAndScissor() != 0) return 1;
        if (verifyPolygonOffsetAndPrimCount() != 0) return 1;
        if (verifyBufferShadowUploadRange() != 0) return 1;
        if (verifyVertexAttribResolve() != 0) return 1;
        if (verifyMetalTypeTables() != 0) return 1;
        if (verifyShaderResourceTextureTypes() != 0) return 1;
        if (verifyTextureCreationTargetPlans() != 0) return 1;
        if (verifyTextureTargetIndices() != 0) return 1;
        if (verifyTextureDataKinds() != 0) return 1;
        if (verifyComputeThreadgroupSize() != 0) return 1;
        if (verifyLevelDimension() != 0) return 1;
        if (verifyLayerPixelFormat() != 0) return 1;
        if (verifyReadTextureRegionClip() != 0) return 1;
        if (verifyGeometryGather() != 0) return 1;
        if (verifyExpandTriangleFan() != 0) return 1;
        if (verifyExpandStripAndLineLoop() != 0) return 1;
        if (verifyExpandQuad() != 0) return 1;
        if (verifyQuadLine() != 0) return 1;
        if (verifyArrayVariants() != 0) return 1;
        if (verifyUInt8ToUInt16() != 0) return 1;
        if (verifyScanIndexRange() != 0) return 1;
        if (verifyComputePreparedByteOffset() != 0) return 1;
        if (verifyComputeIndexByteOffset() != 0) return 1;
        if (verifyGLIndexValueRead() != 0) return 1;
        if (verifyVertexAttribBytes() != 0) return 1;
        if (verifyDrawModePredicates() != 0) return 1;
        if (verifyQuadTriangleCount() != 0) return 1;
        if (verifyAlignStride() != 0) return 1;
        if (verifyDoubleAttribFormat() != 0) return 1;
        if (verifyIntegerAttribConversionFormat() != 0) return 1;
        if (verifyHashStepU64() != 0) return 1;
        if (verifyPrimitiveRestartFixedIndex() != 0) return 1;
        if (verifyGLTypeElementByteSize() != 0) return 1;
        if (verifyMDIScratchOwner() != 0) return 1;
        if (verifyRenderEncoderGetter() != 0) return 1;
        if (verifyCommandBufferGetterAndAdopt() != 0) return 1;
        if (verifyRenderPassIdentityOwner() != 0) return 1;
        if (verifyRenderPassStateOwner(device) != 0) return 1;
        if (verifyTextureStagingOwner() != 0) return 1;
        if (verifyTextureUploadEncoding(device) != 0) return 1;
        if (verifyRenderEncoderOwner(device) != 0) return 1;
        if (verifyQueryUtilities(device) != 0) return 1;
        if (verifyRawRenderAndBlitFacade(device) != 0) return 1;

        // 多 context 引用同一 device：重复 init 增加一个 renderer user。
        if (mglRenderCppInit((__bridge void*)device) != 0) {
            fprintf(stderr, "FAIL: shared-device init\n");
            return 1;
        }

        const uint32_t teardownValue = 0x10203040u;
        Buffer *teardownSlot = mglRenderCppAcquirePackedStructBuffer(
            &teardownValue, sizeof(teardownValue), NULL, 0);
        if (!teardownSlot || !teardownSlot->data.mtl_data) {
            fprintf(stderr, "FAIL: packed struct teardown fixture\n");
            return 1;
        }
        const uint64_t releaseCountBeforeShutdown = s_metalReleaseCount;

        mglRenderCppShutdown();
        if (mglRenderCppGetDevice() == NULL ||
            s_metalReleaseCount != releaseCountBeforeShutdown) {
            fprintf(stderr,
                    "FAIL: first shutdown dropped shared renderer state\n");
            return 1;
        }
        mglRenderCppShutdown();
        if (mglRenderCppGetDevice() != NULL ||
            s_metalReleaseCount != releaseCountBeforeShutdown + 128) {
            fprintf(stderr,
                    "FAIL: final shutdown did not clear renderer state\n");
            return 1;
        }
        mglRenderCppShutdown(); // 二次 shutdown 幂等

        // 重建
        if (mglRenderCppInit((__bridge void*)device) != 0) {
            fprintf(stderr, "FAIL: reinit\n");
            return 1;
        }
        mglRenderCppShutdown();

        printf("SMOKE_DONE\n");
    }
    return 0;
}

#undef EXPECT_EMIT
#undef EXPECT_SKIP
