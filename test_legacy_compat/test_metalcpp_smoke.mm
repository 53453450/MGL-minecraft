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
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"

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
        !detached) {
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
            0u, NULL, 3u, 0u, 0u) != 0 ||
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
            MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS, NULL, 0u, 0u, 0u) == 0 ||
        mglRenderCppSetRenderPassStateAttachmentTexture(
            owner, 0xffffffffu, 0u, NULL, 0u, 0u, 0u) == 0) {
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
     * count overflow and NULL buffer entries are rejected. */
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
        snap.vertex_buffers[snap.vertex_buffer_count++] =
            (MGLRenderCppBindingBufferEntry){
                (__bridge void *)snapshotBuffer, 0, 0};
        snap.fragment_buffers[snap.fragment_buffer_count++] =
            (MGLRenderCppBindingBufferEntry){
                (__bridge void *)snapshotBuffer, 16, 1};
        MGLRenderCppBindingSnapshot overflow = snap;
        overflow.vertex_buffer_count =
            MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_BUFFERS + 1;
        MGLRenderCppBindingSnapshot nullEntry = snap;
        nullEntry.vertex_buffers[0].buffer = NULL;
        char snapError[128] = {0};
        if (mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &snap, snapError, sizeof(snapError)) != 0 ||
            mglRenderCppEncodeBindingSnapshot(NULL, &snap, NULL, 0) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &overflow, snapError, sizeof(snapError)) != -1 ||
            mglRenderCppEncodeBindingSnapshot(
                stateEncoder, &nullEntry, snapError, sizeof(snapError)) != -1) {
            fprintf(stderr, "FAIL: binding snapshot encode\n");
            mglRenderCppDestroyRenderEncoderOwner(&adoptedStateEncoderOwner);
            mglRenderCppDestroyRenderPassStateOwner(&stateOwner);
            return 1;
        }
        printf("BINDING_SNAPSHOT_OK\n");
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
