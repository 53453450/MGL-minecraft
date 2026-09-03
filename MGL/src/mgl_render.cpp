/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------

//


//

//------------------------------------------------------------------------------------------------
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_capability.h"
#include "mgl_sync.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>

extern "C" void mglMetalCountRelease(int kind);
extern "C" void mglMetalCountCreate(int kind);
extern "C" void mglRecordBufferCowSnapshot(uint64_t bytes);

static_assert(MGLTextureType2D == static_cast<uint32_t>(MTL::TextureType2D));
static_assert(MGLTextureType3D == static_cast<uint32_t>(MTL::TextureType3D));
static_assert(MGLTextureUsageRenderTarget ==
              static_cast<uint32_t>(MTL::TextureUsageRenderTarget));
static_assert(MGLStorageModePrivate ==
              static_cast<uint32_t>(MTL::StorageModePrivate));
static_assert(MGLLoadActionClear == static_cast<uint32_t>(MTL::LoadActionClear));
static_assert(MGLStoreActionMultisampleResolve ==
              static_cast<uint32_t>(MTL::StoreActionMultisampleResolve));
static_assert(MGLCompareFunctionAlways ==
              static_cast<uint32_t>(MTL::CompareFunctionAlways));
static_assert(MGLCommandBufferStatusError ==
              static_cast<uint32_t>(MTL::CommandBufferStatusError));
static_assert(MGLPrimitiveTypeTriangleStrip ==
              static_cast<uint32_t>(MTL::PrimitiveTypeTriangleStrip));
static_assert(MGLWindingCounterClockwise ==
              static_cast<uint32_t>(MTL::WindingCounterClockwise));
static_assert(MGLColorWriteMaskAll ==
              static_cast<uint32_t>(MTL::ColorWriteMaskAll));
static_assert(MGLTessellationControlPointIndexTypeUInt32 ==
              static_cast<uint32_t>(MTL::TessellationControlPointIndexTypeUInt32));
static_assert(MGLBlendFactorOneMinusSource1Alpha ==
              static_cast<uint32_t>(MTL::BlendFactorOneMinusSource1Alpha));
static_assert(MGLBlendOperationMax ==
              static_cast<uint32_t>(MTL::BlendOperationMax));
static_assert(MGLVertexFormatHalf ==
              static_cast<uint32_t>(MTL::VertexFormatHalf));

namespace mgl {

MTL::Device* wrapDevice(void* objcDevice) {


    MTL::Device* device = static_cast<MTL::Device*>(objcDevice);
    if (device) {
        device->retain();
    }
    return device;
}

namespace {

constexpr uint32_t kMGLMaxBufferSlots = 31;
constexpr size_t kPackedStructBufferCapacity = 128;
constexpr size_t kMinimumStageBindingSize = 256;

enum MetalObjectKind {
    kMetalKindBuffer = 0,
    kMetalKindTexture = 1,
    kMetalKindSampler = 2,
    kMetalKindLibrary = 3,
    kMetalKindFunction = 4,
    kMetalKindPipeline = 5,
    kMetalKindOther = 6,
};

int metalObjectKind(void* object) {
    if (!object) return kMetalKindOther;
#ifdef __OBJC__
    id objcObject = (__bridge id)object;
#else
    id objcObject = reinterpret_cast<id>(object);
#endif
    const char* className = object_getClassName(objcObject);
    if (!className) return kMetalKindOther;
    if (std::strstr(className, "Buffer")) return kMetalKindBuffer;
    if (std::strstr(className, "Texture")) return kMetalKindTexture;
    if (std::strstr(className, "Sampler")) return kMetalKindSampler;
    if (std::strstr(className, "Library")) return kMetalKindLibrary;
    if (std::strstr(className, "Function")) return kMetalKindFunction;
    if (std::strstr(className, "Pipeline")) return kMetalKindPipeline;
    return kMetalKindOther;
}

void releaseBridgedObject(void** slot) {
    if (!slot || !*slot) return;
    void* object = *slot;
    *slot = nullptr;
    mglMetalCountRelease(metalObjectKind(object));
    static_cast<NS::Object*>(object)->release();
}

int loadAIRMainFunction(MTL::Device* device,
                        const unsigned char* bytes,
                        size_t size,
                        void** libraryOut,
                        void** functionOut,
                        char* err,
                        size_t errcap) {
    void* library = nullptr;
    if (mglAirLoadLibrary(device, bytes, size, &library, err, errcap) != 0 ||
        !library) {
        return -1;
    }
    MTL::Function* function = static_cast<MTL::Library*>(library)->newFunction(
        NS::String::string("main", NS::UTF8StringEncoding));
    if (!function) {
        static_cast<MTL::Library*>(library)->release();
        if (err && errcap) snprintf(err, errcap, "function 'main' not found");
        return -1;
    }
    *libraryOut = library;
    *functionOut = function;
    return 0;
}

struct ComputePipelineKey {
    uintptr_t function = 0;
    uint64_t programInstance = 0;
    uint64_t programGeneration = 0;
    uint32_t stage = 0;

    bool operator<(const ComputePipelineKey& other) const {
        return std::tie(programInstance, programGeneration, stage, function) <
               std::tie(other.programInstance, other.programGeneration,
                        other.stage, other.function);
    }
};

struct AuxComputePipelineKey {
    uint32_t kind = 0;
    uint64_t variant = 0;

    bool operator<(const AuxComputePipelineKey& other) const {
        return std::tie(kind, variant) <
               std::tie(other.kind, other.variant);
    }
};

struct AuxRenderPipelineKey {
    uint32_t kind = 0;
    uint64_t variant = 0;
    uint32_t colorFormat = 0;
    uint32_t depthFormat = 0;
    uint32_t stencilFormat = 0;
    uint32_t colorWriteMask = 0;
    uint32_t rasterSampleCount = 0;
    int icbEnabled = 0;

    bool operator<(const AuxRenderPipelineKey& other) const {
        return std::tie(kind, variant, colorFormat, depthFormat,
                        stencilFormat, colorWriteMask, rasterSampleCount,
                        icbEnabled) <
               std::tie(other.kind, other.variant, other.colorFormat,
                        other.depthFormat, other.stencilFormat,
                        other.colorWriteMask, other.rasterSampleCount,
                        other.icbEnabled);
    }
};

struct PipelineCacheKey {
    std::array<uint64_t, MGL_RENDER_PIPELINE_CACHE_KEY_WORDS> words{};

    bool operator<(const PipelineCacheKey& other) const {
        return words < other.words;
    }

    bool operator==(const PipelineCacheKey& other) const {
        return words == other.words;
    }
};

struct DepthStencilCacheKey {
    std::array<uint32_t, 16> words{};

    bool operator<(const DepthStencilCacheKey& other) const {
        return words < other.words;
    }

    bool operator==(const DepthStencilCacheKey& other) const {
        return words == other.words;
    }
};

struct PipelineCacheEntry {
    ~PipelineCacheEntry() {
        if (pipeline) pipeline->release();
        if (vertexFunction) vertexFunction->release();
        if (fragmentFunction) fragmentFunction->release();
    }

    MTL::RenderPipelineState* pipeline = nullptr;
    MTL::Function* vertexFunction = nullptr;
    MTL::Function* fragmentFunction = nullptr;
};

struct PipelineCacheDescriptorEntry {
    MGLRenderPipelineDescriptorState state{};
};

struct PipelineCacheDepthStencilEntry {
    ~PipelineCacheDepthStencilEntry() {
        if (state) state->release();
    }

    MTL::DepthStencilState* state = nullptr;
};

struct PipelineCacheOwner {
    ~PipelineCacheOwner() {
        clearBinaryArchive();
        reset();
    }

    void clearCaches() {
        releaseObject(active.pipeline_state);
        releaseObject(active.vertex_function);
        releaseObject(active.fragment_function);
        pipelineCache.clear();
        pipelineCacheLRU.clear();
        descriptorCache.clear();
        descriptorCacheLRU.clear();
        depthStencilCache.clear();
        depthStencilCacheLRU.clear();
    }

    void reset() {
        clearCaches();
        blend = {};
        active = {};
        active.color0_format = static_cast<uint32_t>(MTL::PixelFormatInvalid);
        active.depth_format = static_cast<uint32_t>(MTL::PixelFormatInvalid);
        active.stencil_format = static_cast<uint32_t>(MTL::PixelFormatInvalid);
    }

    void clearBinaryArchive() {
        if (binaryArchive) binaryArchive->release();
        binaryArchive = nullptr;
        binaryArchiveKey.clear();
    }

    static void retainObject(void* object) {
        if (object) static_cast<NS::Object*>(object)->retain();
    }

    static void releaseObject(void*& object) {
        if (object) static_cast<NS::Object*>(object)->release();
        object = nullptr;
    }

    static PipelineCacheKey makeKey(
        const uint64_t words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS]) {
        PipelineCacheKey key;
        if (words) {
            std::copy(words,
                      words + MGL_RENDER_PIPELINE_CACHE_KEY_WORDS,
                      key.words.begin());
        }
        return key;
    }

    static DepthStencilCacheKey makeDepthStencilKey(
        const MGLRenderDepthStencilDescriptorState& descriptor) {
        DepthStencilCacheKey key;
        key.words[0] = descriptor.depth_compare_function;
        key.words[1] = descriptor.depth_write_enabled;
        const MGLRenderStencilDescriptorState* stencils[] = {
            &descriptor.front, &descriptor.back};
        size_t cursor = 2;
        for (const MGLRenderStencilDescriptorState* stencil : stencils) {
            key.words[cursor++] = stencil->present;
            key.words[cursor++] = stencil->compare_function;
            key.words[cursor++] = stencil->read_mask;
            key.words[cursor++] = stencil->write_mask;
            key.words[cursor++] = stencil->stencil_failure_operation;
            key.words[cursor++] = stencil->depth_failure_operation;
            key.words[cursor++] = stencil->depth_stencil_pass_operation;
        }
        return key;
    }

    static void touch(std::list<PipelineCacheKey>& lru,
                      const PipelineCacheKey& key) {
        lru.remove(key);
        lru.push_back(key);
    }

    static void touch(std::list<DepthStencilCacheKey>& lru,
                      const DepthStencilCacheKey& key) {
        lru.remove(key);
        lru.push_back(key);
    }

    std::mutex mutex;
    bool psoDedupEnabled = true;
    bool depthStencilCacheEnabled = true;
    bool binaryArchiveEnabled = false;
    MTL::BinaryArchive* binaryArchive = nullptr;
    std::string binaryArchiveKey;
    MGLRenderPipelineActiveState active{
        nullptr, nullptr, nullptr,
        static_cast<uint32_t>(MTL::PixelFormatInvalid),
        static_cast<uint32_t>(MTL::PixelFormatInvalid),
        static_cast<uint32_t>(MTL::PixelFormatInvalid), 0};
    std::array<MGLRenderPipelineBlendState,
               MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS> blend{};
    std::map<PipelineCacheKey, std::unique_ptr<PipelineCacheEntry>>
        pipelineCache;
    std::list<PipelineCacheKey> pipelineCacheLRU;
    std::map<PipelineCacheKey,
             std::unique_ptr<PipelineCacheDescriptorEntry>> descriptorCache;
    std::list<PipelineCacheKey> descriptorCacheLRU;
    std::map<DepthStencilCacheKey,
             std::unique_ptr<PipelineCacheDepthStencilEntry>> depthStencilCache;
    std::list<DepthStencilCacheKey> depthStencilCacheLRU;
};

struct BufferCowSlot {
    MTL::Buffer* buffer = nullptr;
    uint64_t lastUseGeneration = 0;
};

struct BufferCowPool {
    ~BufferCowPool() {
        for (BufferCowSlot& slot : slots) {
            if (slot.buffer) slot.buffer->release();
        }
    }

    std::vector<BufferCowSlot> slots;
};

std::atomic<uint64_t> gBufferFrameGeneration{0};
std::atomic<uint64_t> gBufferCompletedGeneration{0};

struct BufferCowSnapshot {
    MTL::Buffer* buffer = nullptr;
    bool poolOwnsReference = false;
};

struct ConvertedVertexBufferKey {
    uint64_t sourceHash = 0;
    uint64_t copyLength = 0;
    uint64_t originalStride = 0;
    uint64_t convertedStride = 0;
    int64_t bindingOffset = 0;
    int64_t relativeOffset = 0;
    uint32_t sourceName = 0;
    uint32_t kind = 0;
    uint32_t componentCount = 0;
    uint32_t sourceType = 0;
    uint32_t normalized = 0;
    uint32_t destinationSigned = 0;

    bool operator<(const ConvertedVertexBufferKey& other) const {
        return std::tie(sourceName, kind, componentCount, sourceType,
                        normalized, destinationSigned, bindingOffset,
                        relativeOffset, originalStride, convertedStride,
                        copyLength, sourceHash) <
               std::tie(other.sourceName, other.kind, other.componentCount,
                        other.sourceType, other.normalized,
                        other.destinationSigned, other.bindingOffset,
                        other.relativeOffset, other.originalStride,
                        other.convertedStride, other.copyLength,
                        other.sourceHash);
    }
};

uint64_t hashVertexBytes(const uint8_t* bytes, size_t length) {
    uint64_t hash = 1469598103934665603ull;
    if (!bytes) return hash;
    for (size_t i = 0; i < length; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

bool alignVertexStride(size_t stride, size_t* alignedOut) {
    if (!alignedOut || stride > std::numeric_limits<size_t>::max() - 3u) {
        return false;
    }
    *alignedOut = (stride + 3u) & ~size_t{3u};
    return *alignedOut != 0;
}

size_t vertexComponentSize(uint32_t type) {
    switch (type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return 4;
        case GL_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

float decodeUnsignedFloatComponent(uint32_t value, uint32_t mantissaBits) {
    const uint32_t mantissaMask = (1u << mantissaBits) - 1u;
    const uint32_t exponent = (value >> mantissaBits) & 0x1fu;
    const uint32_t mantissa = value & mantissaMask;
    if (exponent == 0u) {
        return static_cast<float>(
            (static_cast<double>(mantissa) / (1u << mantissaBits)) /
            16384.0);
    }
    if (exponent == 31u) {
        return mantissa
            ? std::numeric_limits<float>::quiet_NaN()
            : std::numeric_limits<float>::infinity();
    }
    return std::ldexp(
        static_cast<float>(
            1.0 + static_cast<double>(mantissa) / (1u << mantissaBits)),
        static_cast<int>(exponent) - 15);
}

bool vertexConversionSource(Buffer* source,
                            const uint8_t** bytesOut,
                            size_t* sizeOut) {
    if (!source || !bytesOut || !sizeOut) return false;
    *bytesOut = nullptr;
    *sizeOut = 0;
    if (source->data.buffer_data && source->size > 0) {
        *bytesOut = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(source->data.buffer_data));
        *sizeOut = static_cast<size_t>(source->size);
        return true;
    }
    MTL::Buffer* metal =
        static_cast<MTL::Buffer*>(source->data.mtl_data);
    if (!metal || !metal->contents() || metal->length() == 0) return false;
    *bytesOut = static_cast<const uint8_t*>(metal->contents());
    *sizeOut = static_cast<size_t>(metal->length());
    return true;
}

BufferCowPool* bufferCowPool(Buffer* owner, bool create) {
    if (!owner) return nullptr;
    BufferCowPool* pool =
        static_cast<BufferCowPool*>(owner->mtl_cow_pool);
    if (!pool && create) {
        pool = new (std::nothrow) BufferCowPool();
        owner->mtl_cow_pool = pool;
    }
    return pool;
}

BufferCowSnapshot takeBufferCowSnapshot(MTL::Device* device,
                                        MTL::Buffer* oldBuffer,
                                        size_t length,
                                        MTL::ResourceOptions options,
                                        Buffer* owner) {
    BufferCowPool* pool = bufferCowPool(owner, true);
    const uint64_t completed =
        gBufferCompletedGeneration.load(std::memory_order_acquire);
    if (pool) {
        for (BufferCowSlot& slot : pool->slots) {
            if (!slot.buffer || slot.buffer == oldBuffer ||
                completed < slot.lastUseGeneration) {
                continue;
            }
            return {slot.buffer, true};
        }
    }

    MTL::Buffer* snapshot = device
        ? device->newBuffer(static_cast<NS::UInteger>(length), options)
        : nullptr;
    if (!snapshot) return {};

    if (pool && pool->slots.size() < 4) {
        try {
            pool->slots.push_back({snapshot, 0});
            return {snapshot, true};
        } catch (const std::bad_alloc&) {
            /* Transfer the +1 newBuffer reference directly to Buffer when the
             * reuse pool cannot grow. */
        }
    }
    return {snapshot, false};
}

bool bufferShadowUploadRange(const Buffer* buffer,
                             size_t limit,
                             size_t* offsetOut,
                             size_t* lengthOut) {
    size_t offset = 0;
    size_t length = limit;
    if (buffer->gpu_write_target) {
        if (buffer->written_min < 0 ||
            buffer->written_max <= buffer->written_min) {
            return false;
        }
        offset = std::min(static_cast<size_t>(buffer->written_min), limit);
        const size_t end =
            std::min(static_cast<size_t>(buffer->written_max), limit);
        length = end - offset;
    }
    if (length == 0) return false;
    *offsetOut = offset;
    *lengthOut = length;
    return true;
}

void installBufferCowSnapshot(Buffer* owner,
                              const BufferCowSnapshot& snapshot) {
    if (snapshot.poolOwnsReference) snapshot.buffer->retain();
    releaseBridgedObject(&owner->data.mtl_data);
    owner->data.mtl_data = snapshot.buffer;
}

struct BindingState {
    explicit BindingState(uint32_t textureSlotCount)
        : vertexBuffers(kMGLMaxBufferSlots, nullptr),
          fragmentBuffers(kMGLMaxBufferSlots, nullptr),
          vertexBufferOffsets(kMGLMaxBufferSlots, 0),
          fragmentBufferOffsets(kMGLMaxBufferSlots, 0),
          vertexTextures(textureSlotCount, nullptr),
          fragmentTextures(textureSlotCount, nullptr),
          vertexSamplers(textureSlotCount, nullptr),
          fragmentSamplers(textureSlotCount, nullptr) {}

    ~BindingState() { invalidate(); }

    void invalidate() {
        releaseObjects(vertexBuffers);
        releaseObjects(fragmentBuffers);
        std::fill(vertexBufferOffsets.begin(), vertexBufferOffsets.end(), 0);
        std::fill(fragmentBufferOffsets.begin(), fragmentBufferOffsets.end(), 0);
        vertexBufferMask = 0;
        fragmentBufferMask = 0;
        textureSlotMask[0] = 0;
        textureSlotMask[1] = 0;
        replaceObject(pipelineState, static_cast<MTL::RenderPipelineState*>(nullptr));
        replaceObject(depthStencilState,
                      static_cast<MTL::DepthStencilState*>(nullptr));
        lastCullMode = MTL::CullModeNone;
        lastWinding = MTL::WindingClockwise;
        lastDepthBias = 0.0f;
        lastDepthBiasClamp = 0.0f;
        lastDepthSlopeScale = 0.0f;
        lastBlendColorRed = 0.0f;
        lastBlendColorGreen = 0.0f;
        lastBlendColorBlue = 0.0f;
        lastBlendColorAlpha = 0.0f;
        releaseObjects(vertexTextures);
        releaseObjects(fragmentTextures);
        releaseObjects(vertexSamplers);
        releaseObjects(fragmentSamplers);
        viewport = {};
        viewport.zfar = 1.0;
        viewportCount = 0;
        scissor = {};
        triangleFillMode = MTL::TriangleFillModeFill;
        valid = false;
    }

    template <typename T>
    static void releaseObjects(std::vector<T*>& objects) {
        for (T*& object : objects) {
            if (object) object->release();
            object = nullptr;
        }
    }

    template <typename T>
    static void replaceObject(T*& destination, T* object) {
        if (object) object->retain();
        if (destination) destination->release();
        destination = object;
    }

    std::vector<MTL::Buffer*> vertexBuffers;
    std::vector<MTL::Buffer*> fragmentBuffers;
    std::vector<uint64_t> vertexBufferOffsets;
    std::vector<uint64_t> fragmentBufferOffsets;
    uint32_t vertexBufferMask = 0;
    uint32_t fragmentBufferMask = 0;
    uint64_t textureSlotMask[2] = {0, 0};
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL::CullMode lastCullMode = MTL::CullModeNone;
    MTL::Winding lastWinding = MTL::WindingClockwise;
    float lastDepthBias = 0.0f;
    float lastDepthBiasClamp = 0.0f;
    float lastDepthSlopeScale = 0.0f;
    float lastBlendColorRed = 0.0f;
    float lastBlendColorGreen = 0.0f;
    float lastBlendColorBlue = 0.0f;
    float lastBlendColorAlpha = 0.0f;
    std::vector<MTL::Texture*> vertexTextures;
    std::vector<MTL::Texture*> fragmentTextures;
    std::vector<MTL::SamplerState*> vertexSamplers;
    std::vector<MTL::SamplerState*> fragmentSamplers;
    MTL::Viewport viewport = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    MTL::Viewport viewports[MGL_MAX_VIEWPORTS];
    uint64_t viewportCount = 0;
    MTL::ScissorRect scissor = {0, 0, 0, 0};
    MTL::TriangleFillMode triangleFillMode = MTL::TriangleFillModeFill;
    bool valid = false;
    MGLRenderBindingStats stats = {};
};

struct Renderer {
    MTL::Device* device = nullptr;
    uint32_t users = 0;
    std::mutex mutex;
    std::map<ComputePipelineKey, MTL::ComputePipelineState*> computePipelines;
    std::map<AuxComputePipelineKey, MTL::ComputePipelineState*>
        auxComputePipelines;
    std::map<AuxRenderPipelineKey, MTL::RenderPipelineState*>
        auxRenderPipelines;
    /* Precompiled aux shader asset libraries (mgl_aux_assets table), keyed by
     * their FNV-1a hash, owned by the renderer until shutdown.  Functions from
     * these libraries are always +1 refs handed to callers. */
    std::map<uint64_t, MTL::Library*> auxLibraries;
    /* Process-wide archive registry mirrors the former ObjC shared dictionary.
     * Each map entry owns one reference; PipelineCacheOwner retains its own. */
    std::map<std::string, MTL::BinaryArchive*> binaryArchives;
    std::map<ConvertedVertexBufferKey, MTL::Buffer*>
        convertedVertexBuffers;
    std::array<Buffer*, kPackedStructBufferCapacity> packedStructBuffers{};
    size_t packedStructBufferIndex = 0;
    std::set<BindingState*> bindingStates;
};

struct CommandQueueOwner {
    ~CommandQueueOwner() {
        if (queue) queue->release();
    }

    MTL::CommandQueue* queue = nullptr;
};


struct CommandBufferSyncList {
    ~CommandBufferSyncList() { free(list); }

    Sync** list = nullptr;
    uint32_t count = 0;
    uint32_t size = 0;

    void reset() {
        if (list && count) {
            memset(list, 0, sizeof(Sync*) * count);
        }
        count = 0;
    }
};

struct CommandBufferOwner {
    ~CommandBufferOwner() {
        if (lastSubmitted) lastSubmitted->release();
        if (current) current->release();
        if (queue) queue->release();
    }

    /* Retained only for owners created from the C++ queue facade. Adopted
     * ObjC buffers intentionally leave this null as a fallback. */
    MTL::CommandQueue* queue = nullptr;
    MTL::CommandBuffer* current = nullptr;
    /* Most recently accepted submission.  The owner retains this buffer so
     * finish/readback paths can wait through value-state APIs instead of
     * mirroring command-buffer lifetime in Objective-C ivars. */
    MTL::CommandBuffer* lastSubmitted = nullptr;
    CommandBufferSyncList syncs;
    bool commit_in_progress = false;
    /* Set only by a submit transaction that rotated the owner.  Keeping this
     * bit beside `current` avoids an ObjC lifecycle mirror while allowing the
     * caller to consume the already-created buffer exactly once. */
    bool transaction_created_current = false;
};

void setLastSubmitted(CommandBufferOwner* owner,
                      MTL::CommandBuffer* commandBuffer) {
    if (!owner || owner->lastSubmitted == commandBuffer) return;
    if (commandBuffer) commandBuffer->retain();
    if (owner->lastSubmitted) owner->lastSubmitted->release();
    owner->lastSubmitted = commandBuffer;
}

struct CommandBufferRecoveryOwner {
    std::mutex mutex;
    std::atomic<uint32_t> references{1};
    uint64_t consecutiveErrors = 0;
    uint64_t consecutiveSuccesses = 0;
    double lastErrorTime = 0.0;
    bool recoveryMode = false;
    bool resetRequested = false;
};

void retainCommandRecoveryOwner(CommandBufferRecoveryOwner* owner) {
    if (owner) {
        owner->references.fetch_add(1, std::memory_order_relaxed);
    }
}

void releaseCommandRecoveryOwner(CommandBufferRecoveryOwner* owner) {
    if (owner && owner->references.fetch_sub(
                      1, std::memory_order_acq_rel) == 1) {
        delete owner;
    }
}

void snapshotCommandRecovery(
    const CommandBufferRecoveryOwner& owner,
    MGLRenderCommandRecoverySnapshot* state) {
    if (!state) return;
    state->consecutive_errors = owner.consecutiveErrors;
    state->consecutive_successes = owner.consecutiveSuccesses;
    state->last_error_time = owner.lastErrorTime;
    state->recovery_mode = owner.recoveryMode ? 1u : 0u;
}


struct PendingEventOwner {
    ~PendingEventOwner() { if (event) event->release(); }

    MTL::Event* event = nullptr;
    GLsizei sync_name = 0;
};

struct CommandBufferSubmission {
    ~CommandBufferSubmission() {
        if (buffer) buffer->release();
    }

    MTL::CommandBuffer* buffer = nullptr;
};

struct MDIScratchOwner {
    ~MDIScratchOwner() {
        if (buffer) buffer->release();
    }

    MTL::Buffer* buffer = nullptr;
    uint64_t capacity = 0;
    uint64_t offset = 0;
};

struct RenderPassIdentityOwner {
    MGLRenderPassIdentityState state{};
    MGLRenderFboMatchCacheState cache{};
    bool cache_valid = false;
};

void retainRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->retain();
}

void releaseRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->release();
}

void retainRenderPassStateResources(
    const MGLRenderPassState& state) {
    for (uint32_t index = 0;
         index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        retainRenderPassObject(state.color[index].attachment.texture);
        retainRenderPassObject(state.color[index].attachment.resolve_texture);
    }
    retainRenderPassObject(state.depth.attachment.texture);
    retainRenderPassObject(state.depth.attachment.resolve_texture);
    retainRenderPassObject(state.stencil.attachment.texture);
    retainRenderPassObject(state.stencil.attachment.resolve_texture);
    retainRenderPassObject(state.visibility_result_buffer);
    retainRenderPassObject(state.rasterization_rate_map);
}

void releaseRenderPassStateResources(
    const MGLRenderPassState& state) {
    for (uint32_t index = 0;
         index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        releaseRenderPassObject(state.color[index].attachment.texture);
        releaseRenderPassObject(state.color[index].attachment.resolve_texture);
    }
    releaseRenderPassObject(state.depth.attachment.texture);
    releaseRenderPassObject(state.depth.attachment.resolve_texture);
    releaseRenderPassObject(state.stencil.attachment.texture);
    releaseRenderPassObject(state.stencil.attachment.resolve_texture);
    releaseRenderPassObject(state.visibility_result_buffer);
    releaseRenderPassObject(state.rasterization_rate_map);
}

struct RenderPassStateOwner {
    ~RenderPassStateOwner() {
        releaseRenderPassStateResources(state);
    }

    MGLRenderPassState state{};
};

struct QueryStateOwner {
    ~QueryStateOwner() {
        if (visibilityBuffer) visibilityBuffer->release();
    }

    MTL::Buffer* visibilityBuffer = nullptr;
    uint32_t visibilitySlotCount = 0;
    uint32_t nextVisibilitySlot = 0;
    bool sampleQueryActive = false;
    bool sampleQueryCounting = false;
    uint64_t timerQueryBeginGPU = 0;
};

struct TextureStagingOwner {
    ~TextureStagingOwner() {
        if (buffer) buffer->release();
    }

    MTL::Buffer* buffer = nullptr;
};

struct RenderEncoderOwner {
    ~RenderEncoderOwner() {
        if (encoder) encoder->release();
    }

    MTL::RenderCommandEncoder* encoder = nullptr;
    bool ended = false;
};

void copyString(NS::String* string, char* out, size_t capacity) {
    if (!out || capacity == 0) return;
    out[0] = '\0';
    if (!string) return;
    const char* value = string->utf8String();
    if (value) snprintf(out, capacity, "%s", value);
}

int snapshotCommandBufferState(
    MTL::CommandBuffer* commandBuffer,
    MGLRenderCommandBufferState* state) {
    if (!commandBuffer || !state) return -1;
    memset(state, 0, sizeof(*state));
    state->status = static_cast<uint32_t>(commandBuffer->status());
    NS::Error* error = commandBuffer->error();
    if (!error) return 0;
    state->has_error = 1;
    state->error_code = static_cast<int64_t>(error->code());
    copyString(error->domain(), state->error_domain,
               sizeof(state->error_domain));
    copyString(error->localizedDescription(), state->error_description,
               sizeof(state->error_description));
    return 0;
}

struct CommandBufferCompletionContext {
    ~CommandBufferCompletionContext() { destroy(); }

    void retain() {
        references.fetch_add(1u, std::memory_order_relaxed);
    }

    void release() {
        if (references.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            delete this;
        }
    }

    void abandonCallerContext() {
        std::lock_guard<std::mutex> lock(mutex);
        context = nullptr;
        destroyContext = nullptr;
    }

    void configure(MGLRenderCommandBufferCompletion completionCallback,
                   void* callbackContext,
                   MGLRenderDestroyContext destroyFunction) {
        std::lock_guard<std::mutex> lock(mutex);
        completed = false;
        callback = completionCallback;
        context = callbackContext;
        destroyContext = destroyFunction;
    }

    void destroy() {
        void* value = nullptr;
        MGLRenderDestroyContext destroyFunction = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            value = std::exchange(context, nullptr);
            destroyFunction = std::exchange(destroyContext, nullptr);
        }
        if (value && destroyFunction) destroyFunction(value);
    }

    void complete(MTL::CommandBuffer* commandBuffer) {
        MGLRenderCommandBufferState state = {};
        snapshotCommandBufferState(commandBuffer, &state);
        void* callbackContext = nullptr;
        MGLRenderCommandBufferCompletion completionCallback = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (completed) return;
            completed = true;
            callbackContext = context;
            completionCallback = callback;
        }
        struct DestroyGuard {
            CommandBufferCompletionContext* owner;
            ~DestroyGuard() { owner->destroy(); }
        } guard{this};
        if (completionCallback) completionCallback(callbackContext, &state);
    }

    std::mutex mutex;
    std::atomic<uint32_t> references{1u};
    bool completed = false;
    MGLRenderCommandBufferCompletion callback = nullptr;
    void* context = nullptr;
    MGLRenderDestroyContext destroyContext = nullptr;
};

/* C auto-cleanup may call renderer shutdown after ordinary C++ static
 * destruction. Keep the container alive for the process and release Metal
 * objects only from the explicit shutdown boundary. */
Renderer& renderer() {
    static Renderer* instance = new Renderer();
    return *instance;
}

void copyError(NS::Error* error, char* out, size_t capacity) {
    if (!out || capacity == 0) return;
    if (error && error->localizedDescription()) {
        const char* message = error->localizedDescription()->utf8String();
        if (message) {
            snprintf(out, capacity, "%s", message);
            return;
        }
    }
    snprintf(out, capacity, "unknown Metal error");
}

void releasePipelineCaches(Renderer& renderer) {
    for (auto& entry : renderer.computePipelines) {
        if (entry.second) entry.second->release();
    }
    renderer.computePipelines.clear();
    for (auto& entry : renderer.auxComputePipelines) {
        if (entry.second) entry.second->release();
    }
    renderer.auxComputePipelines.clear();
    for (auto& entry : renderer.auxRenderPipelines) {
        if (entry.second) entry.second->release();
    }
    renderer.auxRenderPipelines.clear();
    for (auto& entry : renderer.auxLibraries) {
        if (entry.second) entry.second->release();
    }
    renderer.auxLibraries.clear();
    for (auto& entry : renderer.binaryArchives) {
        if (entry.second) entry.second->release();
    }
    renderer.binaryArchives.clear();
    for (auto& entry : renderer.convertedVertexBuffers) {
        if (entry.second) entry.second->release();
    }
    renderer.convertedVertexBuffers.clear();
}

void releaseBindingStates(Renderer& renderer) {
    for (BindingState* state : renderer.bindingStates) {
        delete state;
    }
    renderer.bindingStates.clear();
}

void releasePackedStructBuffers(Renderer& renderer) {
    for (Buffer*& buffer : renderer.packedStructBuffers) {
        if (!buffer) continue;
        releaseBridgedObject(&buffer->data.mtl_data);
        std::free(buffer);
        buffer = nullptr;
    }
    renderer.packedStructBufferIndex = 0;
}

void recordBindingResult(BindingState& state, uint32_t setter, bool emitted) {
    if (emitted) {
        state.stats.emitted[setter]++;
    } else {
        state.stats.skipped[setter]++;
    }
}

bool viewportEqual(const MTL::Viewport& lhs, const MTL::Viewport& rhs) {
    return lhs.originX == rhs.originX && lhs.originY == rhs.originY &&
           lhs.width == rhs.width && lhs.height == rhs.height &&
           lhs.znear == rhs.znear && lhs.zfar == rhs.zfar;
}

bool scissorEqual(const MTL::ScissorRect& lhs, const MTL::ScissorRect& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width &&
           lhs.height == rhs.height;
}

MTL::TextureDescriptor* newTextureDescriptor(
    const MGLRenderTextureDescriptorState* state) {
    if (!state || state->width == 0 || state->height == 0 ||
        state->depth == 0 || state->mipmap_level_count == 0 ||
        state->sample_count == 0 || state->array_length == 0) {
        return nullptr;
    }
    MTL::TextureDescriptor* descriptor =
        MTL::TextureDescriptor::alloc()->init();
    if (!descriptor) return nullptr;

    descriptor->setResourceOptions(
        static_cast<MTL::ResourceOptions>(state->resource_options));
    descriptor->setTextureType(
        static_cast<MTL::TextureType>(state->texture_type));
    descriptor->setPixelFormat(
        static_cast<MTL::PixelFormat>(state->pixel_format));
    descriptor->setWidth(static_cast<NS::UInteger>(state->width));
    descriptor->setHeight(static_cast<NS::UInteger>(state->height));
    descriptor->setDepth(static_cast<NS::UInteger>(state->depth));
    descriptor->setMipmapLevelCount(
        static_cast<NS::UInteger>(state->mipmap_level_count));
    descriptor->setSampleCount(
        static_cast<NS::UInteger>(state->sample_count));
    descriptor->setArrayLength(
        static_cast<NS::UInteger>(state->array_length));
    descriptor->setCpuCacheMode(
        static_cast<MTL::CPUCacheMode>(state->cpu_cache_mode));
    descriptor->setStorageMode(
        static_cast<MTL::StorageMode>(state->storage_mode));
    descriptor->setHazardTrackingMode(
        static_cast<MTL::HazardTrackingMode>(state->hazard_tracking_mode));
    descriptor->setUsage(static_cast<MTL::TextureUsage>(state->usage));
    descriptor->setCompressionType(
        static_cast<MTL::TextureCompressionType>(state->compression_type));
    descriptor->setPlacementSparsePageSize(
        static_cast<MTL::SparsePageSize>(
            state->placement_sparse_page_size));
    descriptor->setAllowGPUOptimizedContents(
        state->allow_gpu_optimized_contents != 0);
    if (state->has_swizzle) {
        descriptor->setSwizzle(MTL::TextureSwizzleChannels(
            static_cast<MTL::TextureSwizzle>(state->swizzle_red),
            static_cast<MTL::TextureSwizzle>(state->swizzle_green),
            static_cast<MTL::TextureSwizzle>(state->swizzle_blue),
            static_cast<MTL::TextureSwizzle>(state->swizzle_alpha)));
    }
    return descriptor;
}

void applyRenderPassAttachmentState(
    MTL::RenderPassAttachmentDescriptor* attachment,
    const MGLRenderPassAttachmentState& state) {
    attachment->setTexture(static_cast<MTL::Texture*>(state.texture));
    attachment->setResolveTexture(
        static_cast<MTL::Texture*>(state.resolve_texture));
    attachment->setLevel(static_cast<NS::UInteger>(state.level));
    attachment->setSlice(static_cast<NS::UInteger>(state.slice));
    attachment->setDepthPlane(
        static_cast<NS::UInteger>(state.depth_plane));
    attachment->setResolveLevel(
        static_cast<NS::UInteger>(state.resolve_level));
    attachment->setResolveSlice(
        static_cast<NS::UInteger>(state.resolve_slice));
    attachment->setResolveDepthPlane(
        static_cast<NS::UInteger>(state.resolve_depth_plane));
    attachment->setLoadAction(
        static_cast<MTL::LoadAction>(state.load_action));
    attachment->setStoreAction(
        static_cast<MTL::StoreAction>(state.store_action));
    attachment->setStoreActionOptions(
        static_cast<MTL::StoreActionOptions>(state.store_action_options));
}

MTL::RenderPassDescriptor* newRenderPassDescriptor(
    const MGLRenderPassState* state) {
    if (!state) return nullptr;
    MTL::RenderPassDescriptor* descriptor =
        MTL::RenderPassDescriptor::alloc()->init();
    if (!descriptor) return nullptr;

    MTL::RenderPassColorAttachmentDescriptorArray* colors =
        descriptor->colorAttachments();
    for (uint32_t index = 0;
         index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        MTL::RenderPassColorAttachmentDescriptor* attachment =
            colors->object(index);
        applyRenderPassAttachmentState(attachment,
                                       state->color[index].attachment);
        attachment->setClearColor(MTL::ClearColor::Make(
            state->color[index].clear_red,
            state->color[index].clear_green,
            state->color[index].clear_blue,
            state->color[index].clear_alpha));
    }

    MTL::RenderPassDepthAttachmentDescriptor* depth =
        descriptor->depthAttachment();
    applyRenderPassAttachmentState(depth, state->depth.attachment);
    depth->setClearDepth(state->depth.clear_depth);
    depth->setDepthResolveFilter(
        static_cast<MTL::MultisampleDepthResolveFilter>(
            state->depth.resolve_filter));

    MTL::RenderPassStencilAttachmentDescriptor* stencil =
        descriptor->stencilAttachment();
    applyRenderPassAttachmentState(stencil, state->stencil.attachment);
    stencil->setClearStencil(state->stencil.clear_stencil);
    stencil->setStencilResolveFilter(
        static_cast<MTL::MultisampleStencilResolveFilter>(
            state->stencil.resolve_filter));

    descriptor->setVisibilityResultBuffer(
        static_cast<MTL::Buffer*>(state->visibility_result_buffer));
    descriptor->setRasterizationRateMap(
        static_cast<MTL::RasterizationRateMap*>(
            state->rasterization_rate_map));
    descriptor->setRenderTargetArrayLength(
        static_cast<NS::UInteger>(state->render_target_array_length));
    descriptor->setRenderTargetWidth(
        static_cast<NS::UInteger>(state->render_target_width));
    descriptor->setRenderTargetHeight(
        static_cast<NS::UInteger>(state->render_target_height));
    descriptor->setDefaultRasterSampleCount(
        static_cast<NS::UInteger>(state->default_raster_sample_count));
    descriptor->setImageblockSampleLength(
        static_cast<NS::UInteger>(state->imageblock_sample_length));
    descriptor->setThreadgroupMemoryLength(
        static_cast<NS::UInteger>(state->threadgroup_memory_length));
    descriptor->setTileWidth(static_cast<NS::UInteger>(state->tile_width));
    descriptor->setTileHeight(static_cast<NS::UInteger>(state->tile_height));
    descriptor->setVisibilityResultType(
        static_cast<MTL::VisibilityResultType>(
            state->visibility_result_type));
    descriptor->setSupportColorAttachmentMapping(
        state->support_color_attachment_mapping != 0);

    const uint32_t sampleCount = std::min(
        state->sample_position_count,
        static_cast<uint32_t>(MGL_RENDER_MAX_SAMPLE_POSITIONS));
    if (sampleCount > 0) {
        MTL::SamplePosition positions[MGL_RENDER_MAX_SAMPLE_POSITIONS];
        for (uint32_t index = 0; index < sampleCount; ++index) {
            positions[index] = MTL::SamplePosition::Make(
                state->sample_positions[index].x,
                state->sample_positions[index].y);
        }
        descriptor->setSamplePositions(positions, sampleCount);
    }
    return descriptor;
}

MGLRenderPassState defaultRenderPassState() {
    MGLRenderPassState state = {};
    for (uint32_t index = 0;
         index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        state.color[index].attachment.store_action =
            static_cast<uint32_t>(MTL::StoreActionStore);
        state.color[index].clear_alpha = 1.0;
    }
    state.depth.attachment.store_action =
        static_cast<uint32_t>(MTL::StoreActionStore);
    state.depth.clear_depth = 1.0;
    state.stencil.attachment.store_action =
        static_cast<uint32_t>(MTL::StoreActionStore);
    return state;
}

struct CullDistanceIndexPlan {
    ~CullDistanceIndexPlan() {
        if (indexBuffer) indexBuffer->release();
    }

    MTL::Buffer* indexBuffer = nullptr;
    std::vector<MGLRenderCullDistancePrimitive> primitives;
};

bool readCullDistanceSourceIndex(const uint8_t* bytes,
                                 uint32_t type,
                                 uint64_t index,
                                 uint32_t& value) {
    if (!bytes) return false;
    switch (type) {
    case GL_UNSIGNED_BYTE:
        value = bytes[index];
        return true;
    case GL_UNSIGNED_SHORT: {
        uint16_t source = 0;
        std::memcpy(&source, bytes + index * sizeof(source), sizeof(source));
        value = source;
        return true;
    }
    case GL_UNSIGNED_INT:
        std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
        return true;
    default:
        return false;
    }
}

bool appendCullDistancePrimitive(
    std::vector<uint32_t>& expanded,
    std::vector<MGLRenderCullDistancePrimitive>& primitives,
    MTL::PrimitiveType primitiveType,
    const uint32_t* vertices,
    uint32_t vertexCount,
    const uint32_t* drawIndices,
    uint32_t indexCount,
    int64_t baseVertex) {
    if (!vertices || !drawIndices || vertexCount == 0 || vertexCount > 4 ||
        indexCount == 0 || expanded.size() >
            (std::numeric_limits<uint64_t>::max() / sizeof(uint32_t))) {
        return false;
    }

    MGLRenderCullDistancePrimitive primitive = {};
    primitive.vertex_count = vertexCount;
    primitive.primitive_type = static_cast<uint32_t>(primitiveType);
    primitive.index_count = indexCount;
    primitive.index_buffer_offset =
        static_cast<uint64_t>(expanded.size()) * sizeof(uint32_t);

    for (uint32_t index = 0; index < vertexCount; ++index) {
        const int64_t actual = static_cast<int64_t>(vertices[index]) +
                               baseVertex;
        if (actual < 0 ||
            static_cast<uint64_t>(actual) > UINT32_MAX) {
            return false;
        }
        primitive.vertices[index] = static_cast<uint32_t>(actual);
    }
    for (uint32_t index = 0; index < indexCount; ++index) {
        const int64_t actual = static_cast<int64_t>(drawIndices[index]) +
                               baseVertex;
        if (actual < 0 ||
            static_cast<uint64_t>(actual) > UINT32_MAX) {
            return false;
        }
        expanded.push_back(static_cast<uint32_t>(actual));
    }
    primitives.push_back(primitive);
    return true;
}

bool appendCullDistanceSegment(
    const std::vector<uint32_t>& source,
    size_t begin,
    size_t end,
    uint32_t mode,
    bool polygonLineMode,
    int64_t baseVertex,
    std::vector<uint32_t>& expanded,
    std::vector<MGLRenderCullDistancePrimitive>& primitives) {
    const size_t count = end - begin;
    auto append = [&](MTL::PrimitiveType primitiveType,
                      std::initializer_list<size_t> vertexOffsets,
                      std::initializer_list<size_t> drawOffsets) {
        uint32_t vertices[4] = {};
        uint32_t drawIndices[8] = {};
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        for (size_t offset : vertexOffsets) {
            vertices[vertexCount++] = source[begin + offset];
        }
        for (size_t offset : drawOffsets) {
            drawIndices[indexCount++] = source[begin + offset];
        }
        return appendCullDistancePrimitive(
            expanded, primitives, primitiveType, vertices, vertexCount,
            drawIndices, indexCount, baseVertex);
    };

    switch (mode) {
    case GL_POINTS:
        for (size_t i = 0; i < count; ++i) {
            if (!append(MTL::PrimitiveTypePoint, {i}, {i})) return false;
        }
        return true;
    case GL_LINES:
        for (size_t i = 0; i + 1 < count; i += 2) {
            if (!append(MTL::PrimitiveTypeLine, {i, i + 1}, {i, i + 1}))
                return false;
        }
        return true;
    case GL_LINE_STRIP:
        for (size_t i = 0; i + 1 < count; ++i) {
            if (!append(MTL::PrimitiveTypeLine, {i, i + 1}, {i, i + 1}))
                return false;
        }
        return true;
    case GL_LINE_LOOP:
        if (count < 2) return true;
        for (size_t i = 0; i < count; ++i) {
            const size_t next = (i + 1) % count;
            if (!append(MTL::PrimitiveTypeLine, {i, next}, {i, next}))
                return false;
        }
        return true;
    case GL_TRIANGLES:
        for (size_t i = 0; i + 2 < count; i += 3) {
            if (!append(MTL::PrimitiveTypeTriangle,
                        {i, i + 1, i + 2}, {i, i + 1, i + 2}))
                return false;
        }
        return true;
    case GL_TRIANGLE_STRIP:
        for (size_t i = 0; i + 2 < count; ++i) {
            const bool odd = (i & 1u) != 0u;
            if (!append(MTL::PrimitiveTypeTriangle,
                        {i, i + 1, i + 2},
                        odd ? std::initializer_list<size_t>{i + 1, i, i + 2}
                            : std::initializer_list<size_t>{i, i + 1, i + 2}))
                return false;
        }
        return true;
    case GL_TRIANGLE_FAN:
        for (size_t i = 1; i + 1 < count; ++i) {
            if (!append(MTL::PrimitiveTypeTriangle,
                        {0, i, i + 1}, {0, i, i + 1}))
                return false;
        }
        return true;
    case GL_QUADS:
        for (size_t i = 0; i + 3 < count; i += 4) {
            if (polygonLineMode) {
                if (!append(MTL::PrimitiveTypeLine,
                            {i, i + 1, i + 2, i + 3},
                            {i, i + 1, i + 1, i + 2,
                             i + 2, i + 3, i + 3, i}))
                    return false;
            } else if (!append(MTL::PrimitiveTypeTriangle,
                               {i, i + 1, i + 2, i + 3},
                               {i, i + 1, i + 2, i + 2, i + 3, i})) {
                return false;
            }
        }
        return true;
    default:
        return false;
    }
}

} // namespace

} // namespace mgl

//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
extern "C" {

void mglRenderInitDefaultRenderPassState(
    MGLRenderPassState* state_out) {
    if (!state_out) return;
    *state_out = mgl::defaultRenderPassState();
}

int mglRenderInit(void* objc_device) {
    if (!objc_device) {
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (renderer.device) {
        /* A process may own more than one GL context. They must share the
         * same Metal device, but each renderer balances its init/shutdown. */
        if (renderer.device !=
            static_cast<MTL::Device*>(objc_device)) {
            return -1;
        }
        renderer.users++;
        return 0;
    }
    renderer.device = mgl::wrapDevice(objc_device);
    if (!renderer.device) return -1;
    renderer.users = 1;
    return 0;
}

void mglRenderShutdown(void) {
    mgl::Renderer& renderer = mgl::renderer();
    {
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (renderer.users > 1) {
            renderer.users--;
            return;
        }
        renderer.users = 0;
        mgl::releasePipelineCaches(renderer);
        mgl::releaseBindingStates(renderer);
        mgl::releasePackedStructBuffers(renderer);
        mglAirLoaderShutdown();
        if (renderer.device) {
            renderer.device->release();
            renderer.device = nullptr;
        }
    }
}

int mglRenderIsInitialized(void) {
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    return renderer.device && renderer.users > 0 ? 1 : 0;
}

int mglRenderQueryCapability(void* device_ref,
                                MGLRenderCapabilityState* state_out) {
    if (!device_ref || !state_out) return -1;
    MTL::Device* device = static_cast<MTL::Device*>(device_ref);
    if (!device) return -1;

    MGLRenderCapabilityState state = {};
    NS::String* name_string = device->name();
    const char* name = name_string ? name_string->utf8String() : nullptr;
    const bool virtualized = name && std::strstr(name, "AGX") != nullptr;
    const bool apple_family = device->supportsFamily(MTL::GPUFamilyApple1);
    const bool apple_name = name && std::strncmp(name, "Apple ", 6) == 0;

    if (virtualized) {
        state.family = MGL_GPU_FAMILY_VIRTUALIZED;
        state.is_virtualized = 1;
    } else if (apple_family || apple_name) {
        state.family = MGL_GPU_FAMILY_AGX;
    } else {
        state.family = MGL_GPU_FAMILY_OTHER;
    }

    static constexpr uint64_t sample_counts[] = {32u, 16u, 8u, 4u, 2u};
    state.max_sample_count = 1;
    for (uint64_t sample_count : sample_counts) {
        if (device->supportsTextureSampleCount(
                static_cast<NS::UInteger>(sample_count))) {
            state.max_sample_count = sample_count;
            break;
        }
    }
    state.supports8x_msaa = state.max_sample_count >= 8 ? 1u : 0u;

    const bool agx = state.family == MGL_GPU_FAMILY_VIRTUALIZED ||
                     state.family == MGL_GPU_FAMILY_AGX;
    if (agx) {
        state.bug_3d_getbytes_slice_oob = 1;
        state.bug_3d_replace_region_nonzero_origin = 1;
        state.bug_3d_copy_from_buffer_slice_oob = 1;
        state.bug_msl_pipeline_rejection = 1;
        state.bug_async_shader_compile_in_vm = state.is_virtualized;
        state.conservative_cpu_cache_mode = 1;
        state.max_concurrent_command_buffers =
            state.is_virtualized ? 16u : 64u;
    } else {
        state.max_concurrent_command_buffers = 64u;
    }
    state.texture_alignment_bytes = 256u;
    state.command_buffer_recovery_limit = 4096u;
    *state_out = state;
    return 0;
}

int mglRenderLoadAIRMainFunction(const unsigned char* bytes,
                                    size_t size,
                                    void** library_out,
                                    void** function_out,
                                    char* err,
                                    size_t errcap) {
    if (library_out) *library_out = nullptr;
    if (function_out) *function_out = nullptr;
    if (!bytes || size == 0 || !library_out || !function_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "renderer not initialized");
        return -1;
    }
    return mgl::loadAIRMainFunction(
        renderer.device, bytes, size, library_out, function_out, err, errcap);
}

void mglRenderDeleteMTLObj(GLMContext glm_ctx, void* object) {
    (void)glm_ctx;
    mgl::releaseBridgedObject(&object);
}

void mglRenderReleaseBufferMetalData(GLMContext glm_ctx, Buffer* buffer) {
    if (!buffer || !buffer->data.mtl_data) return;
    (void)glm_ctx;
    mgl::releaseBridgedObject(&buffer->data.mtl_data);
}

void mglRenderReleaseBufferCowPool(Buffer* buffer) {
    if (!buffer || !buffer->mtl_cow_pool) return;
    mgl::BufferCowPool* pool =
        static_cast<mgl::BufferCowPool*>(buffer->mtl_cow_pool);
    buffer->mtl_cow_pool = nullptr;
    delete pool;
}

Buffer* mglRenderAcquirePackedStructBuffer(const void* data,
                                               size_t size,
                                               char* err,
                                               size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!data || size == 0) {
        if (err && errcap) snprintf(err, errcap, "invalid packed struct data");
        return nullptr;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
        return nullptr;
    }

    const size_t paddedSize =
        std::max(size, mgl::kMinimumStageBindingSize);
    MTL::Buffer* metalBuffer = renderer.device->newBuffer(
        static_cast<NS::UInteger>(paddedSize),
        MTL::ResourceStorageModeShared);
    if (!metalBuffer || !metalBuffer->contents()) {
        if (metalBuffer) metalBuffer->release();
        if (err && errcap) {
            snprintf(err, errcap,
                     "packed struct Metal buffer creation failed size=%zu",
                     paddedSize);
        }
        return nullptr;
    }
    std::memcpy(metalBuffer->contents(), data, size);
    if (paddedSize > size) {
        std::memset(static_cast<uint8_t*>(metalBuffer->contents()) + size,
                    0, paddedSize - size);
    }

    const size_t index = renderer.packedStructBufferIndex;
    Buffer*& slot = renderer.packedStructBuffers[index];
    if (!slot) {
        slot = static_cast<Buffer*>(std::calloc(1, sizeof(Buffer)));
        if (!slot) {
            metalBuffer->release();
            if (err && errcap) {
                snprintf(err, errcap,
                         "packed struct Buffer allocation failed");
            }
            return nullptr;
        }
        slot->name = 0xF0000000u | static_cast<GLuint>(index);
        slot->target = GL_UNIFORM_BUFFER;
        slot->usage = GL_STATIC_DRAW;
        slot->written_min = -1;
        slot->written_max = -1;
        slot->transient_batch_buffer = GL_TRUE;
    }

    mgl::releaseBridgedObject(&slot->data.mtl_data);
    mglMetalCountCreate(mgl::kMetalKindBuffer);
    slot->data.mtl_data = metalBuffer;
    slot->size = static_cast<GLsizeiptr>(paddedSize);
    slot->data.buffer_data = 0;
    slot->data.buffer_size = paddedSize;
    slot->data.dirty_bits = 0;
    slot->data.mtl_owns_buffer_data = GL_FALSE;
    slot->has_initialized_data = GL_TRUE;
    slot->ever_written = GL_TRUE;

    renderer.packedStructBufferIndex =
        (index + 1) % mgl::kPackedStructBufferCapacity;
    return slot;
}

uint64_t mglRenderAdvanceBufferGeneration(void) {
    return mgl::gBufferFrameGeneration.fetch_add(
               1, std::memory_order_acq_rel) + 1;
}

void mglRenderRecordBufferGenerationCompleted(uint64_t generation) {
    uint64_t completed =
        mgl::gBufferCompletedGeneration.load(std::memory_order_relaxed);
    while (generation > completed &&
           !mgl::gBufferCompletedGeneration.compare_exchange_weak(
               completed, generation, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
}

uint64_t mglRenderCompletedBufferGeneration(void) {
    return mgl::gBufferCompletedGeneration.load(std::memory_order_acquire);
}

void mglRenderNoteBufferEncoded(Buffer* buffer) {
    if (!buffer || !buffer->data.mtl_data) return;
    mgl::BufferCowPool* pool = mgl::bufferCowPool(buffer, false);
    if (!pool) return;
    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const uint64_t generation =
        mgl::gBufferFrameGeneration.load(std::memory_order_acquire);
    for (mgl::BufferCowSlot& slot : pool->slots) {
        if (slot.buffer == current) {
            slot.lastUseGeneration = generation;
            return;
        }
    }
}

int mglRenderSnapshotSharedDirtyBuffer(Buffer* buffer,
                                          void** metal_buffer_out,
                                          char* err,
                                          size_t errcap) {
    if (metal_buffer_out) *metal_buffer_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!buffer || !metal_buffer_out || !device) {
        if (err && errcap) snprintf(err, errcap, "bad arguments or renderer");
        return -1;
    }

    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    *metal_buffer_out = current;
    uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!current || buffer->transient_batch_buffer ||
        current->storageMode() != MTL::StorageModeShared || !cpuData ||
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0 ||
        cpuData == current->contents()) {
        return 0;
    }

    const size_t metalLength = static_cast<size_t>(current->length());
    size_t snapshotLength = metalLength;
    if (buffer->data.buffer_size > 0) {
        snapshotLength = std::min(snapshotLength, buffer->data.buffer_size);
    }
    if (snapshotLength == 0) return 0;

    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    if (current->cpuCacheMode() == MTL::CPUCacheModeWriteCombined) {
        options = static_cast<MTL::ResourceOptions>(
            options | MTL::ResourceCPUCacheModeWriteCombined);
    }
    mgl::BufferCowSnapshot snapshot = mgl::takeBufferCowSnapshot(
        device, current, metalLength, options, buffer);
    if (!snapshot.buffer || !snapshot.buffer->contents() ||
        !current->contents()) {
        if (snapshot.buffer && !snapshot.poolOwnsReference) {
            snapshot.buffer->release();
        }
        if (err && errcap) snprintf(err, errcap, "snapshot allocation failed");
        return -1;
    }

    uint8_t* snapshotData =
        static_cast<uint8_t*>(snapshot.buffer->contents());
    if (buffer->gpu_write_target) {
        memcpy(snapshotData, current->contents(), metalLength);
        size_t uploadOffset = 0;
        size_t uploadLength = 0;
        if (mgl::bufferShadowUploadRange(
                buffer, snapshotLength, &uploadOffset, &uploadLength)) {
            memcpy(snapshotData + uploadOffset, cpuData + uploadOffset,
                   uploadLength);
        }
    } else {
        memcpy(snapshotData, cpuData, snapshotLength);
        if (snapshotLength < metalLength) {
            memset(snapshotData + snapshotLength, 0,
                   metalLength - snapshotLength);
        }
    }

    mgl::installBufferCowSnapshot(buffer, snapshot);
    mglRenderNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderSnapshotSharedBufferRange(Buffer* buffer,
                                          size_t offset,
                                          size_t length,
                                          void** metal_buffer_out,
                                          char* err,
                                          size_t errcap) {
    if (metal_buffer_out) *metal_buffer_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!buffer || !metal_buffer_out || !device) {
        if (err && errcap) snprintf(err, errcap, "bad arguments or renderer");
        return -1;
    }

    MTL::Buffer* current =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    *metal_buffer_out = current;
    uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!current) return 0;
    const size_t metalLength = static_cast<size_t>(current->length());
    if (offset > metalLength || length > metalLength - offset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu length=%zu exceeds Metal length=%zu",
                     offset, length, metalLength);
        }
        return -1;
    }
    if (buffer->transient_batch_buffer ||
        current->storageMode() != MTL::StorageModeShared || !cpuData ||
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0 ||
        cpuData == current->contents()) {
        return 0;
    }

    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    if (current->cpuCacheMode() == MTL::CPUCacheModeWriteCombined) {
        options = static_cast<MTL::ResourceOptions>(
            options | MTL::ResourceCPUCacheModeWriteCombined);
    }
    mgl::BufferCowSnapshot snapshot = mgl::takeBufferCowSnapshot(
        device, current, metalLength, options, buffer);
    if (!snapshot.buffer || !snapshot.buffer->contents() ||
        !current->contents()) {
        if (snapshot.buffer && !snapshot.poolOwnsReference) {
            snapshot.buffer->release();
        }
        if (err && errcap) snprintf(err, errcap, "snapshot allocation failed");
        return -1;
    }

    memcpy(snapshot.buffer->contents(), current->contents(), metalLength);
    memcpy(static_cast<uint8_t*>(snapshot.buffer->contents()) + offset,
           cpuData + offset, length);
    mgl::installBufferCowSnapshot(buffer, snapshot);
    mglRenderNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderBindBufferStorage(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    constexpr size_t kMaxSafeBufferSize =
        static_cast<size_t>(2) * 1024u * 1024u * 1024u;
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_ERROR;
    }

    if (buffer->size <= 0 ||
        static_cast<size_t>(buffer->size) > kMaxSafeBufferSize) {
        if (err && errcap) {
            snprintf(err, errcap, "suspicious size=%zu",
                     static_cast<size_t>(buffer->size));
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }

    uint64_t options = static_cast<uint64_t>(MTL::ResourceStorageModeShared);
    if ((buffer->storage_flags & GL_MAP_READ_BIT) == 0) {
        options |= static_cast<uint64_t>(
            MTL::ResourceCPUCacheModeWriteCombined);
    }

    size_t allocationSize = static_cast<size_t>(buffer->size);
    const void* bytes = nullptr;
    if (buffer->data.buffer_data != 0) {
        allocationSize = buffer->data.buffer_size;
        if (allocationSize == 0 || allocationSize > kMaxSafeBufferSize) {
            allocationSize = static_cast<size_t>(buffer->size);
        }
        bytes = reinterpret_cast<const void*>(
            static_cast<uintptr_t>(buffer->data.buffer_data));
    }
    if (buffer->transient_batch_buffer && !bytes) {
        if (err && errcap) {
            snprintf(err, errcap, "transient buffer has no CPU backing");
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }
    if (buffer->transient_batch_buffer) {
        allocationSize = static_cast<size_t>(buffer->size);
    }

    const bool clientStorage =
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0;
    const bool persistentNoCopy =
        bytes &&
        (buffer->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) != 0 &&
        (buffer->storage_flags & GL_MAP_PERSISTENT_BIT) != 0;
    const bool noCopy = clientStorage || persistentNoCopy;
    if (noCopy && !bytes) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "no-copy buffer has no CPU backing buffer=%u",
                     static_cast<unsigned>(buffer->name));
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }
    if (clientStorage) {
        allocationSize = static_cast<size_t>(buffer->size);
    }

    void* metalBuffer = nullptr;
    mglMetalCountCreate(mgl::kMetalKindBuffer);
    int result = noCopy
        ? mglRenderCreateBufferWithBytesNoCopy(
              bytes, allocationSize, options, nullptr, 1, &metalBuffer)
        : (bytes
            ? mglRenderCreateBufferWithBytes(
                  bytes, allocationSize, options, nullptr, &metalBuffer)
            : mglRenderCreateBuffer(
                  allocationSize, options, nullptr, &metalBuffer));
    if (result != 0 || !metalBuffer) {
        if (err && errcap) {
            snprintf(err, errcap, "Metal buffer creation failed size=%zu",
                     allocationSize);
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }

    buffer->data.mtl_data = metalBuffer;
    buffer->data.mtl_owns_buffer_data = noCopy ? GL_TRUE : GL_FALSE;
    if (!bytes) buffer->data.buffer_data = 0;
    return MGL_RENDER_BUFFER_BOUND;
}

int mglRenderUpdateDirtyBuffer(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    auto ensureMetalBuffer = [&]() -> int {
        if (buffer->data.mtl_data) {
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
        const int bindResult =
            mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    };

    if (buffer->plain_uniform_slot && !buffer->data.mtl_data &&
        buffer->data.buffer_data && buffer->size > 0 &&
        buffer->size <= 4096) {
        buffer->data.dirty_bits &=
            ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if (buffer->size < 4096) {
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) &&
            !buffer->data.mtl_data) {
            const int result = ensureMetalBuffer();
            if (result != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
                return result;
            }
        }

        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }

        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }

        MTL::Buffer* metalBuffer =
            static_cast<MTL::Buffer*>(buffer->data.mtl_data);
        MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        if (!metalBuffer) {
            if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }

        const size_t metalLength = static_cast<size_t>(metalBuffer->length());
        size_t copyLength = buffer->size > 0
            ? std::min(static_cast<size_t>(buffer->size), metalLength)
            : 0;
        if (buffer->data.buffer_size > 0) {
            copyLength = std::min(copyLength, buffer->data.buffer_size);
        }
        uint8_t* cpuData = buffer->data.buffer_data >= 0x1000u
            ? reinterpret_cast<uint8_t*>(
                  static_cast<uintptr_t>(buffer->data.buffer_data))
            : nullptr;
        uint8_t* metalData =
            static_cast<uint8_t*>(metalBuffer->contents());

        if (metalBuffer == bufferBeforeSnapshot && cpuData && metalData &&
            copyLength > 0) {
            size_t uploadOffset = 0;
            size_t uploadLength = 0;
            if (mgl::bufferShadowUploadRange(
                    buffer, copyLength, &uploadOffset, &uploadLength)) {
                if (cpuData != metalData) {
                    memmove(metalData + uploadOffset,
                            cpuData + uploadOffset, uploadLength);
                }
                if (metalBuffer->storageMode() == MTL::StorageModeManaged) {
                    metalBuffer->didModifyRange(
                        NS::Range::Make(uploadOffset, uploadLength));
                }
            }
        } else if (metalBuffer == bufferBeforeSnapshot && metalData &&
                   copyLength > 0) {
            size_t modifyOffset = 0;
            size_t modifyLength = copyLength;
            if (buffer->mapped_length > 0 && buffer->mapped_offset >= 0 &&
                static_cast<size_t>(buffer->mapped_offset) < metalLength) {
                modifyOffset = static_cast<size_t>(buffer->mapped_offset);
                modifyLength = std::min(
                    static_cast<size_t>(buffer->mapped_length),
                    metalLength - modifyOffset);
            }
            if (modifyLength > 0 &&
                metalBuffer->storageMode() == MTL::StorageModeManaged) {
                metalBuffer->didModifyRange(
                    NS::Range::Make(modifyOffset, modifyLength));
            }
        }

        if ((buffer->access_flags & GL_MAP_COHERENT_BIT) != 0) {
            buffer->data.dirty_bits = DIRTY_BUFFER_DATA;
        } else {
            buffer->data.dirty_bits &=
                ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
            buffer->cpu_shadow_pending = GL_FALSE;
        }
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) != 0) {
        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_BUFFER_OPERATION_HANDLED;
        }
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "buffer %u has no dirty CPU or Metal backing",
                     static_cast<unsigned>(buffer->name));
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    const int bindResult = ensureMetalBuffer();
    if (bindResult != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
        return bindResult;
    }
    void* snapshotBuffer = nullptr;
    if (mglRenderSnapshotSharedDirtyBuffer(
            buffer, &snapshotBuffer, err, errcap) != 0) {
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    MTL::Buffer* metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    if (!metalBuffer) {
        if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    const bool coherentMapped =
        (buffer->access_flags & GL_MAP_COHERENT_BIT) != 0;
    if (coherentMapped) {
        size_t modifyOffset = 0;
        size_t modifyLength = metalLength;
        if (buffer->mapped_length > 0 && buffer->mapped_offset >= 0 &&
            static_cast<size_t>(buffer->mapped_offset) < metalLength) {
            modifyOffset = static_cast<size_t>(buffer->mapped_offset);
            modifyLength = std::min(
                static_cast<size_t>(buffer->mapped_length),
                metalLength - modifyOffset);
        }
        if (modifyLength > 0 &&
            metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(
                NS::Range::Make(modifyOffset, modifyLength));
        }
        buffer->data.dirty_bits = DIRTY_BUFFER_DATA;
    } else {
        size_t modifyLength = metalLength;
        if (buffer->data.buffer_size > 0) {
            modifyLength = std::min(modifyLength, buffer->data.buffer_size);
        }
        if (modifyLength > 0 &&
            metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(NS::Range::Make(0, modifyLength));
        }
        buffer->data.dirty_bits = 0;
        buffer->cpu_shadow_pending = GL_FALSE;
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void mglRenderBindBuffer(GLMContext glm_ctx, Buffer* buffer) {
    char error[256] = {};
    int result = mglRenderBindBufferStorage(
        buffer, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_BOUND) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderBufferSubDataStorage(Buffer* buffer,
                                     size_t offset,
                                     size_t size,
                                     const void* bytes,
                                     char* err,
                                     size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (size == 0) return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    if (!bytes) {
        if (err && errcap) snprintf(err, errcap, "null source bytes");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    uint8_t* cpuBase = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset > metalLength || size > metalLength - offset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu size=%zu exceeds Metal length=%zu",
                     offset, size, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (!metalBase) {
        if (err && errcap) snprintf(err, errcap, "Metal buffer has no contents");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
    if (cpuBase && cpuBase != metalBase) {
        memmove(cpuBase + offset, bytes, size);
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        metalBase = metalBuffer
            ? static_cast<uint8_t*>(metalBuffer->contents())
            : nullptr;
        if (!metalBuffer || !metalBase) {
            if (err && errcap) snprintf(err, errcap, "snapshot has no contents");
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    if (metalBuffer == bufferBeforeSnapshot) {
        memcpy(metalBase + offset, bytes, size);
        if (metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(NS::Range::Make(offset, size));
        }
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void mglRenderBufferSubData(GLMContext glm_ctx,
                               Buffer* buffer,
                               size_t offset,
                               size_t size,
                               const void* bytes) {
    char error[256] = {};
    int result = mglRenderBufferSubDataStorage(
        buffer, offset, size, bytes, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer subdata failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderMapBufferStorage(Buffer* buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map,
                                 void** mapped_out,
                                 char* err,
                                 size_t errcap) {
    if (mapped_out) *mapped_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!buffer || !mapped_out) {
        if (err && errcap) snprintf(err, errcap, "bad arguments");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset > metalLength) {
        if (err && errcap) {
            snprintf(err, errcap, "offset=%zu beyond Metal length=%zu",
                     offset, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    const size_t safeLength = std::min(size, metalLength - offset);
    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    uint8_t* cpuBase = nullptr;
    if (buffer->data.buffer_data >= 0x1000u) {
        cpuBase = reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(buffer->data.buffer_data));
    }

    if (map) {
        const bool reads = access == GL_READ_ONLY || access == GL_READ_WRITE ||
                           (access & GL_MAP_READ_BIT) != 0;
        if (cpuBase) {
            uint8_t* cpuPointer = cpuBase + offset;
            if (reads && metalBase && metalBase != cpuBase && safeLength > 0 &&
                !buffer->cpu_shadow_pending) {
                memcpy(cpuPointer, metalBase + offset, safeLength);
            }
            *mapped_out = cpuPointer;
        } else {
            *mapped_out = metalBase ? metalBase + offset : nullptr;
        }
        return MGL_RENDER_BUFFER_OPERATION_HANDLED;
    }

    if (!cpuBase &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(NS::Range::Make(offset, safeLength));
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void* mglRenderMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer* buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map) {
    void* mapped = nullptr;
    char error[256] = {};
    int result = mglRenderMapBufferStorage(
        buffer, offset, size, access, map, &mapped, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return mapped;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer map failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
    return nullptr;
}

void mglRenderReadBackBuffer(GLMContext glm_ctx,
                                Buffer* buffer,
                                size_t offset,
                                size_t size) {
    (void)glm_ctx;
    if (!buffer || size == 0 || buffer->cpu_shadow_pending ||
        !buffer->data.mtl_data) {
        return;
    }
    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    if (metalBuffer->storageMode() != MTL::StorageModeShared) return;

    uint8_t* metalBase = static_cast<uint8_t*>(metalBuffer->contents());
    uint8_t* cpuBase = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!metalBase || !cpuBase || metalBase == cpuBase) return;

    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (offset >= metalLength) return;
    size_t safeLength = std::min(size, metalLength - offset);
    const size_t shadowLength = buffer->data.buffer_size;
    if (shadowLength > 0) {
        if (offset >= shadowLength) return;
        safeLength = std::min(safeLength, shadowLength - offset);
    }
    memcpy(cpuBase + offset, metalBase + offset, safeLength);
}

int mglRenderFlushBufferRangeStorage(Buffer* buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char* err,
                                         size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer || offset < 0 || length < 0) {
        if (err && errcap) snprintf(err, errcap, "bad buffer or signed range");
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }
    if (length == 0) return MGL_RENDER_BUFFER_OPERATION_HANDLED;

    bool created = false;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_BUFFER_BOUND) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        created = true;
    }

    MTL::Buffer* metalBuffer =
        static_cast<MTL::Buffer*>(buffer->data.mtl_data);
    const size_t safeOffset = static_cast<size_t>(offset);
    const size_t safeLength = static_cast<size_t>(length);
    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    if (safeOffset > metalLength || safeLength > metalLength - safeOffset) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "range offset=%zu length=%zu exceeds Metal length=%zu",
                     safeOffset, safeLength, metalLength);
        }
        return MGL_RENDER_BUFFER_OPERATION_ERROR;
    }

    if (!created) {
        void* snapshotBuffer = nullptr;
        if (mglRenderSnapshotSharedBufferRange(
                buffer, safeOffset, safeLength, &snapshotBuffer,
                err, errcap) != 0) {
            return MGL_RENDER_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    }
    if (metalBuffer &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(
            NS::Range::Make(safeOffset, safeLength));
    }
    return MGL_RENDER_BUFFER_OPERATION_HANDLED;
}

void mglRenderFlushBufferRange(GLMContext glm_ctx,
                                  Buffer* buffer,
                                  intptr_t offset,
                                  intptr_t length) {
    char error[256] = {};
    int result = mglRenderFlushBufferRangeStorage(
        buffer, offset, length, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer range flush failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderConvertVertexBuffer(
    Buffer* sourceBuffer,
    const MGLRenderVertexConversion* conversion,
    uint64_t* convertedStrideOut,
    void** convertedBufferOut,
    char* err,
    size_t errcap) {
    if (convertedStrideOut) *convertedStrideOut = 0;
    if (convertedBufferOut) *convertedBufferOut = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!sourceBuffer || !conversion || !convertedStrideOut ||
        !convertedBufferOut) {
        if (err && errcap) snprintf(err, errcap, "bad arguments");
        return -1;
    }
    if (conversion->kind > MGL_RENDER_VERTEX_INTEGER_TO_32) {
        if (err && errcap) {
            snprintf(err, errcap, "unknown conversion kind=%u",
                     conversion->kind);
        }
        return -1;
    }
    if (conversion->binding_offset < 0 ||
        conversion->relative_offset < 0) {
        if (err && errcap) snprintf(err, errcap, "negative vertex offset");
        return -1;
    }

    const uint8_t* sourceBytes = nullptr;
    size_t sourceSize = 0;
    if (!mgl::vertexConversionSource(
            sourceBuffer, &sourceBytes, &sourceSize) ||
        !sourceBytes || sourceSize == 0) {
        if (err && errcap) snprintf(err, errcap, "missing source bytes");
        return -1;
    }
    const size_t bindingOffset =
        static_cast<size_t>(conversion->binding_offset);
    const size_t relativeOffset =
        static_cast<size_t>(conversion->relative_offset);
    if (bindingOffset >= sourceSize) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "binding offset=%zu exceeds source size=%zu",
                     bindingOffset, sourceSize);
        }
        return -1;
    }

    const uint32_t kind = conversion->kind;
    uint32_t componentCount = conversion->component_count;
    size_t sourceComponentSize = 0;
    size_t defaultStride = 0;
    size_t minimumConvertedStride = 0;
    switch (kind) {
        case MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(double);
            defaultStride = componentCount * sizeof(double);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_INT_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            if (conversion->source_type != GL_INT &&
                conversion->source_type != GL_UNSIGNED_INT) {
                if (err && errcap) snprintf(err, errcap, "invalid int source type");
                return -1;
            }
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = componentCount * sizeof(uint32_t);
            minimumConvertedStride = 0;
            break;
        case MGL_RENDER_VERTEX_FIXED_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(int32_t);
            defaultStride = componentCount * sizeof(int32_t);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT:
            componentCount = 4;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 4u * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT:
            componentCount = 3;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 3u * sizeof(float);
            break;
        case MGL_RENDER_VERTEX_INTEGER_TO_32:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize =
                mgl::vertexComponentSize(conversion->source_type);
            if (sourceComponentSize == 0 || sourceComponentSize > 4) {
                if (err && errcap) {
                    snprintf(err, errcap, "invalid integer source type");
                }
                return -1;
            }
            defaultStride = componentCount * sourceComponentSize;
            minimumConvertedStride = componentCount * sizeof(uint32_t);
            break;
        default:
            return -1;
    }

    {
        if (conversion->stride >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            if (err && errcap) snprintf(err, errcap, "vertex stride overflow");
            return -1;
        }
        const size_t originalStride = conversion->stride > 0
            ? static_cast<size_t>(conversion->stride)
            : defaultStride;
        if (originalStride == 0) {
            if (err && errcap) snprintf(err, errcap, "zero vertex stride");
            return -1;
        }

        size_t convertedStrideBase = originalStride;
        if (kind == MGL_RENDER_VERTEX_INTEGER_TO_32) {
            convertedStrideBase = minimumConvertedStride;
        } else if (minimumConvertedStride > convertedStrideBase) {
            convertedStrideBase = minimumConvertedStride;
        }
        size_t convertedStride = 0;
        if (!mgl::alignVertexStride(convertedStrideBase, &convertedStride)) {
            if (err && errcap) snprintf(err, errcap, "converted stride overflow");
            return -1;
        }

        const size_t copyLength = sourceSize - bindingOffset;
        const size_t vertexCount =
            copyLength / originalStride +
            ((copyLength % originalStride) != 0 ? 1u : 0u);
        if (vertexCount == 0 ||
            vertexCount > std::numeric_limits<size_t>::max() /
                              convertedStride) {
            if (err && errcap) snprintf(err, errcap, "converted size overflow");
            return -1;
        }
        const size_t convertedLength = vertexCount * convertedStride;
        const uint8_t* sourceBase = sourceBytes + bindingOffset;
        const uint64_t sourceHash =
            mgl::hashVertexBytes(sourceBase, copyLength);
        mgl::ConvertedVertexBufferKey key = {};
        key.sourceHash = sourceHash;
        key.copyLength = copyLength;
        key.originalStride = originalStride;
        key.convertedStride = convertedStride;
        key.bindingOffset = conversion->binding_offset;
        key.relativeOffset = conversion->relative_offset;
        key.sourceName = sourceBuffer->name;
        key.kind = kind;
        key.componentCount = componentCount;
        key.sourceType = conversion->source_type;
        key.normalized = conversion->normalized;
        key.destinationSigned = conversion->destination_signed;

        mgl::Renderer& renderer = mgl::renderer();
        {
            std::lock_guard<std::mutex> lock(renderer.mutex);
            if (!renderer.device) {
                if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
                return -1;
            }
            auto found = renderer.convertedVertexBuffers.find(key);
            if (found != renderer.convertedVertexBuffers.end() &&
                found->second) {
                found->second->retain();
                *convertedStrideOut = convertedStride;
                *convertedBufferOut = found->second;
                return 0;
            }
        }

        std::vector<uint8_t> converted;
        try {
            converted.resize(convertedLength, 0);
        } catch (const std::bad_alloc&) {
            if (err && errcap) snprintf(err, errcap, "converted allocation failed");
            return -1;
        }

        const bool preservesVertexBytes =
            kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT ||
            kind == MGL_RENDER_VERTEX_INT_TO_FLOAT ||
            kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT;
        for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
            const size_t sourceOffset = vertex * originalStride;
            const size_t destinationOffset = vertex * convertedStride;
            const size_t remaining = sourceOffset < copyLength
                ? copyLength - sourceOffset
                : 0;
            const size_t copyBytes = std::min(originalStride, remaining);
            if (preservesVertexBytes && copyBytes > 0) {
                memcpy(converted.data() + destinationOffset,
                       sourceBase + sourceOffset, copyBytes);
            }

            if (kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_INT_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT) {
                const size_t inputBytes =
                    componentCount * sourceComponentSize;
                const size_t outputBytes = componentCount * sizeof(float);
                if (relativeOffset > copyBytes ||
                    inputBytes > copyBytes - relativeOffset ||
                    relativeOffset > convertedStride ||
                    outputBytes > convertedStride - relativeOffset) {
                    continue;
                }
                float values[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                for (uint32_t component = 0;
                     component < componentCount; ++component) {
                    const uint8_t* componentBytes =
                        sourceBase + sourceOffset + relativeOffset +
                        component * sourceComponentSize;
                    if (kind == MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT) {
                        double value = 0.0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = static_cast<float>(value);
                    } else if (kind == MGL_RENDER_VERTEX_FIXED_TO_FLOAT) {
                        int32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = static_cast<float>(
                            static_cast<double>(value) / 65536.0);
                    } else if (conversion->source_type == GL_INT) {
                        int32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        if (conversion->normalized) {
                            double normalized =
                                static_cast<double>(value) / 2147483647.0;
                            if (normalized < -1.0) normalized = -1.0;
                            values[component] = static_cast<float>(normalized);
                        } else {
                            values[component] = static_cast<float>(value);
                        }
                    } else {
                        uint32_t value = 0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = conversion->normalized
                            ? static_cast<float>(
                                  static_cast<double>(value) / 4294967295.0)
                            : static_cast<float>(value);
                    }
                }
                memcpy(converted.data() + destinationOffset + relativeOffset,
                       values, outputBytes);
                continue;
            }

            if (kind == MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT ||
                kind == MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT) {
                if (relativeOffset > remaining ||
                    sizeof(uint32_t) > remaining - relativeOffset) {
                    continue;
                }
                const size_t outputBytes = componentCount * sizeof(float);
                if (relativeOffset > convertedStride ||
                    outputBytes > convertedStride - relativeOffset) {
                    continue;
                }
                uint32_t packed = 0;
                memcpy(&packed,
                       sourceBase + sourceOffset + relativeOffset,
                       sizeof(packed));
                float values[4] = {};
                if (kind == MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT) {
                    values[0] = static_cast<float>((packed >> 22) & 0x3ffu) /
                                1023.0f;
                    values[1] = static_cast<float>((packed >> 12) & 0x3ffu) /
                                1023.0f;
                    values[2] = static_cast<float>((packed >> 2) & 0x3ffu) /
                                1023.0f;
                    values[3] = static_cast<float>(packed & 0x3u) / 3.0f;
                } else {
                    values[0] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 0) & 0x7ffu, 6);
                    values[1] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 11) & 0x7ffu, 6);
                    values[2] = mgl::decodeUnsignedFloatComponent(
                        (packed >> 22) & 0x3ffu, 5);
                }
                memcpy(converted.data() + destinationOffset + relativeOffset,
                       values, outputBytes);
                continue;
            }

            uint8_t* destination = converted.data() + destinationOffset;
            for (uint32_t component = 0;
                 component < componentCount; ++component) {
                const size_t componentOffset =
                    sourceOffset + relativeOffset +
                    component * sourceComponentSize;
                if (componentOffset > copyLength ||
                    sourceComponentSize > copyLength - componentOffset) {
                    break;
                }
                const uint8_t* source = sourceBase + componentOffset;
                uint32_t value = 0;
                switch (conversion->source_type) {
                    case GL_BYTE: {
                        int8_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(static_cast<int32_t>(v));
                        break;
                    }
                    case GL_UNSIGNED_BYTE: {
                        uint8_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = v;
                        break;
                    }
                    case GL_SHORT: {
                        int16_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(static_cast<int32_t>(v));
                        break;
                    }
                    case GL_UNSIGNED_SHORT: {
                        uint16_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = v;
                        break;
                    }
                    case GL_INT: {
                        int32_t v = 0;
                        memcpy(&v, source, sizeof(v));
                        value = static_cast<uint32_t>(v);
                        break;
                    }
                    case GL_UNSIGNED_INT:
                        memcpy(&value, source, sizeof(value));
                        break;
                    default:
                        break;
                }
                memcpy(destination + component * sizeof(uint32_t),
                       &value, sizeof(value));
            }
        }

        void* createdObject = nullptr;
        if (mglRenderCreateBufferWithBytes(
                converted.data(), convertedLength,
                static_cast<uint64_t>(MTL::ResourceStorageModeShared),
                nullptr, &createdObject) != 0 || !createdObject) {
            if (err && errcap) snprintf(err, errcap, "Metal buffer creation failed");
            return -1;
        }
        MTL::Buffer* created = static_cast<MTL::Buffer*>(createdObject);
        {
            std::lock_guard<std::mutex> lock(renderer.mutex);
            auto found = renderer.convertedVertexBuffers.find(key);
            if (found != renderer.convertedVertexBuffers.end() &&
                found->second) {
                created->release();
                found->second->retain();
                *convertedBufferOut = found->second;
            } else {
                try {
                    renderer.convertedVertexBuffers.emplace(key, created);
                    created->retain();
                    *convertedBufferOut = created;
                    if (renderer.convertedVertexBuffers.size() > 64) {
                        size_t evictCount =
                            renderer.convertedVertexBuffers.size() / 4;
                        while (evictCount-- > 0 &&
                               !renderer.convertedVertexBuffers.empty()) {
                            auto evict = renderer.convertedVertexBuffers.begin();
                            if (evict->second) evict->second->release();
                            renderer.convertedVertexBuffers.erase(evict);
                        }
                    }
                } catch (const std::bad_alloc&) {
                    *convertedBufferOut = created;
                }
            }
        }
        *convertedStrideOut = convertedStride;
        return 0;
    }

bad_components:
    if (err && errcap) {
        snprintf(err, errcap, "invalid component count=%u",
                 conversion->component_count);
    }
    return -1;
}

int mglRenderBindAIRProgram(Program* program,
                               int* failed_stage_out,
                               char* err,
                               size_t errcap) {
    if (failed_stage_out) *failed_stage_out = -1;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!program || !device) {
        if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
        return MGL_RENDER_AIR_PROGRAM_ERROR;
    }
    program->dirty_bits &= ~DIRTY_PROGRAM;

    bool hasAIRStage = false;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader) continue;
        if (stage == _GEOMETRY_SHADER) {
            if (program->gs_route != MGL_GS_ROUTE_COMPUTE) {
                if (failed_stage_out) *failed_stage_out = stage;
                if (err && errcap) {
                    snprintf(err, errcap,
                             "unsupported geometry shader route %u",
                             (unsigned)program->gs_route);
                }
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
        MGLShaderModule* spirv = &program->modules[stage];
        if (!spirv->metallib_bytes || spirv->metallib_size == 0u) {
            return MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE;
        }
        hasAIRStage = true;
    }
    if (!hasAIRStage) return MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE;

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader) {
            continue;
        }
        MGLShaderModule* spirv = &program->modules[stage];
        if (!spirv->mtl_library || !spirv->mtl_function) {
            mgl::releaseBridgedObject(&spirv->mtl_function);
            mgl::releaseBridgedObject(&spirv->mtl_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_bytes, spirv->metallib_size,
                    &spirv->mtl_library, &spirv->mtl_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }

        if (stage == _VERTEX_SHADER &&
            spirv->metallib_tess_capture_bytes &&
            (!spirv->mtl_tess_capture_library ||
             !spirv->mtl_tess_capture_function)) {
            mgl::releaseBridgedObject(&spirv->mtl_tess_capture_function);
            mgl::releaseBridgedObject(&spirv->mtl_tess_capture_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_tess_capture_bytes,
                    spirv->metallib_tess_capture_size,
                    &spirv->mtl_tess_capture_library,
                    &spirv->mtl_tess_capture_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
        if (stage == _VERTEX_SHADER &&
            spirv->metallib_cull_capture_bytes &&
            (!spirv->mtl_cull_capture_library ||
             !spirv->mtl_cull_capture_function)) {
            mgl::releaseBridgedObject(&spirv->mtl_cull_capture_function);
            mgl::releaseBridgedObject(&spirv->mtl_cull_capture_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_cull_capture_bytes,
                    spirv->metallib_cull_capture_size,
                    &spirv->mtl_cull_capture_library,
                    &spirv->mtl_cull_capture_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
    }
    return MGL_RENDER_AIR_PROGRAM_BOUND;
}

void mglRenderBindProgram(GLMContext glm_ctx, Program* program) {
    (void)glm_ctx;
    int failedStage = -1;
    char error[256] = {};
    int result = mglRenderBindAIRProgram(
        program, &failedStage, error, sizeof(error));
    if (result == MGL_RENDER_AIR_PROGRAM_BOUND) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp program bind failed program=%u "
            "stage=%d: %s\n",
            program ? (unsigned)program->name : 0u, failedStage,
            error[0]
                ? error
                : (result == MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE
                       ? "linked program has no AIR metallib"
                       : "unknown error"));
}

namespace {

MGLRendererBackendHandle* rendererBackend(GLMContext context) {
    return context
        ? static_cast<MGLRendererBackendHandle*>(context->renderer_backend)
        : nullptr;
}

void* rendererOwner(GLMContext context, MGLRendererBackendOwnerKind kind) {
    return mglRendererBackendGetOwner(rendererBackend(context), kind);
}

} // namespace

void mglRenderGetSync(GLMContext glm_ctx, Sync* sync) {
    if (!sync) return;

    mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    mgl::releaseBridgedObject(&sync->mtl_event);
    void* command_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER);
    if (!command_owner) return;

    void* render_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER);
    if (render_owner &&
        mglRenderEncoderOwnerHasCurrent(render_owner) == 1 &&
        mglRenderEndRenderEncoderOwner(render_owner) != 0) {
        return;
    }

    MGLRenderCommandBufferState state = {};
    if (mglRenderGetCommandBufferOwnerState(command_owner, &state) != 0 ||
        state.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued) ||
        state.has_error) {
        void* next = nullptr;
        (void)mglRenderCommandBufferOwnerCreateNext(command_owner, &next);
        return;
    }

    void* submission = nullptr;
    void* command_buffer = nullptr;
    if (mglRenderTakeCommandBufferSubmission(
            command_owner, &submission, &command_buffer) != 0 ||
        !submission || !command_buffer) {
        mglRenderDestroyCommandBufferSubmission(&submission);
        return;
    }

    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    command->retain();
    sync->mtl_command_buffer = command;

    MGLRenderCommandBufferTransaction transaction = {};
    int result = mglRenderCommitCommandBufferTransaction(
        command_owner, &submission, command_buffer,
        rendererOwner(glm_ctx, MGL_RENDERER_BACKEND_OWNER_RECOVERY),
        0u, &transaction);
    mglRenderDestroyCommandBufferSubmission(&submission);
    if (result != 0 &&
        transaction.result !=
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_COMMITTED) {
        mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    }
}

void mglRenderWaitForSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    if (sync->mtl_command_buffer) {
        MGLRenderCommandBufferState state = {};
        (void)mglRenderWaitCommandBufferState(
            sync->mtl_command_buffer, &state);
        mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    }
    mgl::releaseBridgedObject(&sync->mtl_event);
}

unsigned int mglRenderGetSyncStatus(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync || !sync->mtl_command_buffer) return GL_SIGNALED;
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(sync->mtl_command_buffer);
    return commandBuffer->status() == MTL::CommandBufferStatusCompleted
        ? GL_SIGNALED
        : GL_UNSIGNALED;
}

void mglRenderReleaseSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    mgl::releaseBridgedObject(&sync->mtl_event);
}

void mglRenderFlush(GLMContext glm_ctx, bool finish) {
    void* command_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER);
    if (!command_owner) return;

    Sync boundary = {};
    mglRenderGetSync(glm_ctx, &boundary);
    if (finish) {
        if (boundary.mtl_command_buffer) {
            mglRenderWaitForSync(glm_ctx, &boundary);
        } else {
            MGLRenderCommandBufferState state = {};
            (void)mglRenderWaitCommandBufferOwnerLastSubmitted(
                command_owner, &state);
        }
    } else {
        mglRenderReleaseSync(glm_ctx, &boundary);
    }
}

void mglRenderInvalidateRenderPass(GLMContext glm_ctx) {
    void* render_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER);
    if (!render_owner) return;
    if (mglRenderEncoderOwnerHasCurrent(render_owner) == 1) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
}


int mglRenderAttachRuntimeOwners(GLMContext glm_ctx,
                                    void* command_buffer_owner,
                                    void* render_encoder_owner,
                                    void* render_pass_state_owner) {
    MGLRendererBackendHandle* backend = rendererBackend(glm_ctx);
    return backend
        ? mglRendererBackendAttachRuntimeOwners(
              backend, command_buffer_owner,
              render_encoder_owner, render_pass_state_owner)
        : -1;
}

void mglRenderDetachRuntimeOwners(GLMContext glm_ctx) {
    if (MGLRendererBackendHandle* backend = rendererBackend(glm_ctx)) {
        (void)mglRendererBackendAttachRuntimeOwners(
            backend, nullptr, nullptr, nullptr);
    }
}

uint64_t mglRenderGetGPUTimestamp(GLMContext glm_ctx) {
    if (!glm_ctx) return 0;

    /* The GL semantic layer establishes the ordering boundary before entering
     * this callback. Sampling itself is entirely C++ and does not need the
     * ObjC renderer bridge. */
    uint64_t cpu_timestamp = 0;
    uint64_t gpu_timestamp = 0;
    return mglRenderSampleTimestamps(
               &cpu_timestamp, &gpu_timestamp) == 0
        ? gpu_timestamp : 0;
}

void mglRenderBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target) {
    void* query_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY);
    if (!query_owner) return;

    void* visibility_buffer = nullptr;
    if (mglRenderBeginSampleQuery(
            query_owner,
            target == GL_SAMPLES_PASSED ? 1u : 0u,
            "MGL Visibility Result", &visibility_buffer) != 0 ||
        !visibility_buffer) {
        return;
    }

    uint32_t mode = 0;
    uint64_t offset = 0;
    if (mglRenderAcquireSampleQuerySlot(
            query_owner, &mode, &offset) != 0) {
        return;
    }

    bool pass_has_visibility = false;
    MGLRenderPassState pass = {};
    void* render_pass_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_PASS);
    if (render_pass_owner &&
        mglRenderGetRenderPassStateOwner(
            render_pass_owner, &pass) == 0) {
        pass_has_visibility = pass.visibility_result_buffer != nullptr;
    }

    void* render_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER);
    if (!render_owner ||
        mglRenderEncoderOwnerHasCurrent(render_owner) != 1) {
        return;
    }
    if (!pass_has_visibility ||
        mglRenderSetVisibilityResultModeForRenderEncoderOwner(
            render_owner, mode, offset) != 0) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
}

uint64_t mglRenderEndSampleQueryCallback(GLMContext glm_ctx) {
    void* query_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY);
    if (!query_owner) return 0;

    void* render_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER);
    if (render_owner &&
        mglRenderEncoderOwnerHasCurrent(render_owner) == 1) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
    mglRenderEndSampleQuery(query_owner);

    void* visibility_buffer = nullptr;
    if (mglRenderGetQueryVisibilityBuffer(
            query_owner, &visibility_buffer) == 0 &&
        visibility_buffer) {
        Sync boundary = {};
        mglRenderGetSync(glm_ctx, &boundary);
        mglRenderWaitForSync(glm_ctx, &boundary);
    }

    uint64_t result = 0;
    return mglRenderGetSampleQueryResult(
               query_owner, &result) == 0
        ? result : 0;
}

void mglRenderBeginTimerQueryCallback(GLMContext glm_ctx) {
    void* query_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY);
    if (!query_owner || mglRenderBeginTimerQuery(query_owner) != 0) {
        fprintf(stderr, "MGL ERROR: failed to begin Metal-cpp timer query\n");
    }
}

uint64_t mglRenderEndTimerQueryCallback(GLMContext glm_ctx) {
    void* query_owner = rendererOwner(
        glm_ctx, MGL_RENDERER_BACKEND_OWNER_QUERY);
    uint64_t elapsed = 0;
    return query_owner &&
           mglRenderEndTimerQuery(query_owner, &elapsed) == 0
        ? elapsed : 0;
}

int mglRenderCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char* label,
                             void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Buffer* buffer = renderer.device->newBuffer(
        static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options));
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderCreateBufferWithBytes(const void* bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char* label,
                                      void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Buffer* buffer = renderer.device->newBuffer(
        bytes, static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options));
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderGetBufferContents(void *buffer,
                                  void **contents_out,
                                  uint64_t *length_out) {
    if (contents_out) *contents_out = nullptr;
    if (length_out) *length_out = 0;
    MTL::Buffer *object = static_cast<MTL::Buffer *>(buffer);
    if (!object || !contents_out || !length_out) return -1;
    *contents_out = object->contents();
    *length_out = static_cast<uint64_t>(object->length());
    return *contents_out ? 0 : -1;
}

int mglRenderGetBufferInfo(const void *buffer,
                              MGLRenderBufferInfo *info_out) {
    if (info_out) *info_out = {};
    const MTL::Buffer *object = static_cast<const MTL::Buffer *>(buffer);
    if (!object || !info_out) return -1;
    info_out->length = static_cast<uint64_t>(object->length());
    return 0;
}

int mglRenderAddBufferDebugMarker(void *buffer,
                                     const char *marker,
                                     uint64_t location,
                                     uint64_t length) {
    MTL::Buffer *object = static_cast<MTL::Buffer *>(buffer);
    if (!object || !marker || location > object->length() ||
        length > object->length() - location) {
        return -1;
    }
    object->addDebugMarker(
        NS::String::string(marker, NS::UTF8StringEncoding),
        NS::Range(location, length));
    return 0;
}

int mglRenderGetTextureInfo(const void *texture,
                               MGLRenderTextureInfo *info_out) {
    if (info_out) *info_out = {};
    const MTL::Texture *object = static_cast<const MTL::Texture *>(texture);
    if (!object || !info_out) return -1;
    info_out->pixel_format = static_cast<uint32_t>(object->pixelFormat());
    info_out->texture_type = static_cast<uint32_t>(object->textureType());
    info_out->width = object->width();
    info_out->height = object->height();
    info_out->depth = object->depth();
    info_out->mipmap_level_count = object->mipmapLevelCount();
    info_out->array_length = object->arrayLength();
    info_out->usage = static_cast<uint64_t>(object->usage());
    info_out->storage_mode = static_cast<uint32_t>(object->storageMode());
    info_out->sample_count = object->sampleCount();
    return 0;
}

int mglRenderTextureIsFramebufferOnly(const void *texture) {
    const MTL::Texture *object = static_cast<const MTL::Texture *>(texture);
    return object && object->isFramebufferOnly() ? 1 : 0;
}

int mglRenderCreateTextureStagingOwner(
    const void* bytes,
    uint64_t length,
    uint64_t resource_options,
    void** owner_out,
    void** buffer_out) {
    if (owner_out) *owner_out = nullptr;
    if (buffer_out) *buffer_out = nullptr;
    if (!owner_out || !buffer_out || !bytes || length == 0) return -1;
    void* rawBuffer = nullptr;
    if (mglRenderCreateBufferWithBytes(
            bytes, length, resource_options, "MGL.texture_staging",
            &rawBuffer) != 0 || !rawBuffer) {
        return -1;
    }
    mgl::TextureStagingOwner* owner =
        new (std::nothrow) mgl::TextureStagingOwner();
    if (!owner) {
        static_cast<MTL::Buffer*>(rawBuffer)->release();
        return -1;
    }
    owner->buffer = static_cast<MTL::Buffer*>(rawBuffer);
    *owner_out = owner;
    *buffer_out = owner->buffer;
    return 0;
}

void mglRenderDestroyTextureStagingOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::TextureStagingOwner* owner =
        static_cast<mgl::TextureStagingOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateBufferWithBytesNoCopy(const void* bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char* label,
                                            int deallocate_vm,
                                            void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    /* Keep the deallocator in the Metal object rather than releasing the
     * backing VM range when the GL Buffer shell disappears.  Command buffers
     * may retain this MTLBuffer past the GL-side unbind/delete. */
    void (^deallocator)(void*, NS::UInteger) = nil;
    if (deallocate_vm) {
        deallocator = ^(void* pointer, NS::UInteger size) {
            if (!pointer || size == 0) return;
            kern_return_t result = vm_deallocate(
                (vm_map_t)mach_task_self(),
                (vm_address_t)pointer, (vm_size_t)size);
            if (result != KERN_SUCCESS) {
                fprintf(stderr,
                        "MGL WARNING: Metal-cpp no-copy vm_deallocate "
                        "failed err=%d ptr=%p len=%llu\\n",
                        result, pointer,
                        (unsigned long long)size);
            }
        };
    }
    MTL::Buffer* buffer = renderer.device->newBuffer(
        bytes, static_cast<NS::UInteger>(length),
        static_cast<MTL::ResourceOptions>(resource_options), deallocator);
    if (!buffer) return -1;
    if (label && label[0]) {
        buffer->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *buffer_out = buffer;
    return 0;
}

int mglRenderCreateTextureFromState(
    const MGLRenderTextureDescriptorState* texture_descriptor,
    const char* label,
    void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    if (!texture_descriptor || !texture_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::TextureDescriptor* descriptor =
        mgl::newTextureDescriptor(texture_descriptor);
    if (!descriptor) return -1;
    MTL::Texture* texture = renderer.device->newTexture(descriptor);
    descriptor->release();
    if (!texture) return -1;
    if (label && label[0]) {
        texture->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *texture_out = texture;
    return 0;
}

static MGLRenderTextureDescriptorState
mglRenderReadTextureDescriptor(MTL::TextureDescriptor* descriptor) {
    MGLRenderTextureDescriptorState state = {};
    if (!descriptor) return state;
    state.texture_type = static_cast<uint32_t>(descriptor->textureType());
    state.pixel_format = static_cast<uint32_t>(descriptor->pixelFormat());
    state.width = descriptor->width();
    state.height = descriptor->height();
    state.depth = descriptor->depth();
    state.mipmap_level_count = descriptor->mipmapLevelCount();
    state.sample_count = descriptor->sampleCount();
    state.array_length = descriptor->arrayLength();
    state.resource_options = descriptor->resourceOptions();
    state.usage = descriptor->usage();
    state.cpu_cache_mode = static_cast<uint32_t>(descriptor->cpuCacheMode());
    state.storage_mode = static_cast<uint32_t>(descriptor->storageMode());
    state.hazard_tracking_mode =
        static_cast<uint32_t>(descriptor->hazardTrackingMode());
    state.compression_type =
        static_cast<uint32_t>(descriptor->compressionType());
    state.placement_sparse_page_size =
        static_cast<uint32_t>(descriptor->placementSparsePageSize());
    state.allow_gpu_optimized_contents =
        descriptor->allowGPUOptimizedContents() ? 1u : 0u;
    MTL::TextureSwizzleChannels swizzle = descriptor->swizzle();
    state.swizzle_red = static_cast<uint32_t>(swizzle.red);
    state.swizzle_green = static_cast<uint32_t>(swizzle.green);
    state.swizzle_blue = static_cast<uint32_t>(swizzle.blue);
    state.swizzle_alpha = static_cast<uint32_t>(swizzle.alpha);
    state.has_swizzle = 1u;
    return state;
}

int mglRenderCreateTextureFromDescriptor(
    void* descriptor_handle,
    const char* label,
    void** texture_out) {
    auto* descriptor =
        reinterpret_cast<MTL::TextureDescriptor*>(descriptor_handle);
    MGLRenderTextureDescriptorState state =
        mglRenderReadTextureDescriptor(descriptor);
    return mglRenderCreateTextureFromState(&state, label, texture_out);
}

int mglRenderCreateBufferTextureFromState(
    void* buffer,
    const MGLRenderTextureDescriptorState* texture_descriptor,
    uint64_t offset,
    uint64_t bytes_per_row,
    void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    MTL::Buffer* source = static_cast<MTL::Buffer*>(buffer);
    if (!source || !texture_descriptor || !texture_out ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::TextureDescriptor* descriptor =
        mgl::newTextureDescriptor(texture_descriptor);
    if (!descriptor) return -1;
    MTL::Texture* texture = source->newTexture(
        descriptor, static_cast<NS::UInteger>(offset),
        static_cast<NS::UInteger>(bytes_per_row));
    descriptor->release();
    if (!texture) return -1;
    *texture_out = texture;
    return 0;
}

int mglRenderCreateBufferTextureFromDescriptor(
    void* buffer,
    void* descriptor_handle,
    uint64_t offset,
    uint64_t bytes_per_row,
    void** texture_out) {
    MGLRenderTextureDescriptorState state =
        mglRenderReadTextureDescriptor(
            reinterpret_cast<MTL::TextureDescriptor*>(descriptor_handle));
    return mglRenderCreateBufferTextureFromState(
        buffer, &state, offset, bytes_per_row, texture_out);
}

int mglRenderCreateTextureView(void* texture,
                                  uint32_t pixel_format,
                                  void** texture_view_out) {
    if (texture_view_out) *texture_view_out = nullptr;
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !texture_view_out) return -1;
    MTL::Texture* view = source->newTextureView(
        static_cast<MTL::PixelFormat>(pixel_format));
    if (!view) return -1;
    *texture_view_out = view;
    return 0;
}

int mglRenderCreateTextureViewRange(
    void* texture,
    uint32_t pixel_format,
    uint32_t texture_type,
    uint64_t level_location,
    uint64_t level_length,
    uint64_t slice_location,
    uint64_t slice_length,
    int use_swizzle,
    uint32_t swizzle_red,
    uint32_t swizzle_green,
    uint32_t swizzle_blue,
    uint32_t swizzle_alpha,
    void** texture_view_out) {
    if (texture_view_out) *texture_view_out = nullptr;
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !texture_view_out || level_length == 0 ||
        slice_length == 0) {
        return -1;
    }
    const NS::Range levels(level_location, level_length);
    const NS::Range slices(slice_location, slice_length);
    MTL::Texture* view = nullptr;
    if (use_swizzle) {
        const MTL::TextureSwizzleChannels swizzle(
            static_cast<MTL::TextureSwizzle>(swizzle_red),
            static_cast<MTL::TextureSwizzle>(swizzle_green),
            static_cast<MTL::TextureSwizzle>(swizzle_blue),
            static_cast<MTL::TextureSwizzle>(swizzle_alpha));
        view = source->newTextureView(
            static_cast<MTL::PixelFormat>(pixel_format),
            static_cast<MTL::TextureType>(texture_type), levels, slices,
            swizzle);
    } else {
        view = source->newTextureView(
            static_cast<MTL::PixelFormat>(pixel_format),
            static_cast<MTL::TextureType>(texture_type), levels, slices);
    }
    if (!view) return -1;
    *texture_view_out = view;
    return 0;
}

int mglRenderSampledTextureViewForBaseLevel(
    Texture *texture_object,
    void *source_texture,
    void **view_out) {
    if (view_out) *view_out = nullptr;
    MTL::Texture *source = static_cast<MTL::Texture *>(source_texture);
    if (!texture_object || !source || !view_out || texture_object->mipmap_levels == 0u) {
        if (view_out) *view_out = source_texture;
        return 0;
    }
    const uint32_t base = texture_object->params.base_level;
    if (base >= texture_object->mipmap_levels || base >= source->mipmapLevelCount()) {
        *view_out = source_texture;
        return 0;
    }
    uint32_t max_level = texture_object->params.max_level == 1000u
        ? texture_object->mipmap_levels - 1u : texture_object->params.max_level;
    if (max_level < base) max_level = base;
    if (max_level >= texture_object->mipmap_levels) max_level = texture_object->mipmap_levels - 1u;
    if (max_level >= source->mipmapLevelCount()) max_level = source->mipmapLevelCount() - 1u;
    const uint64_t level_count = static_cast<uint64_t>(max_level - base + 1u);
    uint64_t slice_count = source->arrayLength();
    const auto type = source->textureType();
    if (type == MTL::TextureTypeCube || type == MTL::TextureTypeCubeArray) {
        slice_count *= 6u;
    }
    const uint32_t components =
        mglRenderStoredColorComponents(texture_object->internalformat);
    uint32_t swizzle_red = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_r, components);
    uint32_t swizzle_green = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_g, components);
    uint32_t swizzle_blue = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_b, components);
    uint32_t swizzle_alpha = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_a, components);
    /* Formats expanded/baked at upload must not get Metal view swizzle. */
    const bool upload_swizzle_baked =
        texture_object->params.swizzled &&
        !texture_object->is_render_target &&
        mglRenderTextureSwizzleUsesUploadBake(
            texture_object->internalformat, 1,
            static_cast<uint32_t>(source->pixelFormat())) != 0;
    if (upload_swizzle_baked) {
        swizzle_red = static_cast<uint32_t>(MTL::TextureSwizzleRed);
        swizzle_green = static_cast<uint32_t>(MTL::TextureSwizzleGreen);
        swizzle_blue = static_cast<uint32_t>(MTL::TextureSwizzleBlue);
        swizzle_alpha = static_cast<uint32_t>(MTL::TextureSwizzleAlpha);
    }
    const bool identity =
        swizzle_red == static_cast<uint32_t>(MTL::TextureSwizzleRed) &&
        swizzle_green == static_cast<uint32_t>(MTL::TextureSwizzleGreen) &&
        swizzle_blue == static_cast<uint32_t>(MTL::TextureSwizzleBlue) &&
        swizzle_alpha == static_cast<uint32_t>(MTL::TextureSwizzleAlpha);
    if (level_count == 0u ||
        (base == 0u && level_count >= source->mipmapLevelCount() &&
         identity)) {
        *view_out = source_texture;
        return 0;
    }
    if (texture_object->mtl_base_level_view &&
        texture_object->mtl_base_level_view_source == source_texture &&
        texture_object->mtl_base_level_view_base == base &&
        texture_object->mtl_base_level_view_max == max_level &&
        texture_object->mtl_base_level_view_swizzle_r ==
            (GLuint)texture_object->params.swizzle_r &&
        texture_object->mtl_base_level_view_swizzle_g ==
            (GLuint)texture_object->params.swizzle_g &&
        texture_object->mtl_base_level_view_swizzle_b ==
            (GLuint)texture_object->params.swizzle_b &&
        texture_object->mtl_base_level_view_swizzle_a ==
            (GLuint)texture_object->params.swizzle_a) {
        *view_out = texture_object->mtl_base_level_view;
        return 0;
    }

    void *view_handle = nullptr;
    if (mglRenderCreateTextureViewRange(
            source_texture, static_cast<uint32_t>(source->pixelFormat()),
            static_cast<uint32_t>(type), base, level_count, 0u, slice_count,
            identity ? 0 : 1, swizzle_red, swizzle_green, swizzle_blue,
            swizzle_alpha, &view_handle) != 0 || !view_handle) {
        *view_out = source_texture;
        return 0;
    }
    if (texture_object->mtl_base_level_view) {
        static_cast<NS::Object *>(texture_object->mtl_base_level_view)->release();
    }
    static_cast<NS::Object *>(view_handle)->retain();
    texture_object->mtl_base_level_view = view_handle;
    texture_object->mtl_base_level_view_source = source_texture;
    texture_object->mtl_base_level_view_base = base;
    texture_object->mtl_base_level_view_max = max_level;
    texture_object->mtl_base_level_view_swizzle_r =
        (GLuint)texture_object->params.swizzle_r;
    texture_object->mtl_base_level_view_swizzle_g =
        (GLuint)texture_object->params.swizzle_g;
    texture_object->mtl_base_level_view_swizzle_b =
        (GLuint)texture_object->params.swizzle_b;
    texture_object->mtl_base_level_view_swizzle_a =
        (GLuint)texture_object->params.swizzle_a;
    static_cast<NS::Object *>(view_handle)->release();
    *view_out = texture_object->mtl_base_level_view;
    return 0;
}

int mglRenderTextureReplaceRegion(void* texture,
                                     uint64_t x,
                                     uint64_t y,
                                     uint64_t z,
                                     uint64_t width,
                                     uint64_t height,
                                     uint64_t depth,
                                     uint64_t level,
                                     uint64_t slice,
                                     const void* bytes,
                                     uint64_t bytes_per_row,
                                     uint64_t bytes_per_image,
                                     int use_slice) {
    MTL::Texture* destination = static_cast<MTL::Texture*>(texture);
    if (!destination || !bytes || width == 0 || height == 0 || depth == 0 ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::Region region = MTL::Region::Make3D(
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(z), static_cast<NS::UInteger>(width),
        static_cast<NS::UInteger>(height),
        static_cast<NS::UInteger>(depth));
    if (use_slice) {
        if (bytes_per_image == 0) return -1;
        destination->replaceRegion(
            region, static_cast<NS::UInteger>(level),
            static_cast<NS::UInteger>(slice), bytes,
            static_cast<NS::UInteger>(bytes_per_row),
            static_cast<NS::UInteger>(bytes_per_image));
    } else {
        destination->replaceRegion(
            region, static_cast<NS::UInteger>(level), bytes,
            static_cast<NS::UInteger>(bytes_per_row));
    }
    return 0;
}

int mglRenderTextureGetBytes(void* texture,
                                void* bytes,
                                uint64_t bytes_per_row,
                                uint64_t bytes_per_image,
                                uint64_t x,
                                uint64_t y,
                                uint64_t z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                uint64_t level,
                                uint64_t slice,
                                int use_slice) {
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !bytes || width == 0 || height == 0 || depth == 0 ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::Region region = MTL::Region::Make3D(
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(z), static_cast<NS::UInteger>(width),
        static_cast<NS::UInteger>(height),
        static_cast<NS::UInteger>(depth));
    if (use_slice) {
        if (bytes_per_image == 0) return -1;
        source->getBytes(
            bytes, static_cast<NS::UInteger>(bytes_per_row),
            static_cast<NS::UInteger>(bytes_per_image), region,
            static_cast<NS::UInteger>(level),
            static_cast<NS::UInteger>(slice));
    } else {
        source->getBytes(bytes, static_cast<NS::UInteger>(bytes_per_row),
                         region, static_cast<NS::UInteger>(level));
    }
    return 0;
}

extern "C"
int mglRenderTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderTextureTargetPlan* plan_out) {
    if (!plan_out) return -1;
    *plan_out = {};
    plan_out->num_faces = 1u;

    switch (gl_target) {
        case GL_TEXTURE_1D:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType2D);
            plan_out->texture_1d_backed_by_2d = 1u;
            return 0;
        case GL_RENDERBUFFER:
            plan_out->texture_type = static_cast<uint32_t>(
                sample_count > 1u ? MTL::TextureType2DMultisample
                                  : MTL::TextureType2D);
            return 0;
        case GL_TEXTURE_1D_ARRAY:
            /* AIR lowers sampler1DArray to texture2d_array, and Metal cannot
             * view a Texture1DArray as Texture2DArray. */
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DArray);
            plan_out->is_array = 1u;
            plan_out->texture_1d_array_backed_by_2d_array = 1u;
            return 0;
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType2D);
            return 0;
        case GL_TEXTURE_2D_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DArray);
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DMultisample);
            return 0;
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureTypeCube);
            plan_out->num_faces = 6u;
            return 0;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureTypeCubeArray);
            plan_out->num_faces = 6u;
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_3D:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType3D);
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DMultisampleArray);
            plan_out->is_array = 1u;
            return 0;
        default:
            *plan_out = {};
            return -1;
    }
}

extern "C"
int mglRenderTextureSubUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint64_t requested_slice,
    uint64_t xoffset,
    uint64_t yoffset,
    uint64_t zoffset,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    MGLRenderTextureSubUploadPlan* plan_out) {
    if (!plan_out || width == 0u || height == 0u || depth == 0u ||
        source_bytes_per_row == 0u || source_bytes_per_image == 0u) {
        return -1;
    }

    *plan_out = {};
    plan_out->destination_x = xoffset;
    plan_out->copy_width = width;
    plan_out->copy_height = height;
    plan_out->copy_depth = 1u;
    plan_out->layer_count = 1u;

    if (gl_target == GL_TEXTURE_1D_ARRAY) {
        if (yoffset > std::numeric_limits<uint64_t>::max() - (height - 1u)) {
            *plan_out = {};
            return -1;
        }
        plan_out->destination_base_slice = yoffset;
        plan_out->destination_y = 0u;
        plan_out->destination_z = 0u;
        plan_out->copy_height = 1u;
        plan_out->copy_depth = 1u;
        plan_out->layer_count = height;
        plan_out->source_layer_stride = source_bytes_per_row;
        return 0;
    }

    if (gl_target == GL_TEXTURE_2D_ARRAY ||
        gl_target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        if (zoffset > std::numeric_limits<uint64_t>::max() - (depth - 1u)) {
            *plan_out = {};
            return -1;
        }
        plan_out->destination_base_slice = zoffset;
        plan_out->destination_y = yoffset;
        plan_out->destination_z = 0u;
        plan_out->copy_depth = 1u;
        plan_out->layer_count = depth;
        plan_out->source_layer_stride = source_bytes_per_image;
        return 0;
    }

    switch (static_cast<MTL::TextureType>(texture_type)) {
        case MTL::TextureType3D:
            plan_out->destination_y = yoffset;
            plan_out->destination_z = zoffset;
            plan_out->copy_depth = depth;
            return 0;
        case MTL::TextureTypeCube:
        case MTL::TextureTypeCubeArray:
        case MTL::TextureType2DArray:
        case MTL::TextureType1DArray:
        case MTL::TextureType2DMultisampleArray:
            plan_out->destination_base_slice = requested_slice;
            plan_out->destination_y = yoffset;
            return 0;
        default:
            plan_out->destination_y =
                gl_target == GL_TEXTURE_1D ? 0u : yoffset;
            plan_out->copy_height =
                gl_target == GL_TEXTURE_1D ? 1u : height;
            return 0;
    }
}

extern "C"
uint32_t mglRenderTextureTypeForShaderResource(
    uint32_t has_resource,
    uint32_t image_dim,
    uint32_t image_arrayed,
    uint32_t image_multisampled) {
    if (!has_resource) return 0u;
    switch (image_dim) {
        case MGL_IMAGE_DIM_1D:
            /* GL 1D/1D-array textures are backed by Metal 2D/2D-array storage
             * (see mglRenderTextureTargetPlan); AIR lowers sampler1DArray the same way. */
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureType2DArray : MTL::TextureType2D);
        case MGL_IMAGE_DIM_2D:
            if (image_multisampled) {
                return static_cast<uint32_t>(
                    image_arrayed ? MTL::TextureType2DMultisampleArray
                                  : MTL::TextureType2DMultisample);
            }
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureType2DArray : MTL::TextureType2D);
        case MGL_IMAGE_DIM_3D:
            return static_cast<uint32_t>(MTL::TextureType3D);
        case MGL_IMAGE_DIM_CUBE:
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureTypeCubeArray
                              : MTL::TextureTypeCube);
        case MGL_IMAGE_DIM_BUFFER:
            return static_cast<uint32_t>(MTL::TextureTypeTextureBuffer);
        default:
            return 0u;
    }
}

extern "C"
int32_t mglRenderTextureIndexForMetalType(uint32_t texture_type) {
    switch (static_cast<MTL::TextureType>(texture_type)) {
        case MTL::TextureType1D:
            return _TEXTURE_1D;
        case MTL::TextureType1DArray:
            return _TEXTURE_1D_ARRAY;
        case MTL::TextureType2D:
            return _TEXTURE_2D;
        case MTL::TextureType2DMultisample:
            return _TEXTURE_2D_MULTISAMPLE;
        case MTL::TextureType2DArray:
            return _TEXTURE_2D_ARRAY;
        case MTL::TextureType2DMultisampleArray:
            return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case MTL::TextureType3D:
            return _TEXTURE_3D;
        case MTL::TextureTypeCube:
            return _TEXTURE_CUBE_MAP;
        case MTL::TextureTypeCubeArray:
            return _TEXTURE_CUBE_MAP_ARRAY;
        case MTL::TextureTypeTextureBuffer:
            return _TEXTURE_BUFFER;
        default:
            return -1;
    }
}

extern "C"
uint32_t mglRenderTextureDataKindForPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA32Sint:
            return MGL_RENDER_TEXTURE_DATA_KIND_SINT;

        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatR16Uint:
        case MTL::PixelFormatRG16Uint:
        case MTL::PixelFormatRGBA16Uint:
        case MTL::PixelFormatR32Uint:
        case MTL::PixelFormatRG32Uint:
        case MTL::PixelFormatRGBA32Uint:
        case MTL::PixelFormatRGB10A2Uint:
            return MGL_RENDER_TEXTURE_DATA_KIND_UINT;

        case MTL::PixelFormatInvalid:
            return MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN;

        case MTL::PixelFormatDepth16Unorm:
        case MTL::PixelFormatDepth32Float:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
            return MGL_RENDER_TEXTURE_DATA_KIND_DEPTH;

        default:
            return MGL_RENDER_TEXTURE_DATA_KIND_FLOAT;
    }
}

extern "C"
int mglRenderMetalPixelFormatIsDepthOrStencil(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatDepth16Unorm:
        case MTL::PixelFormatDepth32Float:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
        case MTL::PixelFormatStencil8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderMetalPixelFormatIsPackedDepthStencil(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderGLInternalFormatLooksDepthOrStencil(uint32_t internal_format) {
    switch (internal_format) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTexturePixelFormatCompatibleWithExpectedDataKind(
    uint32_t pixel_format, uint32_t expected_kind) {
    if (expected_kind == MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN) {
        return 1;
    }
    return mglRenderTextureDataKindForPixelFormat(pixel_format) ==
           expected_kind;
}

extern "C"
uint64_t mglRenderMetalCompressedBlockHeight(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBC1_RGBA:
        case MTL::PixelFormatBC1_RGBA_sRGB:
        case MTL::PixelFormatBC2_RGBA:
        case MTL::PixelFormatBC2_RGBA_sRGB:
        case MTL::PixelFormatBC3_RGBA:
        case MTL::PixelFormatBC3_RGBA_sRGB:
        case MTL::PixelFormatBC4_RUnorm:
        case MTL::PixelFormatBC4_RSnorm:
        case MTL::PixelFormatBC5_RGUnorm:
        case MTL::PixelFormatBC5_RGSnorm:
        case MTL::PixelFormatBC6H_RGBFloat:
        case MTL::PixelFormatBC6H_RGBUfloat:
        case MTL::PixelFormatBC7_RGBAUnorm:
        case MTL::PixelFormatBC7_RGBAUnorm_sRGB:
        case MTL::PixelFormatASTC_4x4_sRGB:
        case MTL::PixelFormatASTC_4x4_LDR:
        case MTL::PixelFormatASTC_4x4_HDR:
        case MTL::PixelFormatASTC_5x4_sRGB:
        case MTL::PixelFormatASTC_5x4_LDR:
        case MTL::PixelFormatASTC_5x4_HDR:
            return 4u;
        case MTL::PixelFormatASTC_5x5_sRGB:
        case MTL::PixelFormatASTC_5x5_LDR:
        case MTL::PixelFormatASTC_5x5_HDR:
        case MTL::PixelFormatASTC_6x5_sRGB:
        case MTL::PixelFormatASTC_6x5_LDR:
        case MTL::PixelFormatASTC_6x5_HDR:
        case MTL::PixelFormatASTC_8x5_sRGB:
        case MTL::PixelFormatASTC_8x5_LDR:
        case MTL::PixelFormatASTC_8x5_HDR:
        case MTL::PixelFormatASTC_10x5_sRGB:
        case MTL::PixelFormatASTC_10x5_LDR:
        case MTL::PixelFormatASTC_10x5_HDR:
            return 5u;
        case MTL::PixelFormatASTC_6x6_sRGB:
        case MTL::PixelFormatASTC_6x6_LDR:
        case MTL::PixelFormatASTC_6x6_HDR:
        case MTL::PixelFormatASTC_8x6_sRGB:
        case MTL::PixelFormatASTC_8x6_LDR:
        case MTL::PixelFormatASTC_8x6_HDR:
        case MTL::PixelFormatASTC_10x6_sRGB:
        case MTL::PixelFormatASTC_10x6_LDR:
        case MTL::PixelFormatASTC_10x6_HDR:
            return 6u;
        case MTL::PixelFormatASTC_8x8_sRGB:
        case MTL::PixelFormatASTC_8x8_LDR:
        case MTL::PixelFormatASTC_8x8_HDR:
        case MTL::PixelFormatASTC_10x8_sRGB:
        case MTL::PixelFormatASTC_10x8_LDR:
        case MTL::PixelFormatASTC_10x8_HDR:
            return 8u;
        case MTL::PixelFormatASTC_10x10_sRGB:
        case MTL::PixelFormatASTC_10x10_LDR:
        case MTL::PixelFormatASTC_10x10_HDR:
        case MTL::PixelFormatASTC_12x10_sRGB:
        case MTL::PixelFormatASTC_12x10_LDR:
        case MTL::PixelFormatASTC_12x10_HDR:
            return 10u;
        case MTL::PixelFormatASTC_12x12_sRGB:
        case MTL::PixelFormatASTC_12x12_LDR:
        case MTL::PixelFormatASTC_12x12_HDR:
            return 12u;
        default:
            return 1u;
    }
}

extern "C"
uint64_t mglRenderMetalUploadRowsForPixelFormat(uint32_t pixel_format,
                                                   uint64_t pixel_height) {
    const uint64_t height = pixel_height ? pixel_height : 1u;
    const uint64_t block_height =
        mglRenderMetalCompressedBlockHeight(pixel_format);
    if (block_height <= 1u) {
        return height;
    }
    return (height + block_height - 1u) / block_height;
}

extern "C"
const char* mglRenderTextureDataKindName(uint32_t kind) {
    switch (kind) {
        case MGL_RENDER_TEXTURE_DATA_KIND_FLOAT:
            return "float";
        case MGL_RENDER_TEXTURE_DATA_KIND_SINT:
            return "sint";
        case MGL_RENDER_TEXTURE_DATA_KIND_UINT:
            return "uint";
        case MGL_RENDER_TEXTURE_DATA_KIND_DEPTH:
            return "depth";
        default:
            return "unknown";
    }
}

extern "C"
int mglRenderTextureMinFilterUsesMipmaps(uint32_t min_filter) {
    switch (min_filter) {
        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_LINEAR:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderMetalLayerPixelFormatIsSupported(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderSRGBPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA8Unorm:
            return (uint32_t)MTL::PixelFormatRGBA8Unorm_sRGB;
        case MTL::PixelFormatBGRA8Unorm:
            return (uint32_t)MTL::PixelFormatBGRA8Unorm_sRGB;
        default:
            return pixel_format;
    }
}

extern "C"
uint32_t mglRenderLinearPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA8Unorm_sRGB:
            return (uint32_t)MTL::PixelFormatRGBA8Unorm;
        case MTL::PixelFormatBGRA8Unorm_sRGB:
            return (uint32_t)MTL::PixelFormatBGRA8Unorm;
        default:
            return pixel_format;
    }
}

extern "C"
uint32_t mglRenderEffectiveMTLPixelFormat(uint32_t pixel_format,
                                            uint32_t srgb_decode_ext) {
    if (srgb_decode_ext == GL_SKIP_DECODE_EXT) {
        return mglRenderLinearPixelFormat(pixel_format);
    }
    return pixel_format;
}


extern "C"
uint32_t mglRenderReadbackBytesPerPixel(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA32Float:
            return (uint32_t)(sizeof(float) * 4u);
        case MTL::PixelFormatR8Unorm:
            return 1u;
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatRG8Unorm:
        case MTL::PixelFormatABGR4Unorm:
        case MTL::PixelFormatBGR5A1Unorm:
        case MTL::PixelFormatR16Float:
            return 2u;
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRGBA16Float:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
            return 8u;
        case MTL::PixelFormatR8Snorm:
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
            return 1u;
        case MTL::PixelFormatRG8Snorm:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
            return 2u;
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRG16Float:
        case MTL::PixelFormatRGBA8Snorm:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        default:
            return 4u;
    }
}


extern "C"
int mglRenderReadbackFormatIsBGRA8Compatible(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB:
        case MTL::PixelFormatRGBA8Unorm:
        case MTL::PixelFormatRGBA8Unorm_sRGB:
        case MTL::PixelFormatRGBA32Float:
        case MTL::PixelFormatR8Unorm:
        case MTL::PixelFormatRG8Unorm:
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
        case MTL::PixelFormatABGR4Unorm:
        case MTL::PixelFormatBGR5A1Unorm:
        case MTL::PixelFormatRG11B10Float:
        case MTL::PixelFormatR32Float:
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRG16Float:
        case MTL::PixelFormatR16Float:
        case MTL::PixelFormatRGBA16Float:
        case MTL::PixelFormatBGR10A2Unorm:
        case MTL::PixelFormatRGB10A2Unorm:
        case MTL::PixelFormatR8Snorm:
        case MTL::PixelFormatRG8Snorm:
        case MTL::PixelFormatRGBA8Snorm:
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGB9E5Float:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderPixelFormatIsIntegerColor(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatR16Uint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatR32Uint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRG16Uint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRG32Uint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGBA16Uint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA32Uint:
        case MTL::PixelFormatRGBA32Sint:
        case MTL::PixelFormatRGB10A2Uint:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderPixelFormatIsSignedIntegerColor(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA32Sint:
            return 1;
        default:
            return 0;
    }
}


int mglRenderTextureUploadRoute(uint32_t texture_type,
                                   uint32_t storage_mode,
                                   int has_agx_3d_copy_bug) {

    const uint32_t kMTLTextureType1D = 0u;
    const uint32_t kMTLTextureType1DArray = 1u;
    const uint32_t kMTLTextureType3D = 7u;
    const uint32_t kMTLStorageModePrivate = 2u;


    if ((texture_type == kMTLTextureType1D ||
         texture_type == kMTLTextureType1DArray) &&
        storage_mode != kMTLStorageModePrivate) {
        return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D;
    }


    if (texture_type == kMTLTextureType3D && has_agx_3d_copy_bug) {
        if (storage_mode == kMTLStorageModePrivate) {
            return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT;
        }
        return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D;
    }

    return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_BLIT;
}

extern "C"
int mglRenderBuildTextureUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint32_t storage_mode,
    uint32_t pixel_format,
    int has_agx_3d_copy_bug,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t bytes_per_row,
    uint64_t bytes_per_image,
    uint64_t destination_level,
    uint64_t destination_slice,
    MGLRenderTextureUploadPlan* plan_out) {
    if (!plan_out) return -1;
    *plan_out = {};
    if (width == 0u || bytes_per_row == 0u || bytes_per_image == 0u) {
        return -1;
    }

    const MTL::TextureType type = static_cast<MTL::TextureType>(texture_type);
    const bool is_3d = type == MTL::TextureType3D;
    const bool logical_1d =
        gl_target == GL_TEXTURE_1D || gl_target == GL_TEXTURE_1D_ARRAY;
    const bool logical_1d_array = gl_target == GL_TEXTURE_1D_ARRAY;
    const bool is_array_or_cube =
        type == MTL::TextureTypeCube || type == MTL::TextureTypeCubeArray ||
        type == MTL::TextureType2DArray || type == MTL::TextureType1DArray ||
        type == MTL::TextureType2DMultisampleArray;

    MGLRenderTextureUploadPlan plan = {};
    plan.normalized_height = logical_1d
        ? 1u
        : std::max<uint64_t>(height, 1u);
    plan.normalized_depth = std::max<uint64_t>(depth, 1u);
    plan.copy_depth = is_3d ? plan.normalized_depth : 1u;
    plan.upload_rows = mglRenderMetalUploadRowsForPixelFormat(
        pixel_format, plan.normalized_height);
    if (plan.upload_rows == 0u ||
        bytes_per_row > std::numeric_limits<uint64_t>::max() /
                            plan.upload_rows) {
        return -1;
    }
    plan.expected_bytes_per_image = bytes_per_row * plan.upload_rows;
    if (bytes_per_image < plan.expected_bytes_per_image) return -1;
    plan.normalized_bytes_per_image =
        (is_array_or_cube || !is_3d)
            ? plan.expected_bytes_per_image
            : bytes_per_image;
    plan.destination_slice =
        (is_3d || gl_target == GL_TEXTURE_1D) ? 0u : destination_slice;
    plan.destination_level = destination_level;

    const uint32_t private_mode =
        static_cast<uint32_t>(MTL::StorageModePrivate);
    if (logical_1d && storage_mode != private_mode) {
        plan.route = MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D;
        plan.replace_region_dimension =
            (type == MTL::TextureType1D || type == MTL::TextureType1DArray)
                ? 1u
                : 2u;
        plan.replace_use_slice = logical_1d_array ||
                                 type == MTL::TextureType1DArray;
    } else {
        plan.route = static_cast<uint32_t>(mglRenderTextureUploadRoute(
            texture_type, storage_mode, has_agx_3d_copy_bug));
        if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D) {
            plan.replace_region_dimension = 1u;
            plan.replace_use_slice = type == MTL::TextureType1DArray;
        } else if (plan.route ==
                   MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
            plan.replace_region_dimension = 3u;
            plan.requires_repack =
                plan.normalized_bytes_per_image !=
                plan.expected_bytes_per_image;
        }
    }

    if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT) {
        *plan_out = plan;
        return 0;
    }

    if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
        if (plan.requires_repack &&
            plan.copy_depth > std::numeric_limits<uint64_t>::max() /
                                  plan.expected_bytes_per_image) {
            return -1;
        }
    } else {
        if (plan.copy_depth > std::numeric_limits<uint64_t>::max() /
                                  plan.normalized_bytes_per_image) {
            return -1;
        }
        plan.buffer_size =
            plan.normalized_bytes_per_image * plan.copy_depth;
        constexpr uint64_t kMaxTextureUploadStagingBytes =
            512ull * 1024ull * 1024ull;
        if (plan.buffer_size == 0u ||
            plan.buffer_size > kMaxTextureUploadStagingBytes) {
            return -1;
        }
    }

    *plan_out = plan;
    return 0;
}


static uint32_t mglPackRGBToSharedExp(double red, double green, double blue)
{
    const int N     = 9;   /* mantissa bits */
    const int B     = 15;  /* exponent bias */
    const int E_max = 31;  /* max exponent */

    double shared_exp_max = ((double)((1 << N) - 1) / (double)(1 << N)) *
                            ldexp(1.0, E_max - B);

    double red_c   = fmax(0.0, fmin(shared_exp_max, red));
    double green_c = fmax(0.0, fmin(shared_exp_max, green));
    double blue_c  = fmax(0.0, fmin(shared_exp_max, blue));

    double max_c = fmax(fmax(red_c, green_c), blue_c);

    double exp_p;
    if (max_c <= 0.0) {
        exp_p = 0.0;
    } else {
        exp_p = fmax((double)(-B - 1), floor(log2(max_c))) + 1.0 + (double)B;
    }

    double scale_p = ldexp(1.0, (int)exp_p - B - N);
    double max_s = floor(max_c / scale_p + 0.5);

    int exp_s;
    if (max_s >= (double)(1 << N)) {
        exp_s = (int)exp_p + 1;
    } else {
        exp_s = (int)exp_p;
    }
    if (exp_s < 0) exp_s = 0;
    if (exp_s > E_max) exp_s = E_max;

    double scale = ldexp(1.0, exp_s - B - N);

    uint32_t red_s   = (uint32_t)floor(red_c   / scale + 0.5);
    uint32_t green_s = (uint32_t)floor(green_c / scale + 0.5);
    uint32_t blue_s  = (uint32_t)floor(blue_c  / scale + 0.5);

    if (red_s > 511u) red_s = 511u;
    if (green_s > 511u) green_s = 511u;
    if (blue_s > 511u) blue_s = 511u;

    return red_s | (green_s << 9) | (blue_s << 18) | ((uint32_t)exp_s << 27);
}


extern "C"
void mglRenderCopyRows(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t row_bytes, uint64_t height, int flip_y) {
    if (!src || !dst || row_bytes == 0u || height == 0u) {
        return;
    }
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        memcpy(dst_row, src_row, row_bytes);
    }
}


extern "C"
void mglRenderCopyDepthTextureBytesToFloat(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint64_t src_depth_bytes, int is_depth16, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u ||
        src_depth_bytes == 0u) {
        return;
    }
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        float* dst_row = reinterpret_cast<float*>(
            dst_bytes + (dst_y * dst_bytes_per_row));
        for (uint64_t x = 0; x < width; x++) {
            if (is_depth16) {
                uint16_t value = 0u;
                memcpy(&value, src_row + (x * src_depth_bytes),
                       sizeof(value));
                dst_row[x] = (float)value / 65535.0f;
            } else {
                memcpy(&dst_row[x], src_row + (x * src_depth_bytes),
                       sizeof(float));
            }
        }
    }
}


extern "C"
int mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    if (src_bytes_per_row < width * 4u ||
        dst_bytes_per_row < width * 4u) {
        return 0;
    }

    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    bool destinationIsRGBA = (pf == MTL::PixelFormatRGBA8Unorm ||
                              pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    bool destinationIsBGRA = (pf == MTL::PixelFormatBGRA8Unorm ||
                              pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    bool destinationIsRGB9E5 = (pf == MTL::PixelFormatRGB9E5Float);
    bool destinationIsRGB10A2 = (pf == MTL::PixelFormatRGB10A2Unorm ||
                                 pf == MTL::PixelFormatBGR10A2Unorm);
    if (!destinationIsRGBA && !destinationIsBGRA &&
        !destinationIsRGB9E5 && !destinationIsRGB10A2) {
        return 0;
    }

    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* srcRow = srcBytes + (y * src_bytes_per_row);
        uint64_t dstY = flip_y ? (height - 1u - y) : y;
        uint8_t* dstRow = dstBytes + (dstY * dst_bytes_per_row);

        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = srcRow + (x * 4u);
            uint8_t* d = dstRow + (x * 4u);
            uint8_t b = s[0];
            uint8_t g = s[1];
            uint8_t r = s[2];
            uint8_t a = s[3];

            if (destinationIsBGRA) {
                d[0] = b;
                d[1] = g;
                d[2] = r;
                d[3] = a;
            } else if (destinationIsRGB10A2) {
                /* RGB10A2Unorm: bits [0:9]=R, [10:19]=G, [20:29]=B,
                 * [30:31]=A.  BGR10A2Unorm: bits [0:9]=B, [10:19]=G,
                 * [20:29]=R, [30:31]=A. */
                uint32_t r10 = ((uint32_t)r * 1023u + 127u) / 255u;
                uint32_t g10 = ((uint32_t)g * 1023u + 127u) / 255u;
                uint32_t b10 = ((uint32_t)b * 1023u + 127u) / 255u;
                uint32_t a2 = ((uint32_t)a * 3u + 127u) / 255u;
                uint32_t packed;
                if (pf == MTL::PixelFormatBGR10A2Unorm) {
                    packed = b10 | (g10 << 10) | (r10 << 20) | (a2 << 30);
                } else {
                    packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
                }
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else if (destinationIsRGB9E5) {
                /* GL_RGB9_E5 packs three 9-bit mantissas and a 5-bit shared
                 * exponent into a 32-bit word.  Source is BGRA8. */
                uint32_t packed = mglPackRGBToSharedExp(
                    (double)r / 255.0, (double)g / 255.0,
                    (double)b / 255.0);
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else {
                d[0] = r;
                d[1] = g;
                d[2] = b;
                d[3] = a;
            }
        }
    }

    return 1;
}


static float mglHalfToFloat(uint16_t value)
{
    uint32_t sign = (uint32_t)(value >> 15u);
    uint32_t exponent = (value >> 10u) & 31u;
    uint32_t mantissa = value & 1023u;
    float result;
    if (exponent == 0u) {
        result = ldexpf((float)mantissa, -24);
    } else if (exponent == 31u) {
        result = mantissa ? NAN : INFINITY;
    } else {
        result = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    }
    return sign ? -result : result;
}


static uint16_t mglFloatToHalf(float value)
{
    uint32_t f;
    memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 16u) & 0x8000u;
    int32_t exp = ((int32_t)(f >> 23u) & 0xff) - 112;
    uint32_t mant = f & 0x7fffffu;

    if (exp >= 143) {
        if (mant != 0u) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    if (exp <= 0) {
        int shift = 1 - exp;
        if (shift >= 25) {
            return (uint16_t)sign;
        }
        uint32_t m = (mant | 0x800000u) >> shift;
        m += 0x00001000u + ((m >> 13u) & 1u);
        return (uint16_t)(sign | (m >> 13u));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    mant += 0x00001000u + ((mant >> 13u) & 1u);
    if (mant >= 0x800000u) {
        mant = 0;
        exp++;
        if (exp >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10u) | (mant >> 13u));
}


static uint32_t mglFloatToFloat11(float v)
{
    if (isnan(v)) return 0x7e0u;
    if (v <= 0.0f) return 0u;
    if (v >= 65024.0f) return 0x7c0u;
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    int ieee_exp = (int)((bits >> 23) & 0xff) - 127;
    uint32_t ieee_mant = bits & 0x7fffff;
    if (ieee_exp <= -15) {
        int shift = -14 - ieee_exp;
        if (shift >= 11) return 0u;
        uint32_t src = (ieee_mant | 0x800000);
        int rshift = 23 - 6 + shift;
        uint32_t m = src >> rshift;
        uint32_t rem = src & ((1u << rshift) - 1u);
        uint32_t half = 1u << (rshift - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            m += 1u;
        }
        return m & 0x3fu;
    }
    if (ieee_exp >= 16) return 0x7c0u;
    uint32_t exp = (uint32_t)(ieee_exp + 15);
    uint32_t mant = ieee_mant >> (23 - 6);
    uint32_t rem = ieee_mant & ((1u << (23 - 6)) - 1u);
    uint32_t half = 1u << (23 - 6 - 1);
    if (rem > half || (rem == half && (mant & 1u))) {
        mant += 1u;
        if (mant > 0x3fu) {
            mant = 0u;
            exp += 1u;
            if (exp >= 31u) return 0x7c0u;
        }
    }
    return (exp << 6) | mant;
}


static uint32_t mglFloatToFloat10(float v)
{
    if (isnan(v)) return 0x3f0u;
    if (v <= 0.0f) return 0u;
    if (v >= 64512.0f) return 0x3e0u;
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    int ieee_exp = (int)((bits >> 23) & 0xff) - 127;
    uint32_t ieee_mant = bits & 0x7fffff;
    if (ieee_exp <= -15) {
        int shift = -14 - ieee_exp;
        if (shift >= 10) return 0u;
        uint32_t src = (ieee_mant | 0x800000);
        int rshift = 23 - 5 + shift;
        uint32_t m = src >> rshift;
        uint32_t rem = src & ((1u << rshift) - 1u);
        uint32_t half = 1u << (rshift - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            m += 1u;
        }
        return m & 0x1fu;
    }
    if (ieee_exp >= 16) return 0x3e0u;
    uint32_t exp = (uint32_t)(ieee_exp + 15);
    uint32_t mant = ieee_mant >> (23 - 5);
    uint32_t rem = ieee_mant & ((1u << (23 - 5)) - 1u);
    uint32_t half = 1u << (23 - 5 - 1);
    if (rem > half || (rem == half && (mant & 1u))) {
        mant += 1u;
        if (mant > 0x1fu) {
            mant = 0u;
            exp += 1u;
            if (exp >= 31u) return 0x3e0u;
        }
    }
    return (exp << 5) | mant;
}


static float mglUnpackUnsignedFloatComponent(uint32_t value,
                                               uint32_t mantissa_bits)
{
    if (mantissa_bits == 0u || mantissa_bits > 23u) return 0.0f;

    uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
    uint32_t mantissa = value & mantissa_mask;
    uint32_t exponent = (value >> mantissa_bits) & 0x1fu;

    if (exponent == 31u) {
        return (mantissa == 0u) ? INFINITY : NAN;
    }
    if (exponent == 0u) {
        return ldexpf((float)mantissa, 1 - 15 - (int)mantissa_bits);
    }
    float normalized = 1.0f + (float)mantissa / (float)(1u << mantissa_bits);
    return ldexpf(normalized, (int)exponent - 15);
}


extern "C"
void mglRenderCopyTextureBytesToBGRA8(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return;
    }

    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    bool sourceIsRGBA =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    bool sourceIsRGBA32Float = (pf == MTL::PixelFormatRGBA32Float);
    bool sourceIsR8 = (pf == MTL::PixelFormatR8Unorm);
    bool sourceIsRG8 = (pf == MTL::PixelFormatRG8Unorm);
    bool sourceIsR16Unorm = (pf == MTL::PixelFormatR16Unorm);
    bool sourceIsRG16Unorm = (pf == MTL::PixelFormatRG16Unorm);
    bool sourceIsRGBA16Unorm = (pf == MTL::PixelFormatRGBA16Unorm);
    bool sourceIsR16Snorm = (pf == MTL::PixelFormatR16Snorm);
    bool sourceIsRG16Snorm = (pf == MTL::PixelFormatRG16Snorm);
    bool sourceIsRGBA16Snorm = (pf == MTL::PixelFormatRGBA16Snorm);
    bool sourceIsBGR5A1 = (pf == MTL::PixelFormatBGR5A1Unorm);
    bool sourceIsABGR4 = (pf == MTL::PixelFormatABGR4Unorm);
    bool sourceIsRG11B10Float = (pf == MTL::PixelFormatRG11B10Float);
    bool sourceIsR32Float = (pf == MTL::PixelFormatR32Float);
    bool sourceIsRG32Float = (pf == MTL::PixelFormatRG32Float);
    bool sourceIsRG16Float = (pf == MTL::PixelFormatRG16Float);
    bool sourceIsR16Float = (pf == MTL::PixelFormatR16Float);
    bool sourceIsRGBA16Float = (pf == MTL::PixelFormatRGBA16Float);
    bool sourceIsBGR10A2 = (pf == MTL::PixelFormatBGR10A2Unorm);
    bool sourceIsRGB10A2 = (pf == MTL::PixelFormatRGB10A2Unorm);
    bool sourceIsR8Snorm = (pf == MTL::PixelFormatR8Snorm);
    bool sourceIsRG8Snorm = (pf == MTL::PixelFormatRG8Snorm);
    bool sourceIsRGBA8Snorm = (pf == MTL::PixelFormatRGBA8Snorm);
    bool sourceIsR8Uint = (pf == MTL::PixelFormatR8Uint);
    bool sourceIsR8Sint = (pf == MTL::PixelFormatR8Sint);
    bool sourceIsRG8Uint = (pf == MTL::PixelFormatRG8Uint);
    bool sourceIsRG8Sint = (pf == MTL::PixelFormatRG8Sint);
    bool sourceIsRGBA8Uint = (pf == MTL::PixelFormatRGBA8Uint);
    bool sourceIsRGBA8Sint = (pf == MTL::PixelFormatRGBA8Sint);
    bool sourceIsRGB9E5 = (pf == MTL::PixelFormatRGB9E5Float);

    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* srcRow = srcBytes + (y * src_bytes_per_row);
        uint64_t dstY = flip_y ? (height - 1u - y) : y;
        uint8_t* dstRow = dstBytes + (dstY * dst_bytes_per_row);

        if (!sourceIsRGBA && !sourceIsRGBA32Float && !sourceIsR8 && !sourceIsRG8 &&
            !sourceIsR16Unorm && !sourceIsRG16Unorm && !sourceIsRGBA16Unorm &&
            !sourceIsR16Snorm && !sourceIsRG16Snorm && !sourceIsRGBA16Snorm &&
            !sourceIsBGR5A1 && !sourceIsABGR4 && !sourceIsRG11B10Float &&
            !sourceIsR32Float && !sourceIsRG32Float && !sourceIsRG16Float &&
            !sourceIsR16Float && !sourceIsRGBA16Float && !sourceIsBGR10A2 &&
            !sourceIsRGB10A2 &&
            !sourceIsR8Snorm && !sourceIsRG8Snorm && !sourceIsRGBA8Snorm &&
            !sourceIsR8Uint && !sourceIsR8Sint && !sourceIsRG8Uint && !sourceIsRG8Sint &&
            !sourceIsRGBA8Uint && !sourceIsRGBA8Sint && !sourceIsRGB9E5) {
            memcpy(dstRow, srcRow, width * 4u);
            continue;
        }

        for (uint64_t x = 0; x < width; x++) {
            uint8_t* d = dstRow + (x * 4u);
            if (sourceIsRGBA32Float) {
                const float* s = reinterpret_cast<const float*>(
                    srcRow + (x * sizeof(float) * 4u));
                d[0] = mglRenderFloatToUnorm8(s[2]);
                d[1] = mglRenderFloatToUnorm8(s[1]);
                d[2] = mglRenderFloatToUnorm8(s[0]);
                d[3] = mglRenderFloatToUnorm8(s[3]);
            } else if (sourceIsRGBA16Float) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[3]));
            } else if (sourceIsRG11B10Float) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed >> 22u, 5u));
                d[1] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed >> 11u, 6u));
                d[2] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed, 6u));
                d[3] = 255u;
            } else if (sourceIsRG32Float) {
                const float* s = reinterpret_cast<const float*>(
                    srcRow + (x * sizeof(float) * 2u));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(s[1]);
                d[2] = mglRenderFloatToUnorm8(s[0]);
                d[3] = 255u;
            } else if (sourceIsR32Float) {
                float component = 0.0f;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(component);
                d[3] = 255u;
            } else if (sourceIsRG16Float) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Float) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(mglHalfToFloat(component));
                d[3] = 255u;
            } else if (sourceIsRGBA16Unorm) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = (uint8_t)((components[2] * 255u + 32767u) / 65535u);
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = (uint8_t)((components[3] * 255u + 32767u) / 65535u);
            } else if (sourceIsRG16Unorm) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsR16Unorm) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = (uint8_t)((component * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsRGBA16Snorm) {
                int16_t components[4] = {0, 0, 0, 0};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[3]));
            } else if (sourceIsRG16Snorm) {
                int16_t components[2] = {0, 0};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Snorm) {
                int16_t component = 0;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(component));
                d[3] = 255u;
            } else if (sourceIsBGR10A2) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsRGB10A2) {
                /* MTLPixelFormatRGB10A2Unorm: R[0:9], G[10:19], B[20:29],
                 * A[30:31] (LSB-first).  BGRA8: d[0]=B, d[1]=G, d[2]=R,
                 * d[3]=A. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsR8) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8) {
                const uint8_t* s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsBGR5A1) {
                /* MTLPixelFormatBGR5A1Unorm: B[0:4], G[5:9], R[10:14], A[15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 31u) * 255u) / 31u);
                d[1] = (uint8_t)((((packed >> 5u) & 31u) * 255u) / 31u);
                d[2] = (uint8_t)((((packed >> 10u) & 31u) * 255u) / 31u);
                d[3] = ((packed >> 15u) & 1u) ? 255u : 0u;
            } else if (sourceIsABGR4) {
                /* MTLPixelFormatABGR4Unorm: A[0:3], B[4:7], G[8:11], R[12:15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 4u) & 15u) * 255u) / 15u);
                d[1] = (uint8_t)((((packed >> 8u) & 15u) * 255u) / 15u);
                d[2] = (uint8_t)((((packed >> 12u) & 15u) * 255u) / 15u);
                d[3] = (uint8_t)(((packed & 15u) * 255u) / 15u);
            } else if (sourceIsR8Snorm || sourceIsR8Sint) {
                int8_t s = (int8_t)srcRow[x];
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(mglRenderSnorm8ToFloat(s));
                d[3] = 255u;
            } else if (sourceIsRG8Snorm || sourceIsRG8Sint) {
                const int8_t* s = reinterpret_cast<const int8_t*>(
                    srcRow + x * 2u);
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[0]));
                d[3] = 255u;
            } else if (sourceIsRGBA8Snorm || sourceIsRGBA8Sint) {
                const int8_t* s = reinterpret_cast<const int8_t*>(
                    srcRow + x * 4u);
                d[0] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[3]));
            } else if (sourceIsR8Uint) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8Uint) {
                const uint8_t* s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsRGBA8Uint) {
                const uint8_t* s = srcRow + x * 4u;
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            } else if (sourceIsRGB9E5) {
                /* MTLPixelFormatRGB9E5Float: 4 bytes/pixel, shared exponent.
                 * Unpack to float R,G,B then convert to BGRA8 UNORM. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * 4u, sizeof(packed));
                uint32_t exp = (packed >> 27u) & 31u;
                uint32_t mant_r = packed & 511u;
                uint32_t mant_g = (packed >> 9u) & 511u;
                uint32_t mant_b = (packed >> 18u) & 511u;
                float scale = ldexpf(1.0f, (int)exp - 24);
                float rf = (float)mant_r * scale;
                float gf = (float)mant_g * scale;
                float bf = (float)mant_b * scale;
                d[0] = mglRenderFloatToUnorm8(bf);
                d[1] = mglRenderFloatToUnorm8(gf);
                d[2] = mglRenderFloatToUnorm8(rf);
                d[3] = 255u;
            } else {
                const uint8_t* s = srcRow + (x * 4u);
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
    }
}

extern "C"
int mglRenderReadbackGLTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_FLOAT:
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

static int mglReadbackFormatChannelMap(uint32_t format, int* slots,
                                          int src_idx[4]) {
    if (!slots || !src_idx) return 0;
    src_idx[0] = src_idx[1] = src_idx[2] = src_idx[3] = 0;
    switch (format) {
        case GL_RGBA: *slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; return 1;
        case GL_BGRA: *slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; return 1;
        case GL_RGB:  *slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; return 1;
        case GL_BGR:  *slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; return 1;
        case GL_RG:   *slots = 2; src_idx[0]=0; src_idx[1]=1; return 1;
        case GL_RED:  *slots = 1; src_idx[0]=0; return 1;
        case GL_GREEN: *slots = 1; src_idx[0]=1; return 1;
        case GL_BLUE:  *slots = 1; src_idx[0]=2; return 1;
        case GL_ALPHA: *slots = 1; src_idx[0]=3; return 1;
        default: return 0;
    }
}


static uint32_t mglSizeForType(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return sizeof(uint8_t);
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return sizeof(uint16_t);
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
            return sizeof(uint32_t);
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 8u;
        default:
            return sizeof(uint32_t);
    }
}


static uint32_t mglNumComponentsForFormat(uint32_t format) {
    switch (format) {
        case GL_RED:
        case GL_RED_INTEGER:
        case GL_GREEN:
        case GL_BLUE:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_ALPHA:
        case 0x803C: /* GL_ALPHA8 */
        case 0x803E: /* GL_ALPHA16 */
        case 0x8816: /* GL_ALPHA32F_ARB */
        case 0x881C: /* GL_ALPHA16F_ARB */
        case 0x1909: /* GL_LUMINANCE */
        case 0x8040: /* GL_LUMINANCE8 */
        case 0x8048: /* GL_LUMINANCE16 (pixel_utils local define) */
        case 0x8818: /* GL_LUMINANCE32F_ARB */
        case 0x881E: /* GL_LUMINANCE16F_ARB */
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_SR8_EXT:
        case 0x8D7E: /* GL_ALPHA8UI_EXT */
        case 0x9014: /* GL_ALPHA8_SNORM */
        case 0x9018: /* GL_ALPHA16_SNORM */
            return 1u;

        case GL_RG:
        case GL_RG_INTEGER:
        case 0x190A: /* GL_LUMINANCE_ALPHA */
        case 0x8819: /* GL_LUMINANCE_ALPHA32F_ARB */
        case 0x881F: /* GL_LUMINANCE_ALPHA16F_ARB */
        case 0x9016: /* GL_LUMINANCE8_ALPHA8_SNORM */
        case 0x901a: /* GL_LUMINANCE16_ALPHA16_SNORM */
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_SRG8_EXT:
            return 2u;

        case 0x8d7b: /* GL_ALPHA8I_EXT */
        case 0x8d81: /* GL_ALPHA32I_EXT */
        case 0x8d87: /* GL_ALPHA16I_EXT */
        case 0x8d8d: /* GL_ALPHA32UI_EXT */
        case 0x8d93: /* GL_ALPHA16UI_EXT */
        case 0x8d72: /* GL_ALPHA32UI_EXT */
            return 1u;

        case GL_RGB:
        case GL_BGR:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_SRGB8:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB565:
            return 3u;

        case 0x8d75: /* alternate GL_RGB8I */
        case 0x8d7a: /* alternate GL_RGB8UI */
        case 0x8d80: /* alternate GL_RGB32UI */
        case 0x8d86: /* alternate GL_RGB16I */
        case 0x8d8c: /* alternate GL_RGB32I */
        case 0x8d92: /* alternate GL_RGB16UI */
            return 3u;

        case GL_RGBA:
        case GL_BGRA:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGB5_A1:
        case GL_RGBA4:
            return 4u;

        case 0x8d78: /* alternate GL_RGBA8UI */
        case 0x8d84: /* alternate GL_RGBA16I */
        case 0x8d8a: /* alternate GL_RGBA32I */
        case 0x8d90: /* alternate GL_RGBA16UI */
            return 4u;

        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
            return 1u;

        default:
            fprintf(stderr,
                    "MGL WARNING: numComponentsForFormat unknown format 0x%x, "
                    "assuming 4 components\n",
                    format);
            return 4u;
    }
}

static int mglPixelTypeIsPacked(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderCopySnorm8TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatR8Snorm &&
        pf != MTL::PixelFormatRG8Snorm &&
        pf != MTL::PixelFormatRGBA8Snorm) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint64_t src_bpp = mglRenderReadbackBytesPerPixel(pixel_format);
    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    int src_channels = (int)src_bpp;
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const int8_t* s = reinterpret_cast<const int8_t*>(
                src_row + (x * src_bpp));
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);
            for (int c = 0; c < slots; ++c) {
                int idx = src_idx[c];
                if (idx >= src_channels) idx = src_channels - 1;
                int8_t sv = s[idx];
                float fv = mglRenderSnorm8ToFloat(sv);
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
                if (type == GL_BYTE) {
                    int32_t iv = (int32_t)lroundf(fv * 127.0f);
                    if (iv > 127) iv = 127;
                    if (iv < -128) iv = -128;
                    int8_t biv = (int8_t)iv;
                    memcpy(out, &biv, sizeof(biv));
                } else if (type == GL_UNSIGNED_BYTE) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_FLOAT) {
                    memcpy(out, &fv, sizeof(fv));
                } else if (type == GL_HALF_FLOAT) {
                    uint16_t iv = mglFloatToHalf(fv);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_SHORT) {
                    int32_t iv = (int32_t)lroundf(fv * 32767.0f);
                    if (iv > 32767) iv = 32767;
                    if (iv < -32768) iv = -32768;
                    int16_t siv = (int16_t)iv;
                    memcpy(out, &siv, sizeof(siv));
                } else if (type == GL_UNSIGNED_SHORT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_INT) {
                    int64_t iv = (int64_t)llroundf(fv * 2147483647.0f);
                    if (iv > 2147483647LL) iv = 2147483647LL;
                    if (iv < -2147483648LL) iv = -2147483648LL;
                    int32_t iiv = (int32_t)iv;
                    memcpy(out, &iiv, sizeof(iiv));
                } else if (type == GL_UNSIGNED_INT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                    memcpy(out, &iv, sizeof(iv));
                }
            }
        }
    }
    return 1;
}

static int mglReadbackRGB10A2TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderCopyRGB10A2TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatRGB10A2Unorm ||
        !mglReadbackRGB10A2TypeAccepted(type)) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    const uint64_t src_bpp = 4u;
    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            uint32_t packed = 0u;
            memcpy(&packed, src_row + (x * src_bpp), sizeof(packed));
            uint32_t rgb10a2_vals[4] = {
                packed & 1023u,
                (packed >> 10u) & 1023u,
                (packed >> 20u) & 1023u,
                (packed >> 30u) & 3u
            };

            if (type == GL_UNSIGNED_INT_10_10_10_2) {
                uint32_t r10 = rgb10a2_vals[src_idx[0]];
                uint32_t g10 = (slots > 1) ? rgb10a2_vals[src_idx[1]] : 0u;
                uint32_t b10 = (slots > 2) ? rgb10a2_vals[src_idx[2]] : 0u;
                uint32_t a2 = (slots > 3) ? rgb10a2_vals[src_idx[3]] : 0u;
                uint32_t out = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                uint32_t r10 = rgb10a2_vals[src_idx[0]];
                uint32_t g10 = (slots > 1) ? rgb10a2_vals[src_idx[1]] : 0u;
                uint32_t b10 = (slots > 2) ? rgb10a2_vals[src_idx[2]] : 0u;
                uint32_t a2 = (slots > 3) ? rgb10a2_vals[src_idx[3]] : 0u;
                uint32_t out = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                float rf = (float)rgb10a2_vals[src_idx[0]] / 1023.0f;
                float gf = (slots > 1)
                    ? (float)rgb10a2_vals[src_idx[1]] / 1023.0f : 0.0f;
                float bf = (slots > 2)
                    ? (float)rgb10a2_vals[src_idx[2]] / 1023.0f : 0.0f;
                uint32_t out = mglPackRGBToSharedExp(rf, gf, bf);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[src_idx[0]] *
                                       255u / 1023u);
                uint8_t g8 = (slots > 1)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[1]] * 255u / 1023u)
                    : 0u;
                uint8_t b8 = (slots > 2)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[2]] * 255u / 1023u)
                    : 0u;
                uint8_t a8 = (slots > 3)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[3]] * 255u / 3u)
                    : 0u;
                uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) |
                               ((uint32_t)b8 << 8u) | a8;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[src_idx[0]] *
                                       255u / 1023u);
                uint8_t g8 = (slots > 1)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[1]] * 255u / 1023u)
                    : 0u;
                uint8_t b8 = (slots > 2)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[2]] * 255u / 1023u)
                    : 0u;
                uint8_t a8 = (slots > 3)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[3]] * 255u / 3u)
                    : 0u;
                uint32_t out = r8 | ((uint32_t)g8 << 8u) |
                               ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else {
                for (int c = 0; c < slots; ++c) {
                    uint32_t raw = rgb10a2_vals[src_idx[c]];
                    float fv = (src_idx[c] == 3)
                        ? (float)raw / 3.0f : (float)raw / 1023.0f;
                    uint8_t* out = dst_row + (x * dst_pixel_bytes) +
                                   (uint64_t)c * (uint64_t)comp_bytes;
                    if (type == GL_UNSIGNED_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int8_t iv = (int8_t)lroundf(cv * 127.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_FLOAT) {
                        memcpy(out, &fv, sizeof(fv));
                    } else {
                        uint16_t iv = mglFloatToHalf(fv);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
    }
    return 1;
}

static int mglReadbackRG11B10TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderCopyRG11B10TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatRG11B10Float ||
        !mglReadbackRG11B10TypeAccepted(type)) {
        return 0;
    }

    const uint64_t src_bpp = 4u;
    uint32_t comp_bytes = mglSizeForType(type);
    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV && format == GL_RGB) {
        if (dst_bytes_per_row < width * src_bpp) {
            return 0;
        }
        const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
        uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
        for (uint64_t y = 0; y < height; y++) {
            const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
            uint64_t dst_y = flip_y ? (height - 1u - y) : y;
            uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
            memcpy(dst_row, src_row, width * src_bpp);
        }
        return 1;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            uint32_t packed = 0u;
            memcpy(&packed, src_row + (x * src_bpp), sizeof(packed));
            float float_vals[4] = {
                mglUnpackUnsignedFloatComponent(packed, 6u),
                mglUnpackUnsignedFloatComponent(packed >> 11u, 6u),
                mglUnpackUnsignedFloatComponent(packed >> 22u, 5u),
                1.0f
            };

            if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                float r = float_vals[src_idx[0]];
                float g = (slots > 1) ? float_vals[src_idx[1]] : 0.0f;
                float b = (slots > 2) ? float_vals[src_idx[2]] : 0.0f;
                uint32_t out = (mglFloatToFloat11(r) & 0x7ffu) |
                               ((mglFloatToFloat11(g) & 0x7ffu) << 11u) |
                               ((mglFloatToFloat10(b) & 0x3ffu) << 22u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                float r = float_vals[src_idx[0]];
                float g = (slots > 1) ? float_vals[src_idx[1]] : 0.0f;
                float b = (slots > 2) ? float_vals[src_idx[2]] : 0.0f;
                uint32_t out = mglPackRGBToSharedExp(r, g, b);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint8_t r8 = mglRenderFloatToUnorm8(float_vals[src_idx[0]]);
                uint8_t g8 = (slots > 1)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[1]]) : 0u;
                uint8_t b8 = (slots > 2)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[2]]) : 0u;
                uint8_t a8 = (slots > 3)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[3]]) : 0u;
                uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) |
                               ((uint32_t)b8 << 8u) | a8;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint8_t r8 = mglRenderFloatToUnorm8(float_vals[src_idx[0]]);
                uint8_t g8 = (slots > 1)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[1]]) : 0u;
                uint8_t b8 = (slots > 2)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[2]]) : 0u;
                uint8_t a8 = (slots > 3)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[3]]) : 0u;
                uint32_t out = r8 | ((uint32_t)g8 << 8u) |
                               ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else {
                for (int c = 0; c < slots; ++c) {
                    float fv = float_vals[src_idx[c]];
                    uint8_t* out = dst_row + (x * dst_pixel_bytes) +
                                   (uint64_t)c * (uint64_t)comp_bytes;
                    if (type == GL_UNSIGNED_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int8_t iv = (int8_t)lroundf(cv * 127.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_FLOAT) {
                        memcpy(out, &fv, sizeof(fv));
                    } else {
                        uint16_t iv = mglFloatToHalf(fv);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
    }
    return 1;
}

static int mglReadback16or32TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

static int mglWideSrcChannelCount(MTL::PixelFormat pf) {
    switch (pf) {
        case MTL::PixelFormatR32Float:
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatR16Float:
            return 1;
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRG16Float:
            return 2;
        case MTL::PixelFormatRGBA32Float:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
        case MTL::PixelFormatRGBA16Float:
            return 4;
        default:
            return 0;
    }
}


static float mglRead16or32SourceFloat(const uint8_t* s, int idx,
                                         int is16u, int is16s, int is16f) {
    if (is16u) {
        uint16_t uv = 0;
        memcpy(&uv, s + (uint64_t)idx * 2u, sizeof(uv));
        return (float)uv / 65535.0f;
    }
    if (is16s) {
        int16_t sv = 0;
        memcpy(&sv, s + (uint64_t)idx * 2u, sizeof(sv));
        return (float)sv / 32767.0f;
    }
    if (is16f) {
        uint16_t hv = 0;
        memcpy(&hv, s + (uint64_t)idx * 2u, sizeof(hv));
        return mglHalfToFloat(hv);
    }
    float fv = 0.0f;
    memcpy(&fv, s + (uint64_t)idx * 4u, sizeof(fv));
    return fv;
}


extern "C"
int mglRenderCopy16or32TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int is16u =
        (pf == MTL::PixelFormatR16Unorm ||
         pf == MTL::PixelFormatRG16Unorm ||
         pf == MTL::PixelFormatRGBA16Unorm);
    const int is16s =
        (pf == MTL::PixelFormatR16Snorm ||
         pf == MTL::PixelFormatRG16Snorm ||
         pf == MTL::PixelFormatRGBA16Snorm);
    const int is16f =
        (pf == MTL::PixelFormatR16Float ||
         pf == MTL::PixelFormatRG16Float ||
         pf == MTL::PixelFormatRGBA16Float);
    const int is32f =
        (pf == MTL::PixelFormatR32Float ||
         pf == MTL::PixelFormatRG32Float ||
         pf == MTL::PixelFormatRGBA32Float);
    if (!(is16u || is16s || is16f || is32f) ||
        !mglReadback16or32TypeAccepted(type)) {
        return 0;
    }

    const uint64_t src_bpp = mglRenderReadbackBytesPerPixel(pixel_format);
    int src_channels = mglWideSrcChannelCount(pf);
    if (src_bpp == 0u || src_channels == 0) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const int output_is_packed = mglPixelTypeIsPacked(type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * src_bpp);
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);

            if (output_is_packed) {
                float fvals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int c = 0; c < slots; ++c) {
                    int idx = src_idx[c];
                    if (idx >= src_channels) idx = src_channels - 1;
                    fvals[c] = mglRead16or32SourceFloat(
                        s, idx, is16u, is16s, is16f);
                }
                if (slots < 4) {
                    const int needs_alpha =
                        (type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                         type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                         type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                         type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                         type == GL_UNSIGNED_INT_8_8_8_8 ||
                         type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                         type == GL_UNSIGNED_INT_10_10_10_2 ||
                         type == GL_UNSIGNED_INT_2_10_10_10_REV);
                    if (needs_alpha) fvals[3] = 1.0f;
                }

                if (type == GL_UNSIGNED_BYTE_3_3_2) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    dp[0] = (uint8_t)(((uint32_t)lroundf(r * 7.0f) << 5) |
                                      ((uint32_t)lroundf(g * 7.0f) << 2) |
                                      (uint32_t)lroundf(b * 3.0f));
                } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    dp[0] = (uint8_t)((uint32_t)lroundf(r * 7.0f) |
                                      ((uint32_t)lroundf(g * 7.0f) << 3) |
                                      ((uint32_t)lroundf(b * 3.0f) << 6));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                 ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                 (uint32_t)lroundf(b * 31.0f));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                 ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 11));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 15.0f) << 12) |
                                                 ((uint32_t)lroundf(g * 15.0f) << 8) |
                                                 ((uint32_t)lroundf(b * 15.0f) << 4) |
                                                 (uint32_t)lroundf(a * 15.0f));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 15.0f) |
                                                 ((uint32_t)lroundf(g * 15.0f) << 4) |
                                                 ((uint32_t)lroundf(b * 15.0f) << 8) |
                                                 ((uint32_t)lroundf(a * 15.0f) << 12));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                 ((uint32_t)lroundf(g * 31.0f) << 6) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 1) |
                                                 (a >= 0.5f ? 1u : 0u));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                 ((uint32_t)lroundf(g * 31.0f) << 5) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 10) |
                                                 ((a >= 0.5f ? 1u : 0u) << 15));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = ((uint32_t)lroundf(r * 255.0f) << 24) |
                                      ((uint32_t)lroundf(g * 255.0f) << 16) |
                                      ((uint32_t)lroundf(b * 255.0f) << 8) |
                                      (uint32_t)lroundf(a * 255.0f);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = (uint32_t)lroundf(r * 255.0f) |
                                      ((uint32_t)lroundf(g * 255.0f) << 8) |
                                      ((uint32_t)lroundf(b * 255.0f) << 16) |
                                      ((uint32_t)lroundf(a * 255.0f) << 24);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = ((uint32_t)lroundf(r * 1023.0f) << 22) |
                                      ((uint32_t)lroundf(g * 1023.0f) << 12) |
                                      ((uint32_t)lroundf(b * 1023.0f) << 2) |
                                      (uint32_t)lroundf(a * 3.0f);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = (uint32_t)lroundf(r * 1023.0f) |
                                      ((uint32_t)lroundf(g * 1023.0f) << 10) |
                                      ((uint32_t)lroundf(b * 1023.0f) << 20) |
                                      ((uint32_t)lroundf(a * 3.0f) << 30);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                    float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                    float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                    uint32_t packed = mglFloatToFloat11(r) |
                                      (mglFloatToFloat11(g) << 11) |
                                      (mglFloatToFloat10(b) << 22);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                    float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                    float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                    uint32_t packed = mglPackRGBToSharedExp(r, g, b);
                    memcpy(dp, &packed, sizeof(packed));
                }
                continue;
            }

            for (int c = 0; c < slots; ++c) {
                int idx = src_idx[c];
                if (idx >= src_channels) idx = src_channels - 1;
                float fv = mglRead16or32SourceFloat(
                    s, idx, is16u, is16s, is16f);
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
                if (type == GL_UNSIGNED_BYTE) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_BYTE) {
                    float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                    int8_t iv = (int8_t)lroundf(cv * 127.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_UNSIGNED_SHORT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_SHORT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                    int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_UNSIGNED_INT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_INT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                    int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_FLOAT) {
                    memcpy(out, &fv, sizeof(fv));
                } else {
                    uint16_t iv = mglFloatToHalf(fv);
                    memcpy(out, &iv, sizeof(iv));
                }
            }
        }
    }
    return 1;
}

static int mglReadbackUnorm8ScalarTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_FLOAT:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderCopyUnorm8ScalarTextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int source_is_rgba =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    const int source_is_bgra =
        (pf == MTL::PixelFormatBGRA8Unorm ||
         pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    if ((!source_is_rgba && !source_is_bgra) ||
        !mglReadbackUnorm8ScalarTypeAccepted(type)) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * 4u);
            const unsigned cv[4] = {
                source_is_rgba ? s[0] : s[2],
                s[1],
                source_is_rgba ? s[2] : s[0],
                s[3]
            };
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);
            for (int c = 0; c < slots; ++c) {
                unsigned v = cv[src_idx[c]];
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
                if (type == GL_BYTE) {
                    float fv = (float)v / 255.0f;
                    int32_t iv = (int32_t)lroundf(fv * 127.0f);
                    if (iv > 127) iv = 127;
                    if (iv < -128) iv = -128;
                    int8_t biv = (int8_t)iv;
                    memcpy(out, &biv, sizeof(biv));
                } else if (type == GL_UNSIGNED_SHORT) {
                    uint16_t iv = (uint16_t)((uint32_t)v * 257u);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_SHORT) {
                    int32_t scaled = (int32_t)((uint32_t)v * 32767u / 255u);
                    if (scaled > 32767) scaled = 32767;
                    int16_t iv = (int16_t)scaled;
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_UNSIGNED_INT) {
                    uint32_t iv = (uint32_t)v * 16843009u;
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_INT) {
                    int32_t scaled =
                        (int32_t)((uint64_t)v * 2147483647ULL / 255u);
                    if (scaled > 2147483647) scaled = 2147483647;
                    memcpy(out, &scaled, sizeof(scaled));
                } else if (type == GL_FLOAT) {
                    float fv = (float)v / 255.0f;
                    memcpy(out, &fv, sizeof(fv));
                } else {
                    uint16_t iv = mglFloatToHalf((float)v / 255.0f);
                    memcpy(out, &iv, sizeof(iv));
                }
            }
        }
    }
    return 1;
}


static uint32_t mglPackUnsignedFloatFromUNorm8(uint32_t value,
                                                 uint32_t mantissa_bits)
{
    if (value == 0u || mantissa_bits == 0u || mantissa_bits > 23u) {
        return 0u;
    }

    float scaled = (float)value / 255.0f;
    int exponent = 15;
    while (scaled < 1.0f && exponent > 0) {
        scaled *= 2.0f;
        exponent--;
    }
    while (scaled >= 2.0f && exponent < 31) {
        scaled *= 0.5f;
        exponent++;
    }

    uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
    uint32_t mantissa = 0u;
    if (exponent == 0) {
        float subnormal = (float)value / 255.0f;
        for (uint32_t i = 0; i < mantissa_bits + 14u; i++) {
            subnormal *= 2.0f;
        }
        mantissa = (uint32_t)(subnormal + 0.5f);
        if (mantissa > mantissa_mask) {
            mantissa = mantissa_mask;
        }
    } else {
        float frac = (scaled - 1.0f) * (float)(1u << mantissa_bits);
        mantissa = (uint32_t)(frac + 0.5f);
        if (mantissa > mantissa_mask) {
            mantissa = 0u;
            if (exponent < 31) {
                exponent++;
            } else {
                mantissa = mantissa_mask;
            }
        }
    }

    return ((uint32_t)exponent << mantissa_bits) | (mantissa & mantissa_mask);
}

static int mglReadbackUnorm8PackedTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}


extern "C"
int mglRenderCopyUnorm8PackedTextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int source_is_rgba =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    const int source_is_bgra =
        (pf == MTL::PixelFormatBGRA8Unorm ||
         pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    if ((!source_is_rgba && !source_is_bgra) ||
        !mglReadbackUnorm8PackedTypeAccepted(type)) {
        return 0;
    }

    uint64_t dst_pixel_bytes = (uint64_t)mglSizeForType(type);
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * 4u);
            uint32_t r = source_is_rgba ? s[0] : s[2];
            uint32_t g = s[1];
            uint32_t b = source_is_rgba ? s[2] : s[0];
            uint32_t a = s[3];
            uint32_t rr = r, gg = g, bb = b, aa = a;
            if (format == GL_BGRA || format == GL_BGR) {
                uint32_t tmp = rr;
                rr = bb;
                bb = tmp;
            }
            uint8_t* d = dst_row + (x * dst_pixel_bytes);
            if (type == GL_UNSIGNED_BYTE_3_3_2) {
                d[0] = (uint8_t)(((rr >> 5u) << 5u) | ((gg >> 5u) << 2u) | (bb >> 6u));
            } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                d[0] = (uint8_t)((rr >> 5u) | ((gg >> 5u) << 3u) | ((bb >> 6u) << 6u));
            } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 2u) << 5u) | (bb >> 3u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 2u) << 5u) | ((bb >> 3u) << 11u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                uint16_t packed = (uint16_t)(((rr >> 4u) << 12u) | ((gg >> 4u) << 8u) |
                                             ((bb >> 4u) << 4u) | (aa >> 4u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                uint16_t packed = (uint16_t)((rr >> 4u) | ((gg >> 4u) << 4u) |
                                             ((bb >> 4u) << 8u) | ((aa >> 4u) << 12u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 3u) << 6u) |
                                             ((bb >> 3u) << 1u) | (aa >= 128u ? 1u : 0u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 3u) << 5u) |
                                             ((bb >> 3u) << 10u) |
                                             ((aa >= 128u ? 1u : 0u) << 15u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint32_t packed = ((uint32_t)rr << 24u) | ((uint32_t)gg << 16u) |
                                  ((uint32_t)bb << 8u) | aa;
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint32_t packed = rr | ((uint32_t)gg << 8u) |
                                  ((uint32_t)bb << 16u) | ((uint32_t)aa << 24u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                uint32_t r10 = rr * 1023u / 255u;
                uint32_t g10 = gg * 1023u / 255u;
                uint32_t b10 = bb * 1023u / 255u;
                uint32_t a2 = aa * 3u / 255u;
                uint32_t packed = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                uint32_t r10 = rr * 1023u / 255u;
                uint32_t g10 = gg * 1023u / 255u;
                uint32_t b10 = bb * 1023u / 255u;
                uint32_t a2 = aa * 3u / 255u;
                uint32_t packed = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                uint32_t packed = mglPackUnsignedFloatFromUNorm8(rr, 6u) |
                                  (mglPackUnsignedFloatFromUNorm8(gg, 6u) << 11u) |
                                  (mglPackUnsignedFloatFromUNorm8(bb, 5u) << 22u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                uint32_t packed = mglPackRGBToSharedExp(
                    (double)rr / 255.0, (double)gg / 255.0, (double)bb / 255.0);
                memcpy(d, &packed, sizeof(packed));
            }
        }
    }
    return 1;
}


extern "C"
int mglRenderCopyUnorm8SwizzleTextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int source_is_rgba =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    const int source_is_bgra =
        (pf == MTL::PixelFormatBGRA8Unorm ||
         pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    if (!source_is_rgba && !source_is_bgra) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }
    (void)src_idx;

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    if (format == GL_BGRA) {
        if (dst_pixel_bytes != 4u) return 0;
    } else if (format == GL_RGBA) {
        if (dst_pixel_bytes != 4u &&
            !(type == GL_FLOAT && dst_pixel_bytes == 16u)) {
            return 0;
        }
    } else if (format == GL_BGR || format == GL_RGB) {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 3u) return 0;
    } else if (format == GL_RG) {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 2u) return 0;
    } else {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 1u) return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * 4u);
            uint8_t r = source_is_rgba ? s[0] : s[2];
            uint8_t g = s[1];
            uint8_t b = source_is_rgba ? s[2] : s[0];
            uint8_t a = s[3];
            uint8_t* d = dst_row + (x * dst_pixel_bytes);

            switch (format) {
                case GL_BGRA:
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    d[3] = a;
                    break;
                case GL_RGBA:
                    if (type == GL_FLOAT) {
                        float* fd = reinterpret_cast<float*>(d);
                        fd[0] = (float)r / 255.0f;
                        fd[1] = (float)g / 255.0f;
                        fd[2] = (float)b / 255.0f;
                        fd[3] = (float)a / 255.0f;
                    } else {
                        d[0] = r;
                        d[1] = g;
                        d[2] = b;
                        d[3] = a;
                    }
                    break;
                case GL_BGR:
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    break;
                case GL_RGB:
                    d[0] = r;
                    d[1] = g;
                    d[2] = b;
                    break;
                case GL_RG:
                    d[0] = r;
                    d[1] = g;
                    break;
                case GL_RED:
                    d[0] = r;
                    break;
                case GL_GREEN:
                    d[0] = g;
                    break;
                case GL_BLUE:
                    d[0] = b;
                    break;
                case GL_ALPHA:
                    d[0] = a;
                    break;
                default:
                    return 0;
            }
        }
    }
    return 1;
}


/* little-endian packed read + unorm bit expansion (RGBA8 path). */
static uint32_t mglReadPackedUploadLE(const uint8_t* src, size_t bytes) {
    uint32_t value = 0u;
    if (!src) return 0u;
    if (bytes > sizeof(value)) bytes = sizeof(value);
    for (size_t i = 0; i < bytes; i++) {
        value |= ((uint32_t)src[i]) << (i * 8u);
    }
    return value;
}

static uint8_t mglExpandUNormBitsTo8(uint32_t value, uint32_t bits) {
    if (bits == 0u) return 0u;
    if (bits >= 8u) return (uint8_t)(value >> (bits - 8u));
    uint32_t maxv = (1u << bits) - 1u;
    return (uint8_t)((value * 255u + (maxv / 2u)) / maxv);
}

/* legacy packed GL formats -> RGBA8 (pure data transform). */
/* stage-binding copy-back encode + CPU-prefix sync.
 * Pure validation/encode over the caller-bridged entries; the CB
 * sequencing (detach/commit/wait/AGX recovery) stays in the renderer. */
extern "C"
int mglRenderEncodeStageBindingCopyBacks(
    const MGLRenderCopyBackEntry* entries, uint32_t count,
    void* blit_encoder) {
    if (!entries && count) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const MGLRenderCopyBackEntry& entry = entries[i];
        if (entry.length == 0) continue;
        MTL::Buffer* temporary =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.temporary));
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.destination));
        if (!temporary || !destination ||
            entry.length > temporary->length() ||
            entry.destination_offset > destination->length() ||
            entry.length >
                destination->length() - entry.destination_offset) {
            return -1;
        }
        if (blit_encoder &&
            mglRenderBlitCopyBuffer(
                blit_encoder, const_cast<void*>(entry.temporary), 0,
                const_cast<void*>(entry.destination),
                entry.destination_offset, entry.length) != 0) {
            return -1;
        }
    }
    return 0;
}

extern "C"
int mglRenderCopyBackCPUPrefix(
    const MGLRenderCopyBackEntry* entries, uint32_t count,
    uint32_t* failed_index_out) {
    if (failed_index_out) *failed_index_out = count;
    if (!entries && count) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const MGLRenderCopyBackEntry& entry = entries[i];
        if (entry.length == 0 || !entry.destination_buffer) continue;
        Buffer* buffer =
            static_cast<Buffer*>(const_cast<void*>(entry.destination_buffer));
        if (!buffer->data.buffer_data) continue;
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.destination));
        if (!destination || !destination->contents() ||
            entry.destination_offset > buffer->data.buffer_size ||
            entry.length >
                buffer->data.buffer_size - entry.destination_offset) {
            if (failed_index_out) *failed_index_out = i;
            return -1;
        }
        buffer->ever_written = GL_TRUE;
        uint8_t* cpu_bytes =
            (uint8_t*)(uintptr_t)buffer->data.buffer_data;
        const uint8_t* metal_bytes =
            (const uint8_t*)destination->contents();
        if (cpu_bytes != metal_bytes) {
            memmove(cpu_bytes + entry.destination_offset,
                    metal_bytes + entry.destination_offset,
                    entry.length);
        }
        buffer->cpu_shadow_pending = GL_FALSE;
    }
    return 0;
}

extern "C"
int mglRenderBuildRuntimeArraySizes(
    const MGLRenderBufferSizeEntry* entries, uint32_t entry_count,
    uint32_t runtime_buffer_index, uint32_t max_slot,
    uint32_t* out_sizes, uint32_t out_capacity) {
    if (!out_sizes || out_capacity < max_slot ||
        (!entries && entry_count != 0)) {
        return -1;
    }
    for (uint32_t i = 0; i < entry_count; i++) {
        const MGLRenderBufferSizeEntry& entry = entries[i];

        if (entry.metal_slot >= max_slot ||
            entry.metal_slot == runtime_buffer_index) {
            continue;
        }
        if (entry.metal_slot >= out_capacity) {
            continue;
        }
        out_sizes[entry.metal_slot] = (uint32_t)entry.visible_size;
    }
    return 0;
}


extern "C" {
GLuint sizeForInternalFormat(GLenum internalformat, GLenum format,
                             GLenum type);
}

extern "C"
int mglRenderTextureInternalFormatNeedsRGBA8Expansion(
    uint32_t internal_format, uint32_t pixel_format) {
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const bool is_rgba8_variant =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB ||
         pf == MTL::PixelFormatRGBA8Snorm ||
         pf == MTL::PixelFormatRGBA8Sint ||
         pf == MTL::PixelFormatRGBA8Uint);
    if (!is_rgba8_variant) {
        return 0;
    }
    switch (internal_format) {
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_R3_G3_B2:
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB565:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureNeedsChannelExpansion(uint32_t internal_format,
                                             uint32_t pixel_format) {
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const bool is_rgba16_variant =
        (pf == MTL::PixelFormatRGBA16Unorm ||
         pf == MTL::PixelFormatRGBA16Snorm ||
         pf == MTL::PixelFormatRGBA16Float ||
         pf == MTL::PixelFormatRGBA16Sint ||
         pf == MTL::PixelFormatRGBA16Uint);
    const bool is_rgba32_variant =
        (pf == MTL::PixelFormatRGBA32Float ||
         pf == MTL::PixelFormatRGBA32Sint ||
         pf == MTL::PixelFormatRGBA32Uint);
    if (!is_rgba16_variant && !is_rgba32_variant) {
        return 0;
    }
    switch (internal_format) {
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB12:
            return 1;
        default:
            return 0;
    }
}


extern "C"
uint8_t* mglRenderCreateChannelExpandedUpload(
    uint32_t internal_format, uint32_t pixel_format, const void* src_data,
    size_t width, size_t height, size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 || src_bytes_per_row == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return nullptr;
    }

    /* Source and destination parameters (bytes per component / pixel). */
    size_t src_comp_bytes = 0;
    size_t dst_comp_bytes = 0;
    size_t src_pixel_bytes = 0;
    size_t dst_pixel_bytes = 0;
    uint64_t alpha_default = 0;

    switch ((MTL::PixelFormat)pixel_format) {
        case MTL::PixelFormatRGBA16Unorm:
            src_comp_bytes = 2; dst_comp_bytes = 2;
            src_pixel_bytes = 6; dst_pixel_bytes = 8;
            alpha_default = 65535; /* 1.0 in unorm16 */
            break;
        case MTL::PixelFormatRGBA16Snorm:
            src_comp_bytes = 2; dst_comp_bytes = 2;
            src_pixel_bytes = 6; dst_pixel_bytes = 8;
            alpha_default = 32767; /* 1.0 in snorm16 */
            break;
        case MTL::PixelFormatRGBA16Float:
            src_comp_bytes = 2; dst_comp_bytes = 2;
            src_pixel_bytes = 6; dst_pixel_bytes = 8;
            alpha_default = 0x3C00; /* 1.0 in half float */
            break;
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA16Uint:
            src_comp_bytes = 2; dst_comp_bytes = 2;
            src_pixel_bytes = 6; dst_pixel_bytes = 8;
            alpha_default = 1;
            break;
        case MTL::PixelFormatRGBA32Float:
            src_comp_bytes = 4; dst_comp_bytes = 4;
            src_pixel_bytes = 12; dst_pixel_bytes = 16;
            { float f = 1.0f; memcpy(&alpha_default, &f, sizeof(f)); }
            break;
        case MTL::PixelFormatRGBA32Sint:
        case MTL::PixelFormatRGBA32Uint:
            src_comp_bytes = 4; dst_comp_bytes = 4;
            src_pixel_bytes = 12; dst_pixel_bytes = 16;
            alpha_default = 1;
            break;
        default:
            return nullptr;
    }

    /* Verify source pixel bytes match the internal format. */
    size_t expected_src_bytes =
        sizeForInternalFormat((GLenum)internal_format, 0, 0);
    if (expected_src_bytes > 0 && expected_src_bytes != src_pixel_bytes) {
        /* GL_RGB12: sizeForInternalFormat may differ; stored as 3x16-bit. */
        if (internal_format != GL_RGB12 || expected_src_bytes != 6) {
            return nullptr;
        }
    }

    if (src_bytes_per_row < width * src_pixel_bytes) {
        return nullptr;
    }

    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512 * 1024 * 1024)) {
        return nullptr;
    }

    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return nullptr;
    }

    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row =
            (const uint8_t*)src_data + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* src_pixel = src_row + x * src_pixel_bytes;
            uint8_t* dst_pixel = dst_row + x * dst_pixel_bytes;
            memcpy(dst_pixel, src_pixel, src_pixel_bytes);
            memcpy(dst_pixel + src_pixel_bytes, &alpha_default,
                   dst_comp_bytes);
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}


extern "C"
int mglRenderConvertIntegerReadback(
    const MGLRenderIntegerReadbackConvertParams* p) {
    if (!p || !p->src || !p->dst || !p->component_map ||
        !p->packed_bit_widths || !p->packed_shifts) {
        return -1;
    }
    const uint32_t src_pixel_bytes =
        p->source_rgb10a2_uint ? 4u :
        p->source_component_count * p->source_component_bytes;
    for (uint32_t y = 0; y < p->copy_h; y++) {
        const uint8_t* srcRow = p->src + (uint64_t)y * p->src_bytes_per_row;
        uint64_t outputY = p->dst_y + y;
        uint8_t* dstRow = (uint8_t *)p->dst + outputY * p->dst_bytes_per_row;
        for (uint32_t x = 0; x < p->copy_w; x++) {
            const uint8_t* s = srcRow + x * src_pixel_bytes;
            uint8_t* d = dstRow + (p->dst_x + x) * p->dst_pixel_bytes;

            /* Extract source component values (up to 4). */
            uint32_t srcValues[4] = {0, 0, 0, 0};
            for (uint32_t sc = 0; sc < p->source_component_count && sc < 4u; sc++) {
                if (p->source_rgb10a2_uint) {
                    uint32_t packed = *(const uint32_t *)(const void *)s;
                    static const uint8_t rgb10a2_shifts[4] = {0u, 10u, 20u, 30u};
                    static const uint32_t rgb10a2_masks[4] = {0x3ffu, 0x3ffu, 0x3ffu, 0x3u};
                    srcValues[sc] = (packed >> rgb10a2_shifts[sc]) & rgb10a2_masks[sc];
                } else if (p->source_component_bytes == 1u) {
                    srcValues[sc] = p->source_signed
                        ? (uint32_t)(int32_t)*(const int8_t *)(const void *)(s + sc)
                        : (uint32_t)s[sc];
                } else if (p->source_component_bytes == 2u) {
                    srcValues[sc] = p->source_signed
                        ? (uint32_t)(int32_t)*(const int16_t *)(const void *)(s + sc * 2u)
                        : (uint32_t)*(const uint16_t *)(const void *)(s + sc * 2u);
                } else {
                    srcValues[sc] = *(const uint32_t *)(const void *)(s + sc * 4u);
                }
            }

            if (p->is_packed_type) {
                /* Pack values into the packed format.
                 * Per OpenGL spec, integer values are CLAMPED to the bit width, not masked. */
                uint32_t packed = 0u;
                for (uint32_t c = 0; c < p->output_components && c < 4u; c++) {
                    int srcIdx = (c < 4u) ? p->component_map[c] : -1;
                    uint32_t val = 0u;
                    if (srcIdx >= 0 && (uint32_t)srcIdx < p->source_component_count) {
                        val = srcValues[srcIdx];
                    }
                    /* Clamp to bit width (not mask). */
                    uint32_t maxVal = (p->packed_bit_widths[c] >= 32u) ? 0xFFFFFFFFu : ((1u << p->packed_bit_widths[c]) - 1u);
                    if (val > maxVal) val = maxVal;
                    packed |= val << p->packed_shifts[c];
                }
                if (p->packed_output_bytes == 1u) {
                    d[0] = (uint8_t)packed;
                } else if (p->packed_output_bytes == 2u) {
                    ((uint16_t *)(void *)d)[0] = (uint16_t)packed;
                } else {
                    ((uint32_t *)(void *)d)[0] = packed;
                }
            } else {
                /* Non-packed: write each component individually.
                 * Per OpenGL spec, integer values are CLAMPED to the output type range. */
                for (uint32_t c = 0; c < p->output_components; c++) {
                    int srcIdx = (c < 4u) ? p->component_map[c] : -1;
                    uint32_t value = 0u;
                    if (srcIdx >= 0 && (uint32_t)srcIdx < p->source_component_count) {
                        value = srcValues[srcIdx];
                    }
                    if (p->output_component_bytes == 1u) {
                        if (p->packed_type == GL_BYTE) {
                            /* Signed byte: clamp to [-128, 127].
                             * If source is unsigned, values > 127 must clamp
                             * to 127 (not wrap to negative via int32_t cast). */
                            if (p->source_signed) {
                                int32_t sv = (int32_t)value;
                                if (sv > 127) sv = 127;
                                if (sv < -128) sv = -128;
                                d[c] = (uint8_t)(int8_t)sv;
                            } else {
                                if (value > 127u) value = 127u;
                                d[c] = (uint8_t)value;
                            }
                        } else {
                            /* Unsigned byte: clamp to [0, 255] */
                            if (value > 255u) value = 255u;
                            d[c] = (uint8_t)value;
                        }
                    } else if (p->output_component_bytes == 2u) {
                        if (p->packed_type == GL_SHORT) {
                            /* Signed short: clamp to [-32768, 32767].
                             * See comment above re: unsigned source. */
                            if (p->source_signed) {
                                int32_t sv = (int32_t)value;
                                if (sv > 32767) sv = 32767;
                                if (sv < -32768) sv = -32768;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)(int16_t)sv;
                            } else {
                                if (value > 32767u) value = 32767u;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                            }
                        } else {
                            /* Unsigned short: clamp to [0, 65535] */
                            if (value > 65535u) value = 65535u;
                            ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                        }
                    } else {
                        if (p->packed_type == GL_INT) {
                            /* Signed int: if source is unsigned, clamp to
                             * [0, INT32_MAX] to avoid wrap. */
                            if (p->source_signed) {
                                ((uint32_t *)(void *)d)[c] = value;
                            } else {
                                if (value > 0x7FFFFFFFu) value = 0x7FFFFFFFu;
                                ((uint32_t *)(void *)d)[c] = value;
                            }
                        } else {
                            /* Unsigned int: clamp to [0, 4294967295] */
                            ((uint32_t *)(void *)d)[c] = value;
                        }
                    }
                }
            }
        }
    }


    return 0;
}

/* GL 4.6 section 11.2.2.2 patch discard predicate.
 * This is evaluated before any tessellation level is clamped to one. */
extern "C"
bool mglRenderTessFactorsDiscardPatch(uint32_t gen_mode,
                                         const float* edge,
                                         const float* inside) {
    if (!edge || !inside) {
        return true;
    }
    switch (gen_mode) {
        case GL_ISOLINES:
            return edge[0] <= 0.0f || edge[1] <= 0.0f ||
                   isnan(edge[0]) || isnan(edge[1]);
        case GL_QUADS:
            return edge[0] <= 0.0f || edge[1] <= 0.0f ||
                   edge[2] <= 0.0f || edge[3] <= 0.0f ||
                   inside[0] <= 0.0f || inside[1] <= 0.0f ||
                   isnan(edge[0]) || isnan(edge[1]) ||
                   isnan(edge[2]) || isnan(edge[3]) ||
                   isnan(inside[0]) || isnan(inside[1]);
        default: /* GL_TRIANGLES */
            return edge[0] <= 0.0f || edge[1] <= 0.0f ||
                   edge[2] <= 0.0f || inside[0] <= 0.0f ||
                   isnan(edge[0]) || isnan(edge[1]) ||
                   isnan(edge[2]) || isnan(inside[0]);
    }
}

extern "C"
int mglRenderFillDefaultTessFactorBuffer(
    void* dst, uint64_t dst_bytes,
    const float* outer_levels, const float* inner_levels,
    uint32_t patch_count) {
    const uint64_t stride = 12u;
    if (!dst || !outer_levels || !inner_levels || patch_count == 0u ||
        dst_bytes < (uint64_t)patch_count * stride) {
        return -1;
    }
    __fp16* out = (__fp16*)dst;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        for (uint32_t i = 0u; i < 4u; i++) {
            out[patch * 6u + i] = (__fp16)outer_levels[i];
        }
        for (uint32_t i = 0u; i < 2u; i++) {
            out[patch * 6u + 4u + i] = (__fp16)inner_levels[i];
        }
    }
    return 0;
}

extern "C"
int mglRenderRepackTessFactorTriangles(
    const void* src, uint64_t src_bytes,
    void* dst, uint64_t dst_bytes,
    uint32_t patch_count) {
    const uint64_t canonical_stride = 12u;
    const uint64_t triangle_stride = 8u;
    if (!src || !dst || patch_count == 0u ||
        src_bytes < (uint64_t)patch_count * canonical_stride ||
        dst_bytes < (uint64_t)patch_count * triangle_stride) {
        return -1;
    }
    const uint16_t* in_all = (const uint16_t*)src;
    uint16_t* out_all = (uint16_t*)dst;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        const uint16_t* in = in_all + patch * 6u;
        uint16_t* out = out_all + patch * 4u;
        out[0] = in[0];
        out[1] = in[1];
        out[2] = in[2];
        out[3] = in[4];
    }
    return 0;
}

extern "C"
uint64_t mglRenderTessPrimitiveCount(
    const void* factors, uint64_t bytes,
    uint32_t patch_count, uint32_t tess_gen_mode,
    uint32_t instance_count) {
    if (!factors || patch_count == 0u ||
        bytes < (uint64_t)patch_count * 12u) {
        return 0u;
    }
    const uint16_t* recs = (const uint16_t*)factors;
    uint64_t total = 0u;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        const uint16_t* record = recs + patch * 6u;
        float edge[4], inside[2];
        for (int i = 0; i < 4; i++) {
            edge[i] = *(const __fp16*)&record[i];
        }
        for (int i = 0; i < 2; i++) {
            inside[i] = *(const __fp16*)&record[4 + i];
        }
        if (mglRenderTessFactorsDiscardPatch(tess_gen_mode, edge, inside)) {
            continue;
        }
        float inside0 = fmaxf(inside[0], 1.0f);
        float inside1 = fmaxf(inside[1], 1.0f);
        uint64_t per_patch = tess_gen_mode == GL_QUADS
            ? 2ull * (uint64_t)ceilf(inside0) * (uint64_t)ceilf(inside1)
            : (uint64_t)ceilf(inside0) * (uint64_t)ceilf(inside0);
        total += per_patch > 1ull ? per_patch : 1ull;
    }
    return total * (uint64_t)instance_count;
}

extern "C"
int mglRenderNativeTESInterfaceSupported(
    void* tes_function, uint64_t tes_metallib_bytes,
    uint32_t tes_gen_point_mode, uint32_t tes_xfb_varying_count,
    uint32_t tes_gen_mode,
    void* tcs_function, uint64_t tcs_metallib_bytes,
    uint32_t tcs_output_vertices) {
    if (!tes_function || tes_metallib_bytes == 0u ||
        tes_gen_point_mode != 0u || tes_xfb_varying_count > 0u) {
        return 0;
    }
    if (tcs_function && (tcs_metallib_bytes == 0u ||
                         tcs_output_vertices == 0u ||
                         tcs_output_vertices > 32u)) {
        return 0;
    }
    if (tes_gen_mode != GL_TRIANGLES && tes_gen_mode != GL_QUADS) {
        return 0;
    }
    MTL::Function* fn = static_cast<MTL::Function*>(tes_function);
    MTL::PatchType expected = tes_gen_mode == GL_QUADS
        ? MTL::PatchTypeQuad : MTL::PatchTypeTriangle;
    if (fn->patchType() != expected) {
        return 0;
    }
    /* The metallib TESS tag now carries 4*controlPointCount + patchKind;
     * a non-zero patchControlPointCount is the real per-patch control
     * point count and must agree with the TCS output vertices.  Zero
     * (legacy encoding) is also tolerated. */
    if (fn->patchControlPointCount() > 0 && tcs_function &&
        tcs_output_vertices != (uint32_t)fn->patchControlPointCount()) {
        return 0;
    }
    return 1;
}

extern "C"
int mglRenderRasterizationIsEmpty(
    int32_t vx, int32_t vy, int32_t vw, int32_t vh,
    uint32_t pass_width, uint32_t pass_height,
    int32_t scissor_enabled,
    int32_t sx, int32_t sy, int32_t sw, int32_t sh) {
    if (vw <= 0 || vh <= 0) {
        return 1;
    }
    if (pass_width == 0 || pass_height == 0) {
        return 0;
    }
    const int64_t fbW = (int64_t)pass_width;
    const int64_t fbH = (int64_t)pass_height;
    const int64_t vx0 = (int64_t)vx;
    const int64_t vy0 = (int64_t)vy;
    const int64_t vx1 = vx0 + (int64_t)vw;
    const int64_t vy1 = vy0 + (int64_t)vh;
    if (vx1 <= 0 || vy1 <= 0 || vx0 >= fbW || vy0 >= fbH) {
        return 1;
    }
    if (scissor_enabled) {
        if (sw <= 0 || sh <= 0) {
            return 1;
        }
        const int64_t sx0 = (int64_t)sx;
        const int64_t sy0 = (int64_t)sy;
        const int64_t sx1 = sx0 + (int64_t)sw;
        const int64_t sy1 = sy0 + (int64_t)sh;
        if (sx1 <= 0 || sy1 <= 0 || sx0 >= fbW || sy0 >= fbH) {
            return 1;
        }
    }
    return 0;
}

extern "C"
int mglRenderIntegerReadbackSourceClassify(
    uint32_t pixel_format, MGLRenderIntegerReadbackSource* out) {
    if (!out) return -1;
    out->component_count = 0;
    out->component_bytes = 0;
    out->source_signed = 0;
    out->source_rgb10a2_uint = 0;
    out->recognized = 0;
    switch (pixel_format) {
        case MTL::PixelFormatR8Uint:
            out->component_count = 1u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatR8Sint:
            out->component_count = 1u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatR16Uint:
            out->component_count = 1u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatR16Sint:
            out->component_count = 1u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatR32Uint:
            out->component_count = 1u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatR32Sint:
            out->component_count = 1u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRG8Uint:
            out->component_count = 2u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRG8Sint:
            out->component_count = 2u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRG16Uint:
            out->component_count = 2u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRG16Sint:
            out->component_count = 2u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRG32Uint:
            out->component_count = 2u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRG32Sint:
            out->component_count = 2u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA8Uint:
            out->component_count = 4u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA8Sint:
            out->component_count = 4u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA16Uint:
            out->component_count = 4u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA16Sint:
            out->component_count = 4u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA32Uint:
            out->component_count = 4u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case MTL::PixelFormatRGBA32Sint:
            out->component_count = 4u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case MTL::PixelFormatRGB10A2Uint:
            out->component_count = 4u; out->component_bytes = 4u;
            out->source_rgb10a2_uint = 1; out->recognized = 1;
            break;
        default:
            break;
    }
    return 0;
}

extern "C"
int mglRenderIntegerReadbackPackedTypeClassify(
    uint32_t packed_type, MGLRenderIntegerPackedType* out) {
    if (!out) return -1;
    out->is_packed = 0;
    for (int i = 0; i < 4; i++) {
        out->bit_widths[i] = 0;
        out->shifts[i] = 0;
    }
    out->output_bytes = 0;
    out->output_components = 0;
    switch (packed_type) {
        case 0x8032: /* GL_UNSIGNED_BYTE_3_3_2 */
            out->is_packed = 1;
            out->bit_widths[0] = 3; out->bit_widths[1] = 3;
            out->bit_widths[2] = 2; out->bit_widths[3] = 0;
            out->shifts[0] = 5; out->shifts[1] = 2;
            out->shifts[2] = 0; out->shifts[3] = 0;
            out->output_bytes = 1; out->output_components = 3;
            break;
        case 0x8362: /* GL_UNSIGNED_BYTE_2_3_3_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 3; out->bit_widths[1] = 3;
            out->bit_widths[2] = 2; out->bit_widths[3] = 0;
            out->shifts[0] = 0; out->shifts[1] = 3;
            out->shifts[2] = 6; out->shifts[3] = 0;
            out->output_bytes = 1; out->output_components = 3;
            break;
        case 0x8363: /* GL_UNSIGNED_SHORT_5_6_5 */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 6;
            out->bit_widths[2] = 5; out->bit_widths[3] = 0;
            out->shifts[0] = 11; out->shifts[1] = 5;
            out->shifts[2] = 0; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 3;
            break;
        case 0x8364: /* GL_UNSIGNED_SHORT_5_6_5_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 6;
            out->bit_widths[2] = 5; out->bit_widths[3] = 0;
            out->shifts[0] = 0; out->shifts[1] = 5;
            out->shifts[2] = 11; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 3;
            break;
        case 0x8033: /* GL_UNSIGNED_SHORT_4_4_4_4 */
            out->is_packed = 1;
            out->bit_widths[0] = 4; out->bit_widths[1] = 4;
            out->bit_widths[2] = 4; out->bit_widths[3] = 4;
            out->shifts[0] = 12; out->shifts[1] = 8;
            out->shifts[2] = 4; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8365: /* GL_UNSIGNED_SHORT_4_4_4_4_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 4; out->bit_widths[1] = 4;
            out->bit_widths[2] = 4; out->bit_widths[3] = 4;
            out->shifts[0] = 0; out->shifts[1] = 4;
            out->shifts[2] = 8; out->shifts[3] = 12;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8034: /* GL_UNSIGNED_SHORT_5_5_5_1 */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 5;
            out->bit_widths[2] = 5; out->bit_widths[3] = 1;
            out->shifts[0] = 11; out->shifts[1] = 6;
            out->shifts[2] = 1; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8366: /* GL_UNSIGNED_SHORT_1_5_5_5_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 5;
            out->bit_widths[2] = 5; out->bit_widths[3] = 1;
            out->shifts[0] = 0; out->shifts[1] = 5;
            out->shifts[2] = 10; out->shifts[3] = 15;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8035: /* GL_UNSIGNED_INT_8_8_8_8 */
            out->is_packed = 1;
            out->bit_widths[0] = 8; out->bit_widths[1] = 8;
            out->bit_widths[2] = 8; out->bit_widths[3] = 8;
            out->shifts[0] = 24; out->shifts[1] = 16;
            out->shifts[2] = 8; out->shifts[3] = 0;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8367: /* GL_UNSIGNED_INT_8_8_8_8_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 8; out->bit_widths[1] = 8;
            out->bit_widths[2] = 8; out->bit_widths[3] = 8;
            out->shifts[0] = 0; out->shifts[1] = 8;
            out->shifts[2] = 16; out->shifts[3] = 24;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8036: /* GL_UNSIGNED_INT_10_10_10_2 */
            out->is_packed = 1;
            out->bit_widths[0] = 10; out->bit_widths[1] = 10;
            out->bit_widths[2] = 10; out->bit_widths[3] = 2;
            out->shifts[0] = 22; out->shifts[1] = 12;
            out->shifts[2] = 2; out->shifts[3] = 0;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8368: /* GL_UNSIGNED_INT_2_10_10_10_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 10; out->bit_widths[1] = 10;
            out->bit_widths[2] = 10; out->bit_widths[3] = 2;
            out->shifts[0] = 0; out->shifts[1] = 10;
            out->shifts[2] = 20; out->shifts[3] = 30;
            out->output_bytes = 4; out->output_components = 4;
            break;
        default:
            break;
    }
    return 0;
}

extern "C"
int mglRenderIntegerReadbackClassify(
    uint32_t pixel_format, uint32_t gl_format, uint32_t gl_type,
    MGLRenderIntegerReadbackClassify* out) {
    if (!out) return -1;
    out->source_is_integer_texture = 0;
    out->output_is_integer_format = 0;
    out->output_components = 0;
    out->component_map[0] = 0; out->component_map[1] = 1;
    out->component_map[2] = 2; out->component_map[3] = 3;
    out->output_component_bytes = 0;
    switch (pixel_format) {
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatR16Uint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatR32Uint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRG16Uint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRG32Uint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGBA16Uint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA32Uint:
        case MTL::PixelFormatRGBA32Sint:
        case MTL::PixelFormatRGB10A2Uint:
            out->source_is_integer_texture = 1;
            break;
        default:
            break;
    }
    switch (gl_format) {
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
        case 0x8d97: /* GL_ALPHA_INTEGER */
            out->output_is_integer_format = 1;
            break;
        default:
            break;
    }
    if (out->source_is_integer_texture && out->output_is_integer_format) {
        switch (gl_format) {
            case GL_RED_INTEGER:
                out->output_components = 1u;
                out->component_map[0] = 0; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case GL_RG_INTEGER:
                out->output_components = 2u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case GL_RGB_INTEGER:
                out->output_components = 3u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = 2; out->component_map[3] = -1;
                break;
            case GL_BGR_INTEGER:
                out->output_components = 3u;
                out->component_map[0] = 2; out->component_map[1] = 1;
                out->component_map[2] = 0; out->component_map[3] = -1;
                break;
            case GL_RGBA_INTEGER:
                out->output_components = 4u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = 2; out->component_map[3] = 3;
                break;
            case GL_BGRA_INTEGER:
                out->output_components = 4u;
                out->component_map[0] = 2; out->component_map[1] = 1;
                out->component_map[2] = 0; out->component_map[3] = 3;
                break;
            case 0x8d95:
                out->output_components = 1u;
                out->component_map[0] = 1; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case 0x8d96:
                out->output_components = 1u;
                out->component_map[0] = 2; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case 0x8d97:
                out->output_components = 1u;
                out->component_map[0] = 3; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            default:
                break;
        }
        out->output_component_bytes =
            (gl_type == GL_BYTE || gl_type == GL_UNSIGNED_BYTE) ? 1u :
            (gl_type == GL_SHORT || gl_type == GL_UNSIGNED_SHORT) ? 2u : 4u;
    }
    return 0;
}

extern "C"
int mglRenderGetTexImagePlan(
    uint32_t pixel_format, uint32_t gl_format, uint32_t gl_type,
    uint32_t width, uint32_t height, uint32_t depth,
    uint32_t dst_pixel_bytes, uint32_t source_bpp,
    int bgra8_format_compatible,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int storage_private,
    MGLRenderGetTexImagePlan* out) {
    if (!out) return -1;
    out->direct_r32_float_read =
        (pixel_format == MTL::PixelFormatR32Float &&
         gl_format == GL_RED && gl_type == GL_FLOAT) ? 1 : 0;
    out->use_bgra8_conversion =
        (dst_pixel_bytes > 0u && depth == 1u &&
         !out->direct_r32_float_read && bgra8_format_compatible) ? 1 : 0;
    out->source_is_bgra8 =
        (pixel_format == MTL::PixelFormatBGRA8Unorm ||
         pixel_format == MTL::PixelFormatBGRA8Unorm_sRGB ||
         pixel_format == MTL::PixelFormatRGBA8Unorm ||
         pixel_format == MTL::PixelFormatRGBA8Unorm_sRGB) ? 1 : 0;
    uint64_t row_bytes;
    if (out->use_bgra8_conversion && !out->source_is_bgra8 &&
        source_bpp > 0u) {
        row_bytes = (uint64_t)width * (uint64_t)source_bpp;
    } else if (out->use_bgra8_conversion) {
        row_bytes = (uint64_t)width * 4u;
    } else {
        row_bytes = bytes_per_row > 0
            ? (uint64_t)bytes_per_row
            : (uint64_t)width * (dst_pixel_bytes > 0
                                     ? (uint64_t)dst_pixel_bytes : 1u);
    }
    out->row_bytes = row_bytes;
    out->image_bytes = row_bytes * (uint64_t)height;
    out->total_bytes = out->image_bytes;
    if (!out->use_bgra8_conversion && storage_private &&
        bytes_per_image > 0 && depth > 1) {
        out->total_bytes = (uint64_t)bytes_per_image * (uint64_t)depth;
    }
    return 0;
}

static inline uint32_t MGLRenderReadIndexBytes(const uint8_t* bytes, int w, uint32_t i) {
    return (w == 1) ? (uint32_t)bytes[i]
        : (w == 2) ? (uint32_t)((const uint16_t*)bytes)[i]
        : (uint32_t)((const uint32_t*)bytes)[i];
}

extern "C"
uint32_t mglRenderGLIndexElementSize(uint64_t gl_index_type) {
    if (gl_index_type == GL_UNSIGNED_BYTE) return 1u;
    if (gl_index_type == GL_UNSIGNED_SHORT) return 2u;
    if (gl_index_type == GL_UNSIGNED_INT) return 4u;
    return 0u;
}

extern "C"
uint32_t mglRenderReadGLIndexValue(const uint8_t* bytes, uint32_t elem_width,
                                      uint64_t element_index) {
    if (!bytes || elem_width == 0u) {
        return 0u;
    }
    if (elem_width == 1u) {
        uint8_t v = 0u;
        memcpy(&v, bytes + element_index, sizeof(v));
        return (uint32_t)v;
    }
    if (elem_width == 2u) {
        uint16_t v = 0u;
        memcpy(&v, bytes + (element_index * 2u), sizeof(v));
        return (uint32_t)v;
    }
    if (elem_width == 4u) {
        uint32_t v = 0u;
        memcpy(&v, bytes + (element_index * 4u), sizeof(v));
        return v;
    }
    return 0u;
}

extern "C"
uint32_t mglRenderVertexAttribComponentSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1u;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2u;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return 4u;
        case GL_DOUBLE:
            return 8u;
        default:
            return 0u;
    }
}

extern "C"
uint64_t mglRenderVertexAttribElementBytes(uint64_t gl_type, uint32_t size) {
    switch (gl_type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
            return 4u;
        default: {
            const uint32_t comp = mglRenderVertexAttribComponentSize(gl_type);
            if (comp == 0u || size == 0u) {
                return 0u;
            }
            return (uint64_t)comp * (uint64_t)size;
        }
    }
}

extern "C"
int mglRenderDrawModeProducesPolygons(uint64_t gl_mode) {
    switch (gl_mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count) {
    switch (gl_mode) {
        case GL_POINTS:
            return index_count >= 1u ? 1 : 0;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return index_count >= 2u ? 1 : 0;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return index_count >= 3u ? 1 : 0;
        case GL_QUADS:
            return index_count >= 4u ? 1 : 0;
        default:
            return index_count > 0u ? 1 : 0;
    }
}

extern "C"
uint64_t mglRenderQuadTriangleIndexCount(uint64_t source_vertex_count) {
    const uint64_t quad_count = source_vertex_count / 4u;
    if (quad_count > (uint64_t)(SIZE_MAX / 6u)) {
        return 0u;
    }
    return quad_count * 6u;
}

extern "C"
uint64_t mglRenderAlignVertexStrideForMetal(uint64_t stride) {
    return (stride + 3u) & ~(uint64_t)3u;
}

extern "C"
uint32_t mglRenderDoubleVertexAttribFloatFormat(uint32_t size) {
    /* MTLVertexFormat Float/Float2/Float3/Float4 = 28/29/30/31. */
    switch (size) {
        case 1u: return 28u;
        case 2u: return 29u;
        case 3u: return 30u;
        case 4u: return 31u;
        default: return 0u; /* MTLVertexFormatInvalid */
    }
}

extern "C"
uint32_t mglRenderIntegerAttribConversionFormat(
    uint64_t src_type,
    uint64_t shader_gl_type,
    uint32_t size) {
    if (size < 1u || size > 4u) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    const bool shader_is_int =
        shader_gl_type == GL_INT || shader_gl_type == GL_INT_VEC2 ||
        shader_gl_type == GL_INT_VEC3 || shader_gl_type == GL_INT_VEC4;
    const bool shader_is_uint =
        shader_gl_type == GL_UNSIGNED_INT ||
        shader_gl_type == GL_UNSIGNED_INT_VEC2 ||
        shader_gl_type == GL_UNSIGNED_INT_VEC3 ||
        shader_gl_type == GL_UNSIGNED_INT_VEC4;
    if (!shader_is_int && !shader_is_uint) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    const bool src_is_unsigned =
        src_type == GL_UNSIGNED_BYTE || src_type == GL_UNSIGNED_SHORT ||
        src_type == GL_UNSIGNED_INT;
    const bool src_is_signed =
        src_type == GL_BYTE || src_type == GL_SHORT || src_type == GL_INT;
    if (!((shader_is_int && src_is_unsigned) ||
          (shader_is_uint && src_is_signed))) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    if (shader_is_int) {
        switch (size) {
            case 1u: return static_cast<uint32_t>(MTL::VertexFormatInt);
            case 2u: return static_cast<uint32_t>(MTL::VertexFormatInt2);
            case 3u: return static_cast<uint32_t>(MTL::VertexFormatInt3);
            case 4u: return static_cast<uint32_t>(MTL::VertexFormatInt4);
        }
    } else {
        switch (size) {
            case 1u: return static_cast<uint32_t>(MTL::VertexFormatUInt);
            case 2u: return static_cast<uint32_t>(MTL::VertexFormatUInt2);
            case 3u: return static_cast<uint32_t>(MTL::VertexFormatUInt3);
            case 4u: return static_cast<uint32_t>(MTL::VertexFormatUInt4);
        }
    }
    return static_cast<uint32_t>(MTL::VertexFormatInvalid);
}

extern "C"
uint64_t mglRenderHashStepU64(uint64_t hash, uint64_t value) {
    return (hash ^ value) * 1099511628211ull;
}

extern "C"
int mglRenderPrimitiveRestartFixedIndex(uint64_t gl_index_type, uint32_t* out) {
    if (!out) {
        return -1;
    }
    switch (gl_index_type) {
        case GL_UNSIGNED_BYTE: *out = 0xffu; return 1;
        case GL_UNSIGNED_SHORT: *out = 0xffffu; return 1;
        case GL_UNSIGNED_INT: *out = 0xffffffffu; return 1;
        default: return 0;
    }
}

extern "C"
uint32_t mglRenderGLTypeElementByteSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT: case GL_INT: case GL_UNSIGNED_INT: case GL_BOOL:
            return 4u;
        case GL_FLOAT_VEC2: case GL_INT_VEC2: case GL_UNSIGNED_INT_VEC2: case GL_BOOL_VEC2:
            return 8u;
        case GL_FLOAT_VEC3: case GL_INT_VEC3: case GL_UNSIGNED_INT_VEC3: case GL_BOOL_VEC3:
            return 12u;
        case GL_FLOAT_VEC4: case GL_INT_VEC4: case GL_UNSIGNED_INT_VEC4: case GL_BOOL_VEC4:
            return 16u;
        case GL_FLOAT_MAT2:
            return 8u;   /* one column = vec2 */
        case GL_FLOAT_MAT3:
            return 12u;  /* one column = vec3 */
        case GL_FLOAT_MAT4:
            return 16u;  /* one column = vec4 */
        case GL_FLOAT_MAT2x3: return 12u;
        case GL_FLOAT_MAT2x4: return 16u;
        case GL_FLOAT_MAT3x2: return 8u;
        case GL_FLOAT_MAT3x4: return 16u;
        case GL_FLOAT_MAT4x2: return 8u;
        case GL_FLOAT_MAT4x3: return 12u;
        case GL_DOUBLE: return 8u;
        default: return 4u;
    }
}

extern "C"
int mglRenderScanIndexRangeIgnoringRestart(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index,
    uint32_t* out_min, uint32_t* out_max, int* out_valid) {
    if (!bytes || count == 0u || !out_min || !out_max || !out_valid) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    uint32_t min_index = UINT32_MAX;
    uint32_t max_index = 0u;
    const int has_restart = restart_enabled ? 1 : 0;
    for (uint32_t i = 0u; i < count; i++) {
        const uint32_t v = MGLRenderReadIndexBytes(bytes, w, i);
        if (has_restart && v == restart_index) {
            continue;
        }
        if (v < min_index) min_index = v;
        if (v > max_index) max_index = v;
    }
    *out_min = min_index;
    *out_max = max_index;
    *out_valid = (min_index <= max_index) ? 1 : 0;
    return 0;
}

extern "C"
int mglRenderComputePreparedIndexByteOffset(
    uint64_t gl_index_type, uint64_t gl_byte_offset,
    uint64_t* out_prepared_offset) {
    if (!out_prepared_offset) {
        return -1;
    }
    if (gl_index_type == GL_UNSIGNED_BYTE) {
        // GL_UNSIGNED_BYTE indices expand to UInt16 (2 bytes each).
        if (gl_byte_offset > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
            return -1;
        }
        *out_prepared_offset = gl_byte_offset * sizeof(uint16_t);
        return 0;
    }
    *out_prepared_offset = gl_byte_offset;
    return 0;
}

extern "C"
int mglRenderComputeIndexByteOffset(
    uint64_t base_byte_offset, uint64_t first_element, uint64_t index_stride,
    uint64_t* out_byte_offset) {
    if (!out_byte_offset || index_stride == 0u) {
        return -1;
    }
    if (first_element > (uint64_t)SIZE_MAX / index_stride) {
        return -1;
    }
    const uint64_t relative = first_element * index_stride;
    if (base_byte_offset > (uint64_t)SIZE_MAX - relative) {
        return -1;
    }
    *out_byte_offset = base_byte_offset + relative;
    return 0;
}

extern "C"
int mglRenderExpandUInt8ToUInt16(
    const uint8_t* bytes, uint32_t byte_count, uint16_t** out, uint64_t* out_count) {
    if (!bytes || byte_count == 0u || !out || !out_count) {
        return -1;
    }
    if ((uint64_t)byte_count > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
        return -1;
    }
    uint16_t* const dst = (uint16_t*)malloc((size_t)byte_count * sizeof(uint16_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t i = 0u; i < byte_count; i++) {
        dst[i] = (uint16_t)bytes[i];
    }
    *out = dst;
    *out_count = byte_count;
    return 0;
}

extern "C"
int mglRenderExpandTriangleFanArrayIndices(
    uint32_t vertex_count, uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = vertex_count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = 0u;
        dst[t*3u+1u] = t + 1u;
        dst[t*3u+2u] = t + 2u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandTriangleStripArrayIndices(
    uint32_t vertex_count, uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = vertex_count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = t + (t & 1u);
        dst[t*3u+1u] = t + ((t & 1u) ? 0u : 1u);
        dst[t*3u+2u] = t + 2u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandLineLoopArrayIndices(
    uint32_t first_vertex, uint32_t vertex_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 2u || !out_indices || !out_count) {
        return -1;
    }
    if ((uint64_t)first_vertex + (uint64_t)vertex_count >
        (uint64_t)UINT32_MAX + 1u) {
        return -1;
    }
    const uint64_t need = (uint64_t)vertex_count + 1u;
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t i = 0u; i < vertex_count; i++) {
        dst[i] = first_vertex + i;
    }
    dst[vertex_count] = first_vertex;
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandQuadArrayLineIndices(
    uint32_t quad_count, uint32_t** out_indices, uint64_t* out_count) {
    if (quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 8u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t b = q * 4u;
        const uint32_t d = q * 8u;
        if (b + 3u > UINT32_MAX) {
            free(dst);
            return -1;
        }
        dst[d+0]=b+0; dst[d+1]=b+1; dst[d+2]=b+1; dst[d+3]=b+2;
        dst[d+4]=b+2; dst[d+5]=b+3; dst[d+6]=b+3; dst[d+7]=b+0;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandQuadElementLineIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 8u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t src = q * 4u;
        const uint32_t d = q * 8u;
        const uint32_t i0 = MGLRenderReadIndexBytes(bytes, w, src + 0u);
        const uint32_t i1 = MGLRenderReadIndexBytes(bytes, w, src + 1u);
        const uint32_t i2 = MGLRenderReadIndexBytes(bytes, w, src + 2u);
        const uint32_t i3 = MGLRenderReadIndexBytes(bytes, w, src + 3u);
        dst[d+0]=i0; dst[d+1]=i1; dst[d+2]=i1; dst[d+3]=i2;
        dst[d+4]=i2; dst[d+5]=i3; dst[d+6]=i3; dst[d+7]=i0;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int
mglRenderExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t** out_indices, uint64_t* out_count) {
    if (quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 6u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t base = q * 4u;
        const uint32_t d = q * 6u;
        if (base + 3u > UINT32_MAX) {
            free(dst);
            return -1;
        }
        dst[d+0u] = base + 0u;
        dst[d+1u] = base + 1u;
        dst[d+2u] = base + 2u;
        dst[d+3u] = base + 0u;
        dst[d+4u] = base + 2u;
        dst[d+5u] = base + 3u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandQuadElementIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 6u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t src = q * 4u;
        const uint32_t d = q * 6u;
        const uint32_t i0 = MGLRenderReadIndexBytes(bytes, w, src + 0u);
        const uint32_t i1 = MGLRenderReadIndexBytes(bytes, w, src + 1u);
        const uint32_t i2 = MGLRenderReadIndexBytes(bytes, w, src + 2u);
        const uint32_t i3 = MGLRenderReadIndexBytes(bytes, w, src + 3u);
        dst[d+0] = i0;
        dst[d+1] = i1;
        dst[d+2] = i2;
        dst[d+3] = i0;
        dst[d+4] = i2;
        dst[d+5] = i3;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandTriangleStripIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    const uint32_t n = count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        const uint32_t first = t + (t & 1u);
        const uint32_t second = t + ((t & 1u) ? 0u : 1u);
        dst[t*3u+0u] = MGLRenderReadIndexBytes(bytes, w, first);
        dst[t*3u+1u] = MGLRenderReadIndexBytes(bytes, w, second);
        dst[t*3u+2u] = MGLRenderReadIndexBytes(bytes, w, t + 2u);
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandLineLoopIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 2u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)count + 1u;
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = MGLRenderReadIndexBytes(bytes, w, i);
    }
    dst[count] = dst[0];
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandTriangleFanIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    #define RDX(i) ((w == 1) ? (uint32_t)bytes[i]         : (w == 2) ? (uint32_t)((const uint16_t*)bytes)[i]         : (uint32_t)((const uint32_t*)bytes)[i])
    const uint32_t c = RDX(0u);
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = c;
        dst[t*3u+1u] = RDX(t + 1u);
        dst[t*3u+2u] = RDX(t + 2u);
    }
#undef RDX
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderGeometryGatherIndices(
    const uint8_t* bytes,
    uint32_t elem_width,
    uint32_t count,
    int restart_enabled,
    uint32_t restart_index,
    uint32_t input_vertices,
    MGLRenderGeometryGatherResult* out) {
    if (!bytes || count == 0u || input_vertices == 0u || !out) {
        return -1;
    }
    const int has_restart = restart_enabled ? 1 : 0;
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    uint32_t* const gather = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!gather) {
        return -1;
    }
    uint32_t gathered = 0u;
    uint32_t primitives = 0u;
    uint32_t max_index = 0u;
    uint32_t in_prim = 0u;
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t index = 0u;
        if (w == 1) {
            index = bytes[i];
        } else if (w == 2) {
            index = ((const uint16_t*)bytes)[i];
        } else {
            index = ((const uint32_t*)bytes)[i];
        }
        if (has_restart && index == restart_index) {
            /* A restart terminates the current primitive.  Any indices since
             * the last complete primitive form an incomplete fragment and
             * must not become the prefix of the next primitive. */
            gathered -= in_prim;
            in_prim = 0u;
            continue;
        }
        gather[gathered++] = index;
        if (index > max_index) {
            max_index = index;
        }
        if (++in_prim == input_vertices) {
            primitives++;
            in_prim = 0u;
        }
    }
    if (gathered == 0u || primitives == 0u) {
        free(gather);
        return -1;
    }
    if (in_prim != 0u) {
        gathered -= in_prim;
    }
    out->gather = gather;
    out->gather_count = gathered;
    out->primitive_count = primitives;
    out->max_index = max_index;
    return 0;
}

extern "C"
int mglRenderReadTextureRegionClip(
    int64_t region_x, int64_t region_y,
    int64_t region_w, int64_t region_h,
    int64_t level_width, int64_t level_h,
    MGLRenderReadTextureRegionClip* out) {
    if (!out) return -1;
    const int64_t max_x = region_x + region_w;
    const int64_t max_y = region_y + region_h;
    const int64_t min_x = region_x > 0 ? region_x : 0;
    const int64_t min_y = region_y > 0 ? region_y : 0;
    const int64_t clip_x = max_x < level_width ? max_x : level_width;
    const int64_t clip_y = max_y < level_h ? max_y : level_h;
    const int64_t copy_w = clip_x - min_x;
    const int64_t copy_h = clip_y - min_y;
    out->copy_w = copy_w;
    out->copy_h = copy_h;
    out->dst_x = min_x - region_x;
    out->dst_y = min_y - region_y;
    out->metal_src_x = min_x;
    out->metal_src_y = level_h - clip_y;
    out->empty = (copy_w <= 0 || copy_h <= 0) ? 1 : 0;
    return 0;
}

extern "C"
int mglRenderThreadgroupSize(
    uint32_t local_x, uint32_t local_y, uint32_t local_z,
    MGLRenderThreadgroupSize* out) {
    if (!out) return -1;
    out->x = local_x ? local_x : 1u;
    out->y = local_y ? local_y : 1u;
    out->z = local_z ? local_z : 1u;
    return 0;
}

extern "C"
uint32_t mglRenderMTLPrimitiveTypeForGLMode(uint32_t mode) {
    switch (mode) {
        case GL_POINTS: return 0u;          /* MTLPrimitiveTypePoint */
        case GL_LINES: return 1u;           /* MTLPrimitiveTypeLine */
        case GL_LINE_STRIP: return 2u;      /* MTLPrimitiveTypeLineStrip */
        case GL_TRIANGLES: return 3u;       /* MTLPrimitiveTypeTriangle */
        case GL_TRIANGLE_STRIP: return 4u;  /* MTLPrimitiveTypeTriangleStrip */
        /* LINE_LOOP / adjacency / fan / quads / patches route elsewhere */
        default: return 0xFFFFFFFFu;        /* err */
    }
}

extern "C"
uint32_t mglRenderMTLIndexTypeForGLType(uint32_t gl_type) {
    switch (gl_type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
            return 0u;                      /* MTLIndexTypeUInt16 */
        case GL_UNSIGNED_INT:
            return 1u;                      /* MTLIndexTypeUInt32 */
        default:
            return 0xFFFFFFFFu;             /* err */
    }
}

extern "C"
uint64_t mglRenderMetalTextureLevelDimension(uint64_t base, uint64_t level) {
    const uint64_t one = 1u;
    uint64_t value = base > one ? base : one;
    while (level-- > 0u && value > one) {
        value >>= 1u;
    }
    return value > one ? value : one;
}

extern "C"
int mglRenderResolveVertexAttribBinding(
    uint32_t binding_index, int binding_has_buffer,
    int64_t binding_offset, uint32_t binding_stride,
    int64_t attrib_binding_offset, uint32_t attrib_stride,
    uint32_t binding_divisor, uint32_t attrib_divisor,
    MGLRenderVertexAttribResolve* out) {
    if (!out) return -1;
    if (binding_index < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
        binding_has_buffer) {
        out->use_binding_table = 1;
        out->binding_offset = binding_offset;
        out->stride = (binding_stride > 0) ? binding_stride : attrib_stride;
        out->divisor = binding_divisor;
    } else {
        out->use_binding_table = 0;
        out->binding_offset = attrib_binding_offset;
        out->stride = attrib_stride;
        out->divisor = attrib_divisor;
    }
    return 0;
}

extern "C"
int mglRenderBufferShadowUploadRange(
    int gpu_write_target, int64_t written_min, int64_t written_max,
    uint64_t limit, uint64_t* out_offset, uint64_t* out_length) {
    if (!out_offset || !out_length) return -1;
    uint64_t offset = 0;
    uint64_t length = limit;
    if (gpu_write_target) {
        if (written_min < 0 || written_max <= written_min) {
            return -1;
        }
        const uint64_t min = (uint64_t)written_min;
        const uint64_t max = (uint64_t)written_max;
        offset = min < limit ? min : limit;
        const uint64_t clampedMax = max < limit ? max : limit;
        length = clampedMax - offset;
    }
    if (length == 0) {
        return -1;
    }
    *out_offset = offset;
    *out_length = length;
    return 0;
}

extern "C"
int mglRenderPolygonOffsetDecision(
    uint32_t mode, int has_ctx, int produces_polygons,
    uint32_t polygon_mode,
    int cap_point, int cap_line, int cap_fill,
    MGLRenderPolygonOffsetDecision* out) {
    if (!out) return -1;
    const int polygons = (has_ctx && produces_polygons) ? 1 : 0;
    out->triangle_fill_mode =
        (polygons && polygon_mode == GL_LINE) ? 1 : 0;
    /* The repair is the original's else-if AFTER the GL_LINE branch, so a
     * valid GL_LINE mode must not trigger it. */
    out->needs_polygon_mode_repair =
        (polygons && polygon_mode != GL_LINE &&
         polygon_mode != GL_FILL && polygon_mode != GL_POINT)
            ? 1 : 0;
    out->enable_depth_bias = 0;
    if (polygons) {
        switch (polygon_mode) {
            case GL_POINT:
                out->enable_depth_bias = cap_point ? 1 : 0;
                break;
            case GL_LINE:
                out->enable_depth_bias = cap_line ? 1 : 0;
                break;
            case GL_FILL:
            default:
                out->enable_depth_bias = cap_fill ? 1 : 0;
                break;
        }
    }
    (void)mode;
    return 0;
}

extern "C"
uint32_t mglRenderPrimitiveVertexCountForMode(uint32_t mode) {
    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return 3u;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return 2u;
        case GL_QUADS:
            return 4u;
        case GL_POINTS:
        default:
            return 1u;
    }
}

extern "C"
int mglRenderScaledBlitUVs(
    uint32_t src_tex_w, uint32_t src_tex_h,
    double src_min_x, double src_max_x, double src_min_y, double src_max_y,
    int src_x_forward, int src_y_forward,
    int dst_x_forward, int dst_y_forward,
    MGLRenderScaledBlitUVs* out) {
    if (!out) return -1;
    const float invSrcW = src_tex_w ? (1.0f / (float)src_tex_w) : 0.0f;
    const float invSrcH = src_tex_h ? (1.0f / (float)src_tex_h) : 0.0f;
    float uvLeft = fmaxf(0.0f, fminf(1.0f, (float)src_min_x * invSrcW));
    float uvRight = fmaxf(0.0f, fminf(1.0f, (float)src_max_x * invSrcW));
    float uvTop = fmaxf(0.0f, fminf(1.0f, (float)((double)src_tex_h - src_max_y) * invSrcH));
    float uvBottom = fmaxf(0.0f, fminf(1.0f, (float)((double)src_tex_h - src_min_y) * invSrcH));
    if (src_x_forward != dst_x_forward) {
        const float tmp = uvLeft;
        uvLeft = uvRight;
        uvRight = tmp;
    }
    if (src_y_forward != dst_y_forward) {
        const float tmp = uvTop;
        uvTop = uvBottom;
        uvBottom = tmp;
    }
    out->uv_left = uvLeft;
    out->uv_top = uvTop;
    out->uv_right = uvRight;
    out->uv_bottom = uvBottom;
    return 0;
}

extern "C"
int mglRenderBlitScissorRect(
    double dst_min_x, double dst_max_x,
    double scaled_dst_metal_y, double dst_h,
    uint32_t dst_tex_w, uint32_t dst_tex_h,
    MGLRenderBlitScissorRect* out) {
    if (!out) return -1;
    const double scaledDstMetalBottom = scaled_dst_metal_y + dst_h;
    int64_t x0 = (int64_t)floor(dst_min_x + 0.00001);
    int64_t x1 = (int64_t)ceil(dst_max_x - 0.00001);
    int64_t y0 = (int64_t)floor(scaled_dst_metal_y + 0.00001);
    int64_t y1 = (int64_t)ceil(scaledDstMetalBottom - 0.00001);
    x0 = fmax((int64_t)0, fmin(x0, (int64_t)dst_tex_w));
    x1 = fmax((int64_t)0, fmin(x1, (int64_t)dst_tex_w));
    y0 = fmax((int64_t)0, fmin(y0, (int64_t)dst_tex_h));
    y1 = fmax((int64_t)0, fmin(y1, (int64_t)dst_tex_h));
    out->x0 = x0;
    out->x1 = x1;
    out->y0 = y0;
    out->y1 = y1;
    return 0;
}

extern "C"
int mglRenderBlitFramebufferPlan(
    double src_x0, double src_x1, double src_y0, double src_y1,
    double dst_x0, double dst_x1, double dst_y0, double dst_y1,
    uint32_t src_tex_w, uint32_t src_tex_h,
    uint32_t dst_tex_w, uint32_t dst_tex_h,
    int needs_format_conversion_blit, int needs_render_target_sync_blit,
    int scissor_test_enabled,
    MGLRenderBlitFramebufferPlan* out) {
    if (!out) return -1;
    out->src_x_forward = src_x1 >= src_x0 ? 1 : 0;
    out->src_y_forward = src_y1 >= src_y0 ? 1 : 0;
    out->dst_x_forward = dst_x1 >= dst_x0 ? 1 : 0;
    out->dst_y_forward = dst_y1 >= dst_y0 ? 1 : 0;
    out->blit_needs_flip =
        (out->src_x_forward != out->dst_x_forward ||
         out->src_y_forward != out->dst_y_forward) ? 1 : 0;
    out->src_min_x = fmin(src_x0, src_x1);
    out->src_max_x = fmax(src_x0, src_x1);
    out->src_min_y = fmin(src_y0, src_y1);
    out->src_max_y = fmax(src_y0, src_y1);
    out->dst_min_x = fmin(dst_x0, dst_x1);
    out->dst_max_x = fmax(dst_x0, dst_x1);
    out->dst_min_y = fmin(dst_y0, dst_y1);
    out->dst_max_y = fmax(dst_y0, dst_y1);
    out->src_w = fabs(src_x1 - src_x0);
    out->src_h = fabs(src_y1 - src_y0);
    out->dst_w = fabs(dst_x1 - dst_x0);
    out->dst_h = fabs(dst_y1 - dst_y0);
    if (out->src_w <= 0.0 || out->src_h <= 0.0 ||
        out->dst_w <= 0.0 || out->dst_h <= 0.0) {
        return -1;
    }
    out->needs_scaled_blit =
        (needs_format_conversion_blit || needs_render_target_sync_blit ||
         scissor_test_enabled || out->blit_needs_flip ||
         fabs(out->src_w - out->dst_w) > 0.00001 ||
         fabs(out->src_h - out->dst_h) > 0.00001) ? 1 : 0;
    out->copy_src_x = (int64_t)floor(out->src_min_x + 0.00001);
    out->copy_src_y = (int64_t)floor(out->src_min_y + 0.00001);
    out->copy_dst_x = (int64_t)floor(out->dst_min_x + 0.00001);
    out->copy_dst_y = (int64_t)floor(out->dst_min_y + 0.00001);
    out->copy_w = (int64_t)ceil(out->src_max_x - 0.00001) - out->copy_src_x;
    out->copy_h = (int64_t)ceil(out->src_max_y - 0.00001) - out->copy_src_y;
    out->src_metal_y = (int64_t)src_tex_h - (out->copy_src_y + out->copy_h);
    out->dst_metal_y = (int64_t)dst_tex_h - (out->copy_dst_y + out->copy_h);
    out->scaled_dst_metal_y = (double)dst_tex_h - out->dst_max_y;
    return 0;
}


extern "C"
uint32_t mglRenderTessRoundLevelForSpacing(uint32_t spacing,
                                              uint32_t ceil_level) {
    if (spacing == GL_FRACTIONAL_EVEN) {
        const uint32_t r = (ceil_level & 1u) ? ceil_level + 1u : ceil_level;
        return r > 2u ? r : 2u;
    }
    if (spacing == GL_FRACTIONAL_ODD) {
        return (ceil_level & 1u) ? ceil_level : ceil_level + 1u;
    }
    return ceil_level;
}

/* TES XFB field byte size for a GL type (FLOAT/INT/UINT + vec2/3/4; 0 for
 * unsupported).  Matches the ObjC mglTESXFBFieldByteSize and the packed-write
 * stride contract injected by mglFixMSLTesAsComputeKernel: a zero result
 * means the renderer cannot prove the write stride.  Shared by both gates. */
extern "C"
uint64_t mglRenderTESXFBFieldByteSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return 4u;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return 8u;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return 12u;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return 16u;
        default:
            return 0u;
    }
}

/* Overflow-checked product for tessellation size math; matches the ObjC
 * mglCheckedNSUIntegerProduct ((a != 0 && b > UINT64_MAX / a) rejects).
 * Returns 0 with *result set on success, -1 on bad args / overflow. */
extern "C"
int mglRenderCheckedProduct(uint64_t a, uint64_t b, uint64_t* result) {
    if (!result || (a != 0u && b > UINT64_MAX / a)) {
        return -1;
    }
    *result = a * b;
    return 0;
}

/* 11-bit unsigned float unpack (GL_UNSIGNED_INT_10F_11F_11F_REV CPU decode):
 * 5-bit exponent, 6-bit mantissa, no sign; exponent bias 15.  Denormalized
 * values use 2^(1-15) * mant/64, exp==31 is inf (mant==0) or NaN. */
extern "C"
float mglRenderFloat11ToFloat(uint32_t val) {
    if (val == 0u) {
        return 0.0f;
    }
    const uint32_t exp = (val >> 6) & 0x1Fu;
    const uint32_t mant = val & 0x3Fu;
    if (exp == 0u) {
        return (float)((double)mant / 64.0) * (1.0 / 16384.0);
    } else if (exp == 31u) {
        return mant ? NAN : INFINITY;
    }
    return ldexpf((float)(1.0 + (double)mant / 64.0), (int)exp - 15);
}

/* 10-bit unsigned float unpack: 5-bit exponent, 5-bit mantissa, no sign;
 * exponent bias 15.  Denormalized values use 2^(1-15) * mant/32. */
extern "C"
float mglRenderFloat10ToFloat(uint32_t val) {
    if (val == 0u) {
        return 0.0f;
    }
    const uint32_t exp = (val >> 5) & 0x1Fu;
    const uint32_t mant = val & 0x1Fu;
    if (exp == 0u) {
        return (float)((double)mant / 32.0) * (1.0 / 16384.0);
    } else if (exp == 31u) {
        return mant ? NAN : INFINITY;
    }
    return ldexpf((float)(1.0 + (double)mant / 32.0), (int)exp - 15);
}

/* Float -> unorm8 with round-to-nearest (0.5 rounds up); matches
 * mglMetalFloatToUnorm8 exactly. */
extern "C"
uint8_t mglRenderFloatToUnorm8(float value) {
    if (!(value > 0.0f)) {
        return 0u;
    }
    if (value >= 1.0f) {
        return 255u;
    }
    return (uint8_t)(value * 255.0f + 0.5f);
}

/* Snorm16 decode: INT16_MIN maps to -1.0 exactly; matches
 * mglMetalSnorm16ToFloat. */
extern "C"
float mglRenderSnorm16ToFloat(int16_t value) {
    if (value == INT16_MIN) {
        return -1.0f;
    }
    return (float)value / 32767.0f;
}

/* Snorm8 decode: INT8_MIN maps to -1.0 exactly; matches
 * mglMetalSnorm8ToFloat. */
extern "C"
float mglRenderSnorm8ToFloat(int8_t value) {
    if (value == INT8_MIN) {
        return -1.0f;
    }
    return (float)value / 127.0f;
}


extern "C"
uint32_t mglRenderTessControlPointFormat(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT: return (uint32_t)MTL::VertexFormatFloat;
        case GL_FLOAT_VEC2: return (uint32_t)MTL::VertexFormatFloat2;
        case GL_FLOAT_VEC3: return (uint32_t)MTL::VertexFormatFloat3;
        case GL_FLOAT_VEC4: return (uint32_t)MTL::VertexFormatFloat4;
        case GL_INT: return (uint32_t)MTL::VertexFormatInt;
        case GL_INT_VEC2: return (uint32_t)MTL::VertexFormatInt2;
        case GL_INT_VEC3: return (uint32_t)MTL::VertexFormatInt3;
        case GL_INT_VEC4: return (uint32_t)MTL::VertexFormatInt4;
        case GL_UNSIGNED_INT:
        case GL_BOOL: return (uint32_t)MTL::VertexFormatUInt;
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2: return (uint32_t)MTL::VertexFormatUInt2;
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3: return (uint32_t)MTL::VertexFormatUInt3;
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4: return (uint32_t)MTL::VertexFormatUInt4;
        default: return (uint32_t)MTL::VertexFormatInvalid;
    }
}


extern "C"
uint64_t mglRenderTESXFBVertexStride(const void* program_v) {
    const Program* program = (const Program*)program_v;
    if (!program || program->transform_feedback_varying_count <= 0) {
        return 0u;
    }
    const MGLShaderResourceList* outputs =
        &program->shader_resources_list[_TESS_EVALUATION_SHADER]
                                        [_STAGE_OUTPUT_RES];
    uint64_t stride = 0u;
    for (GLsizei varying = 0;
         varying < program->transform_feedback_varying_count;
         varying++) {
        const char* name = program->transform_feedback_varying_names[varying];
        const MGLShaderResource* output = NULL;
        for (GLuint i = 0; name && outputs->list && i < outputs->count; i++) {
            if (outputs->list[i].name && strcmp(outputs->list[i].name, name) == 0) {
                output = &outputs->list[i];
                break;
            }
        }
        const uint64_t field_bytes =
            output ? mglRenderTESXFBFieldByteSize(output->gl_type) : 0u;
        if (field_bytes == 0u || stride > UINT64_MAX - field_bytes) {
            return 0u;
        }
        stride += field_bytes;
    }
    return stride;
}

extern "C"
uint32_t mglRenderTessEvalItemsPerPatch(
    const void* factor_record, uint32_t gen_mode, uint32_t spacing,
    uint32_t point_mode) {
    if (!factor_record) return 0u;
    typedef struct __attribute__((packed)) {
        uint16_t edge[4];
        uint16_t inside[2];
    } MGLTessFactorRecord;
    const MGLTessFactorRecord* tf = (const MGLTessFactorRecord*)factor_record;
    {
        float edge[4], inside[2];
        for (int i = 0; i < 4; i++) {
            edge[i] = *(const __fp16*)&tf->edge[i];
        }
        for (int i = 0; i < 2; i++) {
            inside[i] = *(const __fp16*)&tf->inside[i];
        }
        if (mglRenderTessFactorsDiscardPatch(gen_mode, edge, inside)) {
            return 0u;
        }
    }
    (void)point_mode;
    if (gen_mode == GL_ISOLINES) {
        float e0 = *(const __fp16*)&tf->edge[0];
        float e1 = *(const __fp16*)&tf->edge[1];
        if (e0 < 1.0f) e0 = 1.0f;
        if (e1 < 1.0f) e1 = 1.0f;
        return (uint32_t)ceilf(e0) * (uint32_t)ceilf(e1) * 2u;
    }
    /* Quads/triangles compute expansion (point_mode and XFB-forced): one
     * work item per inner-grid cell.  Must match mgl_air_backend.cpp
     * isTESCompute TessCoord decomposition. */
    float i0 = *(const __fp16*)&tf->inside[0];
    if (i0 < 1.0f) i0 = 1.0f;
    if (gen_mode == GL_QUADS) {
        float i1 = *(const __fp16*)&tf->inside[1];
        if (i1 < 1.0f) i1 = 1.0f;
        return mglRenderTessRoundLevelForSpacing(spacing, (uint32_t)ceilf(i0)) *
               mglRenderTessRoundLevelForSpacing(spacing, (uint32_t)ceilf(i1));
    }
    {
        const uint32_t n =
            mglRenderTessRoundLevelForSpacing(spacing, (uint32_t)ceilf(i0));
        return n * n;
    }
}

extern "C"
int mglRenderCheckedTessCaptureSize(
    int64_t count, int64_t instance_count, uint64_t stride,
    uint64_t min_stride, uint64_t* size_out, uint64_t* offset_out) {
    if (count <= 0 || instance_count <= 0 || stride < min_stride ||
        !size_out || !offset_out) {
        return -1;
    }
    const uint64_t c = (uint64_t)count;
    const uint64_t ic = (uint64_t)instance_count;
    uint64_t records;
    if (__builtin_mul_overflow(c, ic, &records) ||
        records > UINT64_MAX / stride) {
        return -1;
    }
    *size_out = records * stride;
    *offset_out = 0u;
    return 0;
}

extern "C"
int mglRenderBuildLevelUploadOps(
    const TextureLevel* levels, uint32_t level_count,
    uint32_t texture_type, uint32_t internal_format, uint32_t pixel_format,
    MGLRenderLevelUploadOp* ops, uint32_t ops_capacity,
    uint32_t* op_count_out, uint32_t* short_backing_out, uint32_t* bad_out) {
    if (op_count_out) *op_count_out = 0;
    if (short_backing_out) *short_backing_out = 0;
    if (bad_out) *bad_out = 0;
    if (!levels || !ops || !op_count_out || !short_backing_out || !bad_out ||
        level_count == 0 || ops_capacity < level_count) {
        return -1;
    }
    uint32_t op_count = 0, short_count = 0, bad_count = 0;
    for (uint32_t level = 0; level < level_count; level++) {
        const TextureLevel* l = &levels[level];
        /* mglTextureLevelHasUploadableCPUData, inlined (the compat header is
         * ObjC-typed and cannot be included from this TU). */
        if (!l->complete || !l->data || l->data_size == 0u || l->pitch == 0u) {
            continue;
        }
        switch (l->last_init_source) {
            case kTexImageCopy:
            case kTexImagePBO:
            case kTexSubImageCPU:
            case kTexSubImagePBO:
            case kTexMetalFill:
                break;
            case kTexInitNone:
            case kTexImageNull:
            case kTexRenderTargetWrite:
            default:
                continue;
        }
        if (!(l->has_initialized_data || l->ever_written)) continue;

        MGLRenderLevelUploadPrep prep = {0};
        int prepResult = mglRenderTexturePrepareLevelUpload(
            l, texture_type, internal_format, pixel_format, &prep);
        if (prepResult == -2) {
            MGLRenderLevelUploadOp& op = ops[op_count++];
            op.level = level;
            op.kind = 1u;
            op.width = 0;
            op.height = 0;
            op.bytes_per_row = 0;
            op.bytes_per_image = prep.bytes_per_image;
            op.copy_depth = prep.copy_depth;
            op.available_bytes = prep.available_bytes;
            op.needed_bytes = prep.bytes_per_image * prep.copy_depth;
            op.data = nullptr;
            op.owns_data = 0;
            short_count++;
            continue;
        }
        if (prepResult != 0) {
            bad_count++;
            continue;
        }
        MGLRenderLevelUploadOp& op = ops[op_count++];
        op.level = level;
        op.kind = 0u;
        op.width = MAX((uint32_t)1u, (uint32_t)l->width);
        op.height = MAX((uint32_t)1u, (uint32_t)l->height);
        op.bytes_per_row = prep.bytes_per_row;
        op.bytes_per_image = prep.bytes_per_image;
        op.copy_depth = prep.copy_depth;
        op.available_bytes = prep.available_bytes;
        op.needed_bytes = 0;
        op.data = prep.data;
        op.owns_data = prep.owns_data;
    }
    *op_count_out = op_count;
    *short_backing_out = short_count;
    *bad_out = bad_count;
    return 0;
}

/* per-level CPU upload data preparation. */
extern "C"
int mglRenderTexturePrepareLevelUpload(
    const TextureLevel* level, uint32_t texture_type,
    uint32_t internal_format, uint32_t pixel_format,
    MGLRenderLevelUploadPrep* out) {
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (!level || !out) {
        return -1;
    }
    const uint64_t width = level->width;
    const uint64_t height = level->height ? level->height : 1;
    const uint64_t depth = level->depth ? level->depth : 1;
    const uint64_t bytes_per_row = level->pitch;
    const void* src_data = (const void*)(uintptr_t)level->data;
    if (!src_data || width == 0 || height == 0 || bytes_per_row == 0) {
        return -1;
    }
    const uint64_t copy_depth =
        ((MTL::TextureType)texture_type == MTL::TextureType3D) ? depth : 1;
    const uint64_t available_bytes = level->data_size;
    const uint64_t bytes_per_image =
        MIN(available_bytes / copy_depth, bytes_per_row * height);
    out->bytes_per_row = bytes_per_row;
    out->bytes_per_image = bytes_per_image;
    out->copy_depth = copy_depth;
    out->available_bytes = available_bytes;
    if (available_bytes < bytes_per_image * copy_depth) {
        return -2;
    }

    const void* data = src_data;
    uint64_t bpr = bytes_per_row;
    uint64_t bpi = bytes_per_image;
    void* expanded = nullptr;
    if (mglRenderTextureInternalFormatNeedsRGBA8Expansion(
            internal_format, pixel_format)) {
        size_t ebpr = 0;
        size_t ebpi = 0;
        expanded = mglRenderCreateRGBA8ExpandedUpload(
            src_data, (size_t)width, (size_t)height,
            (size_t)bytes_per_row, internal_format, &ebpr, &ebpi);
        if (expanded) {
            data = expanded;
            bpr = ebpr;
            bpi = ebpi;
        }
    } else if (mglRenderTextureNeedsChannelExpansion(
                   internal_format, pixel_format)) {
        size_t ebpr = 0;
        size_t ebpi = 0;
        expanded = mglRenderCreateChannelExpandedUpload(
            internal_format, pixel_format, src_data, (size_t)width,
            (size_t)height, (size_t)bytes_per_row, &ebpr, &ebpi);
        if (expanded) {
            data = expanded;
            bpr = ebpr;
            bpi = ebpi;
        }
    }
    out->data = data;
    out->bytes_per_row = bpr;
    out->bytes_per_image = bpi;
    out->owns_data = expanded ? 1 : 0;
    return 0;
}

static uint32_t mglRenderDepthUint32ToFloatBits(uint32_t raw) {
    const float depth = (float)((double)raw / 4294967295.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

static uint32_t mglRenderDepth24ToFloatBits(const uint8_t* src) {
    const uint32_t v = (uint32_t)src[0] |
                       ((uint32_t)src[1] << 8u) |
                       ((uint32_t)src[2] << 16u);
    const float depth = (float)((double)v / 16777215.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

static uint32_t mglRenderDepth24Stencil8ToFloatBits(const uint8_t* src) {
    uint32_t packed = 0u;
    memcpy(&packed, src, sizeof(uint32_t));
    const float depth = (float)((double)(packed >> 8u) / 16777215.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

extern "C"
uint8_t mglRenderResolveR8SwizzledComponent(uint32_t swizzle, uint8_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0xffu;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0x00u;
    }
}

static uint8_t mglRenderResolveR8SnormSwizzledComponent(uint32_t swizzle,
                                                        uint8_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x7fu;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0x00u;
    }
}

static uint16_t mglRenderResolveR16UnormSwizzledComponent(uint32_t swizzle,
                                                          uint16_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 65535u;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

static uint16_t mglRenderResolveR16SnormSwizzledComponent(uint32_t swizzle,
                                                          int16_t red) {
    switch (swizzle) {
        case GL_RED: return static_cast<uint16_t>(red);
        case GL_ALPHA:
        case GL_ONE: return 32767;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0;
    }
}

static uint16_t mglRenderResolveR16FloatSwizzledComponent(uint32_t swizzle,
                                                          uint16_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x3c00u; /* 1.0 in half float */
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

static uint32_t mglRenderResolveR32FloatSwizzledComponent(uint32_t swizzle,
                                                          uint32_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x3f800000u;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

static int64_t mglRenderResolveIntegerSwizzledComponent(
    uint32_t swizzle, int64_t red, int64_t green, int64_t blue,
    int64_t alpha, uint32_t components) {
    switch (swizzle) {
        case GL_RED:
            return components >= 1u ? red : 0;
        case GL_GREEN:
            return components >= 2u ? green : 0;
        case GL_BLUE:
            return components >= 3u ? blue : 0;
        case GL_ALPHA:
            return components >= 4u ? alpha : 1;
        case GL_ONE:
            return 1;
        case GL_ZERO:
        default:
            return 0;
    }
}

static int32_t mglRenderResolveR8IntegerSwizzledComponent(
    uint32_t swizzle, int32_t red, int is_signed) {
    (void)is_signed;
    return (int32_t)mglRenderResolveIntegerSwizzledComponent(
        swizzle, red, 0, 0, 1, 1u);
}

extern "C"
uint32_t mglRenderSingleChannelSwizzleStoragePixelFormat(
    uint32_t internal_format) {
    switch (internal_format) {
        case GL_R8I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
        case GL_R16I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
        case GL_R32I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
        case GL_R8UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
        case GL_R16UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
        case GL_R32UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
        case GL_R8:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Unorm);
        case GL_R8_SNORM:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Snorm);
        case GL_R16:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Unorm);
        case GL_R16_SNORM:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Snorm);
        case GL_R16F:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Float);
        case GL_R32F:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        case GL_DEPTH_COMPONENT16:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Unorm);
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            return static_cast<uint32_t>(MTL::PixelFormatInvalid);
    }
}

static int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed);

extern "C"
int mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
    uint32_t internal_format, int swizzled) {
    if (!swizzled) {
        return 0;
    }
    uint32_t components = 0;
    uint32_t component_bytes = 0;
    int is_signed = 0;
    if (mglRenderIntegerFormatLayout(
            internal_format, &components, &component_bytes, &is_signed) != 0 ||
        components <= 1u) {
        return 0;
    }
    return 1;
}

extern "C"
uint32_t mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
    uint32_t internal_format) {
    switch (internal_format) {
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
        default:
            return static_cast<uint32_t>(MTL::PixelFormatInvalid);
    }
}

static int mglRenderPixelFormatMatchesSwizzleBakeStorage(
    uint32_t internal_format, uint32_t storage_pixel_format) {
    const uint32_t expected =
        mglRenderSingleChannelSwizzleStoragePixelFormat(internal_format);
    if (expected != static_cast<uint32_t>(MTL::PixelFormatInvalid) &&
        expected == storage_pixel_format) {
        return 1;
    }
    if (mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
            internal_format, 1) != 0) {
        /* Multi-channel integer bake keeps the native storage format. */
        switch (internal_format) {
            case GL_RG8I:
            case GL_RGB8I:
            case GL_RGBA8I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
            case GL_RG8UI:
            case GL_RGB8UI:
            case GL_RGBA8UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
            case GL_RG16I:
            case GL_RGB16I:
            case GL_RGBA16I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
            case GL_RG16UI:
            case GL_RGB16UI:
            case GL_RGBA16UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
            case GL_RG32I:
            case GL_RGB32I:
            case GL_RGBA32I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
            case GL_RG32UI:
            case GL_RGB32UI:
            case GL_RGBA32UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
            default:
                break;
        }
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint) ||
                   storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            break;
    }
    return 0;
}

extern "C"
int mglRenderTextureSwizzleUsesUploadBake(
    uint32_t internal_format, int swizzled, uint32_t storage_pixel_format) {
    if (!swizzled) {
        return 0;
    }
    if (mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
            internal_format, swizzled) != 0) {
        return mglRenderPixelFormatMatchesSwizzleBakeStorage(
            internal_format, storage_pixel_format);
    }
    if (mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
            internal_format, swizzled) != 0) {
        return mglRenderPixelFormatMatchesSwizzleBakeStorage(
            internal_format, storage_pixel_format);
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint) ||
                   storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            break;
    }
    return 0;
}

static int64_t mglRenderReadIntegerTexelComponent(
    const uint8_t* texel, uint32_t component, uint32_t component_bytes,
    int is_signed) {
    const uint8_t* p = texel + component * component_bytes;
    if (component_bytes == 1u) {
        return is_signed ? (int64_t)(int8_t)p[0] : (int64_t)p[0];
    }
    if (component_bytes == 2u) {
        if (is_signed) {
            return (int64_t) * (const int16_t*)(const void*)p;
        }
        return (int64_t) * (const uint16_t*)(const void*)p;
    }
    if (is_signed) {
        return (int64_t) * (const int32_t*)(const void*)p;
    }
    return (int64_t) * (const uint32_t*)(const void*)p;
}

static void mglRenderWriteIntegerTexelComponent(
    uint8_t* texel, uint32_t component, uint32_t component_bytes,
    int is_signed, int64_t value) {
    uint8_t* p = texel + component * component_bytes;
    if (component_bytes == 1u) {
        if (is_signed) {
            int32_t clamped = (int32_t)value;
            if (clamped > 127) clamped = 127;
            if (clamped < -128) clamped = -128;
            p[0] = (uint8_t)(int8_t)clamped;
        } else {
            uint32_t clamped =
                value < 0 ? 0u :
                value > 255 ? 255u : (uint32_t)value;
            p[0] = (uint8_t)clamped;
        }
        return;
    }
    if (component_bytes == 2u) {
        if (is_signed) {
            int32_t clamped = (int32_t)value;
            if (clamped > 32767) clamped = 32767;
            if (clamped < -32768) clamped = -32768;
            *(int16_t*)(void*)p = (int16_t)clamped;
        } else {
            uint32_t clamped =
                value < 0 ? 0u :
                value > 65535 ? 65535u : (uint32_t)value;
            *(uint16_t*)(void*)p = (uint16_t)clamped;
        }
        return;
    }
    if (is_signed) {
        *(int32_t*)(void*)p = (int32_t)value;
    } else {
        uint64_t clamped =
            value < 0 ? 0ull :
            (uint64_t)value > 0xffffffffull ? 0xffffffffull :
            (uint64_t)value;
        *(uint32_t*)(void*)p = (uint32_t)clamped;
    }
}

static int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed) {
    if (!out_components || !out_component_bytes || !out_signed) {
        return -1;
    }
    *out_components = 0;
    *out_component_bytes = 0;
    *out_signed = 0;
    switch (internal_format) {
        case GL_R8I:
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 1;
            return 0;
        case GL_R8UI:
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 0;
            return 0;
        case GL_R16I:
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 1;
            return 0;
        case GL_R16UI:
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 0;
            return 0;
        case GL_R32I:
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 1;
            return 0;
        case GL_R32UI:
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 0;
            return 0;
        default:
            return -1;
    }
}

extern "C"
uint8_t* mglRenderCreateIntegerMultiChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }
    uint32_t components = 0;
    uint32_t component_bytes = 0;
    int is_signed = 0;
    if (mglRenderIntegerFormatLayout(
            internal_format, &components, &component_bytes, &is_signed) != 0 ||
        components < 2u) {
        return NULL;
    }
    const size_t src_pixel_bytes = (size_t)components * component_bytes;
    const size_t dst_pixel_bytes = 4u * component_bytes;
    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u) ||
        src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }
    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }
    const uint8_t* src = static_cast<const uint8_t*>(src_data);
    const int64_t default_alpha = 0; /* missing alpha in signed integer texels */
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* in = src_row + x * src_pixel_bytes;
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            const int64_t ch[4] = {
                mglRenderReadIntegerTexelComponent(in, 0u, component_bytes, is_signed),
                components >= 2u
                    ? mglRenderReadIntegerTexelComponent(in, 1u, component_bytes, is_signed)
                    : 0,
                components >= 3u
                    ? mglRenderReadIntegerTexelComponent(in, 2u, component_bytes, is_signed)
                    : 0,
                components >= 4u
                    ? mglRenderReadIntegerTexelComponent(in, 3u, component_bytes, is_signed)
                    : default_alpha,
            };
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, ch[0], ch[1], ch[2], ch[3], components),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, component_bytes, is_signed, outv[c]);
            }
        }
    }
    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

extern "C"
int mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode) {
    if (!swizzled || depth_stencil_mode != GL_DEPTH_COMPONENT) {
        return 0;
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureUploadNeedsStencilSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode) {
    if (!swizzled || depth_stencil_mode != GL_STENCIL_INDEX) {
        return 0;
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderStencilSwizzleStoragePixelFormat(void) {
    return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
}

extern "C"
uint8_t* mglRenderCreateStencilSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }
    size_t src_pixel_bytes = 0u;
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
            src_pixel_bytes = 4u;
            break;
        case GL_DEPTH32F_STENCIL8:
            src_pixel_bytes = 5u;
            break;
        default:
            return NULL;
    }
    const size_t dst_pixel_bytes = 4u;
    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u) ||
        src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }
    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }
    const uint8_t* src = static_cast<const uint8_t*>(src_data);
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* in = src_row + x * src_pixel_bytes;
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            uint8_t stencil = 0u;
            if (internal_format == GL_DEPTH24_STENCIL8) {
                uint32_t packed = 0u;
                memcpy(&packed, in, sizeof(uint32_t));
                stencil = (uint8_t)(packed & 0xffu);
            } else {
                stencil = in[4u];
            }
            const int64_t red = (int64_t)stencil;
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, red, 0, 0, 1, 1u),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, 1u, 0, outv[c]);
            }
        }
    }
    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

extern "C"
int mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
    uint32_t internal_format, int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R16UI:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderStoredColorComponents(uint32_t internal_format) {
    uint32_t components = mglNumComponentsForFormat(internal_format);
    return components > 0u ? components : 4u;
}

extern "C"
uint32_t mglRenderMTLSwizzleForGLSwizzle(uint32_t gl_swizzle,
                                            uint32_t components) {
    switch (gl_swizzle) {
        case GL_ZERO:
            return (uint32_t)MTL::TextureSwizzleZero;
        case GL_ONE:
            return (uint32_t)MTL::TextureSwizzleOne;
        case GL_RED:
            return components >= 1u
                ? (uint32_t)MTL::TextureSwizzleRed
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_GREEN:
            return components >= 2u
                ? (uint32_t)MTL::TextureSwizzleGreen
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_BLUE:
            return components >= 3u
                ? (uint32_t)MTL::TextureSwizzleBlue
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_ALPHA:
            return components >= 4u
                ? (uint32_t)MTL::TextureSwizzleAlpha
                : (uint32_t)MTL::TextureSwizzleOne;
        default:
            fprintf(stderr,
                    "MGL ERROR: Unknown swizzle value 0x%x in swizzleTexDesc\n",
                    gl_swizzle);
            return (uint32_t)MTL::TextureSwizzleZero;
    }
}

extern "C"
uint8_t* mglRenderCreateSingleChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }
    if (mglRenderTextureUploadNeedsSingleChannelSwizzle(internal_format, 1) == 0 &&
        mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
            internal_format, 1, GL_DEPTH_COMPONENT) == 0) {
        return NULL;
    }
    if (mglRenderTextureUploadNeedsSingleChannelSwizzleBake(internal_format, 1) == 0 &&
        mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
            internal_format, 1, GL_DEPTH_COMPONENT) == 0) {
        return NULL;
    }

    uint32_t dst_component_bytes = 1u;
    int dst_signed = 0;
    uint32_t src_component_bytes = 1u;
    int src_signed = 0;
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            break;
        case GL_R8I:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            dst_signed = 1;
            src_signed = 1;
            break;
        case GL_R8UI:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            break;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_R16I:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            dst_signed = 1;
            src_signed = 1;
            break;
        case GL_R16UI:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_R32F:
        case GL_R32I:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            dst_signed = (internal_format == GL_R32I);
            src_signed = dst_signed;
            break;
        case GL_R32UI:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            break;
        case GL_DEPTH_COMPONENT16:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_DEPTH_COMPONENT24:
            dst_component_bytes = 4u;
            src_component_bytes = 3u;
            break;
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            break;
        case GL_DEPTH32F_STENCIL8:
            dst_component_bytes = 4u;
            src_component_bytes = 5u;
            break;
        default:
            return NULL;
    }

    const size_t dst_pixel_bytes = 4u * dst_component_bytes;
    const size_t src_pixel_bytes = src_component_bytes;
    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u) ||
        src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }

    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }

    const uint8_t* src = static_cast<const uint8_t*>(src_data);
    for (size_t row = 0; row < height; row++) {
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        const uint8_t* src_row = src + row * src_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            if (internal_format == GL_R8) {
                const uint8_t red = src_row[x * src_pixel_bytes];
                out[0] = mglRenderResolveR8SwizzledComponent(swizzle_r, red);
                out[1] = mglRenderResolveR8SwizzledComponent(swizzle_g, red);
                out[2] = mglRenderResolveR8SwizzledComponent(swizzle_b, red);
                out[3] = mglRenderResolveR8SwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R8_SNORM) {
                const uint8_t red = src_row[x * src_pixel_bytes];
                out[0] = mglRenderResolveR8SnormSwizzledComponent(swizzle_r, red);
                out[1] = mglRenderResolveR8SnormSwizzledComponent(swizzle_g, red);
                out[2] = mglRenderResolveR8SnormSwizzledComponent(swizzle_b, red);
                out[3] = mglRenderResolveR8SnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16_SNORM) {
                const int16_t red =
                    *(const int16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16F) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R32F ||
                internal_format == GL_DEPTH_COMPONENT32F) {
                const uint32_t red =
                    *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT16) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT24) {
                const uint32_t red =
                    mglRenderDepth24ToFloatBits(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT32) {
                const uint32_t red =
                    mglRenderDepthUint32ToFloatBits(
                        *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes));
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH24_STENCIL8) {
                const uint32_t red =
                    mglRenderDepth24Stencil8ToFloatBits(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH32F_STENCIL8) {
                const uint32_t red =
                    *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            const int64_t red = mglRenderReadIntegerTexelComponent(
                src_row + x * src_pixel_bytes, 0u, src_component_bytes,
                src_signed);
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, red, 0, 0, 1, 1u),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, dst_component_bytes, dst_signed, outv[c]);
            }
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

uint8_t* mglRenderCreateRGBA8ExpandedUpload(
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row, uint32_t internal_format,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        src_bytes_per_row == 0 || !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }

    size_t src_pixel_bytes = 0u;
    switch (internal_format) {
        case GL_R3_G3_B2:
            src_pixel_bytes = 1u;
            break;
        case GL_RGBA2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB565:
        case GL_RGBA4:
        case GL_RGB5_A1:
            src_pixel_bytes = 2u;
            break;
        case GL_RGB10:
        case GL_RGB12:
            src_pixel_bytes = 4u;
            break;
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
            src_pixel_bytes = 3u;
            break;
        default:
            return NULL;
    }
    if (src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }

    size_t dst_bytes_per_row = width * 4u;
    size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u)) {
        return NULL;
    }

    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }

    const uint8_t* src = (const uint8_t*)src_data;
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* src_pixel = src_row + x * src_pixel_bytes;
            uint32_t packed = mglReadPackedUploadLE(src_pixel,
                                                       src_pixel_bytes);
            uint8_t r = 0u, g = 0u, b = 0u, a = 0xffu;
            switch (internal_format) {
                case GL_RGB8:
                case GL_SRGB8:
                case GL_RGB:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 0xffu;
                    break;
                case GL_RGB8_SNORM:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 0x7fu; /* 1.0 in snorm */
                    break;
                case GL_RGB8I:
                case GL_RGB8UI:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 1u; /* 1 in integer */
                    break;
                case GL_R3_G3_B2:
                    r = mglExpandUNormBitsTo8((packed >> 5u) & 0x7u, 3u);
                    g = mglExpandUNormBitsTo8((packed >> 2u) & 0x7u, 3u);
                    b = mglExpandUNormBitsTo8(packed & 0x3u, 2u);
                    break;
                case GL_RGB4:
                case GL_RGB5:
                case GL_RGB565:
                    r = mglExpandUNormBitsTo8((packed >> 11u) & 0x1fu, 5u);
                    g = mglExpandUNormBitsTo8((packed >> 5u) & 0x3fu, 6u);
                    b = mglExpandUNormBitsTo8(packed & 0x1fu, 5u);
                    break;
                case GL_RGB10:
                    r = mglExpandUNormBitsTo8(packed & 0x3ffu, 10u);
                    g = mglExpandUNormBitsTo8((packed >> 10u) & 0x3ffu, 10u);
                    b = mglExpandUNormBitsTo8((packed >> 20u) & 0x3ffu, 10u);
                    break;
                case GL_RGB12:
                    r = mglExpandUNormBitsTo8(packed & 0xfffu, 12u);
                    g = mglExpandUNormBitsTo8((packed >> 12u) & 0xfffu, 12u);
                    b = mglExpandUNormBitsTo8((packed >> 24u) & 0xfffu, 12u);
                    break;
                case GL_RGBA2:
                case GL_RGBA4:
                    r = mglExpandUNormBitsTo8((packed >> 12u) & 0xfu, 4u);
                    g = mglExpandUNormBitsTo8((packed >> 8u) & 0xfu, 4u);
                    b = mglExpandUNormBitsTo8((packed >> 4u) & 0xfu, 4u);
                    a = mglExpandUNormBitsTo8(packed & 0xfu, 4u);
                    break;
                case GL_RGB5_A1:
                    r = mglExpandUNormBitsTo8((packed >> 11u) & 0x1fu, 5u);
                    g = mglExpandUNormBitsTo8((packed >> 6u) & 0x1fu, 5u);
                    b = mglExpandUNormBitsTo8((packed >> 1u) & 0x1fu, 5u);
                    a = (packed & 0x1u) ? 0xffu : 0x00u;
                    break;
                default:
                    break;
            }
            uint8_t* out = dst_row + x * 4u;
            out[0] = r;
            out[1] = g;
            out[2] = b;
            out[3] = a;
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

/* RGB->RGBA channel expansion into a caller-provided buffer. */
int mglRenderTextureExpandRGBToRGBA(const void* src, void* dst,
                                       size_t texel_count, size_t tex_width,
                                       size_t tex_height,
                                       size_t src_comp_bytes,
                                       size_t dst_comp_bytes,
                                       uint64_t alpha_default) {
    if (!src || !dst || tex_width == 0 || tex_height == 0 ||
        src_comp_bytes == 0 || dst_comp_bytes == 0) {
        return -1;
    }
    const size_t src_pixel = src_comp_bytes * 3;
    const size_t dst_pixel = dst_comp_bytes * 4;
    if (src_pixel == 0 || dst_pixel == 0) {
        return -1;
    }
    uint8_t* out = (uint8_t*)dst;
    for (size_t row = 0; row < tex_height; row++) {
        for (size_t col = 0; col < tex_width; col++) {
            const size_t idx = row * tex_width + col;
            uint8_t* dp = out + (row * tex_width + col) * dst_pixel;
            if (idx >= texel_count) {
                memset(dp, 0, dst_pixel);
                continue;
            }
            const uint8_t* sp = (const uint8_t*)src + idx * src_pixel;
            memcpy(dp, sp, src_pixel);
            memcpy(dp + src_pixel, &alpha_default, dst_comp_bytes);
        }
    }
    return 0;
}

/* tight-pack the 3D depth planes (pure data transform). */
void* mglRenderTextureRepackDepthPlanes(const void* bytes,
                                           size_t bytes_per_image,
                                           size_t expected_bytes_per_image,
                                           size_t copy_depth) {
    if (!bytes || expected_bytes_per_image == 0 ||
        bytes_per_image < expected_bytes_per_image || copy_depth == 0) {
        return NULL;
    }
    if (expected_bytes_per_image > SIZE_MAX / copy_depth) {
        return NULL;
    }
    size_t packed_size = expected_bytes_per_image * copy_depth;
    void* packed = malloc(packed_size);
    if (!packed) {
        return NULL;
    }
    const uint8_t* src = (const uint8_t*)bytes;
    uint8_t* dst = (uint8_t*)packed;
    for (size_t z = 0; z < copy_depth; z++) {
        memcpy(dst + z * expected_bytes_per_image,
               src + z * bytes_per_image, expected_bytes_per_image);
    }
    return packed;
}

int mglRenderCreateSampler(void* sampler_descriptor,
                              void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    MTL::SamplerDescriptor* descriptor =
        static_cast<MTL::SamplerDescriptor*>(sampler_descriptor);
    if (!descriptor || !sampler_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::SamplerState* sampler = renderer.device->newSamplerState(descriptor);
    if (!sampler) return -1;
    *sampler_out = sampler;
    return 0;
}

int mglRenderCreateDefaultSampler(void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    if (!sampler_out) return -1;
    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) return -1;
    int result = mglRenderCreateSampler(descriptor, sampler_out);
    descriptor->release();
    return result;
}

int mglRenderCreateFilterSampler(uint32_t nearest, void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    if (!sampler_out) return -1;
    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) return -1;
    const MTL::SamplerMinMagFilter filter =
        nearest ? MTL::SamplerMinMagFilterNearest
                : MTL::SamplerMinMagFilterLinear;
    descriptor->setMinFilter(filter);
    descriptor->setMagFilter(filter);
    descriptor->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
    descriptor->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    descriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    descriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    int result = mglRenderCreateSampler(descriptor, sampler_out);
    descriptor->release();
    return result;
}

int mglRenderCreateSamplerForGL(const TextureParameter* params,
                                   uint32_t target,
                                   void** sampler_out,
                                   char* err,
                                   size_t errcap) {
    if (sampler_out) *sampler_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!params || !sampler_out) {
        if (err && errcap) snprintf(err, errcap, "invalid sampler parameters");
        return -1;
    }

    MTL::SamplerMinMagFilter minFilter;
    MTL::SamplerMipFilter mipFilter = MTL::SamplerMipFilterNotMipmapped;
    switch (params->min_filter) {
        case GL_NEAREST:
            minFilter = MTL::SamplerMinMagFilterNearest;
            break;
        case GL_LINEAR:
            minFilter = MTL::SamplerMinMagFilterLinear;
            break;
        case GL_NEAREST_MIPMAP_NEAREST:
            minFilter = MTL::SamplerMinMagFilterNearest;
            mipFilter = MTL::SamplerMipFilterNearest;
            break;
        case GL_LINEAR_MIPMAP_NEAREST:
            minFilter = MTL::SamplerMinMagFilterLinear;
            mipFilter = MTL::SamplerMipFilterNearest;
            break;
        case GL_NEAREST_MIPMAP_LINEAR:
            minFilter = MTL::SamplerMinMagFilterNearest;
            mipFilter = MTL::SamplerMipFilterLinear;
            break;
        case GL_LINEAR_MIPMAP_LINEAR:
            minFilter = MTL::SamplerMinMagFilterLinear;
            mipFilter = MTL::SamplerMipFilterLinear;
            break;
        default:
            if (err && errcap) snprintf(err, errcap,
                                        "invalid GL min filter=0x%x",
                                        params->min_filter);
            return -1;
    }

    MTL::SamplerMinMagFilter magFilter;
    switch (params->mag_filter) {
        case GL_NEAREST:
            magFilter = MTL::SamplerMinMagFilterNearest;
            break;
        case GL_LINEAR:
            magFilter = MTL::SamplerMinMagFilterLinear;
            break;
        default:
            if (err && errcap) snprintf(err, errcap,
                                        "invalid GL mag filter=0x%x",
                                        params->mag_filter);
            return -1;
    }

    auto addressModeForGL = [&](GLenum value,
                                MTL::SamplerAddressMode* out) -> bool {
        if (!out) return false;
        switch (value) {
            case GL_CLAMP_TO_EDGE:
                *out = MTL::SamplerAddressModeClampToEdge;
                return true;
            case GL_CLAMP_TO_BORDER:
                *out = MTL::SamplerAddressModeClampToBorderColor;
                return true;
            case GL_MIRRORED_REPEAT:
                *out = MTL::SamplerAddressModeMirrorRepeat;
                return true;
            case GL_REPEAT:
                *out = MTL::SamplerAddressModeRepeat;
                return true;
            case GL_MIRROR_CLAMP_TO_EDGE:
                *out = MTL::SamplerAddressModeMirrorClampToEdge;
                return true;
            default:
                return false;
        }
    };

    MTL::SamplerAddressMode sAddress;
    MTL::SamplerAddressMode tAddress;
    MTL::SamplerAddressMode rAddress;
    if (!addressModeForGL(params->wrap_s, &sAddress) ||
        !addressModeForGL(params->wrap_t, &tAddress) ||
        !addressModeForGL(params->wrap_r, &rAddress)) {
        if (err && errcap) snprintf(err, errcap,
                                    "invalid GL sampler address mode");
        return -1;
    }

    MTL::SamplerBorderColor borderColor =
        MTL::SamplerBorderColorTransparentBlack;
    const bool hasBorder = params->wrap_s == GL_CLAMP_TO_BORDER ||
                           params->wrap_t == GL_CLAMP_TO_BORDER ||
                           params->wrap_r == GL_CLAMP_TO_BORDER;
    if (hasBorder) {
        const float* color = params->border_color;
        if (color[0] == 0.0f && color[1] == 0.0f &&
            color[2] == 0.0f && color[3] == 1.0f) {
            borderColor = MTL::SamplerBorderColorOpaqueBlack;
        } else if (color[0] == 1.0f && color[1] == 1.0f &&
                   color[2] == 1.0f && color[3] == 1.0f) {
            borderColor = MTL::SamplerBorderColorOpaqueWhite;
        } else if (!(color[0] == 0.0f && color[1] == 0.0f &&
                     color[2] == 0.0f && color[3] == 0.0f)) {
            /* Metal exposes only three named border colors. Match the ObjC
             * fallback for arbitrary GL colors. */
            borderColor = color[3] < 0.5f
                ? MTL::SamplerBorderColorTransparentBlack
                : (color[0] >= 0.5f && color[1] >= 0.5f && color[2] >= 0.5f
                       ? MTL::SamplerBorderColorOpaqueWhite
                       : MTL::SamplerBorderColorOpaqueBlack);
        }
    }

    MTL::CompareFunction compare = MTL::CompareFunctionNever;
    if (params->compare_mode == GL_COMPARE_REF_TO_TEXTURE) {
        switch (params->compare_func) {
            case GL_NEVER: compare = MTL::CompareFunctionNever; break;
            case GL_LESS: compare = MTL::CompareFunctionLess; break;
            case GL_EQUAL: compare = MTL::CompareFunctionEqual; break;
            case GL_LEQUAL: compare = MTL::CompareFunctionLessEqual; break;
            case GL_GREATER: compare = MTL::CompareFunctionGreater; break;
            case GL_NOTEQUAL: compare = MTL::CompareFunctionNotEqual; break;
            case GL_GEQUAL: compare = MTL::CompareFunctionGreaterEqual; break;
            case GL_ALWAYS: compare = MTL::CompareFunctionAlways; break;
            default:
                if (err && errcap) snprintf(err, errcap,
                                            "invalid GL compare function");
                return -1;
        }
    } else if (params->compare_mode != GL_NONE) {
        if (err && errcap) snprintf(err, errcap,
                                    "invalid GL compare mode=0x%x",
                                    params->compare_mode);
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap,
                                    "renderer is not initialized");
        return -1;
    }
    MTL::SamplerDescriptor* descriptor =
        MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) {
        if (err && errcap) snprintf(err, errcap,
                                    "sampler descriptor allocation failed");
        return -1;
    }
    descriptor->setMinFilter(minFilter);
    descriptor->setMagFilter(magFilter);
    descriptor->setMipFilter(mipFilter);
    descriptor->setSAddressMode(sAddress);
    descriptor->setTAddressMode(tAddress);
    descriptor->setRAddressMode(rAddress);
    descriptor->setBorderColor(borderColor);
    descriptor->setCompareFunction(compare);
    descriptor->setMaxAnisotropy(
        params->max_anisotropy > 1.0f
            ? std::min<NS::UInteger>(16u,
                                     std::max<NS::UInteger>(
                                         1u, static_cast<NS::UInteger>(
                                                 params->max_anisotropy)))
            : 1u);
    descriptor->setLodMinClamp(params->min_lod < 0.0f ? 0.0f : params->min_lod);
    descriptor->setLodMaxClamp(params->max_lod >= 1000.0f
                                   ? 1e9f : params->max_lod);
    if (target == GL_TEXTURE_RECTANGLE) {
        descriptor->setNormalizedCoordinates(false);
        if (params->wrap_s != GL_CLAMP_TO_EDGE ||
            params->wrap_t != GL_CLAMP_TO_EDGE ||
            params->wrap_r != GL_CLAMP_TO_EDGE) {
            descriptor->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
            descriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
            descriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
        }
    }

    MTL::SamplerState* sampler = renderer.device->newSamplerState(descriptor);
    descriptor->release();
    if (!sampler) {
        if (err && errcap) snprintf(err, errcap,
                                    "Metal sampler creation failed");
        return -1;
    }
    *sampler_out = sampler;
    return 0;
}

int mglRenderCreateDepthStencilState(void* depth_stencil_descriptor,
                                        void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        static_cast<MTL::DepthStencilDescriptor*>(depth_stencil_descriptor);
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::DepthStencilState* state =
        renderer.device->newDepthStencilState(descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

static MGLRenderStencilDescriptorState
mglRenderDescribeStencilDescriptor(const MTL::StencilDescriptor *descriptor) {
    MGLRenderStencilDescriptorState state = {};
    if (!descriptor) return state;
    state.present = 1u;
    state.compare_function = static_cast<uint32_t>(descriptor->stencilCompareFunction());
    state.read_mask = descriptor->readMask();
    state.write_mask = descriptor->writeMask();
    state.stencil_failure_operation =
        static_cast<uint32_t>(descriptor->stencilFailureOperation());
    state.depth_failure_operation =
        static_cast<uint32_t>(descriptor->depthFailureOperation());
    state.depth_stencil_pass_operation =
        static_cast<uint32_t>(descriptor->depthStencilPassOperation());
    return state;
}

int mglRenderDescribeDepthStencilDescriptor(
    const void *depth_stencil_descriptor,
    MGLRenderDepthStencilDescriptorState *state_out) {
    if (!state_out) return -1;
    *state_out = {};
    const MTL::DepthStencilDescriptor *descriptor =
        static_cast<const MTL::DepthStencilDescriptor *>(depth_stencil_descriptor);
    if (!descriptor) return -1;
    state_out->depth_compare_function =
        static_cast<uint32_t>(descriptor->depthCompareFunction());
    state_out->depth_write_enabled = descriptor->isDepthWriteEnabled() ? 1u : 0u;
    state_out->front = mglRenderDescribeStencilDescriptor(
        descriptor->frontFaceStencil());
    state_out->back = mglRenderDescribeStencilDescriptor(
        descriptor->backFaceStencil());
    return 0;
}

int mglRenderGetDeviceIdentity(const void *device,
                                  uint64_t *registry_id_out,
                                  char *name_out,
                                  size_t name_capacity) {
    if (registry_id_out) *registry_id_out = 0u;
    if (name_out && name_capacity) name_out[0] = '\0';
    const MTL::Device *metal_device =
        static_cast<const MTL::Device *>(device);
    if (!metal_device) return -1;
    if (registry_id_out) *registry_id_out = metal_device->registryID();
    if (name_out && name_capacity) {
        NS::String *name = metal_device->name();
        const char *utf8 = name ? name->utf8String() : nullptr;
        if (utf8) {
            std::snprintf(name_out, name_capacity, "%s", utf8);
        }
    }
    return 0;
}

static MTL::StencilDescriptor* mglRenderBuildStencilDescriptor(
    const MGLRenderStencilDescriptorState& state) {
    if (!state.present) return nullptr;
    MTL::StencilDescriptor* descriptor =
        MTL::StencilDescriptor::alloc()->init();
    if (!descriptor) return nullptr;
    descriptor->setStencilCompareFunction(
        static_cast<MTL::CompareFunction>(state.compare_function));
    descriptor->setReadMask(state.read_mask);
    descriptor->setWriteMask(state.write_mask);
    descriptor->setStencilFailureOperation(
        static_cast<MTL::StencilOperation>(
            state.stencil_failure_operation));
    descriptor->setDepthFailureOperation(
        static_cast<MTL::StencilOperation>(state.depth_failure_operation));
    descriptor->setDepthStencilPassOperation(
        static_cast<MTL::StencilOperation>(
            state.depth_stencil_pass_operation));
    return descriptor;
}

static MTL::DepthStencilState* mglRenderCreateDepthStencilFromStateLocked(
    mgl::Renderer& renderer,
    const MGLRenderDepthStencilDescriptorState& state) {
    if (!renderer.device) return nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        MTL::DepthStencilDescriptor::alloc()->init();
    if (!descriptor) return nullptr;
    descriptor->setDepthCompareFunction(
        static_cast<MTL::CompareFunction>(state.depth_compare_function));
    descriptor->setDepthWriteEnabled(state.depth_write_enabled != 0);
    MTL::StencilDescriptor* front =
        mglRenderBuildStencilDescriptor(state.front);
    MTL::StencilDescriptor* back =
        mglRenderBuildStencilDescriptor(state.back);
    if (state.front.present && !front) {
        descriptor->release();
        if (back) back->release();
        return nullptr;
    }
    if (state.back.present && !back) {
        descriptor->release();
        if (front) front->release();
        return nullptr;
    }
    descriptor->setFrontFaceStencil(front);
    descriptor->setBackFaceStencil(back);
    if (front) front->release();
    if (back) back->release();
    MTL::DepthStencilState* result =
        renderer.device->newDepthStencilState(descriptor);
    descriptor->release();
    return result;
}

int mglRenderCreateDepthStencilStateFromState(
    const MGLRenderDepthStencilDescriptorState* descriptor,
    void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCreateDepthStencilFromStateLocked(renderer, *descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

int mglRenderCreatePipelineCacheOwner(
    int pso_dedup_enabled,
    int depth_stencil_cache_enabled,
    int binary_archive_enabled,
    void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    try {
        auto owner = std::make_unique<mgl::PipelineCacheOwner>();
        owner->psoDedupEnabled = pso_dedup_enabled != 0;
        owner->depthStencilCacheEnabled =
            depth_stencil_cache_enabled != 0;
        owner->binaryArchiveEnabled = binary_archive_enabled != 0;
        *owner_out = owner.release();
        return 0;
    } catch (...) {
        return -1;
    }
}

void mglRenderDestroyPipelineCacheOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

void mglRenderResetPipelineCacheOwner(void* owner_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->clearCaches();
}

int mglRenderGetPipelineCacheFlags(
    void* owner_handle,
    int* pso_dedup_enabled_out,
    int* depth_stencil_cache_enabled_out,
    int* binary_archive_enabled_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (pso_dedup_enabled_out) {
        *pso_dedup_enabled_out = owner->psoDedupEnabled ? 1 : 0;
    }
    if (depth_stencil_cache_enabled_out) {
        *depth_stencil_cache_enabled_out =
            owner->depthStencilCacheEnabled ? 1 : 0;
    }
    if (binary_archive_enabled_out) {
        *binary_archive_enabled_out = owner->binaryArchiveEnabled ? 1 : 0;
    }
    return 0;
}

void mglRenderDisablePipelineBinaryArchive(void* owner_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->binaryArchiveEnabled = false;
    owner->clearBinaryArchive();
}

int mglRenderGetPipelineBinaryArchiveState(
    void* owner_handle, int* enabled_out, int* present_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (enabled_out) *enabled_out = 0;
    if (present_out) *present_out = 0;
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (enabled_out) *enabled_out = owner->binaryArchiveEnabled ? 1 : 0;
    if (present_out) *present_out = owner->binaryArchive ? 1 : 0;
    return 0;
}

int mglRenderLoadPipelineBinaryArchive(
    void* owner_handle,
    const char* cache_key,
    void* url,
    int archive_exists,
    int* reused_out,
    char* err,
    size_t errcap) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    auto* archiveURL = static_cast<NS::URL*>(url);
    if (reused_out) *reused_out = 0;
    if (err && errcap) err[0] = '\0';
    if (!owner || !cache_key || !cache_key[0] || !archiveURL) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::scoped_lock lock(renderer.mutex, owner->mutex);
    if (!renderer.device || !owner->binaryArchiveEnabled) return -1;

    auto shared = renderer.binaryArchives.find(cache_key);
    if (shared != renderer.binaryArchives.end() && shared->second) {
        owner->clearBinaryArchive();
        owner->binaryArchive = shared->second;
        owner->binaryArchive->retain();
        owner->binaryArchiveKey = cache_key;
        if (reused_out) *reused_out = 1;
        return 0;
    }

    MTL::BinaryArchiveDescriptor* descriptor =
        MTL::BinaryArchiveDescriptor::alloc()->init();
    if (!descriptor) return -1;
    if (archive_exists) descriptor->setUrl(archiveURL);
    NS::Error* nsError = nullptr;
    MTL::BinaryArchive* archive =
        renderer.device->newBinaryArchive(descriptor, &nsError);
    descriptor->release();
    if (!archive) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    archive->setLabel(NS::String::string(
        "MGL Pipeline Binary Archive", NS::UTF8StringEncoding));

    renderer.binaryArchives.emplace(cache_key, archive);
    owner->clearBinaryArchive();
    owner->binaryArchive = archive;
    owner->binaryArchive->retain();
    owner->binaryArchiveKey = cache_key;
    return 0;
}

int mglRenderSerializePipelineBinaryArchive(
    void* owner_handle, void* url, char* err, size_t errcap) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    auto* archiveURL = static_cast<NS::URL*>(url);
    if (err && errcap) err[0] = '\0';
    if (!owner || !archiveURL) return -1;

    MTL::BinaryArchive* archive = nullptr;
    {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (!owner->binaryArchiveEnabled || !owner->binaryArchive) return -1;
        archive = owner->binaryArchive;
        archive->retain();
    }
    NS::Error* nsError = nullptr;
    const bool serialized = archive->serializeToURL(archiveURL, &nsError);
    archive->release();
    if (!serialized) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    return 0;
}

void mglRenderDiscardPipelineBinaryArchive(
    void* owner_handle, const char* cache_key) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::scoped_lock lock(renderer.mutex, owner->mutex);
    std::string key = cache_key && cache_key[0]
        ? std::string(cache_key) : owner->binaryArchiveKey;
    owner->clearBinaryArchive();
    auto shared = renderer.binaryArchives.find(key);
    if (shared != renderer.binaryArchives.end()) {
        if (shared->second) shared->second->release();
        renderer.binaryArchives.erase(shared);
    }
}

int mglRenderGetPipelineActiveState(
    void* owner_handle, MGLRenderPipelineActiveState* state_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->active;
    return 0;
}

int mglRenderInvalidatePipelineActiveState(void* owner_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.vertex_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.fragment_function);
    owner->active = {};
    owner->active.color0_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    owner->active.depth_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    owner->active.stencil_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    return 0;
}

int mglRenderSetPipelineActiveObject(void* owner_handle,
                                        void* pipeline_state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::retainObject(pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    owner->active.pipeline_state = pipeline_state;
    return 0;
}

int mglRenderActivatePipelineState(
    void* owner_handle, const MGLRenderPipelineActiveState* state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::retainObject(state->pipeline_state);
    mgl::PipelineCacheOwner::retainObject(state->vertex_function);
    mgl::PipelineCacheOwner::retainObject(state->fragment_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.vertex_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.fragment_function);
    owner->active = *state;
    return 0;
}

int mglRenderSetPipelineBlendState(
    void* owner_handle, uint32_t attachment,
    const MGLRenderPipelineBlendState* state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state ||
        attachment >= MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->blend[attachment] = *state;
    return 0;
}

int mglRenderGetPipelineBlendState(
    void* owner_handle, uint32_t attachment,
    MGLRenderPipelineBlendState* state_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state_out ||
        attachment >= MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->blend[attachment];
    return 0;
}

int mglRenderGetOrCreateDepthStencilState(
    void* owner_handle,
    const MGLRenderDepthStencilDescriptorState* descriptor,
    void** depth_stencil_state_out,
    int* created_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    if (created_out) *created_out = 0;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !descriptor || !depth_stencil_state_out) return -1;
    std::lock_guard<std::mutex> ownerLock(owner->mutex);
    if (!owner->depthStencilCacheEnabled) return -1;
    const mgl::DepthStencilCacheKey key =
        mgl::PipelineCacheOwner::makeDepthStencilKey(*descriptor);
    auto found = owner->depthStencilCache.find(key);
    if (found != owner->depthStencilCache.end()) {
        *depth_stencil_state_out = found->second->state;
        mgl::PipelineCacheOwner::touch(owner->depthStencilCacheLRU, key);
        return 0;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> rendererLock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCreateDepthStencilFromStateLocked(renderer, *descriptor);
    if (!state) return -1;
    std::unique_ptr<mgl::PipelineCacheDepthStencilEntry> entry(
        new (std::nothrow) mgl::PipelineCacheDepthStencilEntry());
    if (!entry) {
        state->release();
        return -1;
    }
    entry->state = state;
    try {
        owner->depthStencilCache.emplace(key, std::move(entry));
        mgl::PipelineCacheOwner::touch(owner->depthStencilCacheLRU, key);
        while (owner->depthStencilCache.size() > 64u &&
               !owner->depthStencilCacheLRU.empty()) {
            const mgl::DepthStencilCacheKey oldest =
                owner->depthStencilCacheLRU.front();
            owner->depthStencilCacheLRU.pop_front();
            owner->depthStencilCache.erase(oldest);
        }
    } catch (...) {
        return -1;
    }
    if (created_out) *created_out = 1;
    *depth_stencil_state_out = state;
    return 0;
}

int mglRenderLookupPipeline(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderPipelineActiveState* state_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !key_words || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    auto found = owner->pipelineCache.find(key);
    if (found == owner->pipelineCache.end()) return 0;
    *state_out = {};
    state_out->pipeline_state = found->second->pipeline;
    state_out->vertex_function = found->second->vertexFunction;
    state_out->fragment_function = found->second->fragmentFunction;
    mgl::PipelineCacheOwner::touch(owner->pipelineCacheLRU, key);
    return 1;
}

int mglRenderStorePipeline(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderPipelineActiveState* state,
    uint32_t* evicted_out) {
    if (evicted_out) *evicted_out = 0;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !key_words || !state || !state->pipeline_state) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    try {
        auto entry = std::make_unique<mgl::PipelineCacheEntry>();
        entry->pipeline =
            static_cast<MTL::RenderPipelineState*>(state->pipeline_state);
        entry->vertexFunction =
            static_cast<MTL::Function*>(state->vertex_function);
        entry->fragmentFunction =
            static_cast<MTL::Function*>(state->fragment_function);
        entry->pipeline->retain();
        if (entry->vertexFunction) entry->vertexFunction->retain();
        if (entry->fragmentFunction) entry->fragmentFunction->retain();

        const bool replacing = owner->pipelineCache.find(key) !=
                               owner->pipelineCache.end();
        uint32_t removed = 0;
        if (!replacing && owner->pipelineCache.size() >= 256u) {
            const size_t target =
                std::max<size_t>(1u, owner->pipelineCache.size() / 4u);
            while (removed < target && !owner->pipelineCacheLRU.empty()) {
                const mgl::PipelineCacheKey oldest =
                    owner->pipelineCacheLRU.front();
                owner->pipelineCacheLRU.pop_front();
                removed += owner->pipelineCache.erase(oldest) ? 1u : 0u;
            }
        }
        owner->pipelineCache[key] = std::move(entry);
        mgl::PipelineCacheOwner::touch(owner->pipelineCacheLRU, key);
        if (evicted_out) *evicted_out = removed;
        return 0;
    } catch (...) {
        return -1;
    }
}

int mglRenderLookupPipelineDescriptorState(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderPipelineDescriptorState* state_out) {
    if (state_out) *state_out = {};
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !key_words || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    auto found = owner->descriptorCache.find(key);
    if (found == owner->descriptorCache.end()) return 0;
    *state_out = found->second->state;
    mgl::PipelineCacheOwner::touch(owner->descriptorCacheLRU, key);
    return 1;
}

int mglRenderStorePipelineDescriptorState(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderPipelineDescriptorState* state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !key_words || !state) return -1;
    std::unique_ptr<mgl::PipelineCacheDescriptorEntry> entry(
        new (std::nothrow) mgl::PipelineCacheDescriptorEntry());
    if (!entry) return -1;
    entry->state = *state;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    try {
        owner->descriptorCache[key] = std::move(entry);
        mgl::PipelineCacheOwner::touch(owner->descriptorCacheLRU, key);
        while (owner->descriptorCache.size() > 128u &&
               !owner->descriptorCacheLRU.empty()) {
            const mgl::PipelineCacheKey oldest =
                owner->descriptorCacheLRU.front();
            owner->descriptorCacheLRU.pop_front();
            owner->descriptorCache.erase(oldest);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

int mglRenderCreatePendingEventOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::PendingEventOwner* owner = new (std::nothrow) mgl::PendingEventOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

/* Prepare: create-or-reuse the pending event and record the GL sync name.
 * Returns a BORROWED event pointer (the owner keeps its reference). */
int mglRenderPendingEventPrepare(void* owner_handle,
                                    GLsizei sync_name,
                                    void** event_out) {
    if (event_out) *event_out = nullptr;
    mgl::PendingEventOwner* owner =
        static_cast<mgl::PendingEventOwner*>(owner_handle);
    if (!owner || !event_out) return -1;
    if (!owner->event) {
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        MTL::Event* event = renderer.device->newEvent();
        if (!event) return -1;
        owner->event = event;
    }
    owner->sync_name = sync_name;
    *event_out = owner->event;
    return 0;
}

/* Detach: transfer the owner's reference to the caller
 * (the ObjC side bridges it with __bridge_transfer) and clear the slot. */
int mglRenderPendingEventDetach(void* owner_handle,
                                   GLsizei* sync_name_out,
                                   void** event_out) {
    if (event_out) *event_out = nullptr;
    if (sync_name_out) *sync_name_out = 0;
    mgl::PendingEventOwner* owner =
        static_cast<mgl::PendingEventOwner*>(owner_handle);
    if (!owner || !event_out) return -1;
    if (owner->event) {
        *event_out = owner->event;
        owner->event = nullptr;
    }
    if (sync_name_out) *sync_name_out = owner->sync_name;
    owner->sync_name = 0;
    return 0;
}

/* Clear: discard the pending event (owner keeps its allocation). */
void mglRenderPendingEventClear(void* owner_handle) {
    mgl::PendingEventOwner* owner =
        static_cast<mgl::PendingEventOwner*>(owner_handle);
    if (!owner) return;
    if (owner->event) {
        owner->event->release();
        owner->event = nullptr;
    }
    owner->sync_name = 0;
}

void mglRenderDestroyPendingEventOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::PendingEventOwner* owner =
        static_cast<mgl::PendingEventOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateEvent(void** event_out) {
    if (event_out) *event_out = nullptr;
    if (!event_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Event* event = renderer.device->newEvent();
    if (!event) return -1;
    *event_out = event;
    return 0;
}

int mglRenderCreateFunction(void* library,
                               const char* name,
                               void* function_constant_values,
                               void** function_out,
                               char* err,
                               size_t errcap) {
    if (function_out) *function_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Library* source = static_cast<MTL::Library*>(library);
    if (!source || !name || !name[0] || !function_out) return -1;

    NS::String* functionName =
        NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* function = nullptr;
    if (function_constant_values) {
        NS::Error* nsError = nullptr;
        function = source->newFunction(
            functionName,
            static_cast<MTL::FunctionConstantValues*>(
                function_constant_values),
            &nsError);
        if (!function) mgl::copyError(nsError, err, errcap);
    } else {
        function = source->newFunction(functionName);
        if (!function && err && errcap) {
            snprintf(err, errcap, "function '%s' not found", name);
        }
    }
    if (!function) return -1;
    *function_out = function;
    return 0;
}


int mglRenderCreateRenderPipelineFromState(
    void* vs_function,
    void* fs_function,
    const MGLRenderPipelineDescriptorState* state,
    void* binary_archive,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!vs_function || !state || !pipeline_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return mglAirCreateRenderPipelineWithArchive(
        renderer.device, vs_function, fs_function, state, binary_archive,
        pipeline_out, err, errcap);
}

int mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
    void* owner_handle,
    void* vs_function,
    void* fs_function,
    const MGLRenderPipelineDescriptorState* state,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    MTL::BinaryArchive* archive = nullptr;
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->binaryArchiveEnabled && owner->binaryArchive) {
            archive = owner->binaryArchive;
            archive->retain();
        }
    }
    int result = mglRenderCreateRenderPipelineFromState(
        vs_function, fs_function, state, archive, pipeline_out, err, errcap);
    if (archive) archive->release();
    return result;
}

int mglRenderCreateRenderPipelineState(
    void* render_pipeline_descriptor,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    if (!descriptor || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::RenderPipelineState* pipeline =
        renderer.device->newRenderPipelineState(descriptor, &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderCreateRenderPipelineStateWithArchive(
    void* render_pipeline_descriptor,
    void* binary_archive,
    void** pipeline_out,
    int* archive_hit_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (archive_hit_out) *archive_hit_out = 0;
    if (err && errcap) err[0] = '\0';
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    if (!descriptor || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    const bool archiveEligible = archive && descriptor->vertexFunction() &&
                                 descriptor->fragmentFunction();
    MTL::RenderPipelineState* pipeline = nullptr;
    NS::Error* nsError = nullptr;
    if (archiveEligible) {
        descriptor->setBinaryArchives(NS::Array::array(archive));
        pipeline = renderer.device->newRenderPipelineState(
            descriptor, MTL::PipelineOptionFailOnBinaryArchiveMiss,
            nullptr, &nsError);
        if (pipeline && archive_hit_out) *archive_hit_out = 1;
    }

    const bool archiveMiss = archiveEligible && !pipeline;
    if (!pipeline) {
        nsError = nullptr;
        pipeline = renderer.device->newRenderPipelineState(
            descriptor, &nsError);
    }
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }

    if (archiveMiss) {
        NS::Error* addError = nullptr;
        if (!archive->addRenderPipelineFunctions(descriptor, &addError)) {
            char addMessage[512] = {0};
            mgl::copyError(addError, addMessage, sizeof(addMessage));
            fprintf(stderr,
                    "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                    addMessage[0] ? addMessage : "unknown error");
        }
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderCreateRenderPipelineStateWithArchiveOwner(
    void* owner_handle,
    void* render_pipeline_descriptor,
    void** pipeline_out,
    int* archive_hit_out,
    char* err,
    size_t errcap) {
    if (archive_hit_out) *archive_hit_out = 0;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    MTL::BinaryArchive* archive = nullptr;
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->binaryArchiveEnabled && owner->binaryArchive) {
            archive = owner->binaryArchive;
            archive->retain();
        }
    }
    int result = archive
        ? mglRenderCreateRenderPipelineStateWithArchive(
              render_pipeline_descriptor, archive, pipeline_out,
              archive_hit_out, err, errcap)
        : mglRenderCreateRenderPipelineState(
              render_pipeline_descriptor, pipeline_out, err, errcap);
    if (archive) archive->release();
    return result;
}

int mglRenderCreateComputePipelineState(void* function,
                                           void** pipeline_out,
                                           char* err,
                                           size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Function* computeFunction =
        static_cast<MTL::Function*>(function);
    if (!computeFunction || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(computeFunction, &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *pipeline_out = pipeline;
    return 0;
}

uint32_t mglRenderComputePipelineMaxTotalThreads(void *pipeline) {
    if (!pipeline) return 0;
    return static_cast<uint32_t>(
        static_cast<MTL::ComputePipelineState *>(pipeline)
            ->maxTotalThreadsPerThreadgroup());
}

int mglRenderCreateBinaryArchive(void* binary_archive_descriptor,
                                    const char* label,
                                    void** binary_archive_out,
                                    char* err,
                                    size_t errcap) {
    if (binary_archive_out) *binary_archive_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchiveDescriptor* descriptor =
        static_cast<MTL::BinaryArchiveDescriptor*>(binary_archive_descriptor);
    if (!descriptor || !binary_archive_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::BinaryArchive* archive =
        renderer.device->newBinaryArchive(descriptor, &nsError);
    if (!archive) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    if (label && label[0]) {
        archive->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *binary_archive_out = archive;
    return 0;
}

int mglRenderSerializeBinaryArchive(void* binary_archive,
                                       void* url,
                                       char* err,
                                       size_t errcap) {
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    NS::URL* archiveURL = static_cast<NS::URL*>(url);
    if (!archive || !archiveURL) return -1;
    NS::Error* nsError = nullptr;
    if (!archive->serializeToURL(archiveURL, &nsError)) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    return 0;
}

int mglRenderSetVisibilityResultMode(void* render_encoder,
                                        uint32_t mode,
                                        uint64_t offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setVisibilityResultMode(
        static_cast<MTL::VisibilityResultMode>(mode),
        static_cast<NS::UInteger>(offset));
    return 0;
}

int mglRenderSetVisibilityResultModeForRenderEncoderOwner(
    void* render_encoder_owner,
    uint32_t mode,
    uint64_t offset) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(render_encoder_owner);
    if (!owner || !owner->encoder || owner->ended) return -1;
    return mglRenderSetVisibilityResultMode(
        owner->encoder, mode, offset);
}

int mglRenderSampleTimestamps(uint64_t* cpu_timestamp_out,
                                 uint64_t* gpu_timestamp_out) {
    if (!cpu_timestamp_out || !gpu_timestamp_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    renderer.device->sampleTimestamps(cpu_timestamp_out, gpu_timestamp_out);
    return 0;
}

int mglRenderCreateQueryStateOwner(uint32_t visibility_slot_count,
                                      void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out || visibility_slot_count == 0) return -1;
    mgl::QueryStateOwner* owner =
        new (std::nothrow) mgl::QueryStateOwner();
    if (!owner) return -1;
    owner->visibilitySlotCount = visibility_slot_count;
    *owner_out = owner;
    return 0;
}

int mglRenderBeginSampleQuery(void* owner_handle,
                                 uint32_t counting,
                                 const char* buffer_label,
                                 void** visibility_buffer_out) {
    if (visibility_buffer_out) *visibility_buffer_out = nullptr;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !visibility_buffer_out ||
        owner->visibilitySlotCount == 0) {
        return -1;
    }
    if (!owner->visibilityBuffer) {
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        const uint64_t byteLength =
            static_cast<uint64_t>(owner->visibilitySlotCount) *
            sizeof(uint64_t);
        owner->visibilityBuffer = renderer.device->newBuffer(
            static_cast<NS::UInteger>(byteLength),
            MTL::ResourceStorageModeShared);
        if (!owner->visibilityBuffer) return -1;
        if (buffer_label && buffer_label[0]) {
            owner->visibilityBuffer->setLabel(
                NS::String::string(buffer_label, NS::UTF8StringEncoding));
        }
    }

    std::memset(owner->visibilityBuffer->contents(), 0,
                owner->visibilityBuffer->length());
    owner->sampleQueryActive = true;
    owner->sampleQueryCounting = counting != 0;
    owner->nextVisibilitySlot = 0;
    *visibility_buffer_out = owner->visibilityBuffer;
    return 0;
}

int mglRenderGetQueryVisibilityBuffer(void* owner_handle,
                                         void** visibility_buffer_out) {
    if (visibility_buffer_out) *visibility_buffer_out = nullptr;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !visibility_buffer_out || !owner->visibilityBuffer) {
        return -1;
    }
    *visibility_buffer_out = owner->visibilityBuffer;
    return 0;
}

void mglRenderEndSampleQuery(void* owner_handle) {
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (owner) owner->sampleQueryActive = false;
}

int mglRenderIsSampleQueryActive(void* owner_handle,
                                    uint32_t* active_out) {
    if (active_out) *active_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !active_out) return -1;
    *active_out = owner->sampleQueryActive ? 1u : 0u;
    return 0;
}

int mglRenderAcquireSampleQuerySlot(void* owner_handle,
                                       uint32_t* mode_out,
                                       uint64_t* offset_out) {
    if (mode_out) *mode_out = 0;
    if (offset_out) *offset_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !mode_out || !offset_out ||
        !owner->sampleQueryActive || owner->visibilitySlotCount == 0) {
        return -1;
    }
    uint32_t slot = owner->nextVisibilitySlot;
    if (slot >= owner->visibilitySlotCount) {
        slot = owner->visibilitySlotCount - 1;
    } else {
        owner->nextVisibilitySlot++;
    }
    *mode_out = static_cast<uint32_t>(
        owner->sampleQueryCounting
            ? MTL::VisibilityResultModeCounting
            : MTL::VisibilityResultModeBoolean);
    *offset_out = static_cast<uint64_t>(slot) * sizeof(uint64_t);
    return 0;
}

int mglRenderGetSampleQueryResult(void* owner_handle,
                                     uint64_t* result_out) {
    if (result_out) *result_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !result_out || !owner->visibilityBuffer) return -1;
    const uint64_t* slots =
        static_cast<const uint64_t*>(owner->visibilityBuffer->contents());
    const uint32_t used = std::min(owner->nextVisibilitySlot,
                                   owner->visibilitySlotCount);
    uint64_t result = 0;
    for (uint32_t index = 0; index < used; ++index) {
        result += slots[index];
    }
    *result_out = result;
    return 0;
}

int mglRenderBeginTimerQuery(void* owner_handle) {
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    owner->timerQueryBeginGPU = gpuTimestamp;
    return 0;
}

int mglRenderEndTimerQuery(void* owner_handle,
                              uint64_t* elapsed_out) {
    if (elapsed_out) *elapsed_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !elapsed_out) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    *elapsed_out = gpuTimestamp >= owner->timerQueryBeginGPU
        ? gpuTimestamp - owner->timerQueryBeginGPU
        : 0;
    return 0;
}

void mglRenderDestroyQueryStateOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderGetOrCreateComputePipeline(
    void* function,
    uint64_t program_instance,
    uint64_t program_generation,
    uint32_t stage,
    int cache_enabled,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!function || !pipeline_out || program_instance == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }

    mgl::ComputePipelineKey key = {
        reinterpret_cast<uintptr_t>(function), program_instance,
        program_generation, stage};
    if (cache_enabled) {
        auto found = renderer.computePipelines.find(key);
        if (found != renderer.computePipelines.end()) {
            if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
                fprintf(stderr,
                        "MGL METALCPP: compute PSO cache hit "
                        "program=%llu generation=%llu stage=%u function=%p\n",
                        static_cast<unsigned long long>(program_instance),
                        static_cast<unsigned long long>(program_generation),
                        stage, function);
            }
            found->second->retain();
            *pipeline_out = found->second;
            return 0;
        }
    }

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(
            static_cast<MTL::Function*>(function), &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    if (cache_enabled) {
        pipeline->retain();
        renderer.computePipelines.emplace(key, pipeline);
    }
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: compute PSO create "
                "program=%llu generation=%llu stage=%u function=%p cache=%d\n",
                static_cast<unsigned long long>(program_instance),
                static_cast<unsigned long long>(program_generation),
                stage, function, cache_enabled != 0);
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglGetOrCreateProgramComputePipeline(Program* program,
                                         int stage,
                                         void** pipeline_out,
                                         char* err,
                                         size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    const bool validStage = stage == _COMPUTE_SHADER ||
                            stage == _TESS_CONTROL_SHADER ||
                            stage == _TESS_EVALUATION_SHADER ||
                            stage == _GEOMETRY_SHADER;
    if (!program || !validStage || !pipeline_out) {
        if (err && errcap) snprintf(err, errcap, "invalid Program or shader stage");
        return -1;
    }
    MGLShaderModule* spirv = &program->modules[stage];
    if (!spirv->mtl_function) {
        if (err && errcap) snprintf(err, errcap, "compiled compute function is unavailable");
        return -1;
    }
    return mglRenderGetOrCreateComputePipeline(
        spirv->mtl_function,
        program->pipeline_cache_instance_id,
        program->pipeline_cache_generation,
        static_cast<uint32_t>(stage), 1, pipeline_out, err, errcap);
}

void mglRenderInvalidateProgramPipelines(uint64_t program_instance) {
    if (program_instance == 0) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    size_t invalidated = 0;
    for (auto it = renderer.computePipelines.begin();
         it != renderer.computePipelines.end();) {
        if (it->first.programInstance == program_instance) {
            if (it->second) it->second->release();
            it = renderer.computePipelines.erase(it);
            invalidated++;
        } else {
            ++it;
        }
    }
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: invalidate compute PSOs "
                "program=%llu count=%zu\n",
                static_cast<unsigned long long>(program_instance), invalidated);
    }
}

namespace {

/* Loads (or returns the cached) MTL::Library for an embedded aux shader asset.
 * Validates the table row (non-empty) and the FNV-1a fingerprint before
 * loading from bytes.  The library is owned by the renderer until shutdown;
 * the returned pointer is borrowed.  Assumes renderer.mutex is held. */
MTL::Library* loadAuxLibraryLocked(mgl::Renderer& renderer,
                                   const unsigned char* bytes,
                                   size_t size,
                                   uint64_t asset_hash,
                                   char* err,
                                   size_t errcap) {
    if (!bytes || size == 0) {
        if (err && errcap) {
            snprintf(err, errcap, "aux shader asset row is empty (table not built?)");
        }
        return nullptr;
    }
    auto found = renderer.auxLibraries.find(asset_hash);
    if (found != renderer.auxLibraries.end()) return found->second;
    const uint64_t computedHash = mglAuxAssetHash(bytes, size);
    if (computedHash != asset_hash) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "aux shader asset hash mismatch (table 0x%016llx, computed 0x%016llx)",
                     static_cast<unsigned long long>(asset_hash),
                     static_cast<unsigned long long>(computedHash));
        }
        return nullptr;
    }
    // Same loading path as mglAirLoadLibrary: dispatch_data -> newLibrary.
    dispatch_data_t dispatchData = dispatch_data_create(
        bytes, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!dispatchData) {
        if (err && errcap) {
            snprintf(err, errcap, "aux shader asset dispatch_data_create failed");
        }
        return nullptr;
    }
    NS::Error* nsError = nullptr;
    MTL::Library* library = renderer.device->newLibrary(dispatchData, &nsError);
#ifdef __OBJC__
    // -fobjc-arc builds (test_metalcpp_smoke) manage dispatch objects
    // automatically; the C++ lib build releases the temporary manually.
    (void)dispatchData;
#else
    dispatch_release(dispatchData);
#endif
    if (!library) {
        mgl::copyError(nsError, err, errcap);
        return nullptr;
    }
    library->retain();
    renderer.auxLibraries.emplace(asset_hash, library);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux shader asset library loaded "
                "hash=0x%016llx bytes=%zu\n",
                static_cast<unsigned long long>(asset_hash), size);
    }
    return library;
}

MTL::Function* newAuxEntryFunction(MTL::Library* library,
                                   const char* entry,
                                   char* err,
                                   size_t errcap) {
    if (!entry) return nullptr;
    MTL::Function* function = library->newFunction(
        NS::String::string(entry, NS::UTF8StringEncoding));
    if (!function && err && errcap) {
        snprintf(err, errcap, "aux shader entry function '%s' not found",
                 entry);
    }
    return function;
}

/* Core of mglRenderGetOrCreateAuxComputePipeline: lookup/create against the
 * renderer-lifetime cache.  Assumes renderer.mutex is held. */
int getOrCreateAuxComputePipelineLocked(mgl::Renderer& renderer,
                                        void* function,
                                        uint32_t kind,
                                        uint64_t variant,
                                        void** pipeline_out,
                                        char* err,
                                        size_t errcap) {
    mgl::AuxComputePipelineKey key = {kind, variant};
    auto found = renderer.auxComputePipelines.find(key);
    if (found != renderer.auxComputePipelines.end()) {
        if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
            fprintf(stderr,
                    "MGL METALCPP: aux compute PSO cache hit "
                    "kind=%u variant=%llu\n",
                    kind, static_cast<unsigned long long>(variant));
        }
        found->second->retain();
        *pipeline_out = found->second;
        return 0;
    }
    if (!function) return 1;

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(
            static_cast<MTL::Function*>(function), &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    pipeline->retain();
    renderer.auxComputePipelines.emplace(key, pipeline);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux compute PSO create "
                "kind=%u variant=%llu function=%p\n",
                kind, static_cast<unsigned long long>(variant), function);
    }
    *pipeline_out = pipeline;
    return 0;
}

/* Core of mglRenderGetOrCreateAuxRenderPipeline: descriptor assembly plus
 * lookup/create against the renderer-lifetime cache.  Assumes renderer.mutex
 * is held. */
int getOrCreateAuxRenderPipelineLocked(mgl::Renderer& renderer,
                                       void* vertex_function,
                                       void* fragment_function,
                                       uint32_t kind,
                                       uint64_t variant,
                                       uint32_t color_format,
                                       uint32_t depth_format,
                                       uint32_t stencil_format,
                                       uint32_t color_write_mask,
                                       int icb_enabled,
                                       uint32_t raster_sample_count,
                                       void** pipeline_out,
                                       char* err,
                                       size_t errcap) {
    mgl::AuxRenderPipelineKey key = {
        kind, variant, color_format, depth_format, stencil_format,
        color_write_mask, raster_sample_count, icb_enabled != 0};
    auto found = renderer.auxRenderPipelines.find(key);
    if (found != renderer.auxRenderPipelines.end()) {
        if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
            fprintf(stderr,
                    "MGL METALCPP: aux render PSO cache hit "
                    "kind=%u variant=%llu\n",
                    kind, static_cast<unsigned long long>(variant));
        }
        found->second->retain();
        *pipeline_out = found->second;
        return 0;
    }
    if (!vertex_function || (!fragment_function &&
                             kind != MGL_RENDER_AUX_RENDER_CLEAR_RECT)) {
        return 1;
    }

    MTL::RenderPipelineDescriptor* descriptor =
        MTL::RenderPipelineDescriptor::alloc()->init();
    if (!descriptor) {
        if (err && errcap) {
            snprintf(err, errcap, "render descriptor allocation failed");
        }
        return -1;
    }
    descriptor->setVertexFunction(
        static_cast<MTL::Function*>(vertex_function));
    descriptor->setFragmentFunction(
        static_cast<MTL::Function*>(fragment_function));
    descriptor->setDepthAttachmentPixelFormat(
        (MTL::PixelFormat)depth_format);
    descriptor->setStencilAttachmentPixelFormat(
        (MTL::PixelFormat)stencil_format);
    descriptor->setRasterSampleCount(raster_sample_count);
    descriptor->setSupportIndirectCommandBuffers(icb_enabled != 0);
    MTL::RenderPipelineColorAttachmentDescriptor* color =
        descriptor->colorAttachments()->object(0);
    color->setPixelFormat((MTL::PixelFormat)color_format);
    color->setWriteMask((MTL::ColorWriteMask)color_write_mask);
    color->setBlendingEnabled(false);

    NS::Error* nsError = nullptr;
    MTL::RenderPipelineState* pipeline =
        renderer.device->newRenderPipelineState(descriptor, &nsError);
    descriptor->release();
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    pipeline->retain();
    renderer.auxRenderPipelines.emplace(key, pipeline);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux render PSO create "
                "kind=%u variant=%llu vs=%p fs=%p\n",
                kind, static_cast<unsigned long long>(variant),
                vertex_function, fragment_function);
    }
    *pipeline_out = pipeline;
    return 0;
}

}  // namespace

int mglRenderGetOrCreateAuxComputePipeline(
    void* function,
    uint32_t kind,
    uint64_t variant,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return getOrCreateAuxComputePipelineLocked(
        renderer, function, kind, variant, pipeline_out, err, errcap);
}

int mglRenderGetOrCreateAuxRenderPipeline(
    void* vertex_function,
    void* fragment_function,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || raster_sample_count == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return getOrCreateAuxRenderPipelineLocked(
        renderer, vertex_function, fragment_function, kind, variant,
        color_format, depth_format, stencil_format, color_write_mask,
        icb_enabled, raster_sample_count, pipeline_out, err, errcap);
}

int mglRenderGetOrCreateAuxComputePipelineFromMetallib(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* entry_name,
    uint32_t kind,
    uint64_t variant,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || !entry_name) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    MTL::Library* library = loadAuxLibraryLocked(
        renderer, bytes, size, asset_hash, err, errcap);
    if (!library) return -1;
    MTL::Function* function =
        newAuxEntryFunction(library, entry_name, err, errcap);
    if (!function) return -1;
    int result = getOrCreateAuxComputePipelineLocked(
        renderer, function, kind, variant, pipeline_out, err, errcap);
    function->release();
    return result;
}

int mglRenderGetOrCreateAuxRenderPipelineFromMetallib(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* vertex_entry,
    const char* fragment_entry,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || raster_sample_count == 0 ||
        !vertex_entry) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    MTL::Library* library = loadAuxLibraryLocked(
        renderer, bytes, size, asset_hash, err, errcap);
    if (!library) return -1;
    MTL::Function* vertexFunction =
        newAuxEntryFunction(library, vertex_entry, err, errcap);
    if (!vertexFunction) return -1;
    MTL::Function* fragmentFunction =
        newAuxEntryFunction(library, fragment_entry, err, errcap);
    if (fragment_entry && !fragmentFunction) {
        vertexFunction->release();
        if (err && errcap && !err[0]) {
            snprintf(err, errcap, "aux shader entry functions missing");
        }
        return -1;
    }
    int result = getOrCreateAuxRenderPipelineLocked(
        renderer, vertexFunction, fragmentFunction, kind, variant,
        color_format, depth_format, stencil_format, color_write_mask,
        icb_enabled, raster_sample_count, pipeline_out, err, errcap);
    vertexFunction->release();
    if (fragmentFunction) fragmentFunction->release();
    return result;
}

int mglRenderCreateAuxFunctions(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* vertex_entry,
    const char* fragment_entry,
    void** vertex_out,
    void** fragment_out,
    char* err,
    size_t errcap) {
    if (vertex_out) *vertex_out = nullptr;
    if (fragment_out) *fragment_out = nullptr;
    if (!vertex_out || !vertex_entry) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    MTL::Library* library = loadAuxLibraryLocked(
        renderer, bytes, size, asset_hash, err, errcap);
    if (!library) return -1;
    MTL::Function* vertexFunction =
        newAuxEntryFunction(library, vertex_entry, err, errcap);
    if (!vertexFunction) return -1;
    MTL::Function* fragmentFunction =
        newAuxEntryFunction(library, fragment_entry, err, errcap);
    if (fragment_entry && !fragmentFunction) {
        vertexFunction->release();
        if (err && errcap && !err[0]) {
            snprintf(err, errcap, "aux shader entry function '%s' not found",
                     fragment_entry);
        }
        return -1;
    }
    *vertex_out = vertexFunction;
    if (fragment_out) *fragment_out = fragmentFunction;
    return 0;
}

void* mglRenderBindingCreate(uint32_t max_texture_slots) {
    if (max_texture_slots == 0 || max_texture_slots > 128) return nullptr;
    mgl::BindingState* state =
        new (std::nothrow) mgl::BindingState(max_texture_slots);
    if (!state) return nullptr;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    renderer.bindingStates.insert(state);
    return state;
}

void mglRenderBindingDestroy(void* binding_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    auto found = renderer.bindingStates.find(state);
    if (found == renderer.bindingStates.end()) return;
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: binding dedup "
                "texture=%llu/%llu sampler=%llu/%llu "
                "viewport=%llu/%llu scissor=%llu/%llu fill=%llu/%llu\n",
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VERTEX_TEXTURE] +
                    state->stats.emitted[MGL_RENDER_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VERTEX_TEXTURE] +
                    state->stats.skipped[MGL_RENDER_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VERTEX_SAMPLER] +
                    state->stats.emitted[MGL_RENDER_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VERTEX_SAMPLER] +
                    state->stats.skipped[MGL_RENDER_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_TRIANGLE_FILL]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_TRIANGLE_FILL]));
    }
    renderer.bindingStates.erase(found);
    delete state;
}

void mglRenderBindingInvalidate(void* binding_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->invalidate();
}

void mglRenderBindingSetValid(void* binding_state, int valid) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->valid = valid != 0;
}

int mglRenderBindingGetValid(void* binding_state, uint32_t* valid_out) {
    if (valid_out) *valid_out = 0;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !valid_out) return -1;
    *valid_out = state->valid ? 1u : 0u;
    return 0;
}

int mglRenderBindingGetTextureSlotMask(void* binding_state,
                                          uint64_t mask_out[2]) {
    if (mask_out) {
        mask_out[0] = 0;
        mask_out[1] = 0;
    }
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !mask_out) return -1;
    mask_out[0] = state->textureSlotMask[0];
    mask_out[1] = state->textureSlotMask[1];
    return 0;
}

namespace {

int recordBufferSlot(std::vector<MTL::Buffer*>& buffers,
                     std::vector<uint64_t>& offsets,
                     uint32_t& mask,
                     void* buffer,
                     uint64_t offset,
                     uint32_t index,
                     bool markPresent) {
    if (index >= buffers.size()) return -1;
    mgl::BindingState::replaceObject(
        buffers[index], static_cast<MTL::Buffer*>(buffer));
    offsets[index] = offset;
    if (markPresent) mask |= 1U << index;
    return 0;
}

int clearBufferSlot(std::vector<MTL::Buffer*>& buffers,
                    std::vector<uint64_t>& offsets,
                    uint32_t index,
                    uint64_t offset) {
    if (index >= buffers.size()) return -1;
    mgl::BindingState::replaceObject(
        buffers[index], static_cast<MTL::Buffer*>(nullptr));
    offsets[index] = offset;
    return 0;
}

} // namespace

int mglRenderBindingRecordVertexBuffer(void* binding_state,
                                          void* buffer,
                                          uint64_t offset,
                                          uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderBindingRecordFragmentBuffer(void* binding_state,
                                            void* buffer,
                                            uint64_t offset,
                                            uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderBindingInvalidateVertexBuffer(void* binding_state,
                                              uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || index >= state->vertexBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->vertexBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->vertexBufferOffsets[index] = UINT64_MAX;
    state->vertexBufferMask |= 1U << index;
    return 0;
}

int mglRenderBindingInvalidateFragmentBuffer(void* binding_state,
                                                uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || index >= state->fragmentBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->fragmentBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->fragmentBufferOffsets[index] = UINT64_MAX;
    state->fragmentBufferMask |= 1U << index;
    return 0;
}

int mglRenderBindingUpdateVertexBuffer(void* binding_state,
                                          void* buffer,
                                          uint64_t offset,
                                          uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderBindingUpdateFragmentBuffer(void* binding_state,
                                            void* buffer,
                                            uint64_t offset,
                                            uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderBindingClearVertexBuffer(void* binding_state,
                                         uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? clearBufferSlot(state->vertexBuffers,
                                   state->vertexBufferOffsets, index, 0) : -1;
}

int mglRenderBindingClearFragmentBuffer(void* binding_state,
                                           uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? clearBufferSlot(state->fragmentBuffers,
                                   state->fragmentBufferOffsets, index, 0) : -1;
}

int mglRenderBindingGetBuffer(void* binding_state,
                                 uint32_t stage,
                                 uint32_t index,
                                 void** buffer_out,
                                 uint64_t* offset_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (offset_out) *offset_out = 0;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !buffer_out || !offset_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    const std::vector<MTL::Buffer*>& buffers =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexBuffers : state->fragmentBuffers;
    const std::vector<uint64_t>& offsets =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexBufferOffsets : state->fragmentBufferOffsets;
    if (index >= buffers.size()) return -1;
    *buffer_out = buffers[index];
    *offset_out = offsets[index];
    return 0;
}

void mglRenderBindingOrVertexBufferMask(void* binding_state,
                                           uint32_t mask) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->vertexBufferMask |= mask;
}

void mglRenderBindingOrFragmentBufferMask(void* binding_state,
                                             uint32_t mask) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->fragmentBufferMask |= mask;
}

void mglRenderBindingSetPipelineState(void* binding_state,
                                         void* pipeline_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) {
        mgl::BindingState::replaceObject(
            state->pipelineState,
            static_cast<MTL::RenderPipelineState*>(pipeline_state));
    }
}

void mglRenderBindingSetDepthStencilState(void* binding_state,
                                             void* depth_stencil_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) {
        mgl::BindingState::replaceObject(
            state->depthStencilState,
            static_cast<MTL::DepthStencilState*>(depth_stencil_state));
    }
}

int mglRenderBindingGetPipelineState(void* binding_state,
                                        void** pipeline_state_out) {
    if (pipeline_state_out) *pipeline_state_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !pipeline_state_out) return -1;
    *pipeline_state_out = state->pipelineState;
    return 0;
}

int mglRenderBindingGetDepthStencilState(
    void* binding_state,
    void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !depth_stencil_state_out) return -1;
    *depth_stencil_state_out = state->depthStencilState;
    return 0;
}

void mglRenderBindingSetCullMode(void* binding_state, uint32_t mode) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->lastCullMode = static_cast<MTL::CullMode>(mode);
}

void mglRenderBindingSetWinding(void* binding_state, uint32_t winding) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->lastWinding = static_cast<MTL::Winding>(winding);
}

void mglRenderBindingSetDepthBias(void* binding_state,
                                     float bias,
                                     float clamp,
                                     float slope_scale) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state) return;
    state->lastDepthBias = bias;
    state->lastDepthBiasClamp = clamp;
    state->lastDepthSlopeScale = slope_scale;
}

void mglRenderBindingSetBlendColor(void* binding_state,
                                      float red,
                                      float green,
                                      float blue,
                                      float alpha) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state) return;
    state->lastBlendColorRed = red;
    state->lastBlendColorGreen = green;
    state->lastBlendColorBlue = blue;
    state->lastBlendColorAlpha = alpha;
}

int mglRenderBindingSetPipelineIfNeeded(void* binding_state,
                                           void* render_encoder,
                                           void* pipeline_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::RenderPipelineState* pipeline =
        static_cast<MTL::RenderPipelineState*>(pipeline_state);
    if (!state || !encoder || !pipeline) return -1;
    const bool emitted = !state->valid || state->pipelineState != pipeline;
    if (emitted) {
        encoder->setRenderPipelineState(pipeline);
        mgl::BindingState::replaceObject(state->pipelineState, pipeline);
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetDepthStencilIfNeeded(void* binding_state,
                                               void* render_encoder,
                                               void* depth_stencil_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::DepthStencilState* depthStencil =
        static_cast<MTL::DepthStencilState*>(depth_stencil_state);
    if (!state || !encoder || !depthStencil) return -1;
    const bool emitted = !state->valid ||
                         state->depthStencilState != depthStencil;
    if (emitted) {
        encoder->setDepthStencilState(depthStencil);
        mgl::BindingState::replaceObject(state->depthStencilState, depthStencil);
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetCullIfNeeded(void* binding_state,
                                       void* render_encoder,
                                       uint32_t mode) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::CullMode cullMode = static_cast<MTL::CullMode>(mode);
    const bool emitted = !state->valid || state->lastCullMode != cullMode;
    if (emitted) {
        encoder->setCullMode(cullMode);
        state->lastCullMode = cullMode;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetWindingIfNeeded(void* binding_state,
                                          void* render_encoder,
                                          uint32_t winding) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::Winding frontWinding = static_cast<MTL::Winding>(winding);
    const bool emitted = !state->valid || state->lastWinding != frontWinding;
    if (emitted) {
        encoder->setFrontFacingWinding(frontWinding);
        state->lastWinding = frontWinding;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetDepthBiasIfNeeded(void* binding_state,
                                            void* render_encoder,
                                            float bias,
                                            float clamp,
                                            float slope_scale) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    const bool emitted = !state->valid || state->lastDepthBias != bias ||
                         state->lastDepthBiasClamp != clamp ||
                         state->lastDepthSlopeScale != slope_scale;
    if (emitted) {
        encoder->setDepthBias(bias, slope_scale, clamp);
        state->lastDepthBias = bias;
        state->lastDepthBiasClamp = clamp;
        state->lastDepthSlopeScale = slope_scale;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetBlendColorIfNeeded(void* binding_state,
                                             void* render_encoder,
                                             float red,
                                             float green,
                                             float blue,
                                             float alpha) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    const bool emitted = !state->valid || state->lastBlendColorRed != red ||
                         state->lastBlendColorGreen != green ||
                         state->lastBlendColorBlue != blue ||
                         state->lastBlendColorAlpha != alpha;
    if (emitted) {
        encoder->setBlendColor(red, green, blue, alpha);
        state->lastBlendColorRed = red;
        state->lastBlendColorGreen = green;
        state->lastBlendColorBlue = blue;
        state->lastBlendColorAlpha = alpha;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetTexture(void* binding_state,
                                 void* render_encoder,
                                 void* texture,
                                 uint32_t stage,
                                 uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    MTL::Texture* newTexture = static_cast<MTL::Texture*>(texture);
    const bool emitted = !state->valid || slots[index] != newTexture;
    const uint32_t setter = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? MGL_RENDER_BINDING_VERTEX_TEXTURE
        : MGL_RENDER_BINDING_FRAGMENT_TEXTURE;
    if (emitted) {
        if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
            encoder->setVertexTexture(newTexture, index);
        } else {
            encoder->setFragmentTexture(newTexture, index);
        }
        mgl::BindingState::replaceObject(slots[index], newTexture);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetSampler(void* binding_state,
                                 void* render_encoder,
                                 void* sampler,
                                 uint32_t stage,
                                 uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    MTL::SamplerState* newSampler = static_cast<MTL::SamplerState*>(sampler);
    const bool emitted = !state->valid || slots[index] != newSampler;
    const uint32_t setter = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? MGL_RENDER_BINDING_VERTEX_SAMPLER
        : MGL_RENDER_BINDING_FRAGMENT_SAMPLER;
    if (emitted) {
        if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
            encoder->setVertexSamplerState(newSampler, index);
        } else {
            encoder->setFragmentSamplerState(newSampler, index);
        }
        mgl::BindingState::replaceObject(slots[index], newSampler);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingGetTexture(void* binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !texture_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    const std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    *texture_out = slots[index];
    return 0;
}

int mglRenderBindingGetSampler(void* binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !sampler_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    const std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    *sampler_out = slots[index];
    return 0;
}

int mglRenderBindingSetViewport(void* binding_state,
                                  void* render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::Viewport viewport = {origin_x, origin_y, width, height, znear, zfar};
    const bool emitted = !state->valid ||
                         !mgl::viewportEqual(state->viewport, viewport);
    if (emitted) {
        encoder->setViewport(viewport);
        state->viewport = viewport;
        state->viewports[0] = viewport;
        state->viewportCount = 1;
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_VIEWPORT, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetViewports(void* binding_state,
                                    void* render_encoder,
                                    const double* viewports,
                                    uint64_t count) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || !viewports || count == 0u ||
        count > MGL_MAX_VIEWPORTS) {
        return -1;
    }
    MTL::Viewport vps[MGL_MAX_VIEWPORTS];
    for (uint64_t i = 0; i < count; i++) {
        vps[i] = {viewports[6 * i], viewports[6 * i + 1],
                  viewports[6 * i + 2], viewports[6 * i + 3],
                  viewports[6 * i + 4], viewports[6 * i + 5]};
    }
    bool same = state->valid && state->viewportCount == count;
    for (uint64_t i = 0; same && i < count; i++) {
        same = mgl::viewportEqual(state->viewports[i], vps[i]);
    }
    if (!same) {
        encoder->setViewports(vps, count);
        for (uint64_t i = 0; i < count; i++) {
            state->viewports[i] = vps[i];
        }
        state->viewportCount = count;
        state->viewport = vps[0];
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_VIEWPORT, !same);
    return same ? 0 : 1;
}

int mglRenderBindingSetScissor(void* binding_state,
                                 void* render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::ScissorRect scissor = {
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(width), static_cast<NS::UInteger>(height)};
    const bool emitted = !state->valid ||
                         !mgl::scissorEqual(state->scissor, scissor);
    if (emitted) {
        encoder->setScissorRect(scissor);
        state->scissor = scissor;
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_SCISSOR, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetTriangleFill(void* binding_state,
                                      void* render_encoder,
                                      uint32_t mode) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || mode > MTL::TriangleFillModeLines) return -1;
    MTL::TriangleFillMode fillMode = static_cast<MTL::TriangleFillMode>(mode);
    const bool emitted = !state->valid || state->triangleFillMode != fillMode;
    if (emitted) {
        encoder->setTriangleFillMode(fillMode);
        state->triangleFillMode = fillMode;
    }
    mgl::recordBindingResult(
        *state, MGL_RENDER_BINDING_TRIANGLE_FILL, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingGetStats(void* binding_state,
                               MGLRenderBindingStats* stats_out) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !stats_out) return -1;
    memcpy(stats_out, &state->stats, sizeof(*stats_out));
    return 0;
}

int mglRenderSetComputePipelineState(void* compute_encoder,
                                        void* pipeline_state) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    MTL::ComputePipelineState* pipeline =
        static_cast<MTL::ComputePipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setComputePipelineState(pipeline);
    return 0;
}

int mglRenderSetComputeBuffer(void* compute_encoder,
                                 void* buffer,
                                 uint64_t offset,
                                 uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setBuffer(static_cast<MTL::Buffer*>(buffer),
                       static_cast<NS::UInteger>(offset), index);
    return 0;
}

int mglRenderSetComputeTexture(void* compute_encoder,
                                  void* texture,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setTexture(static_cast<MTL::Texture*>(texture), index);
    return 0;
}

int mglRenderSetComputeSampler(void* compute_encoder,
                                  void* sampler,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setSamplerState(static_cast<MTL::SamplerState*>(sampler), index);
    return 0;
}

int mglRenderSetComputeBytes(void* compute_encoder,
                                const void* bytes,
                                size_t length,
                                uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || (!bytes && length != 0)) return -1;
    encoder->setBytes(bytes, static_cast<NS::UInteger>(length), index);
    return 0;
}

int mglRenderSetComputeThreadgroupMemoryLength(void* compute_encoder,
                                                  uint64_t length,
                                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setThreadgroupMemoryLength(static_cast<NS::UInteger>(length),
                                        index);
    return 0;
}


int mglRenderEncodeComputeBindingSnapshot(
    void* compute_encoder,
    const MGLRenderComputeBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!compute_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (snapshot->op_count >
        MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->op_count; i++) {
        const MGLRenderComputeBindingOp* op = &snapshot->ops[i];
        if (op->kind == 0) {
            /* kind 0: set buffer; NULL buffer clears the slot. */
            encoder->setBuffer(static_cast<MTL::Buffer*>(op->buffer),
                               static_cast<NS::UInteger>(op->offset),
                               op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null compute bytes op %u", i);
                }
                return -1;
            }
            encoder->setBytes(op->bytes, op->length, op->index);
        } else if (op->kind == 2) {
            /* kind 2: set texture; NULL clears the slot. */
            encoder->setTexture(static_cast<MTL::Texture*>(op->buffer),
                                op->index);
        } else if (op->kind == 3) {
            /* kind 3: set sampler state; NULL clears the slot. */
            encoder->setSamplerState(
                static_cast<MTL::SamplerState*>(op->buffer), op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad compute op kind %u", op->kind);
            }
            return -1;
        }
    }
    return 0;
}

int mglRenderDispatchCompute(void* compute_encoder,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z,
                                uint32_t threads_x,
                                uint32_t threads_y,
                                uint32_t threads_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || groups_x == 0 || groups_y == 0 || groups_z == 0 ||
        threads_x == 0 || threads_y == 0 || threads_z == 0) {
        return -1;
    }
    MTL::Size groups = MTL::Size(groups_x, groups_y, groups_z);
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    encoder->dispatchThreadgroups(groups, threads);
    return 0;
}

int mglRenderDispatchComputeIndirect(void* compute_encoder,
                                        void* indirect_buffer,
                                        uint64_t indirect_offset,
                                        uint32_t threads_x,
                                        uint32_t threads_y,
                                        uint32_t threads_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    MTL::Buffer* buffer = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !buffer || threads_x == 0 || threads_y == 0 ||
        threads_z == 0) {
        return -1;
    }
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    encoder->dispatchThreadgroups(buffer,
                                  static_cast<NS::UInteger>(indirect_offset),
                                  threads);
    return 0;
}


int mglRenderDispatchComputePlan(
    void* compute_encoder,
    const MGLRenderComputePlan* plan,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!compute_encoder || !plan) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    const uint32_t local_x = plan->local_x ? plan->local_x : 1u;
    const uint32_t local_y = plan->local_y ? plan->local_y : 1u;
    const uint32_t local_z = plan->local_z ? plan->local_z : 1u;
    MTL::Size threads = MTL::Size(local_x, local_y, local_z);

    if (plan->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        encoder->dispatchThreadgroups(
            MTL::Size(plan->groups_x, plan->groups_y, plan->groups_z),
            threads);
        return 0;
    }
    if (plan->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        MTL::Buffer* buffer = static_cast<MTL::Buffer*>(plan->indirect_buffer);
        if (!buffer) {
            if (err && errcap) {
                snprintf(err, errcap, "null indirect buffer");
            }
            return -1;
        }
        encoder->dispatchThreadgroups(
            buffer, static_cast<NS::UInteger>(plan->indirect_offset), threads);
        return 0;
    }
    if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                (unsigned)plan->dispatch_kind);
    return -1;
}

int mglRenderAppendComputeBindingSnapshotToPlan(
    MGLRenderComputeExecutionPlan* plan,
    const MGLRenderComputeBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!plan || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad compute plan args");
        return -1;
    }
    if (snapshot->op_count > MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS ||
        plan->binding_op_count > MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS ||
        snapshot->op_count >
            MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS -
                plan->binding_op_count) {
        if (err && errcap) snprintf(err, errcap, "compute execution op overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->op_count; i++) {
        const MGLRenderComputeBindingOp* op = &snapshot->ops[i];
        if (op->kind > 3u || (op->kind == 1u && !op->bytes)) {
            if (err && errcap) {
                snprintf(err, errcap, "invalid compute binding op %u", i);
            }
            return -1;
        }
        plan->binding_ops[plan->binding_op_count++] = *op;
    }
    return 0;
}

int mglRenderAppendComputeDispatchToPlan(
    MGLRenderComputeExecutionPlan* plan,
    const MGLRenderComputePlan* dispatch,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!plan || !dispatch ||
        plan->dispatch_op_count >=
            MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES) {
        if (err && errcap) snprintf(err, errcap, "compute dispatch sequence overflow");
        return -1;
    }
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        if (!dispatch->groups_x || !dispatch->groups_y || !dispatch->groups_z) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
    } else if (dispatch->dispatch_kind ==
               MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        if (!dispatch->indirect_buffer) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
    } else {
        if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                    dispatch->dispatch_kind);
        return -1;
    }
    MGLRenderComputeDispatchEntry* entry =
        &plan->dispatch_ops[plan->dispatch_op_count++];
    entry->binding_op_count = plan->binding_op_count;
    entry->dispatch = *dispatch;
    return 0;
}

static int mglRenderEncodeComputeDispatchOnEncoder(
    MTL::ComputeCommandEncoder* encoder,
    const MGLRenderComputePlan* dispatch,
    char* err,
    size_t errcap) {
    if (!encoder || !dispatch) return -1;
    const uint32_t local_x = dispatch->local_x ? dispatch->local_x : 1u;
    const uint32_t local_y = dispatch->local_y ? dispatch->local_y : 1u;
    const uint32_t local_z = dispatch->local_z ? dispatch->local_z : 1u;
    const MTL::Size threads(local_x, local_y, local_z);
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        if (!dispatch->groups_x || !dispatch->groups_y || !dispatch->groups_z) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
        encoder->dispatchThreadgroups(
            MTL::Size(dispatch->groups_x, dispatch->groups_y,
                      dispatch->groups_z),
            threads);
        return 0;
    }
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        MTL::Buffer* indirect =
            static_cast<MTL::Buffer*>(dispatch->indirect_buffer);
        if (!indirect) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
        encoder->dispatchThreadgroups(
            indirect, static_cast<NS::UInteger>(dispatch->indirect_offset),
            threads);
        return 0;
    }
    if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                dispatch->dispatch_kind);
    return -1;
}

int mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(
    void* command_buffer_owner,
    const MGLRenderComputeExecutionPlan* plan,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!command_buffer_owner || !plan || !plan->pipeline) {
        if (err && errcap) snprintf(err, errcap, "bad compute execution plan");
        return -1;
    }
    if (plan->binding_op_count > MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "compute execution op overflow");
        return -1;
    }
    if (plan->dispatch_op_count >
        MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES) {
        if (err && errcap) snprintf(err, errcap, "compute dispatch sequence overflow");
        return -1;
    }
    const uint32_t validBarrierScope =
        MGL_RENDER_COMPUTE_BARRIER_BUFFERS |
        MGL_RENDER_COMPUTE_BARRIER_TEXTURES |
        MGL_RENDER_COMPUTE_BARRIER_RENDER_TARGETS;
    if (plan->barrier_scope & ~validBarrierScope) {
        if (err && errcap) snprintf(err, errcap, "invalid compute barrier scope");
        return -1;
    }
    for (uint32_t i = 0; i < plan->binding_op_count; i++) {
        const MGLRenderComputeBindingOp* op = &plan->binding_ops[i];
        if (op->kind > 3u || (op->kind == 1u && !op->bytes)) {
            if (err && errcap) {
                snprintf(err, errcap, "invalid compute binding op %u", i);
            }
            return -1;
        }
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind ==
            MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
        (!plan->dispatch.groups_x || !plan->dispatch.groups_y ||
         !plan->dispatch.groups_z)) {
        if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
        return -1;
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind ==
            MGL_RENDER_COMPUTE_DISPATCH_INDIRECT &&
        !plan->dispatch.indirect_buffer) {
        if (err && errcap) snprintf(err, errcap, "null indirect buffer");
        return -1;
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind !=
            MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
        plan->dispatch.dispatch_kind !=
            MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        if (err && errcap) {
            snprintf(err, errcap, "bad dispatch kind %u",
                     plan->dispatch.dispatch_kind);
        }
        return -1;
    }
    uint32_t previousDispatchBindingCount = 0u;
    for (uint32_t i = 0; i < plan->dispatch_op_count; i++) {
        const MGLRenderComputeDispatchEntry* entry = &plan->dispatch_ops[i];
        if (entry->binding_op_count < previousDispatchBindingCount ||
            entry->binding_op_count > plan->binding_op_count) {
            if (err && errcap) snprintf(err, errcap, "invalid dispatch ordering %u", i);
            return -1;
        }
        if (entry->dispatch.dispatch_kind ==
                MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
            (!entry->dispatch.groups_x || !entry->dispatch.groups_y ||
             !entry->dispatch.groups_z)) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
        if (entry->dispatch.dispatch_kind ==
                MGL_RENDER_COMPUTE_DISPATCH_INDIRECT &&
            !entry->dispatch.indirect_buffer) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
        if (entry->dispatch.dispatch_kind !=
                MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
            entry->dispatch.dispatch_kind !=
                MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
            if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                        entry->dispatch.dispatch_kind);
            return -1;
        }
        previousDispatchBindingCount = entry->binding_op_count;
    }

    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    MTL::CommandBuffer* command_buffer = owner->current;
    if (!command_buffer) {
        if (err && errcap) snprintf(err, errcap, "no current command buffer");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        if (err && errcap) snprintf(err, errcap, "compute encoder failed");
        return -1;
    }

    encoder->setComputePipelineState(
        static_cast<MTL::ComputePipelineState*>(plan->pipeline));
    uint32_t nextDispatch = 0u;
    for (uint32_t i = 0; i <= plan->binding_op_count; i++) {
        while (nextDispatch < plan->dispatch_op_count &&
               plan->dispatch_ops[nextDispatch].binding_op_count == i) {
            if (mglRenderEncodeComputeDispatchOnEncoder(
                    encoder, &plan->dispatch_ops[nextDispatch].dispatch,
                    err, errcap) != 0) {
                encoder->endEncoding();
                return -1;
            }
            nextDispatch++;
        }
        if (i == plan->binding_op_count) break;
        const MGLRenderComputeBindingOp* op = &plan->binding_ops[i];
        switch (op->kind) {
            case 0u:
                encoder->setBuffer(static_cast<MTL::Buffer*>(op->buffer),
                                   static_cast<NS::UInteger>(op->offset),
                                   op->index);
                break;
            case 1u:
                if (!op->bytes) {
                    encoder->endEncoding();
                    if (err && errcap) snprintf(err, errcap,
                                                "null compute bytes op %u", i);
                    return -1;
                }
                encoder->setBytes(op->bytes, op->length, op->index);
                break;
            case 2u:
                encoder->setTexture(static_cast<MTL::Texture*>(op->buffer),
                                    op->index);
                break;
            case 3u:
                encoder->setSamplerState(
                    static_cast<MTL::SamplerState*>(op->buffer), op->index);
                break;
            default:
                encoder->endEncoding();
                if (err && errcap) snprintf(err, errcap,
                                            "bad compute op kind %u", op->kind);
                return -1;
        }
    }

    if (plan->dispatch_op_count == 0 &&
        mglRenderEncodeComputeDispatchOnEncoder(
            encoder, &plan->dispatch, err, errcap) != 0) {
        encoder->endEncoding();
        return -1;
    }
    if (plan->barrier_scope != MGL_RENDER_COMPUTE_BARRIER_NONE) {
        encoder->memoryBarrier(static_cast<MTL::BarrierScope>(
            plan->barrier_scope));
    }
    encoder->endEncoding();
    return 0;
}

extern "C"
int mglRenderExecuteComputeExecutionPlan(
    void* command_buffer_owner,
    void* recovery_owner,
    const MGLRenderComputeExecutionPlan* plan,
    const MGLRenderCopyBackEntry* copy_backs,
    uint32_t copy_back_count,
    uint32_t require_cpu_visibility,
    MGLRenderComputeExecutionResult* result,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (result) {
        memset(result, 0, sizeof(*result));
        result->failed_copy_back_index = copy_back_count;
    }
    if (!command_buffer_owner || !plan || !result ||
        (!copy_backs && copy_back_count)) {
        if (err && errcap) snprintf(err, errcap, "bad compute transaction args");
        return -1;
    }

    /* Reject malformed copy-backs before opening the compute encoder. */
    if (mglRenderEncodeStageBindingCopyBacks(
            copy_backs, copy_back_count, nullptr) != 0) {
        if (err && errcap) snprintf(err, errcap, "invalid compute copy-back");
        for (uint32_t i = 0; i < copy_back_count; i++) {
            const MGLRenderCopyBackEntry& entry = copy_backs[i];
            if (!entry.length) continue;
            MTL::Buffer* temporary = static_cast<MTL::Buffer*>(
                const_cast<void*>(entry.temporary));
            MTL::Buffer* destination = static_cast<MTL::Buffer*>(
                const_cast<void*>(entry.destination));
            if (!temporary || !destination ||
                entry.length > temporary->length() ||
                entry.destination_offset > destination->length() ||
                entry.length > destination->length() - entry.destination_offset) {
                result->failed_copy_back_index = i;
                break;
            }
        }
        return -1;
    }
    if (mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(
            command_buffer_owner, plan, err, errcap) != 0) {
        return -1;
    }

    bool has_copies = false;
    for (uint32_t i = 0; i < copy_back_count; i++) {
        has_copies = has_copies || copy_backs[i].length != 0;
    }
    if (!has_copies && !require_cpu_visibility) return 0;

    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    MTL::CommandBuffer* command_buffer = owner->current;
    if (!command_buffer) {
        if (err && errcap) snprintf(err, errcap, "no compute command buffer");
        return -1;
    }
    if (has_copies) {
        MTL::BlitCommandEncoder* blit = command_buffer->blitCommandEncoder();
        if (!blit) {
            if (err && errcap) snprintf(err, errcap, "compute copy-back encoder failed");
            return -1;
        }
        int encode_result = mglRenderEncodeStageBindingCopyBacks(
            copy_backs, copy_back_count, blit);
        blit->endEncoding();
        if (encode_result != 0) {
            if (err && errcap) snprintf(err, errcap, "compute copy-back encode failed");
            return -1;
        }
    }

    result->submitted = 1u;
    if (mglRenderCommitCommandBufferTransaction(
            command_buffer_owner, nullptr, command_buffer, recovery_owner,
            1u, &result->transaction) != 0 ||
        result->transaction.has_error) {
        if (err && errcap) snprintf(err, errcap, "compute submit/wait failed");
        return -1;
    }
    if (mglRenderCopyBackCPUPrefix(
            copy_backs, copy_back_count,
            &result->failed_copy_back_index) != 0) {
        if (err && errcap) snprintf(err, errcap, "compute CPU prefix sync failed");
        return -1;
    }
    result->cpu_prefix_synchronized = 1u;
    return 0;
}

int mglRenderDispatchComputeThreads(void* compute_encoder,
                                       uint32_t threads_x,
                                       uint32_t threads_y,
                                       uint32_t threads_z,
                                       uint32_t group_x,
                                       uint32_t group_y,
                                       uint32_t group_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || threads_x == 0 || threads_y == 0 || threads_z == 0 ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        return -1;
    }
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    MTL::Size threadgroup = MTL::Size(group_x, group_y, group_z);
    encoder->dispatchThreads(threads, threadgroup);
    return 0;
}


int mglRenderBeginComputeDispatch(
    void* command_buffer,
    const MGLRenderComputeDispatchSetup* setup,
    void** compute_encoder_out,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    if (!command_buffer || !setup || !compute_encoder_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    if (setup->buffer_count > MGL_RENDER_COMPUTE_DISPATCH_MAX_BUFFERS ||
        setup->bytes_count > MGL_RENDER_COMPUTE_DISPATCH_MAX_BYTES) {
        if (err && errcap) snprintf(err, errcap, "setup count overflow");
        return -1;
    }
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();
    if (!encoder) {
        if (err && errcap) snprintf(err, errcap, "compute encoder failed");
        return -1;
    }
    if (setup->pipeline) {
        encoder->setComputePipelineState(
            static_cast<MTL::ComputePipelineState*>(setup->pipeline));
    }
    for (uint32_t i = 0; i < setup->buffer_count; i++) {
        const MGLRenderComputeBufferEntry* entry = &setup->buffers[i];
        if (!entry->buffer) {
            encoder->endEncoding();
            if (err && errcap) {
                snprintf(err, errcap, "null compute buffer entry %u", i);
            }
            return -1;
        }
        encoder->setBuffer(
            static_cast<MTL::Buffer*>(entry->buffer),
            static_cast<NS::UInteger>(entry->offset), entry->index);
    }
    for (uint32_t i = 0; i < setup->bytes_count; i++) {
        const MGLRenderComputeBytesEntry* entry = &setup->bytes[i];
        if (!entry->bytes || entry->length == 0) {
            encoder->endEncoding();
            if (err && errcap) {
                snprintf(err, errcap, "null compute bytes entry %u", i);
            }
            return -1;
        }
        encoder->setBytes(entry->bytes, entry->length, entry->index);
    }
    *compute_encoder_out = encoder;
    return 0;
}

int mglRenderBeginComputeDispatchForCommandBufferOwner(
    void* command_buffer_owner,
    const MGLRenderComputeDispatchSetup* setup,
    void** compute_encoder_out,
    char* err,
    size_t errcap) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) {
        if (err && errcap) snprintf(err, errcap, "missing current command buffer");
        if (compute_encoder_out) *compute_encoder_out = nullptr;
        return -1;
    }
    return mglRenderBeginComputeDispatch(
        owner->current, setup, compute_encoder_out, err, errcap);
}

int mglRenderEndComputeDispatch(void* compute_encoder,
                                   const uint32_t groups[3],
                                   const uint32_t threads[3],
                                   char* err,
                                   size_t errcap) {
    if (err && errcap) err[0] = '\0';
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || !groups || !threads) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::Size groupsSize =
        MTL::Size(groups[0], groups[1], groups[2]);
    MTL::Size threadsSize =
        MTL::Size(threads[0], threads[1], threads[2]);
    encoder->dispatchThreadgroups(groupsSize, threadsSize);
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateComputeEncoder(void* command_buffer,
                                     void** compute_encoder_out) {
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !compute_encoder_out) return -1;
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();
    if (!encoder) return -1;
    *compute_encoder_out = encoder;
    return 0;
}

int mglRenderEndComputeEncoder(void* compute_encoder) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateCommandBuffer(void* command_queue,
                                    void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!queue || !command_buffer_out) return -1;
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) return -1;
    *command_buffer_out = commandBuffer;
    return 0;
}

int mglRenderGetCommandBufferState(
    void* command_buffer,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    return mgl::snapshotCommandBufferState(
        static_cast<MTL::CommandBuffer*>(command_buffer), state_out);
}

const char *mglRenderCommandBufferErrorDescription(
    const MGLRenderCommandBufferState* state) {
    return state && state->has_error && state->error_description[0]
        ? state->error_description : "unknown command-buffer error";
}

uint32_t mglRenderCommandBufferStatus(void* command_buffer) {
    MGLRenderCommandBufferState state = {};
    return mglRenderGetCommandBufferState(command_buffer, &state) == 0
        ? state.status : static_cast<uint32_t>(MTL::CommandBufferStatusError);
}

int mglRenderGetCommandBufferLabel(const void *command_buffer,
                                      char *label_out,
                                      size_t label_capacity) {
    if (label_out && label_capacity) label_out[0] = '\0';
    const MTL::CommandBuffer *cb =
        static_cast<const MTL::CommandBuffer *>(command_buffer);
    if (!cb || !label_out || !label_capacity) return -1;
    NS::String *label = cb->label();
    const char *utf8 = label ? label->utf8String() : nullptr;
    std::snprintf(label_out, label_capacity, "%s",
                  utf8 && utf8[0] ? utf8 : "(no-label)");
    return 0;
}

int mglRenderSetCommandBufferLabel(void *command_buffer,
                                      const char *label) {
    MTL::CommandBuffer* object = static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!object || !label) return -1;
    object->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    return 0;
}

int mglRenderClassifyCommandBufferCommit(
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCommitDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    if (!state || !decision_out) return -1;

    /* Preserve commitCommandBufferWithAGXRecovery's original ordering. Since
     * Error follows Committed numerically, Error is classified as the legacy
     * already-committed skip rather than changing recovery behavior here. */
    if (state->status >=
        static_cast<uint32_t>(MTL::CommandBufferStatusCommitted)) {
        decision_out->action =
            MGL_RENDER_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED;
    } else {
        decision_out->action =
            MGL_RENDER_COMMAND_BUFFER_COMMIT_PROCEED;
    }
    return 0;
}

namespace {

double commandRecoveryNowSeconds() {
    using Clock = std::chrono::system_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

void snapshotCommandRecoveryOwner(
    mgl::CommandBufferRecoveryOwner* owner,
    MGLRenderCommandRecoverySnapshot* state) {
    if (!owner || !state) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::snapshotCommandRecovery(*owner, state);
}

int applyCommandRecoveryFailure(
    mgl::CommandBufferRecoveryOwner* owner,
    const MGLRenderCommandBufferState* state,
    bool request_reset,
    MGLRenderCommandBufferTransaction* transaction) {
    if (!owner || !transaction) return -1;
    if (transaction->recovery_error_recorded) {
        snapshotCommandRecoveryOwner(owner, &transaction->recovery);
        return 0;
    }

    MGLRenderCommandBufferCompletionDecision decision = {};
    if (state && mglRenderClassifyCommandBufferCompletion(
                     state, &decision) != 0) {
        return -1;
    }
    const bool driver_rejection = decision.is_driver_rejection != 0;
    {
        std::lock_guard<std::mutex> lock(owner->mutex);
        owner->consecutiveErrors++;
        owner->consecutiveSuccesses = 0;
        owner->lastErrorTime = commandRecoveryNowSeconds();
        mgl::snapshotCommandRecovery(*owner, &transaction->recovery);
    }
    transaction->has_error = 1u;
    transaction->is_driver_rejection = driver_rejection ? 1u : 0u;
    transaction->device_reset_requested =
        (request_reset || driver_rejection) ? 1u : 0u;
    transaction->recovery_error_recorded = 1u;
    return 0;
}

struct CommandRecoveryCompletionContext {
    ~CommandRecoveryCompletionContext() {
        mgl::releaseCommandRecoveryOwner(owner);
    }

    void retain() {
        references.fetch_add(1u, std::memory_order_relaxed);
    }

    void release() {
        if (references.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            delete this;
        }
    }

    std::atomic<uint32_t> references{1u};
    std::mutex applyMutex;
    mgl::CommandBufferRecoveryOwner* owner = nullptr;
    bool completionApplied = false;
    bool completionHadError = false;
    bool completionWasDriverRejection = false;
    bool transactionFailureApplied = false;
};

void processCommandRecoveryCompletionLocked(
    CommandRecoveryCompletionContext* completion,
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCompletionResult* result_out) {
    if (!completion || !completion->owner || !state) return;
    MGLRenderCommandBufferCompletionDecision decision = {};
    if (mglRenderClassifyCommandBufferCompletion(state, &decision) != 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(completion->applyMutex);
    if (completion->transactionFailureApplied ||
        completion->completionApplied) {
        if (result_out) {
            result_out->decision = decision;
            snapshotCommandRecoveryOwner(completion->owner,
                                         &result_out->state);
        }
        return;
    }

    MGLRenderCommandBufferCompletionResult result = {};
    if (mglRenderProcessCommandBufferCompletion(
            completion->owner, state, commandRecoveryNowSeconds(), &result) != 0) {
        return;
    }
    completion->completionApplied = true;
    completion->completionHadError = result.decision.has_error != 0;
    completion->completionWasDriverRejection =
        result.decision.is_driver_rejection != 0;
    if (result.decision.is_driver_rejection) {
        std::lock_guard<std::mutex> ownerLock(completion->owner->mutex);
        completion->owner->resetRequested = true;
    }
    if (result_out) *result_out = result;
}

void commandRecoveryCompletion(void* context,
                               const MGLRenderCommandBufferState* state) {
    processCommandRecoveryCompletionLocked(
        static_cast<CommandRecoveryCompletionContext*>(context), state,
        nullptr);
}

void destroyCommandRecoveryCompletionContext(void* context) {
    CommandRecoveryCompletionContext* completion =
        static_cast<CommandRecoveryCompletionContext*>(context);
    if (completion) completion->release();
}

int addCommandBufferRecoveryCompletion(
    void* command_buffer,
    void* recovery_owner,
    CommandRecoveryCompletionContext** context_out) {
    if (context_out) *context_out = nullptr;
    if (!command_buffer || !recovery_owner) return -1;
    CommandRecoveryCompletionContext* context =
        new (std::nothrow) CommandRecoveryCompletionContext();
    if (!context) return -1;
    context->owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(recovery_owner);
    mgl::retainCommandRecoveryOwner(context->owner);
    if (context_out) {
        context->retain();
        *context_out = context;
    }
    int result = mglRenderAddCommandBufferCompletion(
        command_buffer, commandRecoveryCompletion, context,
        destroyCommandRecoveryCompletionContext);
    if (result != 0) {
        if (context_out) {
            *context_out = nullptr;
            context->release();
        }
        context->release();
    }
    return result;
}

int applyCommandRecoveryTransactionFailure(
    CommandRecoveryCompletionContext* completion,
    mgl::CommandBufferRecoveryOwner* owner,
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferTransaction* transaction) {
    if (!owner || !transaction) return 0;
    if (!completion) {
        return applyCommandRecoveryFailure(owner, state, true, transaction);
    }

    std::lock_guard<std::mutex> lock(completion->applyMutex);
    if (completion->transactionFailureApplied ||
        (completion->completionApplied && completion->completionHadError)) {
        snapshotCommandRecoveryOwner(owner, &transaction->recovery);
        transaction->has_error = 1u;
        transaction->is_driver_rejection =
            completion->completionWasDriverRejection ? 1u : 0u;
        transaction->device_reset_requested = 1u;
        transaction->recovery_error_recorded = 1u;
        return 0;
    }
    int result = applyCommandRecoveryFailure(owner, state, true, transaction);
    if (result == 0) completion->transactionFailureApplied = true;
    return result;
}

struct ScopedRecoveryCompletionContext {
    ~ScopedRecoveryCompletionContext() {
        if (context) context->release();
    }
    CommandRecoveryCompletionContext* context = nullptr;
};

}  // namespace

extern "C"
int mglRenderCommitCommandBufferTransaction(
    void* owner_handle,
    void** submission_handle,
    void* command_buffer,
    void* recovery_owner,
    uint32_t wait_for_completion,
    MGLRenderCommandBufferTransaction* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!command_buffer || !result_out) return -1;
    result_out->result = MGL_RENDER_COMMAND_BUFFER_TRANSACTION_ERROR;

    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    mgl::CommandBufferRecoveryOwner* recovery =
        static_cast<mgl::CommandBufferRecoveryOwner*>(recovery_owner);
    ScopedRecoveryCompletionContext recovery_completion;
    if (mgl::snapshotCommandBufferState(command, &result_out->before) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, nullptr, result_out);
        return -1;
    }

    MGLRenderCommandBufferCommitDecision decision = {};
    if (mglRenderClassifyCommandBufferCommit(
            &result_out->before, &decision) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        return -1;
    }
    if (decision.action ==
        MGL_RENDER_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED) {
        result_out->result =
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_SKIPPED;
        result_out->after = result_out->before;
        if (result_out->before.has_error && recovery) {
            applyCommandRecoveryFailure(
                recovery, &result_out->before, false, result_out);
        } else if (recovery) {
            snapshotCommandRecoveryOwner(recovery, &result_out->recovery);
        }
        return 0;
    }

    bool commit_guard_acquired = false;
    if (owner_handle) {
        int guard = mglRenderCommandBufferOwnerBeginCommit(owner_handle);
        if (guard < 0) {
            applyCommandRecoveryTransactionFailure(
                nullptr, recovery, &result_out->before, result_out);
            return -1;
        }
        if (guard == 0) {
            result_out->result =
                MGL_RENDER_COMMAND_BUFFER_TRANSACTION_NESTED;
            result_out->after = result_out->before;
            if (recovery) {
                snapshotCommandRecoveryOwner(recovery,
                                             &result_out->recovery);
            }
            return 0;
        }
        commit_guard_acquired = true;
    }

    struct CommitGuard {
        void* owner = nullptr;
        bool* acquired = nullptr;
        ~CommitGuard() {
            if (owner && acquired && *acquired) {
                mglRenderCommandBufferOwnerEndCommit(owner);
                *acquired = false;
            }
        }
    } commit_guard{owner_handle, &commit_guard_acquired};

    if (submission_handle && *submission_handle &&
        mglRenderCommandBufferSubmissionMatchesBuffer(
            *submission_handle, command_buffer) != 1) {
        result_out->after = result_out->before;
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        if (!recovery) result_out->has_error = 1u;
        return -1;
    }

    int commit_result = -1;
    bool committed = false;
    if (recovery &&
        addCommandBufferRecoveryCompletion(
            command_buffer, recovery_owner,
            &recovery_completion.context) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        return -1;
    }
    result_out->completion_registered = recovery ? 1u : 0u;
    try {
        if (submission_handle && *submission_handle &&
            mglRenderCommandBufferSubmissionMatchesBuffer(
                *submission_handle, command_buffer) == 1) {
            result_out->used_submission = 1u;
            commit_result = mglRenderCommitCommandBufferSubmission(
                submission_handle);
        } else {
            command->commit();
            commit_result = 0;
        }
        committed = commit_result == 0;
    } catch (...) {
        commit_result = -1;
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, &result_out->before,
            result_out);
        if (!recovery) result_out->has_error = 1u;
    }

    if (mgl::snapshotCommandBufferState(command, &result_out->after) != 0) {
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, nullptr, result_out);
        if (!recovery) result_out->has_error = 1u;
    }
    if (!committed) {
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, &result_out->after,
            result_out);
        if (!recovery) result_out->has_error = 1u;
        return -1;
    }
    if (owner_handle) {
        mgl::setLastSubmitted(
            static_cast<mgl::CommandBufferOwner*>(owner_handle), command);
    }
    result_out->result =
        MGL_RENDER_COMMAND_BUFFER_TRANSACTION_COMMITTED;
    result_out->needs_new_command_buffer = 1u;
    if (wait_for_completion) {
        result_out->waited = 1u;
        try {
            command->waitUntilCompleted();
        } catch (...) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        if (mgl::snapshotCommandBufferState(
                command, &result_out->completion) != 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        MGLRenderCommandBufferCompletionDecision completionDecision = {};
        if (mglRenderClassifyCommandBufferCompletion(
                &result_out->completion, &completionDecision) != 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery,
                &result_out->completion, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        result_out->has_error = completionDecision.has_error;
        result_out->is_driver_rejection =
            completionDecision.is_driver_rejection;
        if (recovery_completion.context) {
            MGLRenderCommandBufferCompletionResult completionResult = {};
            processCommandRecoveryCompletionLocked(
                recovery_completion.context, &result_out->completion,
                &completionResult);
            result_out->recovery = completionResult.state;
            result_out->recovery_error_recorded =
                completionDecision.has_error ? 1u : 0u;
        }
        if (recovery) {
            result_out->device_reset_requested =
                mglRenderCommandRecoveryTakeResetRequest(
                    recovery_owner) == 1 ? 1u : 0u;
        }
        if (result_out->has_error) return -1;
    }
    /* Taking a submission leaves the owner without a current command buffer.
     * Owners created from the C++ queue rotate the next buffer here so the
     * lifecycle transaction owns creation as well as submission. Adopted ObjC
     * buffers have no queue and keep the legacy reset adapter. */
    if (owner_handle) {
        void* next = nullptr;
        int nextResult = mglRenderCommandBufferOwnerCreateNext(
            owner_handle, &next);
        if (nextResult == 0 && next) {
            result_out->current_command_buffer_created = 1u;
            static_cast<mgl::CommandBufferOwner*>(owner_handle)
                ->transaction_created_current = true;
        } else if (nextResult < 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
    }
    if (recovery) {
        snapshotCommandRecoveryOwner(recovery, &result_out->recovery);
    }
    return 0;
}

int mglRenderClassifyCommandBufferCompletion(
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCompletionDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    if (!state || !decision_out) return -1;

    decision_out->has_error = state->has_error != 0;
    decision_out->is_driver_rejection =
        decision_out->has_error &&
        strncmp(state->error_domain, "MTLCommandBufferErrorDomain",
                sizeof(state->error_domain)) == 0 &&
        state->error_code == 4;
    return 0;
}

int mglRenderCreateCommandRecoveryOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::CommandBufferRecoveryOwner* owner =
        new (std::nothrow) mgl::CommandBufferRecoveryOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

void mglRenderDestroyCommandRecoveryOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(*owner_handle);
    *owner_handle = nullptr;
    releaseCommandRecoveryOwner(owner);
}

int mglRenderCommandRecoveryRecordError(
    void* owner_handle,
    double now,
    MGLRenderCommandRecoverySnapshot* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->consecutiveErrors++;
    owner->consecutiveSuccesses = 0;
    owner->lastErrorTime = now;
    mgl::snapshotCommandRecovery(*owner, state_out);
    return 0;
}

int mglRenderCommandRecoveryRecordTransactionFailure(
    void* owner_handle,
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferTransaction* transaction_inout) {
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(owner_handle);
    if (!owner || !transaction_inout) return -1;
    transaction_inout->result =
        MGL_RENDER_COMMAND_BUFFER_TRANSACTION_ERROR;
    return applyCommandRecoveryFailure(
        owner, state, true, transaction_inout);
}

int mglRenderCommandRecoveryRecordSuccess(
    void* owner_handle,
    double now,
    MGLRenderCommandRecoverySuccess* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(owner_handle);
    if (!owner || !result_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (owner->consecutiveErrors > 0 || owner->recoveryMode) {
        owner->consecutiveSuccesses++;
        if (owner->consecutiveSuccesses >= 4 &&
            now - owner->lastErrorTime > 0.25) {
            result_out->sustained_recovery = 1;
            result_out->recovered_successes = owner->consecutiveSuccesses;
            result_out->previous_errors = owner->consecutiveErrors;
            owner->consecutiveErrors = 0;
            owner->recoveryMode = false;
            owner->consecutiveSuccesses = 0;
        }
    }
    mgl::snapshotCommandRecovery(*owner, &result_out->state);
    return 0;
}

int mglRenderCommandRecoveryClearMode(void* owner_handle) {
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(owner_handle);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (!owner->recoveryMode) return 0;
    owner->recoveryMode = false;
    return 1;
}

int mglRenderCommandRecoveryShouldSkip(
    void* owner_handle,
    double now,
    MGLRenderCommandRecoverySkipDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(owner_handle);
    if (!owner || !decision_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (now - owner->lastErrorTime > 3.0) {
        decision_out->recovery_timed_out = 1;
        decision_out->previous_errors = owner->consecutiveErrors;
        owner->consecutiveErrors = 0;
        owner->recoveryMode = false;
    } else if (owner->consecutiveErrors >= 8 || owner->recoveryMode) {
        decision_out->should_skip = 1;
        if (!owner->recoveryMode) {
            owner->recoveryMode = true;
            decision_out->entered_recovery_mode = 1;
        }
    }
    mgl::snapshotCommandRecovery(*owner, &decision_out->state);
    return 0;
}

int mglRenderProcessCommandBufferCompletion(
    void* owner_handle,
    const MGLRenderCommandBufferState* state,
    double now,
    MGLRenderCommandBufferCompletionResult* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!owner_handle || !state || !result_out) return -1;
    if (mglRenderClassifyCommandBufferCompletion(
            state, &result_out->decision) != 0) {
        return -1;
    }

    if (result_out->decision.has_error) {
        return mglRenderCommandRecoveryRecordError(
            owner_handle, now, &result_out->state);
    }

    MGLRenderCommandRecoverySuccess success = {};
    if (mglRenderCommandRecoveryRecordSuccess(
            owner_handle, now, &success) != 0) {
        return -1;
    }
    result_out->state = success.state;
    result_out->sustained_recovery = success.sustained_recovery;
    result_out->recovered_successes = success.recovered_successes;
    result_out->previous_errors = success.previous_errors;

    /* Keep this as a distinct owner operation. The legacy completion path
     * acquired its recovery lock once in recordGPUSuccess and once again to
     * clear recovery mode on the first successful completion. */
    int cleared = mglRenderCommandRecoveryClearMode(owner_handle);
    if (cleared < 0) return -1;
    result_out->cleared_recovery_mode = (uint32_t)cleared;
    if (cleared == 1) result_out->state.recovery_mode = 0;
    return 0;
}

int mglRenderAddCommandBufferRecoveryCompletion(
    void* command_buffer,
    void* recovery_owner) {
    return addCommandBufferRecoveryCompletion(
        command_buffer, recovery_owner, nullptr);
}

int mglRenderCommandRecoveryTakeResetRequest(void* recovery_owner) {
    mgl::CommandBufferRecoveryOwner* owner =
        static_cast<mgl::CommandBufferRecoveryOwner*>(recovery_owner);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (!owner->resetRequested) return 0;
    owner->resetRequested = false;
    return 1;
}

int mglRenderAddCommandBufferCompletion(
    void* command_buffer,
    MGLRenderCommandBufferCompletion callback,
    void* context,
    MGLRenderDestroyContext destroy_context) {
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!commandBuffer || !callback) return -1;

    mgl::CommandBufferCompletionContext* completion =
        new (std::nothrow) mgl::CommandBufferCompletionContext();
    if (!completion) return -1;
    completion->configure(callback, context, destroy_context);
    /* The block captures only a raw pointer. A separate reference belongs to
     * the handler, so a completion that runs before addCompletedHandler
     * returns cannot race a block copy helper or destroy this context early. */
    completion->retain();
    MTL::CommandBufferHandler stackHandler =
        ^(MTL::CommandBuffer* completedBuffer) {
            completion->complete(completedBuffer);
            completion->release();
        };
#ifdef __OBJC__
    MTL::CommandBufferHandler handler = [stackHandler copy];
#else
    MTL::CommandBufferHandler handler = Block_copy(stackHandler);
#endif
    if (!handler) {
        completion->abandonCallerContext();
        completion->release();
        completion->release();
        return -1;
    }
    try {
        commandBuffer->addCompletedHandler(handler);
    } catch (...) {
#ifndef __OBJC__
        Block_release(handler);
#endif
        completion->abandonCallerContext();
        completion->release();
        completion->release();
        return -1;
    }
#ifndef __OBJC__
    Block_release(handler);
#endif
    completion->release();
    return 0;
}

int mglRenderAddCommandBufferOwnerCompletion(
    void* owner_handle,
    MGLRenderCommandBufferCompletion callback,
    void* context,
    MGLRenderDestroyContext destroy_context) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !owner->current) return -1;
    return mglRenderAddCommandBufferCompletion(
        owner->current, callback, context, destroy_context);
}

int mglRenderCreateCommandBufferOwner(void* command_queue,
                                         void** owner_out,
                                         void** command_buffer_out) {
    if (owner_out) *owner_out = nullptr;
    if (command_buffer_out) *command_buffer_out = nullptr;
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!queue || !owner_out || !command_buffer_out) return -1;
    mgl::CommandBufferOwner* owner =
        new (std::nothrow) mgl::CommandBufferOwner();
    if (!owner) return -1;
    queue->retain();
    owner->queue = queue;
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) {
        delete owner;
        return -1;
    }
    commandBuffer->retain();
    owner->current = commandBuffer;
    *owner_out = owner;
    *command_buffer_out = commandBuffer;
    return 0;
}

int mglRenderResetCommandBufferOwner(void* owner_handle,
                                        void* command_queue,
                                        void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!owner || !queue || !command_buffer_out) return -1;
    if (owner->queue != queue) {
        queue->retain();
        if (owner->queue) owner->queue->release();
        owner->queue = queue;
    }
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) return -1;
    commandBuffer->retain();
    if (owner->current) owner->current->release();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = commandBuffer;
    return 0;
}

extern "C"
int mglRenderCreateCommandBufferOwnerAdopt(void* command_buffer,
                                              void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!commandBuffer || !owner_out) return -1;
    mgl::CommandBufferOwner* owner = new (std::nothrow) mgl::CommandBufferOwner();
    if (!owner) return -1;
    commandBuffer->retain();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    *owner_out = owner;
    return 0;
}

extern "C"
void* mglRenderCommandBufferOwnerGetCurrent(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    return owner ? static_cast<void*>(owner->current) : nullptr;
}

int mglRenderCommandBufferOwnerHasCurrent(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    return owner ? (owner->current ? 1 : 0) : -1;
}

int mglRenderCommandBufferOwnerCreateNext(
    void* owner_handle,
    void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !command_buffer_out) return -1;
    if (!owner->queue) return 1;
    if (owner->current) {
        /* A direct commit path can leave the committed object in the owner
         * instead of transferring a submission handle.  Drop that owner
         * reference before rotating; an unfinalized current buffer must stay
         * untouched because callers may still be encoding into it. */
        MTL::CommandBufferStatus status = owner->current->status();
        if (status < MTL::CommandBufferStatusCommitted) {
            owner->transaction_created_current = false;
            *command_buffer_out = owner->current;
            return 0;
        }
        owner->current->release();
        owner->current = nullptr;
    }
    MTL::CommandBuffer* commandBuffer = owner->queue->commandBuffer();
    if (!commandBuffer) return -1;
    commandBuffer->retain();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = commandBuffer;
    return 0;
}

int mglRenderGetCommandBufferOwnerState(
    void* owner_handle,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !owner->current || !state_out) return -1;
    return mgl::snapshotCommandBufferState(owner->current, state_out);
}

int mglRenderCommandBufferOwnerHasLastSubmitted(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    return owner ? (owner->lastSubmitted ? 1 : 0) : -1;
}

int mglRenderWaitCommandBufferState(
    void* command_buffer,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !state_out) return -1;

    MGLRenderCommandBufferState before = {};
    if (mgl::snapshotCommandBufferState(command, &before) != 0) return -1;
    if (before.status ==
        static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        *state_out = before;
        return 1;
    }
    try {
        if (before.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusCompleted)) {
            command->waitUntilCompleted();
        }
    } catch (...) {
        (void)mgl::snapshotCommandBufferState(command, state_out);
        return -1;
    }
    if (mgl::snapshotCommandBufferState(command, state_out) != 0) return -1;
    return state_out->has_error ? -1 : 0;
}

int mglRenderWaitCommandBufferOwnerLastSubmitted(
    void* owner_handle,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    MTL::CommandBuffer* commandBuffer = owner->lastSubmitted;
    if (!commandBuffer) return 1;
    return mglRenderWaitCommandBufferState(commandBuffer, state_out);
}

int mglRenderPresentDrawableForCommandBufferOwner(
    void* owner_handle,
    void* drawable,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    MTL::Drawable* surface = static_cast<MTL::Drawable*>(drawable);
    if (!owner || !owner->current || !surface) return -1;

    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0) {
        return -1;
    }
    if (state_out) *state_out = state;
    if (state.status !=
        static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        return 1;
    }
    owner->current->presentDrawable(surface);
    return 0;
}

int mglRenderEncodeWaitForEventForCommandBufferOwner(
    void* owner_handle,
    void* event,
    uint64_t value) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    MTL::Event* metal_event = static_cast<MTL::Event*>(event);
    if (!owner || !owner->current || !metal_event || value == 0u) return -1;
    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0 ||
        state.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        return -1;
    }
    owner->current->encodeWait(metal_event, value);
    return 0;
}

void mglRenderDiscardCommandBufferOwnerCurrent(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !owner->current) return;
    owner->current->release();
    owner->current = nullptr;
    owner->transaction_created_current = false;
    owner->syncs.reset();
}

int mglRenderTakeCommandBufferSubmission(void* owner_handle,
                                             void** submission_out,
                                             void** command_buffer_out) {
    if (submission_out) *submission_out = nullptr;
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !owner->current || !submission_out ||
        !command_buffer_out) {
        return -1;
    }
    mgl::CommandBufferSubmission* submission =
        new (std::nothrow) mgl::CommandBufferSubmission();
    if (!submission) return -1;
    submission->buffer = owner->current;
    owner->current = nullptr;
    owner->transaction_created_current = false;
    *submission_out = submission;
    *command_buffer_out = submission->buffer;
    return 0;
}

int mglRenderCommitCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return -1;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    if (!submission->buffer) return -1;
    submission->buffer->commit();
    *submission_handle = nullptr;
    delete submission;
    return 0;
}

/* does the submission own exactly this command buffer?
 * Replaces the ObjC MGLCommandState.detachedCommandBuffer mirror used to
 * guard commit/release of a detached submission. */
int mglRenderCommandBufferSubmissionMatchesBuffer(
    void* submission_handle, void* command_buffer) {
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(submission_handle);
    if (!submission || !command_buffer) return -1;
    return submission->buffer ==
                   static_cast<MTL::CommandBuffer*>(command_buffer)
               ? 1
               : 0;
}

void mglRenderDestroyCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    *submission_handle = nullptr;
    delete submission;
}

int mglRenderCommandBufferOwnerAppendSync(void* owner_handle,
                                                 Sync* sync) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !sync) return -1;
    mgl::CommandBufferSyncList& list = owner->syncs;
    if (list.count >= list.size) {
        const uint32_t old_size = list.size;
        const uint32_t new_size =
            old_size ? (old_size > (UINT32_MAX / 2) ? 0u : old_size * 2u) : 8u;
        if (new_size == 0u ||
            new_size > (UINT32_MAX / sizeof(Sync*))) {
            return -1;
        }
        Sync** new_list = (Sync**)realloc(
            list.list, sizeof(Sync*) * (size_t)new_size);
        if (!new_list) return -1;
        list.list = new_list;
        list.size = new_size;
    }
    list.list[list.count++] = sync;
    return 0;
}

void mglRenderCommandBufferOwnerClearSyncs(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner) return;
    owner->syncs.reset();
}

int mglRenderCommandBufferOwnerBeginCommit(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner) return -1;
    if (owner->commit_in_progress) return 0;
    owner->commit_in_progress = true;
    return 1;
}

void mglRenderCommandBufferOwnerEndCommit(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner) return;
    owner->commit_in_progress = false;
}

extern "C"
int mglRenderCommandBufferOwnerConsumeTransactionCurrent(
    void* owner_handle,
    void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !command_buffer_out) return -1;
    if (!owner->transaction_created_current || !owner->current) return 0;
    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0 ||
        state.status != static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        owner->transaction_created_current = false;
        return 0;
    }
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = owner->current;
    return 1;
}

void mglRenderDestroyCommandBufferOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateCommandQueueOwner(uint32_t max_command_buffers,
                                        void** owner_out,
                                        void** command_queue_out) {
    if (owner_out) *owner_out = nullptr;
    if (command_queue_out) *command_queue_out = nullptr;
    if (!owner_out || !command_queue_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    mgl::CommandQueueOwner* owner =
        new (std::nothrow) mgl::CommandQueueOwner();
    if (!owner) return -1;
    MTL::CommandQueue* queue = max_command_buffers
        ? renderer.device->newCommandQueue(max_command_buffers)
        : renderer.device->newCommandQueue();
    if (!queue) {
        delete owner;
        return -1;
    }
    owner->queue = queue;
    *owner_out = owner;
    *command_queue_out = queue;
    return 0;
}

int mglRenderResetCommandQueueOwner(void* owner_handle,
                                       uint32_t max_command_buffers,
                                       void** command_queue_out) {
    if (command_queue_out) *command_queue_out = nullptr;
    mgl::CommandQueueOwner* owner =
        static_cast<mgl::CommandQueueOwner*>(owner_handle);
    if (!owner || !command_queue_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::CommandQueue* queue = max_command_buffers
        ? renderer.device->newCommandQueue(max_command_buffers)
        : renderer.device->newCommandQueue();
    if (!queue) return -1;
    if (owner->queue) owner->queue->release();
    owner->queue = queue;
    *command_queue_out = queue;
    return 0;
}

void mglRenderDestroyCommandQueueOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandQueueOwner* owner =
        static_cast<mgl::CommandQueueOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateMDIScratchOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::MDIScratchOwner* owner =
        new (std::nothrow) mgl::MDIScratchOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

int mglRenderAllocateMDIScratch(void* owner_handle,
                                   uint64_t length,
                                   uint64_t alignment,
                                   void** buffer_out,
                                   uint64_t* offset_out,
                                   uint64_t* capacity_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (offset_out) *offset_out = 0;
    if (capacity_out) *capacity_out = 0;
    mgl::MDIScratchOwner* owner =
        static_cast<mgl::MDIScratchOwner*>(owner_handle);
    if (!owner || !buffer_out || !offset_out || length == 0 ||
        alignment == 0 || (alignment & (alignment - 1u)) != 0u) {
        return -1;
    }

    const uint64_t mask = alignment - 1u;
    if (owner->offset > std::numeric_limits<uint64_t>::max() - mask) {
        return -1;
    }
    uint64_t alignedOffset = (owner->offset + mask) & ~mask;
    if (length > std::numeric_limits<uint64_t>::max() - alignedOffset) {
        return -1;
    }
    uint64_t required = alignedOffset + length;
    if (!owner->buffer || required > owner->capacity) {
        uint64_t nextCapacity = owner->capacity;
        if (nextCapacity == 0) nextCapacity = 64u * 1024u;
        while (nextCapacity < required) {
            if (nextCapacity > std::numeric_limits<uint64_t>::max() / 2u) {
                nextCapacity = required;
                break;
            }
            nextCapacity *= 2u;
        }
        if (nextCapacity > std::numeric_limits<NS::UInteger>::max()) {
            return -1;
        }
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        MTL::Buffer* next = renderer.device->newBuffer(
            static_cast<NS::UInteger>(nextCapacity),
            MTL::ResourceStorageModeShared);
        if (!next) return -1;
        if (owner->buffer) owner->buffer->release();
        owner->buffer = next;
        owner->capacity = nextCapacity;
        alignedOffset = 0;
        required = length;
    }

    owner->offset = required;
    *buffer_out = owner->buffer;
    *offset_out = alignedOffset;
    if (capacity_out) *capacity_out = owner->capacity;
    return 0;
}

void mglRenderResetMDIScratchOwner(void* owner_handle) {
    mgl::MDIScratchOwner* owner =
        static_cast<mgl::MDIScratchOwner*>(owner_handle);
    if (!owner) return;
    if (owner->buffer) owner->buffer->release();
    owner->buffer = nullptr;
    owner->capacity = 0;
    owner->offset = 0;
}

void mglRenderDestroyMDIScratchOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::MDIScratchOwner* owner =
        static_cast<mgl::MDIScratchOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCommitCommandBuffer(void* command_buffer) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command) return -1;
    command->commit();
    return 0;
}

int mglRenderWaitCommandBuffer(void* command_buffer) {
    MGLRenderCommandBufferState state = {};
    return mglRenderWaitCommandBufferState(command_buffer, &state);
}

int mglRenderCreateRenderEncoderFromState(
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !render_pass || !render_encoder_out) return -1;
    MTL::RenderPassDescriptor* descriptor =
        mgl::newRenderPassDescriptor(render_pass);
    if (!descriptor) return -1;
    MTL::RenderCommandEncoder* encoder =
        command->renderCommandEncoder(descriptor);
    descriptor->release();
    if (!encoder) return -1;
    *render_encoder_out = encoder;
    return 0;
}

int mglRenderEncodeColorClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double red,
                                 double green,
                                 double blue,
                                 double alpha) {
    if (!command_buffer || !texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    state.color[0].attachment.texture = texture;
    state.color[0].attachment.level = level;
    state.color[0].attachment.slice = slice;
    state.color[0].attachment.depth_plane = depth_plane;
    state.color[0].attachment.load_action =
        static_cast<uint32_t>(MTL::LoadActionClear);
    state.color[0].clear_red = red;
    state.color[0].clear_green = green;
    state.color[0].clear_blue = blue;
    state.color[0].clear_alpha = alpha;
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeColorClearForCommandBufferOwner(
    void* command_buffer_owner,
    void* texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double red,
    double green,
    double blue,
    double alpha) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeColorClear(
        owner->current, texture, level, slice, depth_plane,
        red, green, blue, alpha);
}

int mglRenderEncodeDepthClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth) {
    if (!command_buffer || !texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    state.depth.attachment.texture = texture;
    state.depth.attachment.level = level;
    state.depth.attachment.slice = slice;
    state.depth.attachment.depth_plane = depth_plane;
    state.depth.attachment.load_action =
        static_cast<uint32_t>(MTL::LoadActionClear);
    state.depth.clear_depth = clear_depth;
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeDepthClearForCommandBufferOwner(
    void* command_buffer_owner,
    void* texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double clear_depth) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeDepthClear(
        owner->current, texture, level, slice, depth_plane, clear_depth);
}

int mglRenderEncodeMultisampleResolve(
    void* command_buffer,
    uint32_t attachment_kind,
    void* source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void* resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter) {
    if (!command_buffer || !source_texture || !resolve_texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        attachment = &state.color[0].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &state.depth.attachment;
        state.depth.resolve_filter = resolve_filter;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        attachment = &state.stencil.attachment;
        state.stencil.resolve_filter = resolve_filter;
        break;
    default:
        return -1;
    }
    attachment->texture = source_texture;
    attachment->level = source_level;
    attachment->slice = source_slice;
    attachment->depth_plane = source_depth_plane;
    attachment->resolve_texture = resolve_texture;
    attachment->resolve_level = resolve_level;
    attachment->resolve_slice = resolve_slice;
    attachment->resolve_depth_plane = resolve_depth_plane;
    attachment->load_action = static_cast<uint32_t>(MTL::LoadActionLoad);
    attachment->store_action =
        static_cast<uint32_t>(MTL::StoreActionMultisampleResolve);
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeMultisampleResolveForCommandBufferOwner(
    void* command_buffer_owner,
    uint32_t attachment_kind,
    void* source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void* resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeMultisampleResolve(
        owner->current, attachment_kind, source_texture, source_level,
        source_slice, source_depth_plane, resolve_texture, resolve_level,
        resolve_slice, resolve_depth_plane, resolve_filter);
}

static int mglRenderResetRenderEncoderOwnerImpl(
    mgl::RenderEncoderOwner* owner,
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!owner || !command || !render_pass || !render_encoder_out) return -1;
    MTL::RenderPassDescriptor* descriptor =
        mgl::newRenderPassDescriptor(render_pass);
    if (!descriptor) return -1;
    MTL::RenderCommandEncoder* encoder =
        command->renderCommandEncoder(descriptor);
    descriptor->release();
    if (!encoder) return -1;
    encoder->retain();
    if (owner->encoder) owner->encoder->release();
    owner->encoder = encoder;
    owner->ended = false;
    *render_encoder_out = encoder;
    return 0;
}

int mglRenderCreateRenderEncoderOwnerFromState(
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** owner_out,
    void** render_encoder_out) {
    if (owner_out) *owner_out = nullptr;
    if (render_encoder_out) *render_encoder_out = nullptr;
    if (!owner_out || !render_encoder_out) return -1;
    mgl::RenderEncoderOwner* owner =
        new (std::nothrow) mgl::RenderEncoderOwner();
    if (!owner) return -1;
    if (mglRenderResetRenderEncoderOwnerImpl(
            owner, command_buffer, render_pass, render_encoder_out) != 0) {
        delete owner;
        return -1;
    }
    *owner_out = owner;
    return 0;
}

int mglRenderResetRenderEncoderOwnerFromState(
    void* owner_handle,
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    return mglRenderResetRenderEncoderOwnerImpl(
        static_cast<mgl::RenderEncoderOwner*>(owner_handle),
        command_buffer, render_pass, render_encoder_out);
}

int mglRenderCreateRenderEncoderOwner(
    void* render_encoder,
    void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || !owner_out) return -1;
    mgl::RenderEncoderOwner* owner =
        new (std::nothrow) mgl::RenderEncoderOwner();
    if (!owner) return -1;
    encoder->retain();
    owner->encoder = encoder;
    *owner_out = owner;
    return 0;
}

int mglRenderResetRenderEncoderOwner(
    void* owner_handle,
    void* render_encoder) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!owner || !encoder) return -1;
    encoder->retain();
    if (owner->encoder) owner->encoder->release();
    owner->encoder = encoder;
    owner->ended = false;
    return 0;
}

int mglRenderEndRenderEncoderOwner(void* owner_handle) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    if (!owner) return -1;
    if (!owner->encoder) return owner->ended ? 0 : -1;
    if (!owner->ended) {
        owner->encoder->endEncoding();
        owner->ended = true;
    }
    owner->encoder->release();
    owner->encoder = nullptr;
    return 0;
}

static void* mglRenderActiveRenderEncoder(void* owner_handle) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    return owner && owner->encoder && !owner->ended
        ? static_cast<void*>(owner->encoder)
        : nullptr;
}

int mglRenderEncoderOwnerHasCurrent(void* owner_handle) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    return owner && owner->encoder && !owner->ended ? 1 : 0;
}

int mglRenderSetRenderEncoderOwnerLabel(void* owner_handle,
                                           const char* label) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    if (!owner || !owner->encoder || owner->ended || !label) return -1;
    owner->encoder->setLabel(
        NS::String::string(label, NS::UTF8StringEncoding));
    return 0;
}

void mglRenderDestroyRenderEncoderOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateRenderPassIdentityOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::RenderPassIdentityOwner* owner =
        new (std::nothrow) mgl::RenderPassIdentityOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

int mglRenderUpdateRenderPassIdentity(
    void* owner_handle,
    const MGLRenderPassIdentityState* state) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !state ||
        state->draw_buffer_count > MGL_RENDER_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    owner->state = *state;
    owner->cache = {};
    owner->cache_valid = false;
    return 0;
}

int mglRenderGetRenderPassIdentity(
    void* owner_handle,
    MGLRenderPassIdentityState* state_out) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

int mglRenderSetFboMatchCache(
    void* owner_handle,
    const MGLRenderFboMatchCacheState* cache) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !cache || cache->fbo_name == 0) return -1;
    owner->cache = *cache;
    owner->cache.result = cache->result != 0;
    owner->cache_valid = true;
    return 0;
}

int mglRenderGetFboMatchCache(
    void* owner_handle,
    MGLRenderFboMatchCacheState* cache_out) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !cache_out || !owner->cache_valid) return 1;
    *cache_out = owner->cache;
    return 0;
}

void mglRenderClearFboMatchCache(void* owner_handle) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner) return;
    owner->cache = {};
    owner->cache_valid = false;
}

void mglRenderDestroyRenderPassIdentityOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateRenderPassStateOwner(
    const MGLRenderPassState* state,
    void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!state || !owner_out ||
        state->sample_position_count > MGL_RENDER_MAX_SAMPLE_POSITIONS) {
        return -1;
    }
    mgl::RenderPassStateOwner* owner =
        new (std::nothrow) mgl::RenderPassStateOwner();
    if (!owner) return -1;
    owner->state = *state;
    mgl::retainRenderPassStateResources(owner->state);
    *owner_out = owner;
    return 0;
}

int mglRenderCreateDefaultRenderPassStateOwner(void** owner_out) {
    MGLRenderPassState state = mgl::defaultRenderPassState();
    return mglRenderCreateRenderPassStateOwner(&state, owner_out);
}

int mglRenderSetRenderPassStateAttachment(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    const MGLRenderPassAttachmentState* attachment) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !attachment) return -1;

    MGLRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        destination = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    MGLRenderPassAttachmentState next = *attachment;
    mgl::retainRenderPassObject(next.texture);
    mgl::retainRenderPassObject(next.resolve_texture);
    mgl::releaseRenderPassObject(destination->texture);
    mgl::releaseRenderPassObject(destination->resolve_texture);
    *destination = next;
    return 0;
}

static uint64_t mglRenderTargetLayerCount(
    MTL::Texture* texture,
    uint64_t level) {
    if (!texture) return 0u;
    switch (texture->textureType()) {
    case MTL::TextureType1DArray:
    case MTL::TextureType2DArray:
    case MTL::TextureType2DMultisampleArray:
        return static_cast<uint64_t>(texture->arrayLength());
    case MTL::TextureTypeCube:
        return 6u;
    case MTL::TextureTypeCubeArray:
        return static_cast<uint64_t>(texture->arrayLength()) * 6u;
    case MTL::TextureType3D:
        return mglRenderMetalTextureLevelDimension(
            static_cast<uint64_t>(texture->depth()), level);
    default:
        return 1u;
    }
}

static uint64_t mglRenderPassArrayLength(
    const MGLRenderPassState& state) {
    uint64_t commonArrayLength = 0u;
    bool hasAttachment = false;
    bool hasLayeredAttachment = false;
    auto accumulate = [&commonArrayLength, &hasAttachment,
                       &hasLayeredAttachment](
                          const MGLRenderPassAttachmentState& attachment) {
        MTL::Texture* texture = static_cast<MTL::Texture*>(attachment.texture);
        if (!texture) return;
        hasAttachment = true;
        if (!attachment.layered) return;
        uint64_t layerCount =
            mglRenderTargetLayerCount(texture, attachment.level);
        if (layerCount == 0u) return;
        hasLayeredAttachment = true;
        commonArrayLength = commonArrayLength == 0u
            ? layerCount
            : std::min(commonArrayLength, layerCount);
    };
    for (uint32_t i = 0u; i < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++i) {
        accumulate(state.color[i].attachment);
    }
    accumulate(state.depth.attachment);
    accumulate(state.stencil.attachment);
    return hasLayeredAttachment ? commonArrayLength
                                : (hasAttachment ? 1u : 0u);
}

int mglRenderSetRenderPassStateAttachmentTexture(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    void* texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    uint32_t layered) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;

    MGLRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        destination = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    mgl::retainRenderPassObject(texture);
    mgl::releaseRenderPassObject(destination->texture);
    destination->texture = texture;
    destination->level = level;
    destination->slice = slice;
    destination->depth_plane = depth_plane;
    destination->layered = layered != 0u;

    owner->state.render_target_array_length =
        mglRenderPassArrayLength(owner->state);
    if (layered) {
        destination->slice = 0u;
        destination->depth_plane = 0u;
    }
    return 0;
}

int mglRenderSetRenderPassStateAttachmentActions(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    uint32_t load_action,
    uint32_t store_action,
    uint64_t store_action_options) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;

    MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        attachment = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        attachment = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    attachment->load_action = load_action;
    attachment->store_action = store_action;
    attachment->store_action_options = store_action_options;
    return 0;
}

int mglRenderSetRenderPassStateColorClear(
    void* owner_handle,
    uint32_t color_index,
    double red,
    double green,
    double blue,
    double alpha) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    MGLRenderPassColorState& color = owner->state.color[color_index];
    color.clear_red = red;
    color.clear_green = green;
    color.clear_blue = blue;
    color.clear_alpha = alpha;
    return 0;
}

int mglRenderSetRenderPassStateDepthClear(
    void* owner_handle,
    double clear_depth) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    owner->state.depth.clear_depth = clear_depth;
    return 0;
}

int mglRenderSetRenderPassStateStencilClear(
    void* owner_handle,
    uint32_t clear_stencil) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    owner->state.stencil.clear_stencil = clear_stencil;
    return 0;
}

int mglRenderSetRenderPassStateVisibility(
    void* owner_handle,
    void* visibility_result_buffer,
    uint32_t visibility_result_type) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    mgl::retainRenderPassObject(visibility_result_buffer);
    mgl::releaseRenderPassObject(owner->state.visibility_result_buffer);
    owner->state.visibility_result_buffer = visibility_result_buffer;
    owner->state.visibility_result_type = visibility_result_type;
    return 0;
}

int mglRenderSetRenderPassStateDimensions(
    void* owner_handle,
    uint64_t width,
    uint64_t height) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    owner->state.render_target_width = width;
    owner->state.render_target_height = height;
    return 0;
}

int mglRenderGetRenderPassStateOwner(
    void* owner_handle,
    MGLRenderPassState* state_out) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

int mglRenderCommandBufferOwnerHasState(
    void* owner_handle,
    MGLRenderCommandBufferState* state_out) {
    return owner_handle && state_out &&
               mglRenderGetCommandBufferOwnerState(owner_handle, state_out) == 0
        ? 1 : 0;
}

int mglRenderGetRenderPassAttachmentStateOwner(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    MGLRenderPassAttachmentState* attachment_out) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !attachment_out) return -1;

    const MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
            attachment = &owner->state.color[color_index].attachment;
            break;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            attachment = &owner->state.depth.attachment;
            break;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            attachment = &owner->state.stencil.attachment;
            break;
        default:
            return -1;
    }
    *attachment_out = *attachment;
    return 0;
}

int mglRenderCreateRenderEncoderFromStateOwner(
    void* command_buffer,
    void* owner_handle,
    void** render_encoder_out) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) {
        if (render_encoder_out) *render_encoder_out = nullptr;
        return -1;
    }
    return mglRenderCreateRenderEncoderFromState(
        command_buffer, &owner->state, render_encoder_out);
}

int mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
    void* command_buffer_owner,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderCreateRenderEncoderFromState(
        owner->current, render_pass, render_encoder_out);
}

void* mglRenderCreateRenderEncoderBorrowed(
    void* command_buffer_owner,
    const MGLRenderPassState* render_pass) {
    void* encoder = nullptr;
    return mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
               command_buffer_owner, render_pass, &encoder) == 0
        ? encoder : nullptr;
}

void* mglRenderCreateBlitEncoderBorrowed(void* command_buffer_owner) {
    void* encoder = nullptr;
    return mglRenderCreateBlitEncoderFromCommandBufferOwner(
               command_buffer_owner, &encoder) == 0
        ? encoder : nullptr;
}

void* mglRenderCreateComputeEncoderBorrowed(void* command_buffer_owner) {
    void* encoder = nullptr;
    return mglRenderCreateComputeEncoderFromCommandBufferOwner(
               command_buffer_owner, &encoder) == 0
        ? encoder : nullptr;
}

static MGLRenderPassAttachmentState*
mglRenderAttachmentForOwner(
    mgl::RenderPassStateOwner* owner,
    uint32_t attachment_kind,
    uint32_t color_index) {
    if (!owner) return nullptr;
    switch (attachment_kind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            return color_index < MGL_RENDER_MAX_COLOR_ATTACHMENTS
                ? &owner->state.color[color_index].attachment : nullptr;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            return &owner->state.depth.attachment;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            return &owner->state.stencil.attachment;
        default:
            return nullptr;
    }
}

void* mglRenderGetRenderPassAttachmentTextureOwner(
    void* owner_handle, uint32_t attachment_kind, uint32_t color_index) {
    auto* owner = static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    return attachment ? attachment->texture : nullptr;
}

int mglRenderGetRenderPassAttachmentSubresourceOwner(
    void* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint64_t* level_out, uint64_t* slice_out, uint64_t* depth_plane_out) {
    auto* owner = static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    if (!attachment) return -1;
    if (level_out) *level_out = attachment->level;
    if (slice_out) *slice_out = attachment->slice;
    if (depth_plane_out) *depth_plane_out = attachment->depth_plane;
    return 0;
}

int mglRenderGetRenderTargetSizeOwner(
    void* owner_handle, uint64_t* width_out, uint64_t* height_out) {
    auto* owner = static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    if (width_out) *width_out = owner->state.render_target_width;
    if (height_out) *height_out = owner->state.render_target_height;
    return 0;
}

int mglRenderPassUsesColorTextureOwner(
    void* owner_handle, void* texture, uint32_t* attachment_index_out) {
    auto* owner = static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !texture) return 0;
    for (uint32_t index = 0; index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        if (owner->state.color[index].attachment.texture == texture) {
            if (attachment_index_out) *attachment_index_out = index;
            return 1;
        }
    }
    return 0;
}

int mglRenderGetRenderPassAttachmentActionsOwner(
    void* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint32_t* load_action_out, uint32_t* store_action_out,
    uint64_t* store_action_options_out) {
    auto* owner = static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    if (!attachment) return -1;
    if (load_action_out) *load_action_out = attachment->load_action;
    if (store_action_out) *store_action_out = attachment->store_action;
    if (store_action_options_out) {
        *store_action_options_out = attachment->store_action_options;
    }
    return 0;
}

uint32_t mglRenderPassLoadActionForTrace(
    void* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_load_action) {
    uint32_t load_action = 0;
    return mglRenderGetRenderPassAttachmentActionsOwner(
               owner_handle, attachment_kind, color_index,
               &load_action, nullptr, nullptr) == 0
        ? load_action : default_load_action;
}

uint32_t mglRenderPassStoreActionForTrace(
    void* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_store_action) {
    uint32_t store_action = 0;
    return mglRenderGetRenderPassAttachmentActionsOwner(
               owner_handle, attachment_kind, color_index,
               nullptr, &store_action, nullptr) == 0
        ? store_action : default_store_action;
}

void mglRenderDestroyRenderPassStateOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderEndRenderEncoder(void* render_encoder) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateBlitEncoder(void* command_buffer,
                                  void** blit_encoder_out) {
    if (blit_encoder_out) *blit_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !blit_encoder_out) return -1;
    MTL::BlitCommandEncoder* encoder = command->blitCommandEncoder();
    if (!encoder) return -1;
    *blit_encoder_out = encoder;
    return 0;
}

int mglRenderCreateBlitEncoderFromCommandBufferOwner(
    void* command_buffer_owner,
    void** blit_encoder_out) {
    if (blit_encoder_out) *blit_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderCreateBlitEncoder(
        owner->current, blit_encoder_out);
}

int mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(
    void* command_buffer_owner,
    void* source_texture,
    void* destination_texture) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!owner || !owner->current || !source || !destination ||
        source->width() != destination->width() ||
        source->height() != destination->height() ||
        source->depth() != destination->depth()) {
        return -1;
    }

    const NS::UInteger slice_count =
        std::min(source->arrayLength(), destination->arrayLength());
    const NS::UInteger level_count = std::min(
        source->mipmapLevelCount(), destination->mipmapLevelCount());
    if (slice_count == 0 || level_count == 0) return -1;

    MTL::BlitCommandEncoder* encoder =
        owner->current->blitCommandEncoder();
    if (!encoder) return -1;
    for (NS::UInteger slice = 0; slice < slice_count; ++slice) {
        for (NS::UInteger level = 0; level < level_count; ++level) {
            const NS::UInteger width =
                std::max<NS::UInteger>(1u, source->width() >> level);
            const NS::UInteger height =
                std::max<NS::UInteger>(1u, source->height() >> level);
            const NS::UInteger depth =
                std::max<NS::UInteger>(1u, source->depth() >> level);
            encoder->copyFromTexture(
                source, slice, level, MTL::Origin(0, 0, 0),
                MTL::Size(width, height, depth), destination, slice, level,
                MTL::Origin(0, 0, 0));
        }
    }
    encoder->endEncoding();
    return 0;
}

int mglRenderEncodeBufferCopiesForCommandBufferOwner(
    void* command_buffer_owner,
    const MGLRenderBufferCopyEntry* entries,
    uint32_t entry_count) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current || !entries || entry_count == 0u) {
        return -1;
    }
    for (uint32_t i = 0; i < entry_count; ++i) {
        const MGLRenderBufferCopyEntry& entry = entries[i];
        MTL::Buffer* source = static_cast<MTL::Buffer*>(entry.source_buffer);
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(entry.destination_buffer);
        if (!source || !destination || entry.length == 0u ||
            entry.source_offset > source->length() ||
            entry.length > source->length() - entry.source_offset ||
            entry.destination_offset > destination->length() ||
            entry.length > destination->length() - entry.destination_offset) {
            return -1;
        }
    }
    MTL::BlitCommandEncoder* encoder =
        owner->current->blitCommandEncoder();
    if (!encoder) return -1;
    for (uint32_t i = 0; i < entry_count; ++i) {
        const MGLRenderBufferCopyEntry& entry = entries[i];
        encoder->copyFromBuffer(
            static_cast<MTL::Buffer*>(entry.source_buffer),
            static_cast<NS::UInteger>(entry.source_offset),
            static_cast<MTL::Buffer*>(entry.destination_buffer),
            static_cast<NS::UInteger>(entry.destination_offset),
            static_cast<NS::UInteger>(entry.length));
    }
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateComputeEncoderFromCommandBufferOwner(
    void* command_buffer_owner,
    void** compute_encoder_out) {
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderCreateComputeEncoder(
        owner->current, compute_encoder_out);
}

int mglRenderEndBlitEncoder(void* blit_encoder) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderEncodeTextureUploadLayers(
    void* command_buffer,
    void* source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void* destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!command || !source || !destination || source_width == 0 ||
        source_height == 0 || source_depth == 0 ||
        source_bytes_per_row == 0 || source_bytes_per_image == 0 ||
        layer_count == 0 || (layer_count > 1u && source_layer_stride == 0u)) {
        return -1;
    }

    const uint64_t last_layer = layer_count - 1u;
    if ((source_layer_stride != 0u &&
         last_layer > (std::numeric_limits<uint64_t>::max() - source_offset) /
                          source_layer_stride) ||
        last_layer > std::numeric_limits<uint64_t>::max() -
                         destination_base_slice) {
        return -1;
    }

    uint64_t source_layer_span = 0u;
    if (source_depth > std::numeric_limits<uint64_t>::max() /
                           source_bytes_per_image) {
        return -1;
    }
    source_layer_span = source_bytes_per_image * source_depth;
    const uint64_t last_source_offset =
        source_offset + last_layer * source_layer_stride;
    if (last_source_offset > source->length() ||
        source_layer_span > source->length() - last_source_offset) {
        return -1;
    }

    if (destination_level >= destination->mipmapLevelCount()) {
        return -1;
    }
    uint64_t destination_slice_count = destination->arrayLength();
    switch (destination->textureType()) {
        case MTL::TextureTypeCube:
            destination_slice_count = 6u;
            break;
        case MTL::TextureTypeCubeArray:
            if (destination_slice_count >
                std::numeric_limits<uint64_t>::max() / 6u) {
                return -1;
            }
            destination_slice_count *= 6u;
            break;
        default:
            break;
    }
    if (destination_base_slice >= destination_slice_count ||
        layer_count > destination_slice_count - destination_base_slice) {
        return -1;
    }

    const uint64_t mip_width =
        std::max<uint64_t>(1u, destination->width() >> destination_level);
    const uint64_t mip_height =
        std::max<uint64_t>(1u, destination->height() >> destination_level);
    const uint64_t mip_depth =
        std::max<uint64_t>(1u, destination->depth() >> destination_level);
    if (destination_x > mip_width || source_width > mip_width - destination_x ||
        destination_y > mip_height ||
        source_height > mip_height - destination_y ||
        destination_z > mip_depth || source_depth > mip_depth - destination_z) {
        return -1;
    }

    MTL::BlitCommandEncoder* encoder = command->blitCommandEncoder();
    if (!encoder) return -1;
    for (uint64_t layer = 0u; layer < layer_count; ++layer) {
        encoder->copyFromBuffer(
            source,
            static_cast<NS::UInteger>(source_offset +
                                      layer * source_layer_stride),
            static_cast<NS::UInteger>(source_bytes_per_row),
            static_cast<NS::UInteger>(source_bytes_per_image),
            MTL::Size(source_width, source_height, source_depth), destination,
            static_cast<NS::UInteger>(destination_base_slice + layer),
            static_cast<NS::UInteger>(destination_level),
            MTL::Origin(destination_x, destination_y, destination_z));
    }
    encoder->endEncoding();
    return 0;
}

int mglRenderEncodeTextureUploadLayersForCommandBufferOwner(
    void* command_buffer_owner,
    void* source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void* destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(command_buffer_owner);
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeTextureUploadLayers(
        owner->current, source_buffer, source_offset, source_bytes_per_row,
        source_bytes_per_image, source_layer_stride, source_width,
        source_height, source_depth, destination_texture,
        destination_base_slice, layer_count, destination_level,
        destination_x, destination_y, destination_z);
}

int mglRenderEncodeTextureUpload(void* command_buffer,
                                    void* source_buffer,
                                    uint64_t source_offset,
                                    uint64_t source_bytes_per_row,
                                    uint64_t source_bytes_per_image,
                                    uint64_t source_width,
                                    uint64_t source_height,
                                    uint64_t source_depth,
                                    void* destination_texture,
                                    uint64_t destination_slice,
                                    uint64_t destination_level,
                                    uint64_t destination_x,
                                    uint64_t destination_y,
                                    uint64_t destination_z) {
    return mglRenderEncodeTextureUploadLayers(
        command_buffer, source_buffer, source_offset, source_bytes_per_row,
        source_bytes_per_image, 0u, source_width, source_height, source_depth,
        destination_texture, destination_slice, 1u, destination_level,
        destination_x, destination_y, destination_z);
}

int mglRenderBlitCopyBuffer(void* blit_encoder,
                               void* source_buffer,
                               uint64_t source_offset,
                               void* destination_buffer,
                               uint64_t destination_offset,
                               uint64_t size) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Buffer* destination =
        static_cast<MTL::Buffer*>(destination_buffer);
    if (!encoder || !source || !destination || size == 0) return -1;
    encoder->copyFromBuffer(source, static_cast<NS::UInteger>(source_offset),
                            destination,
                            static_cast<NS::UInteger>(destination_offset),
                            static_cast<NS::UInteger>(size));
    return 0;
}

int mglRenderBlitCopyBufferToTexture(void* blit_encoder,
                                        void* source_buffer,
                                        uint64_t source_offset,
                                        uint64_t source_bytes_per_row,
                                        uint64_t source_bytes_per_image,
                                        uint64_t source_width,
                                        uint64_t source_height,
                                        uint64_t source_depth,
                                        void* destination_texture,
                                        uint64_t destination_slice,
                                        uint64_t destination_level,
                                        uint64_t destination_x,
                                        uint64_t destination_y,
                                        uint64_t destination_z) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!encoder || !source || !destination || source_width == 0 ||
        source_height == 0 || source_depth == 0 ||
        source_bytes_per_row == 0 || source_bytes_per_image == 0) {
        return -1;
    }
    encoder->copyFromBuffer(
        source, static_cast<NS::UInteger>(source_offset),
        static_cast<NS::UInteger>(source_bytes_per_row),
        static_cast<NS::UInteger>(source_bytes_per_image),
        MTL::Size(source_width, source_height, source_depth), destination,
        static_cast<NS::UInteger>(destination_slice),
        static_cast<NS::UInteger>(destination_level),
        MTL::Origin(destination_x, destination_y, destination_z));
    return 0;
}

int mglRenderBlitSynchronizeTexture(void* blit_encoder,
                                       void* texture,
                                       uint64_t slice,
                                       uint64_t level) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!encoder || !source) return -1;
    encoder->synchronizeTexture(source, static_cast<NS::UInteger>(slice),
                                static_cast<NS::UInteger>(level));
    return 0;
}

int mglRenderBlitGenerateMipmaps(void* blit_encoder,
                                    void* texture) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!encoder || !source) return -1;
    encoder->generateMipmaps(source);
    return 0;
}

int mglRenderBlitCopyTexture(void* blit_encoder,
                                void* source_texture,
                                uint64_t source_slice,
                                uint64_t source_level,
                                uint64_t source_x,
                                uint64_t source_y,
                                uint64_t source_z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                void* destination_texture,
                                uint64_t destination_slice,
                                uint64_t destination_level,
                                uint64_t destination_x,
                                uint64_t destination_y,
                                uint64_t destination_z) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!encoder || !source || !destination || width == 0 || height == 0 ||
        depth == 0) {
        return -1;
    }
    encoder->copyFromTexture(
        source, static_cast<NS::UInteger>(source_slice),
        static_cast<NS::UInteger>(source_level),
        MTL::Origin(source_x, source_y, source_z),
        MTL::Size(width, height, depth), destination,
        static_cast<NS::UInteger>(destination_slice),
        static_cast<NS::UInteger>(destination_level),
        MTL::Origin(destination_x, destination_y, destination_z));
    return 0;
}

int mglRenderBlitCopyTextureToBuffer(
    void* blit_encoder,
    void* source_texture,
    uint64_t source_slice,
    uint64_t source_level,
    uint64_t source_x,
    uint64_t source_y,
    uint64_t source_z,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    void* destination_buffer,
    uint64_t destination_offset,
    uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Buffer* destination =
        static_cast<MTL::Buffer*>(destination_buffer);
    if (!encoder || !source || !destination || width == 0 || height == 0 ||
        depth == 0 || destination_bytes_per_row == 0 ||
        destination_bytes_per_image == 0) {
        return -1;
    }
    encoder->copyFromTexture(
        source, static_cast<NS::UInteger>(source_slice),
        static_cast<NS::UInteger>(source_level),
        MTL::Origin(source_x, source_y, source_z),
        MTL::Size(width, height, depth), destination,
        static_cast<NS::UInteger>(destination_offset),
        static_cast<NS::UInteger>(destination_bytes_per_row),
        static_cast<NS::UInteger>(destination_bytes_per_image));
    return 0;
}

int mglRenderDrawPrimitives(void* render_encoder,
                               uint32_t primitive_type,
                               uint64_t vertex_start,
                               uint64_t vertex_count,
                               uint64_t instance_count,
                               uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || instance_count == 0) return -1;
    encoder->drawPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<NS::UInteger>(vertex_start),
        static_cast<NS::UInteger>(vertex_count),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawIndexedPrimitives(void* render_encoder,
                                      uint32_t primitive_type,
                                      uint64_t index_count,
                                      uint32_t index_type,
                                      void* index_buffer,
                                      uint64_t index_buffer_offset,
                                      uint64_t instance_count,
                                      int64_t base_vertex,
                                      uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    if (!encoder || !indices || instance_count == 0) return -1;
    encoder->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<NS::UInteger>(index_count),
        static_cast<MTL::IndexType>(index_type), indices,
        static_cast<NS::UInteger>(index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::Integer>(base_vertex),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawPrimitivesIndirect(void* render_encoder,
                                       uint32_t primitive_type,
                                       void* indirect_buffer,
                                       uint64_t indirect_buffer_offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indirect = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !indirect) return -1;
    encoder->drawPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type), indirect,
        static_cast<NS::UInteger>(indirect_buffer_offset));
    return 0;
}

int mglRenderDrawIndexedPrimitivesIndirect(
    void* render_encoder,
    uint32_t primitive_type,
    uint32_t index_type,
    void* index_buffer,
    uint64_t index_buffer_offset,
    void* indirect_buffer,
    uint64_t indirect_buffer_offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    MTL::Buffer* indirect = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !indices || !indirect) return -1;
    encoder->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<MTL::IndexType>(index_type), indices,
        static_cast<NS::UInteger>(index_buffer_offset), indirect,
        static_cast<NS::UInteger>(indirect_buffer_offset));
    return 0;
}


int mglRenderEncodeDraw(void* render_encoder,
                           const MGLRenderDrawPlan* plan,
                           char* err,
                           size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !plan) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    switch (plan->kind) {
        case MGL_RENDER_DRAW_ARRAY:
            return mglRenderDrawPrimitives(
                render_encoder, plan->primitive_type,
                plan->vertex_start, plan->vertex_count,
                plan->instance_count, plan->base_instance);
        case MGL_RENDER_DRAW_INDEXED:
            return mglRenderDrawIndexedPrimitives(
                render_encoder, plan->primitive_type,
                plan->index_count, plan->index_type, plan->index_buffer,
                plan->index_buffer_offset, plan->instance_count,
                plan->base_vertex, plan->base_instance);
        case MGL_RENDER_DRAW_ARRAY_INDIRECT:
            return mglRenderDrawPrimitivesIndirect(
                render_encoder, plan->primitive_type,
                plan->indirect_buffer, plan->indirect_buffer_offset);
        case MGL_RENDER_DRAW_INDEXED_INDIRECT:
            return mglRenderDrawIndexedPrimitivesIndirect(
                render_encoder, plan->primitive_type, plan->index_type,
                plan->index_buffer, plan->index_buffer_offset,
                plan->indirect_buffer, plan->indirect_buffer_offset);
        case MGL_RENDER_DRAW_PATCHES:
            return mglRenderDrawPatches(
                render_encoder, plan->control_point_count, plan->patch_start,
                plan->patch_count, plan->patch_index_buffer,
                plan->patch_index_buffer_offset, plan->instance_count,
                plan->base_instance);
        case MGL_RENDER_DRAW_INDEXED_PATCHES:
            return mglRenderDrawIndexedPatches(
                render_encoder, plan->control_point_count, plan->patch_start,
                plan->patch_count, plan->patch_index_buffer,
                plan->patch_index_buffer_offset,
                plan->control_point_index_buffer,
                plan->control_point_index_buffer_offset,
                plan->instance_count, plan->base_instance);
        default:
            if (err && errcap) {
                snprintf(err, errcap, "unknown draw plan kind %u",
                         (unsigned)plan->kind);
            }
            return -1;
    }
}

int mglRenderEncodeDrawForRenderEncoderOwner(
    void* render_encoder_owner,
    const MGLRenderDrawPlan* plan,
    char* err,
    size_t errcap) {
    return mglRenderEncodeDraw(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        plan, err, errcap);
}

int mglRenderCreateCullDistanceIndexPlan(
    void* device,
    const void* source_indices,
    uint32_t source_index_type,
    uint64_t source_index_count,
    uint32_t draw_mode,
    int primitive_restart_enabled,
    uint32_t primitive_restart_index,
    int64_t base_vertex,
    int polygon_line_mode,
    void** owner_out,
    void** index_buffer_out,
    uint64_t* primitive_count_out) {
    if (owner_out) *owner_out = nullptr;
    if (index_buffer_out) *index_buffer_out = nullptr;
    if (primitive_count_out) *primitive_count_out = 0;
    if (!source_indices || !owner_out || !index_buffer_out ||
        !primitive_count_out || source_index_count == 0 ||
        source_index_count > static_cast<uint64_t>(SIZE_MAX)) {
        return -1;
    }

    const uint8_t* bytes = static_cast<const uint8_t*>(source_indices);
    std::vector<uint32_t> source;
    std::vector<uint32_t> expanded;
    std::unique_ptr<mgl::CullDistanceIndexPlan> plan(
        new (std::nothrow) mgl::CullDistanceIndexPlan());
    if (!plan) return -1;
    try {
        source.reserve(static_cast<size_t>(source_index_count));
        expanded.reserve(static_cast<size_t>(source_index_count));
        plan->primitives.reserve(static_cast<size_t>(source_index_count));
        for (uint64_t index = 0; index < source_index_count; ++index) {
            uint32_t value = 0;
            if (!mgl::readCullDistanceSourceIndex(
                    bytes, source_index_type, index, value)) {
                return -1;
            }
            source.push_back(value);
        }

        size_t segmentBegin = 0;
        for (size_t index = 0; index <= source.size(); ++index) {
            const bool atEnd = index == source.size();
            const bool atRestart = !atEnd && primitive_restart_enabled &&
                                   source[index] == primitive_restart_index;
            if (!atEnd && !atRestart) continue;
            if (index > segmentBegin &&
                !mgl::appendCullDistanceSegment(
                    source, segmentBegin, index, draw_mode,
                    polygon_line_mode != 0, base_vertex, expanded,
                    plan->primitives)) {
                return -1;
            }
            segmentBegin = index + 1u;
        }
    } catch (...) {
        return -1;
    }

    if (!expanded.empty()) {
        if (expanded.size() > SIZE_MAX / sizeof(uint32_t)) return -1;
        MTL::Device* metalDevice = static_cast<MTL::Device*>(device);
        if (!metalDevice) {
            mgl::Renderer& renderer = mgl::renderer();
            std::lock_guard<std::mutex> lock(renderer.mutex);
            metalDevice = renderer.device;
            if (!metalDevice) return -1;
            plan->indexBuffer = metalDevice->newBuffer(
                expanded.data(), expanded.size() * sizeof(uint32_t),
                MTL::ResourceStorageModeShared);
        } else {
            plan->indexBuffer = metalDevice->newBuffer(
                expanded.data(), expanded.size() * sizeof(uint32_t),
                MTL::ResourceStorageModeShared);
        }
        if (!plan->indexBuffer) return -1;
        plan->indexBuffer->setLabel(NS::String::string(
            "MGL CullDistance expanded indices", NS::UTF8StringEncoding));
    }

    *index_buffer_out = plan->indexBuffer;
    *primitive_count_out = plan->primitives.size();
    *owner_out = plan.release();
    return 0;
}

int mglRenderGetCullDistanceIndexPrimitive(
    void* owner,
    uint64_t primitive_index,
    MGLRenderCullDistancePrimitive* primitive_out) {
    mgl::CullDistanceIndexPlan* plan =
        static_cast<mgl::CullDistanceIndexPlan*>(owner);
    if (!plan || !primitive_out ||
        primitive_index >= plan->primitives.size()) {
        return -1;
    }
    *primitive_out = plan->primitives[primitive_index];
    return 0;
}

void mglRenderDestroyCullDistanceIndexPlan(void** owner) {
    if (!owner || !*owner) return;
    delete static_cast<mgl::CullDistanceIndexPlan*>(*owner);
    *owner = nullptr;
}

int mglRenderSetRenderBuffer(void* render_encoder,
                                void* buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Buffer* resource = static_cast<MTL::Buffer*>(buffer);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexBuffer(resource, static_cast<NS::UInteger>(offset),
                                 index);
    } else {
        encoder->setFragmentBuffer(resource,
                                   static_cast<NS::UInteger>(offset), index);
    }
    return 0;
}


int mglRenderEncodeBindingSnapshot(
    void* render_encoder,
    const MGLRenderBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (snapshot->vertex_op_count >
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS ||
        snapshot->fragment_op_count >
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->vertex_op_count; i++) {
        const MGLRenderBindingOp* op = &snapshot->vertex_ops[i];
        if (op->kind == 0) {
            /* kind 0: set buffer; NULL buffer clears the slot (the ObjC
             * skip paths emit nil clears through the same op). */
            encoder->setVertexBuffer(
                static_cast<MTL::Buffer*>(op->buffer),
                static_cast<NS::UInteger>(op->offset), op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null vertex bytes op %u", i);
                }
                return -1;
            }
            encoder->setVertexBytes(op->bytes, op->length, op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad vertex op kind %u", op->kind);
            }
            return -1;
        }
    }
    for (uint32_t i = 0; i < snapshot->fragment_op_count; i++) {
        const MGLRenderBindingOp* op = &snapshot->fragment_ops[i];
        if (op->kind == 0) {
            encoder->setFragmentBuffer(
                static_cast<MTL::Buffer*>(op->buffer),
                static_cast<NS::UInteger>(op->offset), op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null fragment bytes op %u", i);
                }
                return -1;
            }
            encoder->setFragmentBytes(op->bytes, op->length, op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad fragment op kind %u", op->kind);
            }
            return -1;
        }
    }
    return 0;
}

int mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
    void* render_encoder_owner,
    const MGLRenderBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    return mglRenderEncodeBindingSnapshot(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        snapshot, err, errcap);
}

int mglRenderEncodeResourceBindingSnapshot(
    void* binding_state,
    void* render_encoder,
    const MGLRenderResourceBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!binding_state || !render_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    if (snapshot->vertex_op_count >
            MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS ||
        snapshot->fragment_op_count >
            MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }

    const auto encodeStage = [&](const MGLRenderResourceBindingOp* ops,
                                 uint32_t count,
                                 uint32_t stage) -> int {
        for (uint32_t i = 0; i < count; ++i) {
            const MGLRenderResourceBindingOp& op = ops[i];
            int result = -1;
            if (op.kind == MGL_RENDER_RESOURCE_BINDING_TEXTURE) {
                result = mglRenderBindingSetTexture(
                    binding_state, render_encoder, op.resource, stage,
                    op.index);
            } else if (op.kind == MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
                result = mglRenderBindingSetSampler(
                    binding_state, render_encoder, op.resource, stage,
                    op.index);
            } else {
                if (err && errcap) {
                    snprintf(err, errcap, "bad resource op kind %u at %u",
                             op.kind, i);
                }
                return -1;
            }
            if (result < 0) {
                if (err && errcap) {
                    snprintf(err, errcap,
                             "resource op failed stage=%u kind=%u index=%u",
                             stage, op.kind, op.index);
                }
                return -1;
            }
        }
        return 0;
    };

    if (encodeStage(snapshot->vertex_ops, snapshot->vertex_op_count,
                    MGL_RENDER_BINDING_STAGE_VERTEX) != 0) {
        return -1;
    }
    return encodeStage(snapshot->fragment_ops, snapshot->fragment_op_count,
                       MGL_RENDER_BINDING_STAGE_FRAGMENT);
}

int mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
    void* binding_state,
    void* render_encoder_owner,
    const MGLRenderResourceBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    return mglRenderEncodeResourceBindingSnapshot(
        binding_state,
        mglRenderActiveRenderEncoder(render_encoder_owner),
        snapshot, err, errcap);
}


namespace {
enum {
    kCmdDrawArrays = 0,
    kCmdDrawElements = 1,
    kCmdDrawArraysInstanced = 2,
    kCmdDrawElementsInstanced = 3,
    kCmdDrawElementsBaseVertex = 4,
    kCmdDrawElementsInstancedBaseVertex = 5,
    kCmdDrawArraysInstancedBaseInstance = 6,
    kCmdDrawElementsInstancedBaseInstance = 7,
    kCmdDrawElementsInstancedBaseVertexBaseInstance = 8,
};
}

int mglRenderReplayBatchDraws(void* render_encoder,
                                 const MGLRenderReplayBatch* batch,
                                 char* err,
                                 size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !batch || !batch->commands ||
        batch->command_count == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return MGL_RENDER_REPLAY_BATCH_ERROR;
    }
    if (batch->command_count > MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS) {
        return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLRenderReplayBatchCommand* cmd = &batch->commands[i];
        if (cmd->count == 0) {
            continue;
        }
        MGLRenderDrawPlan plan = {};
        plan.primitive_type = batch->primitive_type;
        switch (cmd->cmd_type) {

            case kCmdDrawArrays:
            case kCmdDrawArraysInstanced:
            case kCmdDrawArraysInstancedBaseInstance:
                plan.kind = MGL_RENDER_DRAW_ARRAY;
                plan.vertex_start = static_cast<uint64_t>(cmd->first);
                plan.vertex_count = cmd->count;
                plan.instance_count = cmd->instance_count;
                plan.base_instance = cmd->base_instance;
                break;

            case kCmdDrawElements:
            case kCmdDrawElementsInstanced:
            case kCmdDrawElementsBaseVertex:
            case kCmdDrawElementsInstancedBaseVertex:
            case kCmdDrawElementsInstancedBaseInstance:
            case kCmdDrawElementsInstancedBaseVertexBaseInstance:
                if (!cmd->index_buffer ||
                    cmd->index_type == 0xFFFFFFFFu) {
                    if (err && errcap) {
                        snprintf(err, errcap,
                                 "replay command %u: unready index buffer",
                                 i);
                    }
                    return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
                }
                plan.kind = MGL_RENDER_DRAW_INDEXED;
                plan.index_count = cmd->count;
                plan.index_type = cmd->index_type;
                plan.index_buffer = cmd->index_buffer;
                plan.index_buffer_offset = cmd->index_buffer_offset;
                plan.base_vertex = cmd->base_vertex;
                plan.instance_count = cmd->instance_count;
                plan.base_instance = cmd->base_instance;
                break;
            default:
                if (err && errcap) {
                    snprintf(err, errcap,
                             "replay command %u: unknown cmd_type %u",
                             i, cmd->cmd_type);
                }
                return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
        }
        if (mglRenderEncodeDraw(render_encoder, &plan, err, errcap) != 0) {
            if (err && errcap && !err[0]) {
                snprintf(err, errcap, "replay command %u encode failed", i);
            }
            return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
        }
    }
    return MGL_RENDER_REPLAY_BATCH_OK;
}

int mglRenderSetRenderBytes(void* render_encoder,
                               const void* bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || (!bytes && length != 0) ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexBytes(bytes, static_cast<NS::UInteger>(length), index);
    } else {
        encoder->setFragmentBytes(bytes, static_cast<NS::UInteger>(length),
                                  index);
    }
    return 0;
}

int mglRenderSetRenderPipelineState(void* render_encoder,
                                       void* pipeline_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::RenderPipelineState* pipeline =
        static_cast<MTL::RenderPipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setRenderPipelineState(pipeline);
    return 0;
}

int mglRenderSetRenderDepthStencilState(void* render_encoder,
                                           void* depth_stencil_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::DepthStencilState* state =
        static_cast<MTL::DepthStencilState*>(depth_stencil_state);
    if (!encoder || !state) return -1;
    encoder->setDepthStencilState(state);
    return 0;
}

int mglRenderSetRenderTexture(void* render_encoder,
                                 void* texture,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Texture* resource = static_cast<MTL::Texture*>(texture);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexTexture(resource, index);
    } else {
        encoder->setFragmentTexture(resource, index);
    }
    return 0;
}

int mglRenderSetRenderSampler(void* render_encoder,
                                 void* sampler,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::SamplerState* resource = static_cast<MTL::SamplerState*>(sampler);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexSamplerState(resource, index);
    } else {
        encoder->setFragmentSamplerState(resource, index);
    }
    return 0;
}

int mglRenderSetRenderViewport(void* render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setViewport(MTL::Viewport(origin_x, origin_y, width, height,
                                       znear, zfar));
    return 0;
}

int mglRenderSetRenderScissor(void* render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setScissorRect(MTL::ScissorRect(x, y, width, height));
    return 0;
}

int mglRenderSetDepthClipMode(void* render_encoder, uint32_t mode) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setDepthClipMode(static_cast<MTL::DepthClipMode>(mode));
    return 0;
}

int mglRenderSetStencilReferenceValues(void* render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setStencilReferenceValues(front_reference, back_reference);
    return 0;
}

int mglRenderSetTessellationFactorBuffer(void* render_encoder,
                                            void* buffer,
                                            uint64_t offset,
                                            uint64_t instance_stride) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* factors = static_cast<MTL::Buffer*>(buffer);
    if (!encoder || !factors) return -1;
    encoder->setTessellationFactorBuffer(
        factors, static_cast<NS::UInteger>(offset),
        static_cast<NS::UInteger>(instance_stride));
    return 0;
}

int mglRenderSetRenderBufferForOwner(void* render_encoder_owner,
                                        void* buffer,
                                        uint64_t offset,
                                        uint32_t stage,
                                        uint32_t index) {
    return mglRenderSetRenderBuffer(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        buffer, offset, stage, index);
}

int mglRenderBindingSetTextureForOwner(void* binding_state,
                                         void* render_encoder_owner,
                                         void* texture,
                                         uint32_t stage,
                                         uint32_t index) {
    return mglRenderBindingSetTexture(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        texture, stage, index);
}

int mglRenderBindingSetSamplerForOwner(void* binding_state,
                                         void* render_encoder_owner,
                                         void* sampler,
                                         uint32_t stage,
                                         uint32_t index) {
    return mglRenderBindingSetSampler(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        sampler, stage, index);
}

int mglRenderBindingSetPipelineIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    void* pipeline_state) {
    return mglRenderBindingSetPipelineIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        pipeline_state);
}

int mglRenderBindingSetDepthStencilIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    void* depth_stencil_state) {
    return mglRenderBindingSetDepthStencilIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_stencil_state);
}

int mglRenderBindingSetCullIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    uint32_t mode) {
    return mglRenderBindingSetCullIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        mode);
}

int mglRenderBindingSetWindingIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    uint32_t winding) {
    return mglRenderBindingSetWindingIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        winding);
}

int mglRenderBindingSetBlendColorIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    float red,
    float green,
    float blue,
    float alpha) {
    return mglRenderBindingSetBlendColorIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        red, green, blue, alpha);
}

int mglRenderBindingSetDepthBiasIfNeededForOwner(
    void* binding_state,
    void* render_encoder_owner,
    float depth_bias,
    float clamp,
    float slope_scale) {
    return mglRenderBindingSetDepthBiasIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_bias, clamp, slope_scale);
}

int mglRenderBindingSetViewportForOwner(void* binding_state,
                                          void* render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar) {
    return mglRenderBindingSetViewport(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        origin_x, origin_y, width, height, znear, zfar);
}

int mglRenderBindingSetViewportsForOwner(void* binding_state,
                                            void* render_encoder_owner,
                                            const double* viewports,
                                            uint64_t count) {
    return mglRenderBindingSetViewports(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        viewports, count);
}

int mglRenderBindingSetScissorForOwner(void* binding_state,
                                         void* render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height) {
    return mglRenderBindingSetScissor(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        x, y, width, height);
}

int mglRenderBindingSetTriangleFillForOwner(void* binding_state,
                                              void* render_encoder_owner,
                                              uint32_t mode) {
    return mglRenderBindingSetTriangleFill(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        mode);
}

int mglRenderSetRenderBytesForOwner(void* render_encoder_owner,
                                       const void* bytes,
                                       size_t length,
                                       uint32_t stage,
                                       uint32_t index) {
    return mglRenderSetRenderBytes(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        bytes, length, stage, index);
}

int mglRenderSetRenderPipelineStateForOwner(void* render_encoder_owner,
                                               void* pipeline_state) {
    return mglRenderSetRenderPipelineState(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        pipeline_state);
}

int mglRenderSetRenderDepthStencilStateForOwner(
    void* render_encoder_owner,
    void* depth_stencil_state) {
    return mglRenderSetRenderDepthStencilState(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_stencil_state);
}

int mglRenderSetRenderTextureForOwner(void* render_encoder_owner,
                                         void* texture,
                                         uint32_t stage,
                                         uint32_t index) {
    return mglRenderSetRenderTexture(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        texture, stage, index);
}

int mglRenderSetRenderSamplerForOwner(void* render_encoder_owner,
                                         void* sampler,
                                         uint32_t stage,
                                         uint32_t index) {
    return mglRenderSetRenderSampler(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        sampler, stage, index);
}

int mglRenderSetRenderViewportForOwner(void* render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar) {
    return mglRenderSetRenderViewport(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        origin_x, origin_y, width, height, znear, zfar);
}

int mglRenderSetRenderScissorForOwner(void* render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height) {
    return mglRenderSetRenderScissor(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        x, y, width, height);
}

int mglRenderSetDepthClipModeForOwner(void* render_encoder_owner,
                                         uint32_t mode) {
    return mglRenderSetDepthClipMode(
        mglRenderActiveRenderEncoder(render_encoder_owner), mode);
}

int mglRenderSetStencilReferenceValuesForOwner(
    void* render_encoder_owner,
    uint32_t front_reference,
    uint32_t back_reference) {
    return mglRenderSetStencilReferenceValues(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        front_reference, back_reference);
}

int mglRenderSetTessellationFactorBufferForOwner(
    void* render_encoder_owner,
    void* buffer,
    uint64_t offset,
    uint64_t instance_stride) {
    return mglRenderSetTessellationFactorBuffer(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        buffer, offset, instance_stride);
}

int mglRenderDrawPatches(void* render_encoder,
                            uint64_t control_point_count,
                            uint64_t patch_start,
                            uint64_t patch_count,
                            void* patch_index_buffer,
                            uint64_t patch_index_buffer_offset,
                            uint64_t instance_count,
                            uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || control_point_count == 0 || instance_count == 0) return -1;
    encoder->drawPatches(
        static_cast<NS::UInteger>(control_point_count),
        static_cast<NS::UInteger>(patch_start),
        static_cast<NS::UInteger>(patch_count),
        static_cast<MTL::Buffer*>(patch_index_buffer),
        static_cast<NS::UInteger>(patch_index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawIndexedPatches(void* render_encoder,
                                   uint64_t control_point_count,
                                   uint64_t patch_start,
                                   uint64_t patch_count,
                                   void* patch_index_buffer,
                                   uint64_t patch_index_buffer_offset,
                                   void* control_point_index_buffer,
                                   uint64_t control_point_index_buffer_offset,
                                   uint64_t instance_count,
                                   uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || patch_count == 0 || instance_count == 0 ||
        !control_point_index_buffer) return -1;
    encoder->drawIndexedPatches(
        static_cast<NS::UInteger>(control_point_count),
        static_cast<NS::UInteger>(patch_start),
        static_cast<NS::UInteger>(patch_count),
        static_cast<MTL::Buffer*>(patch_index_buffer),
        static_cast<NS::UInteger>(patch_index_buffer_offset),
        static_cast<MTL::Buffer*>(control_point_index_buffer),
        static_cast<NS::UInteger>(control_point_index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderCreateIndirectCommandBuffer(
    uint32_t command_types,
    int inherit_pipeline_state,
    int inherit_buffers,
    uint32_t max_vertex_buffer_bind_count,
    uint32_t max_fragment_buffer_bind_count,
    uint64_t max_command_count,
    uint64_t resource_options,
    void** indirect_buffer_out) {
    if (indirect_buffer_out) *indirect_buffer_out = nullptr;
    if (!indirect_buffer_out || max_command_count == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::IndirectCommandBufferDescriptor* allocated =
        MTL::IndirectCommandBufferDescriptor::alloc();
    if (!allocated) return -1;
    MTL::IndirectCommandBufferDescriptor* descriptor = allocated->init();
    if (!descriptor) return -1;
    descriptor->setCommandTypes(
        static_cast<MTL::IndirectCommandType>(command_types));
    descriptor->setInheritPipelineState(inherit_pipeline_state != 0);
    descriptor->setInheritBuffers(inherit_buffers != 0);
    descriptor->setMaxVertexBufferBindCount(max_vertex_buffer_bind_count);
    descriptor->setMaxFragmentBufferBindCount(max_fragment_buffer_bind_count);
    MTL::IndirectCommandBuffer* buffer = renderer.device->newIndirectCommandBuffer(
        descriptor, static_cast<NS::UInteger>(max_command_count),
        static_cast<MTL::ResourceOptions>(resource_options));
    descriptor->release();
    if (!buffer) return -1;
    *indirect_buffer_out = buffer;
    return 0;
}

int mglRenderResetIndirectCommandBuffer(void* indirect_buffer,
                                           uint64_t location,
                                           uint64_t length) {
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!buffer || length == 0) return -1;
    buffer->reset(NS::Range(location, length));
    return 0;
}

int mglRenderGetIndirectRenderCommand(void* indirect_buffer,
                                         uint64_t command_index,
                                         void** command_out) {
    if (command_out) *command_out = nullptr;
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!buffer || !command_out) return -1;
    MTL::IndirectRenderCommand* command =
        buffer->indirectRenderCommand(command_index);
    if (!command) return -1;
    *command_out = command;
    return 0;
}

int mglRenderSetIndirectDrawIndexed(void* indirect_command,
                                       uint32_t primitive_type,
                                       uint64_t index_count,
                                       uint32_t index_type,
                                       void* index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance) {
    MTL::IndirectRenderCommand* command =
        static_cast<MTL::IndirectRenderCommand*>(indirect_command);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    if (!command || !indices || instance_count == 0) return -1;
    command->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type), index_count,
        static_cast<MTL::IndexType>(index_type), indices,
        index_buffer_offset, instance_count,
        static_cast<NS::Integer>(base_vertex), base_instance);
    return 0;
}

int mglRenderSetIndirectDraw(void* indirect_command,
                                uint32_t primitive_type,
                                uint64_t vertex_start,
                                uint64_t vertex_count,
                                uint64_t instance_count,
                                uint64_t base_instance) {
    MTL::IndirectRenderCommand* command =
        static_cast<MTL::IndirectRenderCommand*>(indirect_command);
    if (!command || instance_count == 0) return -1;
    command->drawPrimitives(static_cast<MTL::PrimitiveType>(primitive_type),
                            vertex_start, vertex_count, instance_count,
                            base_instance);
    return 0;
}

int mglRenderUseRenderResource(void* render_encoder,
                                  void* resource,
                                  uint32_t usage,
                                  uint32_t stages) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Resource* object = static_cast<MTL::Resource*>(resource);
    if (!encoder || !object) return -1;
    encoder->useResource(object, static_cast<MTL::ResourceUsage>(usage),
                         static_cast<MTL::RenderStages>(stages));
    return 0;
}

int mglRenderExecuteIndirectCommands(void* render_encoder,
                                        void* indirect_buffer,
                                        uint64_t location,
                                        uint64_t length) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!encoder || !buffer || length == 0) return -1;
    encoder->executeCommandsInBuffer(buffer, NS::Range(location, length));
    return 0;
}

int mglRenderReplayBatchDrawsForRenderEncoderOwner(
    void* render_encoder_owner,
    const MGLRenderReplayBatch* batch,
    char* err,
    size_t errcap) {
    return mglRenderReplayBatchDraws(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        batch, err, errcap);
}

int mglRenderUseRenderResourceForOwner(void* render_encoder_owner,
                                          void* resource,
                                          uint32_t usage,
                                          uint32_t stages) {
    return mglRenderUseRenderResource(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        resource, usage, stages);
}

int mglRenderExecuteIndirectCommandsForOwner(void* render_encoder_owner,
                                                 void* indirect_buffer,
                                                 uint64_t location,
                                                 uint64_t length) {
    return mglRenderExecuteIndirectCommands(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        indirect_buffer, location, length);
}

const char *mglRenderVertexFormatName(uint32_t format) {
    switch (static_cast<MTL::VertexFormat>(format)) {
        case MTL::VertexFormatFloat: return "Float";
        case MTL::VertexFormatFloat2: return "Float2";
        case MTL::VertexFormatFloat3: return "Float3";
        case MTL::VertexFormatFloat4: return "Float4";
        case MTL::VertexFormatUChar4: return "UChar4";
        case MTL::VertexFormatUChar4Normalized: return "UChar4Normalized";
        case MTL::VertexFormatUChar3: return "UChar3";
        case MTL::VertexFormatUChar3Normalized: return "UChar3Normalized";
        case MTL::VertexFormatUChar2: return "UChar2";
        case MTL::VertexFormatUChar2Normalized: return "UChar2Normalized";
        case MTL::VertexFormatUChar: return "UChar";
        case MTL::VertexFormatUCharNormalized: return "UCharNormalized";
        case MTL::VertexFormatShort: return "Short";
        case MTL::VertexFormatShort2: return "Short2";
        case MTL::VertexFormatShort3: return "Short3";
        case MTL::VertexFormatShort4: return "Short4";
        case MTL::VertexFormatShortNormalized: return "ShortNormalized";
        case MTL::VertexFormatShort2Normalized: return "Short2Normalized";
        case MTL::VertexFormatShort3Normalized: return "Short3Normalized";
        case MTL::VertexFormatShort4Normalized: return "Short4Normalized";
        case MTL::VertexFormatUShort: return "UShort";
        case MTL::VertexFormatUShort2: return "UShort2";
        case MTL::VertexFormatUShort3: return "UShort3";
        case MTL::VertexFormatUShort4: return "UShort4";
        case MTL::VertexFormatUShortNormalized: return "UShortNormalized";
        case MTL::VertexFormatUShort2Normalized: return "UShort2Normalized";
        case MTL::VertexFormatUShort3Normalized: return "UShort3Normalized";
        case MTL::VertexFormatUShort4Normalized: return "UShort4Normalized";
        case MTL::VertexFormatUInt1010102Normalized: return "UInt1010102Normalized";
        case MTL::VertexFormatInt1010102Normalized: return "Int1010102Normalized";
        default: return "Unknown";
    }
}

uint64_t mglRenderVertexDescriptorSignature(const void *descriptor) {
    const MTL::VertexDescriptor *vertex =
        static_cast<const MTL::VertexDescriptor *>(descriptor);
    uint64_t hash = 1469598103934665603ull;
    if (!vertex) return hash;
    MTL::VertexAttributeDescriptorArray *attributes = vertex->attributes();
    MTL::VertexBufferLayoutDescriptorArray *layouts = vertex->layouts();
    for (uint32_t i = 0; i < 32u; ++i) {
        MTL::VertexAttributeDescriptor *attrib = attributes ? attributes->object(i) : nullptr;
        if (!attrib) continue;
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->format()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->offset()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->bufferIndex()));
    }
    for (uint32_t i = 0; i < 31u; ++i) {
        MTL::VertexBufferLayoutDescriptor *layout = layouts ? layouts->object(i) : nullptr;
        if (!layout) continue;
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stride()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stepFunction()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stepRate()));
    }
    return hash;
}

uint64_t mglRenderPipelineDescriptorSignature(const void *descriptor) {
    const MTL::RenderPipelineDescriptor *pipeline =
        static_cast<const MTL::RenderPipelineDescriptor *>(descriptor);
    uint64_t hash = 1469598103934665603ull;
    if (!pipeline) return hash;
    hash = mglRenderHashStepU64(hash, pipeline->rasterSampleCount());
    hash = mglRenderHashStepU64(hash, pipeline->isRasterizationEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->isAlphaToCoverageEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->isAlphaToOneEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->depthAttachmentPixelFormat());
    hash = mglRenderHashStepU64(hash, pipeline->stencilAttachmentPixelFormat());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationPartitionMode());
    hash = mglRenderHashStepU64(hash, pipeline->maxTessellationFactor());
    hash = mglRenderHashStepU64(hash, pipeline->isTessellationFactorScaleEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationFactorFormat());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationControlPointIndexType());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationFactorStepFunction());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationOutputWindingOrder());
    MTL::RenderPipelineColorAttachmentDescriptorArray *attachments = pipeline->colorAttachments();
    for (uint32_t i = 0; i < 8u; ++i) {
        MTL::RenderPipelineColorAttachmentDescriptor *attachment =
            attachments ? attachments->object(i) : nullptr;
        if (!attachment) continue;
        hash = mglRenderHashStepU64(hash, attachment->pixelFormat());
        hash = mglRenderHashStepU64(hash, attachment->isBlendingEnabled());
        hash = mglRenderHashStepU64(hash, attachment->sourceRGBBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->destinationRGBBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->rgbBlendOperation());
        hash = mglRenderHashStepU64(hash, attachment->sourceAlphaBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->destinationAlphaBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->alphaBlendOperation());
        hash = mglRenderHashStepU64(hash, attachment->writeMask());
    }
    return hash;
}

bool mglRenderPassAttachmentMatchesSubresource(
    const void *descriptor,
    const MGLMetalAttachmentSubresource *subresource) {
    const MTL::RenderPassAttachmentDescriptor *attachment =
        static_cast<const MTL::RenderPassAttachmentDescriptor *>(descriptor);
    if (!attachment || !subresource) return false;
    return static_cast<uint64_t>(attachment->level()) == subresource->level &&
           static_cast<uint64_t>(attachment->slice()) == subresource->slice &&
           static_cast<uint64_t>(attachment->depthPlane()) == subresource->depthPlane;
}

const char *mglRenderCommandBufferStatusName(uint32_t status) {
    switch (static_cast<MTL::CommandBufferStatus>(status)) {
        case MTL::CommandBufferStatusNotEnqueued: return "NotEnqueued";
        case MTL::CommandBufferStatusEnqueued: return "Enqueued";
        case MTL::CommandBufferStatusCommitted: return "Committed";
        case MTL::CommandBufferStatusScheduled: return "Scheduled";
        case MTL::CommandBufferStatusCompleted: return "Completed";
        case MTL::CommandBufferStatusError: return "Error";
        default: return "Unknown";
    }
}

const char *mglRenderLoadActionName(uint32_t action) {
    switch (static_cast<MTL::LoadAction>(action)) {
        case MTL::LoadActionDontCare: return "DontCare";
        case MTL::LoadActionLoad: return "Load";
        case MTL::LoadActionClear: return "Clear";
        default: return "Unknown";
    }
}

const char *mglRenderStoreActionName(uint32_t action) {
    switch (static_cast<MTL::StoreAction>(action)) {
        case MTL::StoreActionDontCare: return "DontCare";
        case MTL::StoreActionStore: return "Store";
        case MTL::StoreActionMultisampleResolve: return "MSResolve";
        case MTL::StoreActionStoreAndMultisampleResolve: return "Store+MSResolve";
        case MTL::StoreActionUnknown: return "Unknown";
        default: return "Other";
    }
}

} // extern "C"
