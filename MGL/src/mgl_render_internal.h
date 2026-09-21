/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_INTERNAL_H
#define MGL_RENDER_INTERNAL_H

#include <atomic>
#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <objc/runtime.h>

#include "mgl_renderer_backend.h"

extern "C" void mglMetalCountRelease(int kind);
extern "C" void mglMetalCountCreate(int kind);
extern "C" void mglRecordBufferCowSnapshot(uint64_t bytes);

/* Types shared by the render translation units. Include after mgl_metal.h,
 * mgl_renderer_backend.h, and glm_context.h. Not for C translation units. */

namespace mgl {


/* kMGLMaxBufferSlots / kMGLMinimumStageBindingSize live in mgl_buffer_slots.h
 * (the C-safe home) so the C batch drivers and this file cannot drift. */
constexpr size_t kPackedStructBufferCapacity = 128;
constexpr size_t kMinimumStageBindingSize = kMGLMinimumStageBindingSize;

enum MetalObjectKind {
    kMetalKindBuffer = 0,
    kMetalKindTexture = 1,
    kMetalKindSampler = 2,
    kMetalKindLibrary = 3,
    kMetalKindFunction = 4,
    kMetalKindPipeline = 5,
    kMetalKindOther = 6,
};

inline int metalObjectKind(void* object) {
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

inline void releaseBridgedObject(void** slot) {
    if (!slot || !*slot) return;
    void* object = *slot;
    *slot = nullptr;
    mglMetalCountRelease(metalObjectKind(object));
    static_cast<NS::Object*>(object)->release();
}

inline int loadAIRMainFunction(MTL::Device* device,
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

inline std::atomic<uint64_t> gBufferFrameGeneration{0};
inline std::atomic<uint64_t> gBufferCompletedGeneration{0};

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

inline uint64_t hashVertexBytes(const uint8_t* bytes, size_t length) {
    uint64_t hash = 1469598103934665603ull;
    if (!bytes) return hash;
    for (size_t i = 0; i < length; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

inline bool alignVertexStride(size_t stride, size_t* alignedOut) {
    if (!alignedOut || stride > std::numeric_limits<size_t>::max() - 3u) {
        return false;
    }
    *alignedOut = (stride + 3u) & ~size_t{3u};
    return *alignedOut != 0;
}

inline size_t vertexComponentSize(uint32_t type) {
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

inline float decodeUnsignedFloatComponent(uint32_t value, uint32_t mantissaBits) {
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

inline bool vertexConversionSource(Buffer* source,
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

inline BufferCowPool* bufferCowPool(Buffer* owner, bool create) {
    if (!owner) return nullptr;
    BufferCowPool* pool =
        static_cast<BufferCowPool*>(owner->mtl_cow_pool);
    if (!pool && create) {
        pool = new (std::nothrow) BufferCowPool();
        owner->mtl_cow_pool = pool;
    }
    return pool;
}

inline BufferCowSnapshot takeBufferCowSnapshot(MTL::Device* device,
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

inline bool bufferShadowUploadRange(const Buffer* buffer,
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

inline void installBufferCowSnapshot(Buffer* owner,
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

inline void setLastSubmitted(CommandBufferOwner* owner,
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

inline void retainCommandRecoveryOwner(CommandBufferRecoveryOwner* owner) {
    if (owner) {
        owner->references.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void releaseCommandRecoveryOwner(CommandBufferRecoveryOwner* owner) {
    if (owner && owner->references.fetch_sub(
                      1, std::memory_order_acq_rel) == 1) {
        delete owner;
    }
}

inline void snapshotCommandRecovery(
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

inline void retainRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->retain();
}

inline void releaseRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->release();
}

inline void retainRenderPassStateResources(
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

inline void releaseRenderPassStateResources(
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

inline void copyString(NS::String* string, char* out, size_t capacity) {
    if (!out || capacity == 0) return;
    out[0] = '\0';
    if (!string) return;
    const char* value = string->utf8String();
    if (value) snprintf(out, capacity, "%s", value);
}

inline int snapshotCommandBufferState(
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
inline Renderer& renderer() {
    static Renderer* instance = new Renderer();
    return *instance;
}

inline void copyError(NS::Error* error, char* out, size_t capacity) {
    if (!out || capacity == 0) return;
    if (!error) {
        snprintf(out, capacity, "unknown Metal error");
        return;
    }
    const char* message = nullptr;
    if (error->localizedDescription())
        message = error->localizedDescription()->utf8String();
    const char* domain = nullptr;
    if (error->domain()) domain = error->domain()->utf8String();
    const char* reason = nullptr;
    if (error->localizedFailureReason())
        reason = error->localizedFailureReason()->utf8String();
    const char* info = nullptr;
    if (NS::Dictionary* userInfo = error->userInfo()) {
        if (NS::String* desc = userInfo->description())
            info = desc->utf8String();
    }
    /* Prefer a dense single-line dump so CI logs keep domain/code/userInfo
     * even when localizedDescription is the opaque "Compilation failed". */
    if (message && message[0] && (domain || reason || info)) {
        snprintf(out, capacity,
                 "%s (domain=%s code=%ld reason=%s userInfo=%s)",
                 message, domain ? domain : "?", (long)error->code(),
                 reason && reason[0] ? reason : "-",
                 info && info[0] ? info : "-");
        return;
    }
    if (message && message[0]) {
        snprintf(out, capacity, "%s", message);
        return;
    }
    snprintf(out, capacity, "unknown Metal error (domain=%s code=%ld)",
             domain ? domain : "?", (long)error->code());
}

inline void releasePipelineCaches(Renderer& renderer) {
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

inline void releaseBindingStates(Renderer& renderer) {
    for (BindingState* state : renderer.bindingStates) {
        delete state;
    }
    renderer.bindingStates.clear();
}

inline void releasePackedStructBuffers(Renderer& renderer) {
    for (Buffer*& buffer : renderer.packedStructBuffers) {
        if (!buffer) continue;
        releaseBridgedObject(&buffer->data.mtl_data);
        std::free(buffer);
        buffer = nullptr;
    }
    renderer.packedStructBufferIndex = 0;
}

inline void recordBindingResult(BindingState& state, uint32_t setter, bool emitted) {
    if (emitted) {
        state.stats.emitted[setter]++;
    } else {
        state.stats.skipped[setter]++;
    }
}

inline bool viewportEqual(const MTL::Viewport& lhs, const MTL::Viewport& rhs) {
    return lhs.originX == rhs.originX && lhs.originY == rhs.originY &&
           lhs.width == rhs.width && lhs.height == rhs.height &&
           lhs.znear == rhs.znear && lhs.zfar == rhs.zfar;
}

inline bool scissorEqual(const MTL::ScissorRect& lhs, const MTL::ScissorRect& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width &&
           lhs.height == rhs.height;
}

inline MTL::TextureDescriptor* newTextureDescriptor(
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

inline void applyRenderPassAttachmentState(
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

inline MTL::RenderPassDescriptor* newRenderPassDescriptor(
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

inline MGLRenderPassState defaultRenderPassState() {
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

inline bool readCullDistanceSourceIndex(const uint8_t* bytes,
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

inline bool appendCullDistancePrimitive(
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

inline bool appendCullDistanceSegment(
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


} // namespace mgl

inline MGLRendererBackendHandle* rendererBackend(GLMContext context) {
    return context
        ? static_cast<MGLRendererBackendHandle*>(context->renderer_backend)
        : nullptr;
}

struct BackendLeaseScope {
    MGLRendererBackendLease lease{};
    bool held = false;

    explicit BackendLeaseScope(GLMContext context)
    {
        held = mglRendererBackendBeginContext(context, &lease) == 0;
    }

    ~BackendLeaseScope()
    {
        if (held) mglRendererBackendEnd(&lease);
    }

    BackendLeaseScope(const BackendLeaseScope&) = delete;
    BackendLeaseScope& operator=(const BackendLeaseScope&) = delete;
};

inline void* rendererOwner(GLMContext context, MGLRendererBackendOwnerKind kind) {
    return mglRendererBackendGetOwner(rendererBackend(context), kind);
}

/* Cross-TU helpers formerly file-static in mgl_render.cpp. */
void* mglRenderActiveRenderEncoder(MGLRenderEncoderOwner* owner_handle);
MTL::Library* loadAuxLibraryLocked(mgl::Renderer& renderer,
                                   const unsigned char* bytes,
                                   size_t size,
                                   uint64_t asset_hash,
                                   char* err,
                                   size_t errcap);
MTL::Function* newAuxEntryFunction(MTL::Library* library,
                                   const char* entry,
                                   char* err,
                                   size_t errcap);
int getOrCreateAuxComputePipelineLocked(mgl::Renderer& renderer,
                                        void* function,
                                        uint32_t kind,
                                        uint64_t variant,
                                        void** pipeline_out,
                                        char* err,
                                        size_t errcap);
MTL::DepthStencilState* mglRenderCreateDepthStencilFromStateLocked(
    mgl::Renderer& renderer,
    const MGLRenderDepthStencilDescriptorState& state);
MGLRenderStencilDescriptorState mglRenderDescribeStencilDescriptor(
    const MTL::StencilDescriptor* descriptor);
MTL::StencilDescriptor* mglRenderBuildStencilDescriptor(
    const MGLRenderStencilDescriptorState& state);
uint32_t mglReadPackedUploadLE(const uint8_t* src, size_t bytes);
int mglWideSrcChannelCount(MTL::PixelFormat pf);

inline uint32_t MGLRenderReadIndexBytes(const uint8_t* bytes, int w,
                                       uint32_t i) {
    return (w == 1) ? (uint32_t)bytes[i]
                    : (w == 2) ? (uint32_t)((const uint16_t*)bytes)[i]
                               : (uint32_t)((const uint32_t*)bytes)[i];
}

MGLRenderPassAttachmentState* mglRenderAttachmentForOwner(
    mgl::RenderPassStateOwner* owner,
    uint32_t attachment_kind,
    uint32_t color_index);

#endif
