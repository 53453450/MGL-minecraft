//------------------------------------------------------------------------------------------------
// mgl_render_cpp.cpp — Metal-cpp 渲染门面与 renderer-owned PSO 缓存
//
// 本 TU 是 NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION 的唯一定义点
// （私有类/选择器符号由此产出）；其他 TU 仅 include mgl_metal_cpp.h 拿声明。
//
// 持有桥接后的 MTL::Device*，并逐步接管 renderer-owned Metal caches。
//------------------------------------------------------------------------------------------------
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mgl_metal_cpp.h"
#include "mgl_render_cpp.h"
#include "mgl_air_loader.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"

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
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <objc/runtime.h>

extern "C" void mglMetalCountRelease(int kind);
extern "C" void mglMetalCountCreate(int kind);
extern "C" void mglRecordBufferCowSnapshot(uint64_t bytes);
extern "C" void mtlBindBuffer(GLMContext glm_ctx, Buffer* buffer);
extern "C" void mtlBufferSubData(GLMContext glm_ctx,
                                  Buffer* buffer,
                                  size_t offset,
                                  size_t size,
                                  const void* bytes);
extern "C" void* mtlMapUnmapBuffer(GLMContext glm_ctx,
                                     Buffer* buffer,
                                     size_t offset,
                                     size_t size,
                                     unsigned int access,
                                     bool map);
extern "C" void mtlFlushBufferRange(GLMContext glm_ctx,
                                     Buffer* buffer,
                                     intptr_t offset,
                                     intptr_t length);
extern "C" void mtlBindProgram(GLMContext glm_ctx, Program* program);

namespace mgl {

MTL::Device* wrapDevice(void* objcDevice) {
    // MTL::Device* 与 id<MTLDevice> 指针同地址（metal-cpp 薄包装）。
    // C++ 侧 +1 retain，所有权由渲染器持有；ObjC 侧保留自己那份。
    MTL::Device* device = static_cast<MTL::Device*>(objcDevice);
    if (device) {
        device->retain();
    }
    return device;
}

namespace {

constexpr uint32_t kMGLCppMaxBufferSlots = 31;
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

bool geometryShaderIsPassthrough(const Shader* shader) {
    const char* source = shader ? shader->src : nullptr;
    if (!source) return false;
    return std::strstr(source, "EmitVertex()") &&
           std::strstr(source, "EndPrimitive()") &&
           std::strstr(source,
                       "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !std::strstr(source, "gl_PrimitiveID") &&
           !std::strstr(source, "gl_Layer") &&
           !std::strstr(source, "gl_ViewportIndex");
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
    std::array<uint64_t, MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS> words{};

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
    ~PipelineCacheDescriptorEntry() {
        if (descriptor) descriptor->release();
    }

    MTL::RenderPipelineDescriptor* descriptor = nullptr;
};

struct PipelineCacheDepthStencilEntry {
    ~PipelineCacheDepthStencilEntry() {
        if (state) state->release();
    }

    MTL::DepthStencilState* state = nullptr;
};

struct PipelineCacheOwner {
    ~PipelineCacheOwner() { reset(); }

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

    static void retainObject(void* object) {
        if (object) static_cast<NS::Object*>(object)->retain();
    }

    static void releaseObject(void*& object) {
        if (object) static_cast<NS::Object*>(object)->release();
        object = nullptr;
    }

    static PipelineCacheKey makeKey(
        const uint64_t words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS]) {
        PipelineCacheKey key;
        if (words) {
            std::copy(words,
                      words + MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS,
                      key.words.begin());
        }
        return key;
    }

    static DepthStencilCacheKey makeDepthStencilKey(
        const MGLRenderCppDepthStencilDescriptorState& descriptor) {
        DepthStencilCacheKey key;
        key.words[0] = descriptor.depth_compare_function;
        key.words[1] = descriptor.depth_write_enabled;
        const MGLRenderCppStencilDescriptorState* stencils[] = {
            &descriptor.front, &descriptor.back};
        size_t cursor = 2;
        for (const MGLRenderCppStencilDescriptorState* stencil : stencils) {
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
    MGLRenderCppPipelineActiveState active{
        nullptr, nullptr, nullptr,
        static_cast<uint32_t>(MTL::PixelFormatInvalid),
        static_cast<uint32_t>(MTL::PixelFormatInvalid),
        static_cast<uint32_t>(MTL::PixelFormatInvalid), 0};
    std::array<MGLRenderCppPipelineBlendState,
               MGL_RENDER_CPP_PIPELINE_COLOR_ATTACHMENTS> blend{};
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
        static_cast<BufferCowPool*>(owner->mtl_cpp_cow_pool);
    if (!pool && create) {
        pool = new (std::nothrow) BufferCowPool();
        owner->mtl_cpp_cow_pool = pool;
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
        : vertexBuffers(kMGLCppMaxBufferSlots, nullptr),
          fragmentBuffers(kMGLCppMaxBufferSlots, nullptr),
          vertexBufferOffsets(kMGLCppMaxBufferSlots, 0),
          fragmentBufferOffsets(kMGLCppMaxBufferSlots, 0),
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
    MGLRenderCppBindingStats stats = {};
};

struct RendererCpp {
    MTL::Device* device = nullptr;
    uint32_t users = 0;
    std::mutex mutex;
    std::map<ComputePipelineKey, MTL::ComputePipelineState*> computePipelines;
    std::map<AuxComputePipelineKey, MTL::ComputePipelineState*>
        auxComputePipelines;
    std::map<AuxRenderPipelineKey, MTL::RenderPipelineState*>
        auxRenderPipelines;
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

struct CommandBufferOwner {
    ~CommandBufferOwner() {
        if (current) current->release();
    }

    MTL::CommandBuffer* current = nullptr;
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
    MGLRenderCppRenderPassIdentityState state{};
    MGLRenderCppFboMatchCacheState cache{};
    bool cache_valid = false;
};

void retainRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->retain();
}

void releaseRenderPassObject(void* object) {
    if (object) static_cast<NS::Object*>(object)->release();
}

void retainRenderPassStateResources(
    const MGLRenderCppRenderPassState& state) {
    for (uint32_t index = 0;
         index < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; ++index) {
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
    const MGLRenderCppRenderPassState& state) {
    for (uint32_t index = 0;
         index < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; ++index) {
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

    MGLRenderCppRenderPassState state{};
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
    MGLRenderCppCommandBufferState* state) {
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

    void destroy() {
        void* value = std::exchange(context, nullptr);
        MGLRenderCppDestroyContext destroyFunction =
            std::exchange(destroyContext, nullptr);
        if (value && destroyFunction) destroyFunction(value);
    }

    void complete(MTL::CommandBuffer* commandBuffer) {
        MGLRenderCppCommandBufferState state = {};
        snapshotCommandBufferState(commandBuffer, &state);
        void* callbackContext = context;
        struct DestroyGuard {
            CommandBufferCompletionContext* owner;
            ~DestroyGuard() { owner->destroy(); }
        } guard{this};
        callback(callbackContext, &state);
    }

    MGLRenderCppCommandBufferCompletion callback = nullptr;
    void* context = nullptr;
    MGLRenderCppDestroyContext destroyContext = nullptr;
};

/* C auto-cleanup may call renderer shutdown after ordinary C++ static
 * destruction. Keep the container alive for the process and release Metal
 * objects only from the explicit shutdown boundary. */
RendererCpp& renderer() {
    static RendererCpp* instance = new RendererCpp();
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

void releasePipelineCaches(RendererCpp& renderer) {
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
    for (auto& entry : renderer.convertedVertexBuffers) {
        if (entry.second) entry.second->release();
    }
    renderer.convertedVertexBuffers.clear();
}

void releaseBindingStates(RendererCpp& renderer) {
    for (BindingState* state : renderer.bindingStates) {
        delete state;
    }
    renderer.bindingStates.clear();
}

void releasePackedStructBuffers(RendererCpp& renderer) {
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
    const MGLRenderCppTextureDescriptorState* state) {
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
    descriptor->setSwizzle(MTL::TextureSwizzleChannels(
        static_cast<MTL::TextureSwizzle>(state->swizzle_red),
        static_cast<MTL::TextureSwizzle>(state->swizzle_green),
        static_cast<MTL::TextureSwizzle>(state->swizzle_blue),
        static_cast<MTL::TextureSwizzle>(state->swizzle_alpha)));
    return descriptor;
}

void applyRenderPassAttachmentState(
    MTL::RenderPassAttachmentDescriptor* attachment,
    const MGLRenderCppRenderPassAttachmentState& state) {
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
    const MGLRenderCppRenderPassState* state) {
    if (!state) return nullptr;
    MTL::RenderPassDescriptor* descriptor =
        MTL::RenderPassDescriptor::alloc()->init();
    if (!descriptor) return nullptr;

    MTL::RenderPassColorAttachmentDescriptorArray* colors =
        descriptor->colorAttachments();
    for (uint32_t index = 0;
         index < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; ++index) {
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
        static_cast<uint32_t>(MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS));
    if (sampleCount > 0) {
        MTL::SamplePosition positions[MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS];
        for (uint32_t index = 0; index < sampleCount; ++index) {
            positions[index] = MTL::SamplePosition::Make(
                state->sample_positions[index].x,
                state->sample_positions[index].y);
        }
        descriptor->setSamplePositions(positions, sampleCount);
    }
    return descriptor;
}

MGLRenderCppRenderPassState defaultRenderPassState() {
    MGLRenderCppRenderPassState state = {};
    for (uint32_t index = 0;
         index < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; ++index) {
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
    std::vector<MGLRenderCppCullDistancePrimitive> primitives;
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
    std::vector<MGLRenderCppCullDistancePrimitive>& primitives,
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

    MGLRenderCppCullDistancePrimitive primitive = {};
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
    std::vector<MGLRenderCppCullDistancePrimitive>& primitives) {
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
// 纯 C 入口（mgl_render_cpp.h）
//------------------------------------------------------------------------------------------------
extern "C" {

void mglRenderCppInitDefaultRenderPassState(
    MGLRenderCppRenderPassState* state_out) {
    if (!state_out) return;
    *state_out = mgl::defaultRenderPassState();
}

int mglRenderCppInit(void* objc_device) {
    if (!objc_device) {
        return -1;
    }
    mgl::RendererCpp& renderer = mgl::renderer();
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

void mglRenderCppShutdown(void) {
    mgl::RendererCpp& renderer = mgl::renderer();
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

void* mglRenderCppGetDevice(void) {
    return mgl::renderer().device;
}

void mglRenderCppDeleteMTLObj(GLMContext glm_ctx, void* object) {
    (void)glm_ctx;
    mgl::releaseBridgedObject(&object);
}

void mglRenderCppReleaseBufferMetalData(GLMContext glm_ctx, Buffer* buffer) {
    if (!buffer || !buffer->data.mtl_data) return;
    (void)glm_ctx;
    mgl::releaseBridgedObject(&buffer->data.mtl_data);
}

void mglRenderCppReleaseBufferCowPool(Buffer* buffer) {
    if (!buffer || !buffer->mtl_cpp_cow_pool) return;
    mgl::BufferCowPool* pool =
        static_cast<mgl::BufferCowPool*>(buffer->mtl_cpp_cow_pool);
    buffer->mtl_cpp_cow_pool = nullptr;
    delete pool;
}

Buffer* mglRenderCppAcquirePackedStructBuffer(const void* data,
                                               size_t size,
                                               char* err,
                                               size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!data || size == 0) {
        if (err && errcap) snprintf(err, errcap, "invalid packed struct data");
        return nullptr;
    }

    mgl::RendererCpp& renderer = mgl::renderer();
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

uint64_t mglRenderCppAdvanceBufferGeneration(void) {
    return mgl::gBufferFrameGeneration.fetch_add(
               1, std::memory_order_acq_rel) + 1;
}

void mglRenderCppRecordBufferGenerationCompleted(uint64_t generation) {
    uint64_t completed =
        mgl::gBufferCompletedGeneration.load(std::memory_order_relaxed);
    while (generation > completed &&
           !mgl::gBufferCompletedGeneration.compare_exchange_weak(
               completed, generation, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
}

uint64_t mglRenderCppCompletedBufferGeneration(void) {
    return mgl::gBufferCompletedGeneration.load(std::memory_order_acquire);
}

void mglRenderCppNoteBufferEncoded(Buffer* buffer) {
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

int mglRenderCppSnapshotSharedDirtyBuffer(Buffer* buffer,
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
    mglRenderCppNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderCppSnapshotSharedBufferRange(Buffer* buffer,
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
    mglRenderCppNoteBufferEncoded(buffer);
    mglRecordBufferCowSnapshot(metalLength);
    *metal_buffer_out = buffer->data.mtl_data;
    return 0;
}

int mglRenderCppBindBufferStorage(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    constexpr size_t kMaxSafeBufferSize =
        static_cast<size_t>(2) * 1024u * 1024u * 1024u;
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_CPP_BUFFER_ERROR;
    }

    const bool clientStorage =
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0;
    const bool persistentNoCopy =
        buffer->data.buffer_data != 0 &&
        (buffer->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) != 0 &&
        (buffer->storage_flags & GL_MAP_PERSISTENT_BIT) != 0;
    if (clientStorage || persistentNoCopy) {
        return MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE;
    }

    if (buffer->size <= 0 ||
        static_cast<size_t>(buffer->size) > kMaxSafeBufferSize) {
        if (err && errcap) {
            snprintf(err, errcap, "suspicious size=%zu",
                     static_cast<size_t>(buffer->size));
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_CPP_BUFFER_ERROR;
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
        return MGL_RENDER_CPP_BUFFER_ERROR;
    }
    if (buffer->transient_batch_buffer) {
        allocationSize = static_cast<size_t>(buffer->size);
    }

    void* metalBuffer = nullptr;
    mglMetalCountCreate(mgl::kMetalKindBuffer);
    int result = bytes
        ? mglRenderCppCreateBufferWithBytes(
              bytes, allocationSize, options, nullptr, &metalBuffer)
        : mglRenderCppCreateBuffer(
              allocationSize, options, nullptr, &metalBuffer);
    if (result != 0 || !metalBuffer) {
        if (err && errcap) {
            snprintf(err, errcap, "Metal buffer creation failed size=%zu",
                     allocationSize);
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_CPP_BUFFER_ERROR;
    }

    buffer->data.mtl_data = metalBuffer;
    buffer->data.mtl_owns_buffer_data = GL_FALSE;
    if (!bytes) buffer->data.buffer_data = 0;
    return MGL_RENDER_CPP_BUFFER_BOUND;
}

int mglRenderCppUpdateDirtyBuffer(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    auto ensureMetalBuffer = [&]() -> int {
        if (buffer->data.mtl_data) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
        }
        const int bindResult =
            mglRenderCppBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_CPP_BUFFER_BOUND) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
        }
        if (bindResult == MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    };

    if (buffer->plain_uniform_slot && !buffer->data.mtl_data &&
        buffer->data.buffer_data && buffer->size > 0 &&
        buffer->size <= 4096) {
        buffer->data.dirty_bits &=
            ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
        return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
    }

    if (buffer->size < 4096) {
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) &&
            !buffer->data.mtl_data) {
            const int result = ensureMetalBuffer();
            if (result != MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
                return result;
            }
        }

        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
        }

        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }

        MTL::Buffer* metalBuffer =
            static_cast<MTL::Buffer*>(buffer->data.mtl_data);
        MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
        void* snapshotBuffer = nullptr;
        if (mglRenderCppSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        if (!metalBuffer) {
            if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
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

        if ((buffer->access & GL_MAP_COHERENT_BIT) != 0) {
            buffer->data.dirty_bits = DIRTY_BUFFER_DATA;
        } else {
            buffer->data.dirty_bits &=
                ~(DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR);
            buffer->cpu_shadow_pending = GL_FALSE;
        }
        return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_ADDR) != 0) {
        const int bindResult = ensureMetalBuffer();
        if (bindResult != MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
            return bindResult;
        }
        if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
            buffer->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
            return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
        }
    }

    if ((buffer->data.dirty_bits & DIRTY_BUFFER_DATA) == 0) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "buffer %u has no dirty CPU or Metal backing",
                     static_cast<unsigned>(buffer->name));
        }
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    const int bindResult = ensureMetalBuffer();
    if (bindResult != MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) {
        return bindResult;
    }
    void* snapshotBuffer = nullptr;
    if (mglRenderCppSnapshotSharedDirtyBuffer(
            buffer, &snapshotBuffer, err, errcap) != 0) {
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }
    MTL::Buffer* metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    if (!metalBuffer) {
        if (err && errcap) snprintf(err, errcap, "missing Metal buffer");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    const size_t metalLength = static_cast<size_t>(metalBuffer->length());
    const bool coherentMapped =
        ((buffer->access_flags & GL_MAP_COHERENT_BIT) != 0) ||
        ((buffer->access & GL_MAP_COHERENT_BIT) != 0);
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
    return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
}

void mglRenderCppBindBuffer(GLMContext glm_ctx, Buffer* buffer) {
    char error[256] = {};
    int result = mglRenderCppBindBufferStorage(
        buffer, error, sizeof(error));
    if (result == MGL_RENDER_CPP_BUFFER_BOUND) return;
    if (result == MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
        mtlBindBuffer(glm_ctx, buffer);
        return;
    }
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderCppBufferSubDataStorage(Buffer* buffer,
                                     size_t offset,
                                     size_t size,
                                     const void* bytes,
                                     char* err,
                                     size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }
    if (size == 0) return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
    if (!bytes) {
        if (err && errcap) snprintf(err, errcap, "null source bytes");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    uint8_t* cpuBase = buffer->data.buffer_data >= 0x1000u
        ? reinterpret_cast<uint8_t*>(
              static_cast<uintptr_t>(buffer->data.buffer_data))
        : nullptr;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderCppBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_CPP_BUFFER_BOUND) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
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
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }
    if (!metalBase) {
        if (err && errcap) snprintf(err, errcap, "Metal buffer has no contents");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    MTL::Buffer* bufferBeforeSnapshot = metalBuffer;
    if (cpuBase && cpuBase != metalBase) {
        memmove(cpuBase + offset, bytes, size);
        void* snapshotBuffer = nullptr;
        if (mglRenderCppSnapshotSharedDirtyBuffer(
                buffer, &snapshotBuffer, err, errcap) != 0) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
        metalBase = metalBuffer
            ? static_cast<uint8_t*>(metalBuffer->contents())
            : nullptr;
        if (!metalBuffer || !metalBase) {
            if (err && errcap) snprintf(err, errcap, "snapshot has no contents");
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
        }
    }

    if (metalBuffer == bufferBeforeSnapshot) {
        memcpy(metalBase + offset, bytes, size);
        if (metalBuffer->storageMode() == MTL::StorageModeManaged) {
            metalBuffer->didModifyRange(NS::Range::Make(offset, size));
        }
    }
    return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
}

void mglRenderCppBufferSubData(GLMContext glm_ctx,
                               Buffer* buffer,
                               size_t offset,
                               size_t size,
                               const void* bytes) {
    char error[256] = {};
    int result = mglRenderCppBufferSubDataStorage(
        buffer, offset, size, bytes, error, sizeof(error));
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) return;
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE) {
        mtlBufferSubData(glm_ctx, buffer, offset, size, bytes);
        return;
    }
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer subdata failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderCppMapBufferStorage(Buffer* buffer,
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
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderCppBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_CPP_BUFFER_BOUND) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
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
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
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
        return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
    }

    if (!cpuBase &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(NS::Range::Make(offset, safeLength));
    }
    return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
}

void* mglRenderCppMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer* buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map) {
    void* mapped = nullptr;
    char error[256] = {};
    int result = mglRenderCppMapBufferStorage(
        buffer, offset, size, access, map, &mapped, error, sizeof(error));
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) return mapped;
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE) {
        return mtlMapUnmapBuffer(
            glm_ctx, buffer, offset, size, access, map);
    }
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer map failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
    return nullptr;
}

void mglRenderCppReadBackBuffer(GLMContext glm_ctx,
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

int mglRenderCppFlushBufferRangeStorage(Buffer* buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char* err,
                                         size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!buffer || offset < 0 || length < 0) {
        if (err && errcap) snprintf(err, errcap, "bad buffer or signed range");
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }
    if (length == 0) return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;

    bool created = false;
    if (!buffer->data.mtl_data) {
        int bindResult = mglRenderCppBindBufferStorage(buffer, err, errcap);
        if (bindResult == MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE;
        }
        if (bindResult != MGL_RENDER_CPP_BUFFER_BOUND) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
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
        return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
    }

    if (!created) {
        void* snapshotBuffer = nullptr;
        if (mglRenderCppSnapshotSharedBufferRange(
                buffer, safeOffset, safeLength, &snapshotBuffer,
                err, errcap) != 0) {
            return MGL_RENDER_CPP_BUFFER_OPERATION_ERROR;
        }
        metalBuffer = static_cast<MTL::Buffer*>(snapshotBuffer);
    }
    if (metalBuffer &&
        metalBuffer->storageMode() == MTL::StorageModeManaged) {
        metalBuffer->didModifyRange(
            NS::Range::Make(safeOffset, safeLength));
    }
    return MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED;
}

void mglRenderCppFlushBufferRange(GLMContext glm_ctx,
                                  Buffer* buffer,
                                  intptr_t offset,
                                  intptr_t length) {
    char error[256] = {};
    int result = mglRenderCppFlushBufferRangeStorage(
        buffer, offset, length, error, sizeof(error));
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED) return;
    if (result == MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE) {
        mtlFlushBufferRange(glm_ctx, buffer, offset, length);
        return;
    }
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer range flush failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderCppConvertVertexBuffer(
    Buffer* sourceBuffer,
    const MGLRenderCppVertexConversion* conversion,
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
    if (conversion->kind > MGL_RENDER_CPP_VERTEX_INTEGER_TO_32) {
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
        case MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(double);
            defaultStride = componentCount * sizeof(double);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_CPP_VERTEX_INT_TO_FLOAT:
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
        case MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT:
            if (componentCount == 0 || componentCount > 4) goto bad_components;
            sourceComponentSize = sizeof(int32_t);
            defaultStride = componentCount * sizeof(int32_t);
            minimumConvertedStride = componentCount * sizeof(float);
            break;
        case MGL_RENDER_CPP_VERTEX_PACKED_1010102_TO_FLOAT:
            componentCount = 4;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 4u * sizeof(float);
            break;
        case MGL_RENDER_CPP_VERTEX_PACKED_10F11F11F_TO_FLOAT:
            componentCount = 3;
            sourceComponentSize = sizeof(uint32_t);
            defaultStride = sizeof(uint32_t);
            minimumConvertedStride = 3u * sizeof(float);
            break;
        case MGL_RENDER_CPP_VERTEX_INTEGER_TO_32:
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
        if (kind == MGL_RENDER_CPP_VERTEX_INTEGER_TO_32) {
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

        mgl::RendererCpp& renderer = mgl::renderer();
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
            kind == MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT ||
            kind == MGL_RENDER_CPP_VERTEX_INT_TO_FLOAT ||
            kind == MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT;
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

            if (kind == MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT ||
                kind == MGL_RENDER_CPP_VERTEX_INT_TO_FLOAT ||
                kind == MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT) {
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
                    if (kind == MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT) {
                        double value = 0.0;
                        memcpy(&value, componentBytes, sizeof(value));
                        values[component] = static_cast<float>(value);
                    } else if (kind == MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT) {
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

            if (kind == MGL_RENDER_CPP_VERTEX_PACKED_1010102_TO_FLOAT ||
                kind == MGL_RENDER_CPP_VERTEX_PACKED_10F11F11F_TO_FLOAT) {
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
                if (kind == MGL_RENDER_CPP_VERTEX_PACKED_1010102_TO_FLOAT) {
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
        if (mglRenderCppCreateBufferWithBytes(
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

int mglRenderCppBindAIRProgram(Program* program,
                               int* failed_stage_out,
                               char* err,
                               size_t errcap) {
    if (failed_stage_out) *failed_stage_out = -1;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!program || !device) {
        if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
        return MGL_RENDER_CPP_AIR_PROGRAM_ERROR;
    }
    program->dirty_bits &= ~DIRTY_PROGRAM;

    bool hasAIRStage = false;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader) continue;
        if (stage == _GEOMETRY_SHADER) {
            if (mgl::geometryShaderIsPassthrough(shader)) continue;
            if (program->gs_route != MGL_GS_ROUTE_COMPUTE) {
                if (failed_stage_out) *failed_stage_out = stage;
                if (err && errcap) {
                    snprintf(err, errcap,
                             "unsupported geometry shader route %u",
                             (unsigned)program->gs_route);
                }
                return MGL_RENDER_CPP_AIR_PROGRAM_ERROR;
            }
        }
        Spirv* spirv = &program->spirv[stage];
        if (!spirv->metallib_bytes || spirv->metallib_size == 0u) {
            return MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE;
        }
        hasAIRStage = true;
    }
    if (!hasAIRStage) return MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE;

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader || (stage == _GEOMETRY_SHADER &&
                        mgl::geometryShaderIsPassthrough(shader))) {
            continue;
        }
        Spirv* spirv = &program->spirv[stage];
        if (!spirv->mtl_library || !spirv->mtl_function) {
            mgl::releaseBridgedObject(&spirv->mtl_function);
            mgl::releaseBridgedObject(&spirv->mtl_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_bytes, spirv->metallib_size,
                    &spirv->mtl_library, &spirv->mtl_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_CPP_AIR_PROGRAM_ERROR;
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
                return MGL_RENDER_CPP_AIR_PROGRAM_ERROR;
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
                return MGL_RENDER_CPP_AIR_PROGRAM_ERROR;
            }
        }
    }
    return MGL_RENDER_CPP_AIR_PROGRAM_BOUND;
}

void mglRenderCppBindProgram(GLMContext glm_ctx, Program* program) {
    int failedStage = -1;
    char error[256] = {};
    int result = mglRenderCppBindAIRProgram(
        program, &failedStage, error, sizeof(error));
    if (result == MGL_RENDER_CPP_AIR_PROGRAM_BOUND) return;
    if (result == MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE) {
        mtlBindProgram(glm_ctx, program);
        return;
    }
    fprintf(stderr,
            "MGL ERROR: Metal-cpp AIR program bind failed program=%u "
            "stage=%d: %s\n",
            program ? (unsigned)program->name : 0u, failedStage,
            error[0] ? error : "unknown error");
}

void mglRenderCppWaitForSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    if (sync->mtl_command_buffer) {
        MTL::CommandBuffer* commandBuffer =
            static_cast<MTL::CommandBuffer*>(sync->mtl_command_buffer);
        if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
            commandBuffer->waitUntilCompleted();
        }
        mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    }
    mgl::releaseBridgedObject(&sync->mtl_event);
}

unsigned int mglRenderCppGetSyncStatus(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync || !sync->mtl_command_buffer) return GL_SIGNALED;
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(sync->mtl_command_buffer);
    return commandBuffer->status() == MTL::CommandBufferStatusCompleted
        ? GL_SIGNALED
        : GL_UNSIGNALED;
}

void mglRenderCppReleaseSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    mgl::releaseBridgedObject(&sync->mtl_event);
}

int mglRenderCppCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char* label,
                             void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || length == 0) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateBufferWithBytes(const void* bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char* label,
                                      void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateTextureStagingOwner(
    const void* bytes,
    uint64_t length,
    uint64_t resource_options,
    void** owner_out,
    void** buffer_out) {
    if (owner_out) *owner_out = nullptr;
    if (buffer_out) *buffer_out = nullptr;
    if (!owner_out || !buffer_out || !bytes || length == 0) return -1;
    void* rawBuffer = nullptr;
    if (mglRenderCppCreateBufferWithBytes(
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

void mglRenderCppDestroyTextureStagingOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::TextureStagingOwner* owner =
        static_cast<mgl::TextureStagingOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCreateBufferWithBytesNoCopy(const void* bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char* label,
                                            int deallocate_vm,
                                            void** buffer_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (!buffer_out || !bytes || length == 0) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateTextureFromState(
    const MGLRenderCppTextureDescriptorState* texture_descriptor,
    const char* label,
    void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    if (!texture_descriptor || !texture_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateBufferTextureFromState(
    void* buffer,
    const MGLRenderCppTextureDescriptorState* texture_descriptor,
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

int mglRenderCppCreateTextureView(void* texture,
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

int mglRenderCppCreateTextureViewRange(
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

int mglRenderCppTextureReplaceRegion(void* texture,
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

int mglRenderCppTextureGetBytes(void* texture,
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

int mglRenderCppCreateSampler(void* sampler_descriptor,
                              void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    MTL::SamplerDescriptor* descriptor =
        static_cast<MTL::SamplerDescriptor*>(sampler_descriptor);
    if (!descriptor || !sampler_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::SamplerState* sampler = renderer.device->newSamplerState(descriptor);
    if (!sampler) return -1;
    *sampler_out = sampler;
    return 0;
}

int mglRenderCppCreateSamplerForGL(const TextureParameter* params,
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

    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateDepthStencilState(void* depth_stencil_descriptor,
                                        void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        static_cast<MTL::DepthStencilDescriptor*>(depth_stencil_descriptor);
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::DepthStencilState* state =
        renderer.device->newDepthStencilState(descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

static MTL::StencilDescriptor* mglRenderCppBuildStencilDescriptor(
    const MGLRenderCppStencilDescriptorState& state) {
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

static MTL::DepthStencilState* mglRenderCppCreateDepthStencilFromStateLocked(
    mgl::RendererCpp& renderer,
    const MGLRenderCppDepthStencilDescriptorState& state) {
    if (!renderer.device) return nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        MTL::DepthStencilDescriptor::alloc()->init();
    if (!descriptor) return nullptr;
    descriptor->setDepthCompareFunction(
        static_cast<MTL::CompareFunction>(state.depth_compare_function));
    descriptor->setDepthWriteEnabled(state.depth_write_enabled != 0);
    MTL::StencilDescriptor* front =
        mglRenderCppBuildStencilDescriptor(state.front);
    MTL::StencilDescriptor* back =
        mglRenderCppBuildStencilDescriptor(state.back);
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

int mglRenderCppCreateDepthStencilStateFromState(
    const MGLRenderCppDepthStencilDescriptorState* descriptor,
    void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCppCreateDepthStencilFromStateLocked(renderer, *descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

int mglRenderCppCreatePipelineCacheOwner(
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

void mglRenderCppDestroyPipelineCacheOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

void mglRenderCppResetPipelineCacheOwner(void* owner_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->clearCaches();
}

int mglRenderCppGetPipelineCacheFlags(
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

void mglRenderCppDisablePipelineBinaryArchive(void* owner_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->binaryArchiveEnabled = false;
}

int mglRenderCppGetPipelineActiveState(
    void* owner_handle, MGLRenderCppPipelineActiveState* state_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->active;
    return 0;
}

int mglRenderCppInvalidatePipelineActiveState(void* owner_handle) {
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

int mglRenderCppSetPipelineActiveObject(void* owner_handle,
                                        void* pipeline_state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::retainObject(pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    owner->active.pipeline_state = pipeline_state;
    return 0;
}

int mglRenderCppActivatePipelineState(
    void* owner_handle, const MGLRenderCppPipelineActiveState* state) {
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

int mglRenderCppSetPipelineBlendState(
    void* owner_handle, uint32_t attachment,
    const MGLRenderCppPipelineBlendState* state) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state ||
        attachment >= MGL_RENDER_CPP_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->blend[attachment] = *state;
    return 0;
}

int mglRenderCppGetPipelineBlendState(
    void* owner_handle, uint32_t attachment,
    MGLRenderCppPipelineBlendState* state_out) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !state_out ||
        attachment >= MGL_RENDER_CPP_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->blend[attachment];
    return 0;
}

int mglRenderCppGetOrCreateDepthStencilState(
    void* owner_handle,
    const MGLRenderCppDepthStencilDescriptorState* descriptor,
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

    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> rendererLock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCppCreateDepthStencilFromStateLocked(renderer, *descriptor);
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

int mglRenderCppLookupPipeline(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderCppPipelineActiveState* state_out) {
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

int mglRenderCppStorePipeline(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderCppPipelineActiveState* state,
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

int mglRenderCppLookupPipelineDescriptor(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    void** descriptor_out) {
    if (descriptor_out) *descriptor_out = nullptr;
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    if (!owner || !key_words || !descriptor_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    auto found = owner->descriptorCache.find(key);
    if (found == owner->descriptorCache.end()) return 0;
    *descriptor_out = found->second->descriptor;
    mgl::PipelineCacheOwner::touch(owner->descriptorCacheLRU, key);
    return 1;
}

int mglRenderCppStorePipelineDescriptor(
    void* owner_handle,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    void* descriptor_handle) {
    auto* owner = static_cast<mgl::PipelineCacheOwner*>(owner_handle);
    auto* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(descriptor_handle);
    if (!owner || !key_words || !descriptor) return -1;
    MTL::RenderPipelineDescriptor* descriptorCopy = descriptor->copy();
    if (!descriptorCopy) return -1;
    std::unique_ptr<mgl::PipelineCacheDescriptorEntry> entry(
        new (std::nothrow) mgl::PipelineCacheDescriptorEntry());
    if (!entry) {
        descriptorCopy->release();
        return -1;
    }
    entry->descriptor = descriptorCopy;
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

int mglRenderCppCreateEvent(void** event_out) {
    if (event_out) *event_out = nullptr;
    if (!event_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Event* event = renderer.device->newEvent();
    if (!event) return -1;
    *event_out = event;
    return 0;
}

int mglRenderCppCreateMetal4Compiler(const char* label,
                                     void** compiler_out,
                                     char* err,
                                     size_t errcap) {
    if (compiler_out) *compiler_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!compiler_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    MTL4::CompilerDescriptor* descriptor =
        MTL4::CompilerDescriptor::alloc()->init();
    if (!descriptor) return -1;
    if (label && label[0]) {
        descriptor->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    NS::Error* nsError = nullptr;
    MTL4::Compiler* compiler =
        renderer.device->newCompiler(descriptor, &nsError);
    descriptor->release();
    if (!compiler) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *compiler_out = compiler;
    return 0;
}

int mglRenderCppCompileLibrary(void* compiler,
                               void* source_string,
                               void* compile_options,
                               const char* label,
                               void** library_out,
                               char* err,
                               size_t errcap) {
    if (library_out) *library_out = nullptr;
    if (err && errcap) err[0] = '\0';
    NS::String* source = static_cast<NS::String*>(source_string);
    if (!source || !library_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::Library* library = nullptr;
    MTL::CompileOptions* options =
        static_cast<MTL::CompileOptions*>(compile_options);
    if (compiler) {
        MTL4::LibraryDescriptor* descriptor =
            MTL4::LibraryDescriptor::alloc()->init();
        if (!descriptor) return -1;
        descriptor->setSource(source);
        descriptor->setOptions(options);
        if (label && label[0]) {
            descriptor->setName(
                NS::String::string(label, NS::UTF8StringEncoding));
        }
        library = static_cast<MTL4::Compiler*>(compiler)->newLibrary(
            descriptor, &nsError);
        descriptor->release();
    } else {
        library = renderer.device->newLibrary(source, options, &nsError);
    }
    if (!library) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *library_out = library;
    return 0;
}

int mglRenderCppCreateFunction(void* library,
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

int mglRenderCppCreateRenderPipelineState(
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

    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateComputePipelineState(void* function,
                                           void** pipeline_out,
                                           char* err,
                                           size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Function* computeFunction =
        static_cast<MTL::Function*>(function);
    if (!computeFunction || !pipeline_out) return -1;

    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppCreateBinaryArchive(void* binary_archive_descriptor,
                                    const char* label,
                                    void** binary_archive_out,
                                    char* err,
                                    size_t errcap) {
    if (binary_archive_out) *binary_archive_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchiveDescriptor* descriptor =
        static_cast<MTL::BinaryArchiveDescriptor*>(binary_archive_descriptor);
    if (!descriptor || !binary_archive_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppSetRenderPipelineBinaryArchive(
    void* render_pipeline_descriptor,
    void* binary_archive) {
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    if (!descriptor || !archive) return -1;
    descriptor->setBinaryArchives(NS::Array::array(archive));
    return 0;
}

int mglRenderCppAddRenderPipelineFunctionsToBinaryArchive(
    void* binary_archive,
    void* render_pipeline_descriptor,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    if (!archive || !descriptor) return -1;
    NS::Error* nsError = nullptr;
    if (!archive->addRenderPipelineFunctions(descriptor, &nsError)) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    return 0;
}

int mglRenderCppSerializeBinaryArchive(void* binary_archive,
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

int mglRenderCppSetVisibilityResultMode(void* render_encoder,
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

int mglRenderCppSampleTimestamps(uint64_t* cpu_timestamp_out,
                                 uint64_t* gpu_timestamp_out) {
    if (!cpu_timestamp_out || !gpu_timestamp_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    renderer.device->sampleTimestamps(cpu_timestamp_out, gpu_timestamp_out);
    return 0;
}

int mglRenderCppCreateQueryStateOwner(uint32_t visibility_slot_count,
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

int mglRenderCppBeginSampleQuery(void* owner_handle,
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
        mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppGetQueryVisibilityBuffer(void* owner_handle,
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

void mglRenderCppEndSampleQuery(void* owner_handle) {
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (owner) owner->sampleQueryActive = false;
}

int mglRenderCppIsSampleQueryActive(void* owner_handle,
                                    uint32_t* active_out) {
    if (active_out) *active_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !active_out) return -1;
    *active_out = owner->sampleQueryActive ? 1u : 0u;
    return 0;
}

int mglRenderCppAcquireSampleQuerySlot(void* owner_handle,
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

int mglRenderCppGetSampleQueryResult(void* owner_handle,
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

int mglRenderCppBeginTimerQuery(void* owner_handle) {
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderCppSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    owner->timerQueryBeginGPU = gpuTimestamp;
    return 0;
}

int mglRenderCppEndTimerQuery(void* owner_handle,
                              uint64_t* elapsed_out) {
    if (elapsed_out) *elapsed_out = 0;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(owner_handle);
    if (!owner || !elapsed_out) return -1;
    uint64_t cpuTimestamp = 0;
    uint64_t gpuTimestamp = 0;
    if (mglRenderCppSampleTimestamps(
            &cpuTimestamp, &gpuTimestamp) != 0) {
        return -1;
    }
    *elapsed_out = gpuTimestamp >= owner->timerQueryBeginGPU
        ? gpuTimestamp - owner->timerQueryBeginGPU
        : 0;
    return 0;
}

void mglRenderCppDestroyQueryStateOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::QueryStateOwner* owner =
        static_cast<mgl::QueryStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppGetOrCreateComputePipeline(
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

    mgl::RendererCpp& renderer = mgl::renderer();
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
    Spirv* spirv = &program->spirv[stage];
    if (!spirv->mtl_function) {
        if (err && errcap) snprintf(err, errcap, "compiled compute function is unavailable");
        return -1;
    }
    return mglRenderCppGetOrCreateComputePipeline(
        spirv->mtl_function,
        program->pipeline_cache_instance_id,
        program->pipeline_cache_generation,
        static_cast<uint32_t>(stage), 1, pipeline_out, err, errcap);
}

void mglRenderCppInvalidateProgramPipelines(uint64_t program_instance) {
    if (program_instance == 0) return;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppGetOrCreateAuxComputePipeline(
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

    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }

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

int mglRenderCppGetOrCreateAuxRenderPipeline(
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
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }

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
                             kind != MGL_RENDER_CPP_AUX_RENDER_CLEAR_RECT)) {
        return 1;
    }

    MTL::RenderPipelineDescriptor* descriptor =
        MTL::RenderPipelineDescriptor::alloc()->init();
    if (!descriptor) {
        if (err && errcap) snprintf(err, errcap, "render descriptor allocation failed");
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

void* mglRenderCppBindingCreate(uint32_t max_texture_slots) {
    if (max_texture_slots == 0 || max_texture_slots > 128) return nullptr;
    mgl::BindingState* state =
        new (std::nothrow) mgl::BindingState(max_texture_slots);
    if (!state) return nullptr;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    renderer.bindingStates.insert(state);
    return state;
}

void mglRenderCppBindingDestroy(void* binding_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state) return;
    mgl::RendererCpp& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    auto found = renderer.bindingStates.find(state);
    if (found == renderer.bindingStates.end()) return;
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: binding dedup "
                "texture=%llu/%llu sampler=%llu/%llu "
                "viewport=%llu/%llu scissor=%llu/%llu fill=%llu/%llu\n",
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_VERTEX_TEXTURE] +
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_VERTEX_TEXTURE] +
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_VERTEX_SAMPLER] +
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_VERTEX_SAMPLER] +
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_CPP_BINDING_TRIANGLE_FILL]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_CPP_BINDING_TRIANGLE_FILL]));
    }
    renderer.bindingStates.erase(found);
    delete state;
}

void mglRenderCppBindingInvalidate(void* binding_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->invalidate();
}

void mglRenderCppBindingSetValid(void* binding_state, int valid) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->valid = valid != 0;
}

int mglRenderCppBindingGetValid(void* binding_state, uint32_t* valid_out) {
    if (valid_out) *valid_out = 0;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !valid_out) return -1;
    *valid_out = state->valid ? 1u : 0u;
    return 0;
}

int mglRenderCppBindingGetTextureSlotMask(void* binding_state,
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

int mglRenderCppBindingRecordVertexBuffer(void* binding_state,
                                          void* buffer,
                                          uint64_t offset,
                                          uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderCppBindingRecordFragmentBuffer(void* binding_state,
                                            void* buffer,
                                            uint64_t offset,
                                            uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderCppBindingInvalidateVertexBuffer(void* binding_state,
                                              uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || index >= state->vertexBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->vertexBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->vertexBufferOffsets[index] = UINT64_MAX;
    state->vertexBufferMask |= 1U << index;
    return 0;
}

int mglRenderCppBindingInvalidateFragmentBuffer(void* binding_state,
                                                uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || index >= state->fragmentBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->fragmentBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->fragmentBufferOffsets[index] = UINT64_MAX;
    state->fragmentBufferMask |= 1U << index;
    return 0;
}

int mglRenderCppBindingUpdateVertexBuffer(void* binding_state,
                                          void* buffer,
                                          uint64_t offset,
                                          uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderCppBindingUpdateFragmentBuffer(void* binding_state,
                                            void* buffer,
                                            uint64_t offset,
                                            uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderCppBindingClearVertexBuffer(void* binding_state,
                                         uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? clearBufferSlot(state->vertexBuffers,
                                   state->vertexBufferOffsets, index, 0) : -1;
}

int mglRenderCppBindingClearFragmentBuffer(void* binding_state,
                                           uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    return state ? clearBufferSlot(state->fragmentBuffers,
                                   state->fragmentBufferOffsets, index, 0) : -1;
}

int mglRenderCppBindingGetBuffer(void* binding_state,
                                 uint32_t stage,
                                 uint32_t index,
                                 void** buffer_out,
                                 uint64_t* offset_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (offset_out) *offset_out = 0;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !buffer_out || !offset_out ||
        stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    const std::vector<MTL::Buffer*>& buffers =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexBuffers : state->fragmentBuffers;
    const std::vector<uint64_t>& offsets =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexBufferOffsets : state->fragmentBufferOffsets;
    if (index >= buffers.size()) return -1;
    *buffer_out = buffers[index];
    *offset_out = offsets[index];
    return 0;
}

void mglRenderCppBindingOrVertexBufferMask(void* binding_state,
                                           uint32_t mask) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->vertexBufferMask |= mask;
}

void mglRenderCppBindingOrFragmentBufferMask(void* binding_state,
                                             uint32_t mask) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->fragmentBufferMask |= mask;
}

void mglRenderCppBindingSetPipelineState(void* binding_state,
                                         void* pipeline_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) {
        mgl::BindingState::replaceObject(
            state->pipelineState,
            static_cast<MTL::RenderPipelineState*>(pipeline_state));
    }
}

void mglRenderCppBindingSetDepthStencilState(void* binding_state,
                                             void* depth_stencil_state) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) {
        mgl::BindingState::replaceObject(
            state->depthStencilState,
            static_cast<MTL::DepthStencilState*>(depth_stencil_state));
    }
}

int mglRenderCppBindingGetPipelineState(void* binding_state,
                                        void** pipeline_state_out) {
    if (pipeline_state_out) *pipeline_state_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !pipeline_state_out) return -1;
    *pipeline_state_out = state->pipelineState;
    return 0;
}

int mglRenderCppBindingGetDepthStencilState(
    void* binding_state,
    void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !depth_stencil_state_out) return -1;
    *depth_stencil_state_out = state->depthStencilState;
    return 0;
}

void mglRenderCppBindingSetCullMode(void* binding_state, uint32_t mode) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->lastCullMode = static_cast<MTL::CullMode>(mode);
}

void mglRenderCppBindingSetWinding(void* binding_state, uint32_t winding) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (state) state->lastWinding = static_cast<MTL::Winding>(winding);
}

void mglRenderCppBindingSetDepthBias(void* binding_state,
                                     float bias,
                                     float clamp,
                                     float slope_scale) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state) return;
    state->lastDepthBias = bias;
    state->lastDepthBiasClamp = clamp;
    state->lastDepthSlopeScale = slope_scale;
}

void mglRenderCppBindingSetBlendColor(void* binding_state,
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

int mglRenderCppBindingSetPipelineIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetDepthStencilIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetCullIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetWindingIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetDepthBiasIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetBlendColorIfNeeded(void* binding_state,
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

int mglRenderCppBindingSetTexture(void* binding_state,
                                 void* render_encoder,
                                 void* texture,
                                 uint32_t stage,
                                 uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    MTL::Texture* newTexture = static_cast<MTL::Texture*>(texture);
    const bool emitted = !state->valid || slots[index] != newTexture;
    const uint32_t setter = stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
        ? MGL_RENDER_CPP_BINDING_VERTEX_TEXTURE
        : MGL_RENDER_CPP_BINDING_FRAGMENT_TEXTURE;
    if (emitted) {
        if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
            encoder->setVertexTexture(newTexture, index);
        } else {
            encoder->setFragmentTexture(newTexture, index);
        }
        mgl::BindingState::replaceObject(slots[index], newTexture);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderCppBindingSetSampler(void* binding_state,
                                 void* render_encoder,
                                 void* sampler,
                                 uint32_t stage,
                                 uint32_t index) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    MTL::SamplerState* newSampler = static_cast<MTL::SamplerState*>(sampler);
    const bool emitted = !state->valid || slots[index] != newSampler;
    const uint32_t setter = stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
        ? MGL_RENDER_CPP_BINDING_VERTEX_SAMPLER
        : MGL_RENDER_CPP_BINDING_FRAGMENT_SAMPLER;
    if (emitted) {
        if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
            encoder->setVertexSamplerState(newSampler, index);
        } else {
            encoder->setFragmentSamplerState(newSampler, index);
        }
        mgl::BindingState::replaceObject(slots[index], newSampler);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderCppBindingGetTexture(void* binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !texture_out ||
        stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    const std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    *texture_out = slots[index];
    return 0;
}

int mglRenderCppBindingGetSampler(void* binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !sampler_out ||
        stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    const std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    *sampler_out = slots[index];
    return 0;
}

int mglRenderCppBindingSetViewport(void* binding_state,
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
    mgl::recordBindingResult(*state, MGL_RENDER_CPP_BINDING_VIEWPORT, emitted);
    return emitted ? 1 : 0;
}

int mglRenderCppBindingSetViewports(void* binding_state,
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
    mgl::recordBindingResult(*state, MGL_RENDER_CPP_BINDING_VIEWPORT, !same);
    return same ? 0 : 1;
}

int mglRenderCppBindingSetScissor(void* binding_state,
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
    mgl::recordBindingResult(*state, MGL_RENDER_CPP_BINDING_SCISSOR, emitted);
    return emitted ? 1 : 0;
}

int mglRenderCppBindingSetTriangleFill(void* binding_state,
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
        *state, MGL_RENDER_CPP_BINDING_TRIANGLE_FILL, emitted);
    return emitted ? 1 : 0;
}

int mglRenderCppBindingGetStats(void* binding_state,
                               MGLRenderCppBindingStats* stats_out) {
    mgl::BindingState* state = static_cast<mgl::BindingState*>(binding_state);
    if (!state || !stats_out) return -1;
    memcpy(stats_out, &state->stats, sizeof(*stats_out));
    return 0;
}

int mglRenderCppSetComputePipelineState(void* compute_encoder,
                                        void* pipeline_state) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    MTL::ComputePipelineState* pipeline =
        static_cast<MTL::ComputePipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setComputePipelineState(pipeline);
    return 0;
}

int mglRenderCppSetComputeBuffer(void* compute_encoder,
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

int mglRenderCppSetComputeTexture(void* compute_encoder,
                                  void* texture,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setTexture(static_cast<MTL::Texture*>(texture), index);
    return 0;
}

int mglRenderCppSetComputeSampler(void* compute_encoder,
                                  void* sampler,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setSamplerState(static_cast<MTL::SamplerState*>(sampler), index);
    return 0;
}

int mglRenderCppSetComputeBytes(void* compute_encoder,
                                const void* bytes,
                                size_t length,
                                uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || (!bytes && length != 0)) return -1;
    encoder->setBytes(bytes, static_cast<NS::UInteger>(length), index);
    return 0;
}

int mglRenderCppSetComputeThreadgroupMemoryLength(void* compute_encoder,
                                                  uint64_t length,
                                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setThreadgroupMemoryLength(static_cast<NS::UInteger>(length),
                                        index);
    return 0;
}

int mglRenderCppDispatchCompute(void* compute_encoder,
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

int mglRenderCppDispatchComputeIndirect(void* compute_encoder,
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

int mglRenderCppDispatchComputeThreads(void* compute_encoder,
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

int mglRenderCppCreateComputeEncoder(void* command_buffer,
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

int mglRenderCppEndComputeEncoder(void* compute_encoder) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCppCreateCommandBuffer(void* command_queue,
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

int mglRenderCppGetCommandBufferState(
    void* command_buffer,
    MGLRenderCppCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    return mgl::snapshotCommandBufferState(
        static_cast<MTL::CommandBuffer*>(command_buffer), state_out);
}

int mglRenderCppAddCommandBufferCompletion(
    void* command_buffer,
    MGLRenderCppCommandBufferCompletion callback,
    void* context,
    MGLRenderCppDestroyContext destroy_context) {
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!commandBuffer || !callback) return -1;

    std::shared_ptr<mgl::CommandBufferCompletionContext> completion;
    try {
        completion =
            std::make_shared<mgl::CommandBufferCompletionContext>();
    } catch (const std::bad_alloc&) {
        return -1;
    }
    completion->callback = callback;
    completion->context = context;
    completion->destroyContext = destroy_context;
    MTL::HandlerFunction handler =
        [completion](MTL::CommandBuffer* completedBuffer) {
            completion->complete(completedBuffer);
        };
    commandBuffer->addCompletedHandler(handler);
    return 0;
}

int mglRenderCppCreateCommandBufferOwner(void* command_queue,
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

int mglRenderCppResetCommandBufferOwner(void* owner_handle,
                                        void* command_queue,
                                        void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!owner || !queue || !command_buffer_out) return -1;
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) return -1;
    commandBuffer->retain();
    if (owner->current) owner->current->release();
    owner->current = commandBuffer;
    *command_buffer_out = commandBuffer;
    return 0;
}

void mglRenderCppDiscardCommandBufferOwnerCurrent(void* owner_handle) {
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(owner_handle);
    if (!owner || !owner->current) return;
    owner->current->release();
    owner->current = nullptr;
}

int mglRenderCppTakeCommandBufferSubmission(void* owner_handle,
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
    *submission_out = submission;
    *command_buffer_out = submission->buffer;
    return 0;
}

int mglRenderCppCommitCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return -1;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    if (!submission->buffer) return -1;
    submission->buffer->commit();
    *submission_handle = nullptr;
    delete submission;
    return 0;
}

void mglRenderCppDestroyCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    *submission_handle = nullptr;
    delete submission;
}

void mglRenderCppDestroyCommandBufferOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandBufferOwner* owner =
        static_cast<mgl::CommandBufferOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCreateCommandQueueOwner(uint32_t max_command_buffers,
                                        void** owner_out,
                                        void** command_queue_out) {
    if (owner_out) *owner_out = nullptr;
    if (command_queue_out) *command_queue_out = nullptr;
    if (!owner_out || !command_queue_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppResetCommandQueueOwner(void* owner_handle,
                                       uint32_t max_command_buffers,
                                       void** command_queue_out) {
    if (command_queue_out) *command_queue_out = nullptr;
    mgl::CommandQueueOwner* owner =
        static_cast<mgl::CommandQueueOwner*>(owner_handle);
    if (!owner || !command_queue_out) return -1;
    mgl::RendererCpp& renderer = mgl::renderer();
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

void mglRenderCppDestroyCommandQueueOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandQueueOwner* owner =
        static_cast<mgl::CommandQueueOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCreateMDIScratchOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::MDIScratchOwner* owner =
        new (std::nothrow) mgl::MDIScratchOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

int mglRenderCppAllocateMDIScratch(void* owner_handle,
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
        mgl::RendererCpp& renderer = mgl::renderer();
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

void mglRenderCppResetMDIScratchOwner(void* owner_handle) {
    mgl::MDIScratchOwner* owner =
        static_cast<mgl::MDIScratchOwner*>(owner_handle);
    if (!owner) return;
    if (owner->buffer) owner->buffer->release();
    owner->buffer = nullptr;
    owner->capacity = 0;
    owner->offset = 0;
}

void mglRenderCppDestroyMDIScratchOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::MDIScratchOwner* owner =
        static_cast<mgl::MDIScratchOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCommitCommandBuffer(void* command_buffer) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command) return -1;
    command->commit();
    return 0;
}

int mglRenderCppWaitCommandBuffer(void* command_buffer) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command) return -1;
    command->waitUntilCompleted();
    return 0;
}

int mglRenderCppPresentDrawable(void* command_buffer, void* drawable) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::Drawable* surface = static_cast<MTL::Drawable*>(drawable);
    if (!command || !surface) return -1;
    command->presentDrawable(surface);
    return 0;
}

int mglRenderCppCreateRenderEncoderFromState(
    void* command_buffer,
    const MGLRenderCppRenderPassState* render_pass,
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

int mglRenderCppEncodeColorClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double red,
                                 double green,
                                 double blue,
                                 double alpha) {
    if (!command_buffer || !texture) return -1;
    MGLRenderCppRenderPassState state = mgl::defaultRenderPassState();
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
    if (mglRenderCppCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderCppEncodeDepthClear(void* command_buffer,
                                 void* texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth) {
    if (!command_buffer || !texture) return -1;
    MGLRenderCppRenderPassState state = mgl::defaultRenderPassState();
    state.depth.attachment.texture = texture;
    state.depth.attachment.level = level;
    state.depth.attachment.slice = slice;
    state.depth.attachment.depth_plane = depth_plane;
    state.depth.attachment.load_action =
        static_cast<uint32_t>(MTL::LoadActionClear);
    state.depth.clear_depth = clear_depth;
    void* encoder_handle = nullptr;
    if (mglRenderCppCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderCppEncodeMultisampleResolve(
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
    MGLRenderCppRenderPassState state = mgl::defaultRenderPassState();
    MGLRenderCppRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
        attachment = &state.color[0].attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &state.depth.attachment;
        state.depth.resolve_filter = resolve_filter;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
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
    if (mglRenderCppCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

static int mglRenderCppResetRenderEncoderOwnerImpl(
    mgl::RenderEncoderOwner* owner,
    void* command_buffer,
    const MGLRenderCppRenderPassState* render_pass,
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

int mglRenderCppCreateRenderEncoderOwnerFromState(
    void* command_buffer,
    const MGLRenderCppRenderPassState* render_pass,
    void** owner_out,
    void** render_encoder_out) {
    if (owner_out) *owner_out = nullptr;
    if (render_encoder_out) *render_encoder_out = nullptr;
    if (!owner_out || !render_encoder_out) return -1;
    mgl::RenderEncoderOwner* owner =
        new (std::nothrow) mgl::RenderEncoderOwner();
    if (!owner) return -1;
    if (mglRenderCppResetRenderEncoderOwnerImpl(
            owner, command_buffer, render_pass, render_encoder_out) != 0) {
        delete owner;
        return -1;
    }
    *owner_out = owner;
    return 0;
}

int mglRenderCppResetRenderEncoderOwnerFromState(
    void* owner_handle,
    void* command_buffer,
    const MGLRenderCppRenderPassState* render_pass,
    void** render_encoder_out) {
    return mglRenderCppResetRenderEncoderOwnerImpl(
        static_cast<mgl::RenderEncoderOwner*>(owner_handle),
        command_buffer, render_pass, render_encoder_out);
}

int mglRenderCppCreateRenderEncoderOwner(
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

int mglRenderCppResetRenderEncoderOwner(
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

int mglRenderCppEndRenderEncoderOwner(void* owner_handle) {
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(owner_handle);
    if (!owner || !owner->encoder) return -1;
    if (!owner->ended) {
        owner->encoder->endEncoding();
        owner->ended = true;
    }
    return 0;
}

void mglRenderCppDestroyRenderEncoderOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderEncoderOwner* owner =
        static_cast<mgl::RenderEncoderOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCreateRenderPassIdentityOwner(void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::RenderPassIdentityOwner* owner =
        new (std::nothrow) mgl::RenderPassIdentityOwner();
    if (!owner) return -1;
    *owner_out = owner;
    return 0;
}

int mglRenderCppUpdateRenderPassIdentity(
    void* owner_handle,
    const MGLRenderCppRenderPassIdentityState* state) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !state ||
        state->draw_buffer_count > MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    owner->state = *state;
    owner->cache = {};
    owner->cache_valid = false;
    return 0;
}

int mglRenderCppGetRenderPassIdentity(
    void* owner_handle,
    MGLRenderCppRenderPassIdentityState* state_out) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

int mglRenderCppSetFboMatchCache(
    void* owner_handle,
    const MGLRenderCppFboMatchCacheState* cache) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !cache || cache->fbo_name == 0) return -1;
    owner->cache = *cache;
    owner->cache.result = cache->result != 0;
    owner->cache_valid = true;
    return 0;
}

int mglRenderCppGetFboMatchCache(
    void* owner_handle,
    MGLRenderCppFboMatchCacheState* cache_out) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner || !cache_out || !owner->cache_valid) return 1;
    *cache_out = owner->cache;
    return 0;
}

void mglRenderCppClearFboMatchCache(void* owner_handle) {
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(owner_handle);
    if (!owner) return;
    owner->cache = {};
    owner->cache_valid = false;
}

void mglRenderCppDestroyRenderPassIdentityOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassIdentityOwner* owner =
        static_cast<mgl::RenderPassIdentityOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppCreateRenderPassStateOwner(
    const MGLRenderCppRenderPassState* state,
    void** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!state || !owner_out ||
        state->sample_position_count > MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS) {
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

int mglRenderCppCreateDefaultRenderPassStateOwner(void** owner_out) {
    MGLRenderCppRenderPassState state = mgl::defaultRenderPassState();
    return mglRenderCppCreateRenderPassStateOwner(&state, owner_out);
}

int mglRenderCppSetRenderPassStateAttachment(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    const MGLRenderCppRenderPassAttachmentState* attachment) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !attachment) return -1;

    MGLRenderCppRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
        destination = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    MGLRenderCppRenderPassAttachmentState next = *attachment;
    mgl::retainRenderPassObject(next.texture);
    mgl::retainRenderPassObject(next.resolve_texture);
    mgl::releaseRenderPassObject(destination->texture);
    mgl::releaseRenderPassObject(destination->resolve_texture);
    *destination = next;
    return 0;
}

int mglRenderCppSetRenderPassStateAttachmentTexture(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    void* texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;

    MGLRenderCppRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
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

    /* Layered rendering: keep renderTargetArrayLength capped at the largest
     * arrayLength among attached color textures (>= 1). */
    uint64_t maxArrayLength = 1u;
    for (uint32_t i = 0u; i < MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS; ++i) {
        MTL::Texture* t = static_cast<MTL::Texture*>(
            owner->state.color[i].attachment.texture);
        if (t && t->arrayLength() > maxArrayLength) {
            maxArrayLength = t->arrayLength();
        }
    }
    owner->state.render_target_array_length = maxArrayLength;
    /* Layered pass: the layer comes from the VS
     * [[render_target_array_index]] output; a non-zero attachment slice is
     * ignored (or dropped) by Metal, so keep it at 0. */
    if (maxArrayLength > 0u && destination) {
        destination->slice = 0u;
    }
    return 0;
}

int mglRenderCppSetRenderPassStateAttachmentActions(
    void* owner_handle,
    uint32_t attachment_kind,
    uint32_t color_index,
    uint32_t load_action,
    uint32_t store_action,
    uint64_t store_action_options) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;

    MGLRenderCppRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) return -1;
        attachment = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL:
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

int mglRenderCppSetRenderPassStateColorClear(
    void* owner_handle,
    uint32_t color_index,
    double red,
    double green,
    double blue,
    double alpha) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || color_index >= MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    MGLRenderCppRenderPassColorState& color = owner->state.color[color_index];
    color.clear_red = red;
    color.clear_green = green;
    color.clear_blue = blue;
    color.clear_alpha = alpha;
    return 0;
}

int mglRenderCppSetRenderPassStateDepthClear(
    void* owner_handle,
    double clear_depth) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    owner->state.depth.clear_depth = clear_depth;
    return 0;
}

int mglRenderCppSetRenderPassStateStencilClear(
    void* owner_handle,
    uint32_t clear_stencil) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) return -1;
    owner->state.stencil.clear_stencil = clear_stencil;
    return 0;
}

int mglRenderCppSetRenderPassStateVisibility(
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

int mglRenderCppSetRenderPassStateDimensions(
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

int mglRenderCppGetRenderPassStateOwner(
    void* owner_handle,
    MGLRenderCppRenderPassState* state_out) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

int mglRenderCppCreateRenderEncoderFromStateOwner(
    void* command_buffer,
    void* owner_handle,
    void** render_encoder_out) {
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(owner_handle);
    if (!owner) {
        if (render_encoder_out) *render_encoder_out = nullptr;
        return -1;
    }
    return mglRenderCppCreateRenderEncoderFromState(
        command_buffer, &owner->state, render_encoder_out);
}

void mglRenderCppDestroyRenderPassStateOwner(void** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassStateOwner* owner =
        static_cast<mgl::RenderPassStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCppEndRenderEncoder(void* render_encoder) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCppCreateBlitEncoder(void* command_buffer,
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

int mglRenderCppEndBlitEncoder(void* blit_encoder) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCppEncodeTextureUpload(void* command_buffer,
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
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!command || !source || !destination || source_width == 0 ||
        source_height == 0 || source_depth == 0 ||
        source_bytes_per_row == 0 || source_bytes_per_image == 0) {
        return -1;
    }

    MTL::BlitCommandEncoder* encoder = command->blitCommandEncoder();
    if (!encoder) return -1;
    encoder->copyFromBuffer(
        source, static_cast<NS::UInteger>(source_offset),
        static_cast<NS::UInteger>(source_bytes_per_row),
        static_cast<NS::UInteger>(source_bytes_per_image),
        MTL::Size(source_width, source_height, source_depth), destination,
        static_cast<NS::UInteger>(destination_slice),
        static_cast<NS::UInteger>(destination_level),
        MTL::Origin(destination_x, destination_y, destination_z));
    encoder->endEncoding();
    return 0;
}

int mglRenderCppBlitCopyBuffer(void* blit_encoder,
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

int mglRenderCppBlitCopyBufferToTexture(void* blit_encoder,
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

int mglRenderCppBlitSynchronizeTexture(void* blit_encoder,
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

int mglRenderCppBlitGenerateMipmaps(void* blit_encoder,
                                    void* texture) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!encoder || !source) return -1;
    encoder->generateMipmaps(source);
    return 0;
}

int mglRenderCppBlitCopyTexture(void* blit_encoder,
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

int mglRenderCppBlitCopyTextureToBuffer(
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

int mglRenderCppDrawPrimitives(void* render_encoder,
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

int mglRenderCppDrawIndexedPrimitives(void* render_encoder,
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

int mglRenderCppDrawPrimitivesIndirect(void* render_encoder,
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

int mglRenderCppDrawIndexedPrimitivesIndirect(
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

int mglRenderCppCreateCullDistanceIndexPlan(
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
            mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppGetCullDistanceIndexPrimitive(
    void* owner,
    uint64_t primitive_index,
    MGLRenderCppCullDistancePrimitive* primitive_out) {
    mgl::CullDistanceIndexPlan* plan =
        static_cast<mgl::CullDistanceIndexPlan*>(owner);
    if (!plan || !primitive_out ||
        primitive_index >= plan->primitives.size()) {
        return -1;
    }
    *primitive_out = plan->primitives[primitive_index];
    return 0;
}

void mglRenderCppDestroyCullDistanceIndexPlan(void** owner) {
    if (!owner || !*owner) return;
    delete static_cast<mgl::CullDistanceIndexPlan*>(*owner);
    *owner = nullptr;
}

int mglRenderCppSetRenderBuffer(void* render_encoder,
                                void* buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Buffer* resource = static_cast<MTL::Buffer*>(buffer);
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        encoder->setVertexBuffer(resource, static_cast<NS::UInteger>(offset),
                                 index);
    } else {
        encoder->setFragmentBuffer(resource,
                                   static_cast<NS::UInteger>(offset), index);
    }
    return 0;
}

int mglRenderCppSetRenderBytes(void* render_encoder,
                               const void* bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || (!bytes && length != 0) ||
        stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        encoder->setVertexBytes(bytes, static_cast<NS::UInteger>(length), index);
    } else {
        encoder->setFragmentBytes(bytes, static_cast<NS::UInteger>(length),
                                  index);
    }
    return 0;
}

int mglRenderCppSetRenderPipelineState(void* render_encoder,
                                       void* pipeline_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::RenderPipelineState* pipeline =
        static_cast<MTL::RenderPipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setRenderPipelineState(pipeline);
    return 0;
}

int mglRenderCppSetRenderDepthStencilState(void* render_encoder,
                                           void* depth_stencil_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::DepthStencilState* state =
        static_cast<MTL::DepthStencilState*>(depth_stencil_state);
    if (!encoder || !state) return -1;
    encoder->setDepthStencilState(state);
    return 0;
}

int mglRenderCppSetRenderTexture(void* render_encoder,
                                 void* texture,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Texture* resource = static_cast<MTL::Texture*>(texture);
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        encoder->setVertexTexture(resource, index);
    } else {
        encoder->setFragmentTexture(resource, index);
    }
    return 0;
}

int mglRenderCppSetRenderSampler(void* render_encoder,
                                 void* sampler,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT) return -1;
    MTL::SamplerState* resource = static_cast<MTL::SamplerState*>(sampler);
    if (stage == MGL_RENDER_CPP_BINDING_STAGE_VERTEX) {
        encoder->setVertexSamplerState(resource, index);
    } else {
        encoder->setFragmentSamplerState(resource, index);
    }
    return 0;
}

int mglRenderCppSetRenderViewport(void* render_encoder,
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

int mglRenderCppSetRenderScissor(void* render_encoder,
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

int mglRenderCppSetDepthClipMode(void* render_encoder, uint32_t mode) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setDepthClipMode(static_cast<MTL::DepthClipMode>(mode));
    return 0;
}

int mglRenderCppSetStencilReferenceValues(void* render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setStencilReferenceValues(front_reference, back_reference);
    return 0;
}

int mglRenderCppSetTessellationFactorBuffer(void* render_encoder,
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

int mglRenderCppDrawPatches(void* render_encoder,
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

int mglRenderCppDrawIndexedPatches(void* render_encoder,
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

int mglRenderCppCreateIndirectCommandBuffer(
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
    mgl::RendererCpp& renderer = mgl::renderer();
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

int mglRenderCppResetIndirectCommandBuffer(void* indirect_buffer,
                                           uint64_t location,
                                           uint64_t length) {
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!buffer || length == 0) return -1;
    buffer->reset(NS::Range(location, length));
    return 0;
}

int mglRenderCppGetIndirectRenderCommand(void* indirect_buffer,
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

int mglRenderCppSetIndirectDrawIndexed(void* indirect_command,
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

int mglRenderCppSetIndirectDraw(void* indirect_command,
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

int mglRenderCppUseRenderResource(void* render_encoder,
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

int mglRenderCppExecuteIndirectCommands(void* render_encoder,
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

} // extern "C"
