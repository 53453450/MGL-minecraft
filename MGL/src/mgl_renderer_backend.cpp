#include "mgl_renderer_backend.h"

#include <algorithm>
#include <mutex>

#include "glm_context.h"
#include "mgl_program_resource.h"
#include "mgl_render_cpp.h"
#include "mgl_shader_resource.h"

extern "C" Program *mglResolveProgramForStageFromState(
    GLMContext context, int stage);
extern "C" void mglRendererCompatDispatchCompute(
    void *compat_context, GLMContext context,
    unsigned int groups_x, unsigned int groups_y, unsigned int groups_z);
extern "C" void mglRendererCompatDispatchComputeIndirect(
    void *compat_context, GLMContext context, intptr_t indirect);
extern "C" void mglRendererCompatDraw(
    void *compat_context, GLMContext context,
    const MGLRenderCppDrawCallbackArgs *args);
extern "C" void mglRendererCompatBindTexture(
    void *compat_context, GLMContext context, Texture *texture);
extern "C" void mglRendererCompatFlushDrawBuffer(
    void *compat_context, GLMContext context);
extern "C" void mglRendererCompatSwapBuffers(
    void *compat_context, GLMContext context);
extern "C" void mglRendererCompatClearBuffer(
    void *compat_context, GLMContext context,
    unsigned int type, unsigned int mask);
extern "C" void mglRendererCompatBlitFramebuffer(
    void *compat_context, GLMContext context,
    int src_x0, int src_y0, int src_x1, int src_y1,
    int dst_x0, int dst_y0, int dst_x1, int dst_y1,
    unsigned int mask, unsigned int filter);
extern "C" int mglRendererCompatResource(
    void *compat_context, GLMContext context,
    const MGLRenderCppResourceCallbackArgs *args);

struct MGLRendererBackendHandle {
    std::mutex mutex;
    GLMContext context = nullptr;
    void *command_queue_owner = nullptr;
    void *command_buffer_owner = nullptr;
    void *render_encoder_owner = nullptr;
    void *render_pass_state_owner = nullptr;
    void *query_owner = nullptr;
    void *recovery_owner = nullptr;
    void *binding_owner = nullptr;
    void *operation_context = nullptr;
    bool renderer_initialized = false;
    bool shutdown_started = false;
    bool destroying = false;
};

static void *mglRendererBackendCompatContext(GLMContext context)
{
    if (!context || !context->renderer_backend) return nullptr;
    return mglRendererBackendGetOperationContext(
        static_cast<MGLRendererBackendHandle *>(context->renderer_backend));
}

static void mglRendererBackendReleaseOwnedState(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return;
    if (backend->operation_context) {
        mglRendererReleaseOperationContext(backend->operation_context);
        backend->operation_context = nullptr;
    }
    mglRenderCppDestroyCommandQueueOwner(&backend->command_queue_owner);
    mglRenderCppBindingDestroy(backend->binding_owner);
    backend->binding_owner = nullptr;
    mglRenderCppDestroyQueryStateOwner(&backend->query_owner);
    mglRenderCppDestroyCommandRecoveryOwner(&backend->recovery_owner);
    backend->command_buffer_owner = nullptr;
    backend->render_encoder_owner = nullptr;
    backend->render_pass_state_owner = nullptr;
    if (backend->renderer_initialized) {
        mglRenderCppShutdown();
        backend->renderer_initialized = false;
    }
}

extern "C" int mglRendererBackendCreate(
    const MGLRendererBackendCreateInfo *info,
    MGLRendererBackendHandle **backend_out)
{
    if (backend_out) *backend_out = nullptr;
    if (!info || !backend_out || !info->objc_device ||
        info->binding_slot_count == 0 || info->query_capacity == 0) {
        return -1;
    }

    MGLRendererBackendHandle *backend = new MGLRendererBackendHandle();
    backend->context = info->context;
    if (mglRenderCppInit(info->objc_device) != 0) {
        delete backend;
        return -1;
    }
    backend->renderer_initialized = true;

    backend->binding_owner =
        mglRenderCppBindingCreate(info->binding_slot_count);
    if (!backend->binding_owner ||
        mglRenderCppCreateQueryStateOwner(
            info->query_capacity, &backend->query_owner) != 0 ||
        mglRenderCppCreateCommandRecoveryOwner(
            &backend->recovery_owner) != 0) {
        mglRendererBackendReleaseOwnedState(backend);
        delete backend;
        return -1;
    }
    *backend_out = backend;
    return 0;
}

extern "C" int mglRendererBackendIsReady(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->renderer_initialized && !backend->shutdown_started &&
           backend->command_queue_owner && backend->binding_owner &&
           backend->query_owner && backend->recovery_owner;
}

extern "C" int mglRendererBackendResetCommandQueue(
    MGLRendererBackendHandle *backend,
    uint32_t max_command_buffers,
    void **command_queue_out)
{
    if (command_queue_out) *command_queue_out = nullptr;
    if (!backend || !command_queue_out) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    if (!backend->renderer_initialized || backend->shutdown_started) return -1;
    void *queue = nullptr;
    int result = backend->command_queue_owner
        ? mglRenderCppResetCommandQueueOwner(
              backend->command_queue_owner, max_command_buffers, &queue)
        : mglRenderCppCreateCommandQueueOwner(
              max_command_buffers, &backend->command_queue_owner, &queue);
    if (result != 0 || !queue) return -1;
    *command_queue_out = queue;
    return 0;
}

extern "C" int mglRendererBackendAttachRuntimeOwners(
    MGLRendererBackendHandle *backend,
    void *command_buffer_owner,
    void *render_encoder_owner,
    void *render_pass_state_owner)
{
    if (!backend) return -1;
    std::lock_guard<std::mutex> lock(backend->mutex);
    backend->command_buffer_owner = command_buffer_owner;
    backend->render_encoder_owner = render_encoder_owner;
    backend->render_pass_state_owner = render_pass_state_owner;
    return 0;
}

extern "C" int mglRendererBackendInstallOperationContext(
    MGLRendererBackendHandle *backend,
    void *operation_context)
{
    if (!backend || !operation_context) return -1;
    void *previous = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->shutdown_started || backend->destroying) return -1;
        previous = backend->operation_context;
        backend->operation_context = operation_context;
    }
    if (previous) mglRendererReleaseOperationContext(previous);
    return 0;
}

extern "C" void *mglRendererBackendGetOperationContext(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->operation_context;
}

extern "C" int mglRendererBackendIsDestroying(
    const MGLRendererBackendHandle *backend)
{
    if (!backend) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    return backend->destroying ? 1 : 0;
}

extern "C" void *mglRendererBackendGetOwner(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendOwnerKind kind)
{
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(
        const_cast<MGLRendererBackendHandle *>(backend)->mutex);
    switch (kind) {
        case MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE:
            return backend->command_queue_owner;
        case MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER:
            return backend->command_buffer_owner;
        case MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER:
            return backend->render_encoder_owner;
        case MGL_RENDERER_BACKEND_OWNER_RENDER_PASS:
            return backend->render_pass_state_owner;
        case MGL_RENDERER_BACKEND_OWNER_QUERY:
            return backend->query_owner;
        case MGL_RENDERER_BACKEND_OWNER_RECOVERY:
            return backend->recovery_owner;
        case MGL_RENDERER_BACKEND_OWNER_BINDING:
            return backend->binding_owner;
    }
    return nullptr;
}

extern "C" int mglRendererBackendShutdown(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendShutdownResult *result_out)
{
    if (result_out) *result_out = {};
    if (!backend) return -1;

    void *command_owner = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->shutdown_started) return 0;
        backend->shutdown_started = true;
        command_owner = backend->command_buffer_owner;
    }

    if (command_owner &&
        mglRenderCppCommandBufferOwnerHasLastSubmitted(command_owner) == 1) {
        MGLRenderCppCommandBufferState state = {};
        int wait_result = mglRenderCppWaitCommandBufferOwnerLastSubmitted(
            command_owner, &state);
        if (result_out) {
            result_out->waited_for_last_submission = 1;
            result_out->last_submission_has_error = state.has_error;
            result_out->last_submission_error_code = state.error_code;
            result_out->status = wait_result;
        }
        return wait_result;
    }
    return 0;
}

extern "C" void mglRendererBackendDestroy(
    MGLRendererBackendHandle **backend_ptr)
{
    if (!backend_ptr || !*backend_ptr) return;
    MGLRendererBackendHandle *backend = *backend_ptr;
    *backend_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->destroying) return;
        backend->destroying = true;
        if (backend->context && backend->context->renderer_backend == backend) {
            backend->context->renderer_backend = nullptr;
        }
    }
    (void)mglRendererBackendShutdown(backend, nullptr);
    mglRendererBackendReleaseOwnedState(backend);
    delete backend;
}

extern "C" void mglRendererBindBuffer(GLMContext context, Buffer *buffer)
{
    mglRenderCppBindBuffer(context, buffer);
}

extern "C" void mglRendererBindTexture(GLMContext context, Texture *texture)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) mglRendererCompatBindTexture(compat, context, texture);
}

extern "C" void mglRendererBindProgram(GLMContext context, Program *program)
{
    mglRenderCppBindProgram(context, program);
}

extern "C" void mglRendererDeleteMetalObject(GLMContext context, void *object)
{
    mglRenderCppDeleteMTLObj(context, object);
}

extern "C" void mglRendererReleaseBufferMetalData(
    GLMContext context, Buffer *buffer)
{
    mglRenderCppReleaseBufferMetalData(context, buffer);
}

extern "C" void mglRendererGetSync(GLMContext context, Sync *sync)
{
    mglRenderCppGetSync(context, sync);
}

extern "C" void mglRendererWaitForSync(GLMContext context, Sync *sync)
{
    mglRenderCppWaitForSync(context, sync);
}

extern "C" uint32_t mglRendererGetSyncStatus(
    GLMContext context, Sync *sync)
{
    return mglRenderCppGetSyncStatus(context, sync);
}

extern "C" void mglRendererReleaseSync(GLMContext context, Sync *sync)
{
    mglRenderCppReleaseSync(context, sync);
}

extern "C" void mglRendererFlush(GLMContext context, bool finish)
{
    mglRenderCppFlush(context, finish);
}

extern "C" void mglRendererSwapBuffers(GLMContext context)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) mglRendererCompatSwapBuffers(compat, context);
}

extern "C" void mglRendererFlushDrawBuffer(GLMContext context)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) mglRendererCompatFlushDrawBuffer(compat, context);
}

extern "C" void mglRendererInvalidateRenderPass(GLMContext context)
{
    mglRenderCppInvalidateRenderPass(context);
}

extern "C" void mglRendererClearBuffer(
    GLMContext context, uint32_t type, uint32_t mask)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) mglRendererCompatClearBuffer(compat, context, type, mask);
}

extern "C" void mglRendererBlitFramebuffer(
    GLMContext context,
    int32_t src_x0, int32_t src_y0, int32_t src_x1, int32_t src_y1,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
    uint32_t mask, uint32_t filter)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) {
        mglRendererCompatBlitFramebuffer(
            compat, context, src_x0, src_y0, src_x1, src_y1,
            dst_x0, dst_y0, dst_x1, dst_y1, mask, filter);
    }
}

extern "C" void mglRendererBufferSubData(
    GLMContext context, Buffer *buffer,
    size_t offset, size_t size, const void *bytes)
{
    mglRenderCppBufferSubData(context, buffer, offset, size, bytes);
}

extern "C" void *mglRendererMapUnmapBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size,
    uint32_t access, bool map)
{
    return mglRenderCppMapUnmapBuffer(
        context, buffer, offset, size, access, map);
}

extern "C" void mglRendererReadBackBuffer(
    GLMContext context, Buffer *buffer, size_t offset, size_t size)
{
    mglRenderCppReadBackBuffer(context, buffer, offset, size);
}

extern "C" void mglRendererFlushBufferRange(
    GLMContext context, Buffer *buffer, intptr_t offset, intptr_t length)
{
    mglRenderCppFlushBufferRange(context, buffer, offset, length);
}

static int mglRendererInvokeResource(
    GLMContext context, MGLRenderCppResourceCallbackArgs args)
{
    void *compat = mglRendererBackendCompatContext(context);
    return compat ? mglRendererCompatResource(compat, context, &args) : 0;
}

extern "C" void mglRendererReadDrawable(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DRAWABLE,
        .pixel_bytes = pixel_bytes,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
        .x = x,
        .y = y,
    });
}

extern "C" void mglRendererReadIntegerPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_INTEGER_PIXELS,
        .pixel_bytes = pixel_bytes,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
        .format = format,
        .type = type,
        .x = x,
        .y = y,
    });
}

extern "C" void mglRendererReadDepthPixels(
    GLMContext context, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DEPTH_PIXELS,
        .pixel_bytes = pixel_bytes,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
        .x = x,
        .y = y,
    });
}

extern "C" void mglRendererGetTexImage(
    GLMContext context, Texture *texture, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_GET_TEX_IMAGE,
        .texture = texture,
        .pixel_bytes = pixel_bytes,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
        .format = format,
        .type = type,
        .slice = slice,
        .level = level,
        .x = x,
        .y = y,
    });
}

extern "C" void mglRendererGenerateMipmaps(
    GLMContext context, Texture *texture)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_GENERATE_MIPMAPS,
        .texture = texture,
    });
}

extern "C" void mglRendererTexSubImage(
    GLMContext context, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch,
    size_t source_image_size, size_t source_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE,
        .texture = texture,
        .buffer = buffer,
        .source_offset = source_offset,
        .source_pitch = source_pitch,
        .source_image_size = source_image_size,
        .source_size = source_size,
        .width = width,
        .height = height,
        .depth = depth,
        .x_offset = x_offset,
        .y_offset = y_offset,
        .z_offset = z_offset,
        .slice = slice,
        .level = level,
    });
}

extern "C" bool mglRendererTexSubImageBytes(
    GLMContext context, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    return mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE_BYTES,
        .texture = texture,
        .bytes = bytes,
        .bytes_size = bytes_size,
        .source_offset = source_offset,
        .source_pitch = source_pitch,
        .source_image_size = source_image_size,
        .width = width,
        .height = height,
        .depth = depth,
        .x_offset = x_offset,
        .y_offset = y_offset,
        .z_offset = z_offset,
        .slice = slice,
        .level = level,
    }) != 0;
}

extern "C" void mglRendererCopyTexSubImage(
    GLMContext context, Texture *texture,
    uint32_t slice, int32_t level,
    int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_TEX_SUB_IMAGE,
        .texture = texture,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .x_offset = static_cast<size_t>(x_offset),
        .y_offset = static_cast<size_t>(y_offset),
        .slice = slice,
        .level = static_cast<uint32_t>(level),
        .x = x,
        .y = y,
    });
}

extern "C" void mglRendererCopyImageSubData(
    GLMContext context, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture,
    int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    mglRendererInvokeResource(context, {
        .kind = MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_IMAGE_SUB_DATA,
        .source_texture = source_texture,
        .destination_texture = destination_texture,
        .width = static_cast<size_t>(width),
        .height = static_cast<size_t>(height),
        .depth = static_cast<size_t>(depth),
        .source_level = source_level,
        .source_x = source_x,
        .source_y = source_y,
        .source_z = source_z,
        .destination_level = destination_level,
        .destination_x = destination_x,
        .destination_y = destination_y,
        .destination_z = destination_z,
    });
}

static void mglRendererInvokeDraw(
    GLMContext context, MGLRenderCppDrawCallbackArgs args)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) mglRendererCompatDraw(compat, context, &args);
}

extern "C" void mglRendererDrawArrays(
    GLMContext context, uint32_t mode, int32_t first, int32_t count)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS,
        .mode = mode, .first = first, .count = count,
    });
}

extern "C" void mglRendererDrawElements(
    GLMContext context, uint32_t mode, int32_t count,
    uint32_t type, const void *indices)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS,
        .mode = mode, .type = type, .count = count,
        .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawRangeElements(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS,
        .mode = mode, .type = type, .start = start, .end = end,
        .count = count, .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED,
        .mode = mode, .first = first, .count = count,
        .instance_count = instance_count,
    });
}

extern "C" void mglRendererDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED,
        .mode = mode, .type = type, .count = count,
        .instance_count = instance_count, .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_BASE_VERTEX,
        .mode = mode, .type = type, .count = count,
        .base_vertex = base_vertex, .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS_BASE_VERTEX,
        .mode = mode, .type = type, .start = start, .end = end,
        .count = count, .base_vertex = base_vertex,
        .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX,
        .mode = mode, .type = type, .count = count,
        .instance_count = instance_count, .base_vertex = base_vertex,
        .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INDIRECT,
        .mode = mode, .indices_or_indirect = indirect,
    });
}

extern "C" void mglRendererDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type, const void *indirect)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INDIRECT,
        .mode = mode, .type = type, .indices_or_indirect = indirect,
    });
}

extern "C" void mglRendererDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED_BASE_INSTANCE,
        .mode = mode, .first = first, .count = count,
        .instance_count = instance_count, .base_instance = base_instance,
    });
}

extern "C" void mglRendererDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_INSTANCE,
        .mode = mode, .type = type, .count = count,
        .instance_count = instance_count, .base_instance = base_instance,
        .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance)
{
    mglRendererInvokeDraw(context, {
        .kind =
            MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE,
        .mode = mode, .type = type, .count = count,
        .instance_count = instance_count, .base_vertex = base_vertex,
        .base_instance = base_instance, .indices_or_indirect = indices,
    });
}

extern "C" void mglRendererMultiDrawArrays(
    GLMContext context, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS,
        .mode = mode, .draw_count = draw_count,
        .firsts = firsts, .counts = counts,
    });
}

extern "C" void mglRendererMultiDrawElements(
    GLMContext context, uint32_t mode, const int32_t *counts,
    uint32_t type, const void *const *indices, int32_t draw_count)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS,
        .mode = mode, .type = type, .draw_count = draw_count,
        .indices_or_indirect = indices, .counts = counts,
    });
}

extern "C" void mglRendererMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts,
    uint32_t type, const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_BASE_VERTEX,
        .mode = mode, .type = type, .draw_count = draw_count,
        .indices_or_indirect = indices, .counts = counts,
        .base_vertices = base_vertices,
    });
}

extern "C" void mglRendererMultiDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect,
    int32_t draw_count, int32_t stride)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS_INDIRECT,
        .mode = mode, .draw_count = draw_count, .stride = stride,
        .indices_or_indirect = indirect,
    });
}

extern "C" void mglRendererMultiDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    mglRendererInvokeDraw(context, {
        .kind = MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_INDIRECT,
        .mode = mode, .type = type,
        .draw_count = draw_count, .stride = stride,
        .indices_or_indirect = indirect,
    });
}

extern "C" void mglRendererDispatchCompute(
    GLMContext context, uint32_t groups_x,
    uint32_t groups_y, uint32_t groups_z)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) {
        mglRendererCompatDispatchCompute(
            compat, context, groups_x, groups_y, groups_z);
    }
}

extern "C" void mglRendererDispatchComputeIndirect(
    GLMContext context, intptr_t indirect)
{
    void *compat = mglRendererBackendCompatContext(context);
    if (compat) {
        mglRendererCompatDispatchComputeIndirect(
            compat, context, indirect);
    }
}

extern "C" void mglRendererBeginSampleQuery(
    GLMContext context, uint32_t target)
{
    mglRenderCppBeginSampleQueryCallback(context, target);
}

extern "C" uint64_t mglRendererEndSampleQuery(GLMContext context)
{
    return mglRenderCppEndSampleQueryCallback(context);
}

extern "C" void mglRendererBeginTimerQuery(GLMContext context)
{
    mglRenderCppBeginTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererEndTimerQuery(GLMContext context)
{
    return mglRenderCppEndTimerQueryCallback(context);
}

extern "C" uint64_t mglRendererGetGPUTimestamp(GLMContext context)
{
    return mglRenderCppGetGPUTimestamp(context);
}

namespace {

bool mglRendererProgramResourceTypeIsSupported(int32_t type,
                                               bool include_separate)
{
    switch (type) {
        case _UNIFORM_BUFFER_RES:
        case _UNIFORM_CONSTANT_RES:
        case _STORAGE_BUFFER_RES:
        case _ATOMIC_COUNTER_RES:
        case _PUSH_CONSTANT_RES:
        case _STAGE_INPUT_RES:
        case _STAGE_OUTPUT_RES:
        case _SAMPLED_IMAGE_RES:
        case _STORAGE_IMAGE_RES:
            return true;
        case _SEPARATE_IMAGE_RES:
        case _SEPARATE_SAMPLERS_RES:
            return include_separate;
        default:
            return false;
    }
}

MGLShaderResource *mglRendererProgramResource(GLMContext context,
                                               int32_t stage,
                                               int32_t type,
                                               int32_t index,
                                               Program **program_out)
{
    if (program_out) *program_out = nullptr;
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return nullptr;
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return nullptr;
    MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
    if (index < 0 || index >= static_cast<int32_t>(list->count)) return nullptr;
    if (program_out) *program_out = program;
    return &list->list[index];
}

}  // namespace

extern "C" uint32_t mglDeclaredTextureTypeFromResource(
    const MGLShaderResource *resource)
{
    return mglRenderCppTextureTypeForShaderResource(
        resource != nullptr,
        resource ? static_cast<uint32_t>(resource->image_dim) : 0u,
        resource ? static_cast<uint32_t>(resource->image_arrayed) : 0u,
        resource ? static_cast<uint32_t>(resource->image_multisampled) : 0u);
}

extern "C" uint32_t mglExpectedTextureTypeForResource(
    Program *program, int32_t stage, MGLShaderResource *resource)
{
    if (!program || !resource || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    return mglDeclaredTextureTypeFromResource(resource);
}

extern "C" uint32_t mglExpectedTextureDataKindForResource(
    Program *program, int32_t stage, MGLShaderResource *resource)
{
    if (!program || !resource || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGL_SHADER_TEXTURE_DATA_UNKNOWN;
    }
    return resource->texture_data_kind != MGL_SHADER_TEXTURE_DATA_UNKNOWN
        ? static_cast<uint32_t>(resource->texture_data_kind)
        : static_cast<uint32_t>(MGL_SHADER_TEXTURE_DATA_FLOAT);
}

extern "C" int32_t mglRendererGetProgramBindingCount(
    GLMContext context, int32_t stage, int32_t type)
{
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        !mglRendererProgramResourceTypeIsSupported(type, true)) {
        return 0;
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    return program ? static_cast<int32_t>(
        program->shader_resources_list[stage][type].count) : 0;
}

extern "C" int32_t mglRendererGetProgramBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    if (!mglRendererProgramResourceTypeIsSupported(type, true)) return 0;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->binding) : 0;
}

extern "C" int32_t mglRendererGetProgramGLBinding(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->gl_binding) : 0;
}

extern "C" int32_t mglRendererGetProgramLocation(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    switch (type) {
        case _UNIFORM_BUFFER_RES:
        case _UNIFORM_CONSTANT_RES:
        case _STORAGE_BUFFER_RES:
        case _ATOMIC_COUNTER_RES:
        case _PUSH_CONSTANT_RES:
        case _STAGE_INPUT_RES:
        case _SAMPLED_IMAGE_RES:
        case _STORAGE_IMAGE_RES:
            break;
        default:
            return 0;
    }
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<int32_t>(resource->location) : 0;
}

extern "C" size_t mglRendererGetProgramBindingRequiredSize(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? static_cast<size_t>(resource->required_size) : 0u;
}

extern "C" intptr_t mglRendererGetProgramMetalBufferIndexForStage(
    GLMContext context, int32_t stage, uint32_t client_binding)
{
    static constexpr int32_t resource_types[] = {
        _UNIFORM_BUFFER_RES, _UNIFORM_CONSTANT_RES, _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES, _PUSH_CONSTANT_RES,
    };
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return static_cast<intptr_t>(client_binding);
    }
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return static_cast<intptr_t>(client_binding);
    for (int32_t type : resource_types) {
        MGLShaderResourceList *list =
            &program->shader_resources_list[stage][type];
        for (uint32_t i = 0; i < list->count; ++i) {
            MGLShaderResource *resource = &list->list[i];
            if (mglShouldSkipStageBufferResource(
                    program, stage, type, resource)) {
                continue;
            }
            if (mglClientBufferBindingForResource(type, resource) ==
                client_binding) {
                return static_cast<intptr_t>(mglMetalResourceSlot(resource));
            }
        }
    }
    return -1;
}

extern "C" size_t mglRendererGetProgramBindingRequiredSizeForStage(
    GLMContext context, int32_t stage, uint32_t client_binding)
{
    static constexpr int32_t resource_types[] = {
        _UNIFORM_BUFFER_RES, _UNIFORM_CONSTANT_RES, _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES, _PUSH_CONSTANT_RES,
    };
    if (!context || stage < 0 || stage >= _MAX_SHADER_TYPES) return 0u;
    Program *program = mglResolveProgramForStageFromState(context, stage);
    if (!program) return 0u;
    size_t required = 0u;
    for (int32_t type : resource_types) {
        MGLShaderResourceList *list =
            &program->shader_resources_list[stage][type];
        for (uint32_t i = 0; i < list->count; ++i) {
            MGLShaderResource *resource = &list->list[i];
            if (mglShouldSkipStageBufferResource(
                    program, stage, type, resource) ||
                mglClientBufferBindingForResource(type, resource) !=
                    client_binding) {
                continue;
            }
            required = std::max(
                required, static_cast<size_t>(resource->required_size));
        }
    }
    return required;
}

extern "C" uint32_t mglRendererGetProgramDeclaredTextureType(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, nullptr);
    return resource ? mglDeclaredTextureTypeFromResource(resource) : 0u;
}

extern "C" uint32_t mglRendererGetProgramExpectedTextureType(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    Program *program = nullptr;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, &program);
    return resource
        ? mglExpectedTextureTypeForResource(program, stage, resource)
        : 0u;
}

extern "C" uint32_t mglRendererGetProgramExpectedTextureDataKind(
    GLMContext context, int32_t stage, int32_t type, int32_t index)
{
    Program *program = nullptr;
    MGLShaderResource *resource = mglRendererProgramResource(
        context, stage, type, index, &program);
    return resource
        ? mglExpectedTextureDataKindForResource(program, stage, resource)
        : static_cast<uint32_t>(MGL_SHADER_TEXTURE_DATA_UNKNOWN);
}
