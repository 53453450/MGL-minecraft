#include "mgl_renderer_backend.h"

#include <mutex>

#include "glm_context.h"
#include "mgl_render_cpp.h"

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
    bool renderer_initialized = false;
    bool shutdown_started = false;
};

static void mglRendererBackendReleaseOwnedState(
    MGLRendererBackendHandle *backend)
{
    if (!backend) return;
    if (backend->context && backend->query_owner) {
        mglRenderCppUnregisterContextQueryStateOwner(
            backend->context, backend->query_owner);
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
    if (backend->context &&
        mglRenderCppRegisterContextQueryStateOwner(
            backend->context, backend->query_owner) != 0) {
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
    (void)mglRendererBackendShutdown(backend, nullptr);
    mglRendererBackendReleaseOwnedState(backend);
    delete backend;
    *backend_ptr = nullptr;
}
