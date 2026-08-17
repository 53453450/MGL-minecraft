#ifndef MGL_RENDERER_BACKEND_H
#define MGL_RENDERER_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GLMContextRec_t *GLMContext;
typedef struct MGLRendererBackendHandle MGLRendererBackendHandle;

typedef enum MGLRendererBackendOwnerKind {
    MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE = 0,
    MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER = 1,
    MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER = 2,
    MGL_RENDERER_BACKEND_OWNER_RENDER_PASS = 3,
    MGL_RENDERER_BACKEND_OWNER_QUERY = 4,
    MGL_RENDERER_BACKEND_OWNER_RECOVERY = 5,
    MGL_RENDERER_BACKEND_OWNER_BINDING = 6,
} MGLRendererBackendOwnerKind;

typedef struct MGLRendererBackendCreateInfo {
    void *objc_device;
    GLMContext context;
    uint32_t binding_slot_count;
    uint32_t query_capacity;
} MGLRendererBackendCreateInfo;

typedef struct MGLRendererBackendShutdownResult {
    int32_t status;
    uint32_t waited_for_last_submission;
    uint32_t last_submission_has_error;
    int64_t last_submission_error_code;
} MGLRendererBackendShutdownResult;

int mglRendererBackendCreate(const MGLRendererBackendCreateInfo *info,
                             MGLRendererBackendHandle **backend_out);
int mglRendererBackendIsReady(const MGLRendererBackendHandle *backend);
int mglRendererBackendResetCommandQueue(MGLRendererBackendHandle *backend,
                                        uint32_t max_command_buffers,
                                        void **command_queue_out);
int mglRendererBackendAttachRuntimeOwners(MGLRendererBackendHandle *backend,
                                          void *command_buffer_owner,
                                          void *render_encoder_owner,
                                          void *render_pass_state_owner);
void *mglRendererBackendGetOwner(const MGLRendererBackendHandle *backend,
                                 MGLRendererBackendOwnerKind kind);
int mglRendererBackendShutdown(MGLRendererBackendHandle *backend,
                               MGLRendererBackendShutdownResult *result_out);
void mglRendererBackendDestroy(MGLRendererBackendHandle **backend);

#ifdef __cplusplus
}
#endif

#endif
