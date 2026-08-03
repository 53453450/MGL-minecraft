/*
  mgl_context_enums.h
  mgl

  Single source of truth for the MGLContext enums, shared by
  MGLContext.h (external API) and glm_context.h (internal GL layer).
*/

#ifndef MGL_CONTEXT_ENUMS_DEFINED
#define MGL_CONTEXT_ENUMS_DEFINED
enum {
    MGL_PIXEL_FORMAT,
    MGL_PIXEL_TYPE,
    MGL_DEPTH_FORMAT,
    MGL_DEPTH_TYPE,
    MGL_STENCIL_FORMAT,
    MGL_STENCIL_TYPE,
    MGL_CONTEXT_FLAGS
};
#endif /* MGL_CONTEXT_ENUMS_DEFINED */