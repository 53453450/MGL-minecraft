/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer.h
 * MGL
 *
 * GLFW-side thin facade for the real MGL renderer.
 *
 * IMPORTANT: Do NOT import mgl_metal_bridge.h / glcorearb.h from this header.
 * mgl_context.m includes internal.h (which defines a tiny set of GL_* macros
 * for GLFW's own loader) and then this file. Pulling glcorearb.h here redefines
 * GL_VERSION / GL_EXTENSIONS / etc. and produces -Wmacro-redefined warnings
 * (and can break GLFW's constant set). The real Metal bridge protocol lives
 * in MGL proper; GLFW only needs a minimal ObjC surface to create/bind a
 * renderer to an NSWindow/NSView.
 */

#ifndef MGLRenderer_h
#define MGLRenderer_h

#ifdef __OBJC__

#import <AppKit/AppKit.h>

#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

/* Forward-declare GLenum so we don't need the full OpenGL registry headers
 * in the GLFW compile unit. Values match glcorearb.h. */
#ifndef GL_ENUM_DEFINED_FOR_MGL_RENDERER
typedef unsigned int GLenum;
#define GL_ENUM_DEFINED_FOR_MGL_RENDERER 1
#endif

@interface MGLRenderer : NSObject
{
}

- (id) initMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
- (id) createMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view;

@end

/* Pixel-format helpers are implemented in MGL; declare without pulling Metal
 * or glcorearb into this TU. */
GLenum mtlPixelFormatForGLFormatType(GLenum gl_format, GLenum gl_type);

#else /* !__OBJC__ */

#ifdef __cplusplus
extern "C" {
#endif

unsigned int mtlPixelFormatForGLFormatType(unsigned int gl_format, unsigned int gl_type);

#ifdef __cplusplus
}
#endif

#endif /* __OBJC__ */

#ifdef __cplusplus
extern "C" {
#endif
void* CppCreateMGLRendererFromContextAndBindToWindow (void *glm_ctx, void *window);
void* CppCreateMGLRendererHeadless (void *glm_ctx);
void* CppCreateMGLRendererAndBindToContext (void *glm_ctx);
#ifdef __cplusplus
}
#endif

#endif /* MGLRenderer_h */
