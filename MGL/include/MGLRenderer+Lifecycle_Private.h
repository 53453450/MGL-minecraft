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
 * MGLRenderer+Lifecycle_Private.h
 * MGL
 *
 * Private method declarations for the Lifecycle category
 * (MGLRenderer+Lifecycle.m).  Covers renderer construction
 * (createMGLRendererAndBindToContext:view:), the glm_ctx mtl_funcs binding
 * table (bindObjFuncsToGLMContext:), proactive texture priming, Metal frame
 * capture helpers, and dealloc.  The CppCreateMGLRenderer* C entry points are
 * declared in MGLRenderer.h.
 *
 * Imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_Lifecycle_Private_h
#define MGLRenderer_Lifecycle_Private_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer ()

@end

#endif /* MGLRenderer_Lifecycle_Private_h */
