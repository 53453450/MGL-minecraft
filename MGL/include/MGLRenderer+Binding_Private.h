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
 * MGLRenderer+Binding_Private.h
 * MGL
 *
 * Private method declarations for the Binding category
 * (MGLRenderer+Binding.m).  Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access
 * and shared types.
 */

#ifndef MGLRenderer_Binding_Private_h
#define MGLRenderer_Binding_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer ()

/* the buffer bind is now the C function mglRendererBindMTLBuffer
 * (mgl_renderer_ports.h), and the texture bind body is the C function
 * mglRendererBindMTLTexture (mgl_texture_bind.h); this method only keeps the
 * lock/assert frame for the Objective-C call sites.  Its implementation lives in
 * the Objective-C shell TU now (MGLRenderer+Binding.m is gone). */
- (bool)bindMTLTexture:(Texture *)tex;

/* Sampler cascade for a sampled texture: the GL sampler object at the unit,
 * else the texture parameters, else `defaultSampler`.  Shared by the
 * vertex / fragment spine and the compute texture loop (O5.2); `stage` only
 * feeds the decision "vertex requires texture parameters" and the trace tag
 * (see mglBindingTextureSamplerStageTag). */
/* materializeSampledSamplerForTexture: is C now (mglSampledSamplerMaterialize). */

@end

#endif /* MGLRenderer_Binding_Private_h */
