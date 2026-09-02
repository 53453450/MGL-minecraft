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

bool mglRendererTextureBindLocked(MGLRenderer *self, Texture *tex);

@interface MGLRenderer ()

- (void)bindMTLBuffer:(Buffer *)ptr;
- (void)bindMTLBufferLocked:(Buffer *)ptr;
- (bool)bindMTLTexture:(Texture *)tex;
- (bool)bindMTLTextureLocked:(Texture *)tex;

@end

#endif /* MGLRenderer_Binding_Private_h */
