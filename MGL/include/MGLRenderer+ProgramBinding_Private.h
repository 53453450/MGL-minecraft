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
 * MGLRenderer+ProgramBinding_Private.h
 * MGL
 *
 * Private method declarations for the ProgramBinding category
 * (MGLRenderer+ProgramBinding.m).  All methods are read-only queries over
 * the active program's spirv_resources_list.  Imports MGLRenderer_Private.h
 * for ivar access and shared types.
 */

#ifndef MGLRenderer_ProgramBinding_Private_h
#define MGLRenderer_ProgramBinding_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer (ProgramBinding)

/* === SPIR-V resource list queries === */
- (int)getProgramBindingCount:(int)stage type:(int)type;
- (int)getProgramBinding:(int)stage type:(int)type index:(int)index;
- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index;
- (int)getProgramLocation:(int)stage type:(int)type index:(int)index;

/* === Buffer binding size / Metal slot queries === */
- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index;
- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage
                                      clientBinding:(GLuint)clientBinding;
- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage
                                   clientBinding:(GLuint)clientBinding;

/* === Texture type / data kind queries === */
- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index;
- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index;
- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index;

@end

/* === program-resolved texture type / data kind helpers ===
 *
 * These C helpers accept an already-resolved Program pointer and a
 * SpirvResource pointer, skipping the per-call mglResolveProgramForStageFromState
 * that the ObjC query methods perform.  Used by the hot sampled-texture binding
 * loops in MGLRenderer+Draw.m to eliminate 3-5 redundant program re-resolves per
 * resource.  Behavior matches the ObjC query methods (caching + MSL fallback +
 * rate-limited override logging). */
MTLTextureType mglDeclaredTextureTypeFromResource(const SpirvResource *res);
MTLTextureType mglExpectedTextureTypeForResource(Program *program, int stage, SpirvResource *res);
MGLTextureDataKind mglExpectedTextureDataKindForResource(Program *program, int stage, SpirvResource *res);

#endif /* MGLRenderer_ProgramBinding_Private_h */
