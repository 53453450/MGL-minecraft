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
 * MGLRenderer+VertexLayout_Private.h
 * MGL
 *
 * Private method declarations for the VertexLayout category
 * (MGLRenderer+VertexLayout.m).  Imports MGLRenderer_Private.h for ivar
 * access and shared types.
 */

#ifndef MGLRenderer_VertexLayout_Private_h
#define MGLRenderer_VertexLayout_Private_h

#import "MGLRenderer.h"

/* P4.2: value-state 版顶点布局填充（完整定义见 mgl_air_loader.h）。 */
typedef struct MGLRenderCppPipelineDescriptorState
    MGLRenderCppPipelineDescriptorState;

@interface MGLRenderer ()

- (BOOL)generateVertexDescriptorState:(MGLRenderCppPipelineDescriptorState *)state;
- (void)updateBlendStateCache;

@end

#endif /* MGLRenderer_VertexLayout_Private_h */
