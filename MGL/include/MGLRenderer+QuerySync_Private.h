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
 * MGLRenderer+QuerySync_Private.h
 * MGL
 *
 * Private method declarations for the QuerySync category (MGLRenderer+QuerySync.m).
 *
 * These selectors are temporary private operation targets until P5.4 moves
 * query and sync semantics into the C++ backend.
 *
 * Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_QuerySync_Private_h
#define MGLRenderer_QuerySync_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer ()

- (void)mtlGetSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)mtlWaitForSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (GLenum)mtlGetSyncStatus:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)mtlReleaseSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)mtlBeginSampleQuery:(GLMContext)glm_ctx target:(GLenum)target;
- (GLuint64)mtlEndSampleQuery:(GLMContext)glm_ctx;
- (void)mtlBeginTimerQuery:(GLMContext)glm_ctx;
- (GLuint64)mtlEndTimerQuery:(GLMContext)glm_ctx;
- (GLuint64)mtlGetGPUTimestamp:(GLMContext)glm_ctx;

@end

#endif /* MGLRenderer_QuerySync_Private_h */
