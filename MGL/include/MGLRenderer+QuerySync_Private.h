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
 * Note: The public QuerySync method declarations (mtlGetSync:sync:,
 * mtlWaitForSync:sync:, mtlGetSyncStatus:sync:, mtlReleaseSync:sync:,
 * mtlBeginSampleQuery:, mtlEndSampleQuery:, mtlBeginTimerQuery:,
 * mtlEndTimerQuery:, mtlGetGPUTimestamp:) are declared in mgl_metal_bridge.h
 * and called through GLMContext function pointers.  This header exists for
 * structural symmetry with the other per-category private headers and
 * MGLRenderer.h provides the MGLRenderer interface;
 * ivar access comes from MGLRenderer_Private.h in the implementing file.
 *
 * Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_QuerySync_Private_h
#define MGLRenderer_QuerySync_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer ()

@end

#endif /* MGLRenderer_QuerySync_Private_h */
