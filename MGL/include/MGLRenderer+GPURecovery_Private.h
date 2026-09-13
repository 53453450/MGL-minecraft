/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * MGLRenderer+GPURecovery_Private.h
 * MGL
 *
 * Private method declarations for the GPURecovery category
 * (MGLRenderer+GPURecovery.m).  Imports MGLRenderer_Private.h for ivar
 * access and shared types.
 */

#ifndef MGLRenderer_GPURecovery_Private_h
#define MGLRenderer_GPURecovery_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer ()

/* AGX recovery: recreate the command queue (implemented in MGLRenderer.m). */
- (int)mglRecreateCommandQueue;
- (void *)mglMetalDevicePointer;
- (int)mglMetalObjectsPresent;


/* now the C function mglRendererValidateMetalObjects (mgl_gpu_recovery.h) */
/* now the C functions in mgl_gpu_recovery.h */
- (void)cleanupCommandBuffer;
/* now the C function mglRendererResetMetalState (mgl_gpu_recovery.h) */
- (void)emergencyResetMetalState;
- (void)commitCommandBufferWithAGXRecovery:(id)commandBuffer;
/* now the C functions in mgl_gpu_recovery.h */
/* now the C functions in mgl_gpu_recovery.h */
/* now the C functions in mgl_gpu_recovery.h */
/* now the C functions in mgl_gpu_recovery.h */
/* now the C functions in mgl_gpu_recovery.h */

@end

#endif /* MGLRenderer_GPURecovery_Private_h */
