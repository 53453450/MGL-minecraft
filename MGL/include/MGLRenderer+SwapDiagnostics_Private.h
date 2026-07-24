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
 * MGLRenderer+SwapDiagnostics_Private.h
 * MGL
 *
 * Private method declarations for the SwapDiagnostics category
 * (MGLRenderer+SwapDiagnostics.m).  These methods run at mtlSwapBuffers time
 * to (1) copy offscreen render-pass color into the drawable when the default
 * framebuffer's blit path was bypassed, and (2) sample both source and
 * destination textures for low-frequency black-screen diagnostics.
 */

#ifndef MGLRenderer_SwapDiagnostics_Private_h
#define MGLRenderer_SwapDiagnostics_Private_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer (SwapDiagnostics)

/* Copies rpColor0 into drawableTexture when the default framebuffer's render
 * pass still targets an offscreen texture at swap time. */
- (void)copyRenderPassColorToDrawableIfNeeded:(id<MTLTexture>)rpColor0
                              drawableTexture:(id<MTLTexture>)drawableTexture
                                      swapCall:(uint64_t)swapCall
                                    traceSwap:(bool)traceSwap;

/* Samples both render-pass color source and drawable target at low frequency
 * for black-screen diagnostics. */
- (void)scheduleSwapTextureSampleDiagnostics:(id<MTLTexture>)rpColor0
                             drawableTexture:(id<MTLTexture>)drawableTexture
                                     swapCall:(uint64_t)swapCall;

@end

#endif /* MGLRenderer_SwapDiagnostics_Private_h */
