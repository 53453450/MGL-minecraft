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
 * MGLRenderer+Tessellation_Private.h
 * MGL
 *
 * Private method declarations for the Tessellation category
 * (MGLRenderer+Tessellation.m).  The tessellation compute path (TCS/TES
 * dispatch) runs GL_PATCHES draws as consecutive Metal compute encoders.
 *
 * dispatchTessControlShader:/dispatchTessEvaluationShader: are the entry
 * points called from MGLRenderer+Draw.m; the remaining methods are internal
 * helpers used only within the category.
 *
 * Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_Tessellation_Private_h
#define MGLRenderer_Tessellation_Private_h

#import "MGLRenderer.h"

struct MGLAIRTessDrawContract;
typedef struct MGLAIRTessDrawContract MGLAIRTessDrawContract;

@interface MGLRenderer ()

/* Isolines / point-mode TES: expand one vertex record per work item with
 * the AIR TES compute kernel, then rasterize through the passthrough
 * vertex stage as lines / points.  Called from the airTES branch of the
 * GL_PATCHES draw path (MGLRenderer+DrawSupport.m). */
- (BOOL)dispatchAIRTessEvalCompute:(GLMContext)glm_ctx
                          program:(Program *)tesProgram
                         contract:(const MGLAIRTessDrawContract *)contract
                       patchCount:(GLuint)patchCount
                    instanceCount:(GLsizei)instanceCount
                     baseInstance:(GLuint)baseInstance;

@end

#endif /* MGLRenderer_Tessellation_Private_h */
