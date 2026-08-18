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
 * MGLRenderer+Lifecycle_Private.h
 * MGL
 *
 * Private readiness declarations for the compatibility renderer shell.
 * The CppCreateMGLRenderer* C entry points are declared in MGLRenderer.h.
 *
 * Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_Lifecycle_Private_h
#define MGLRenderer_Lifecycle_Private_h

#import "MGLRenderer.h"

@interface MGLRenderer ()

- (BOOL)mglRendererIsReady;

@end

#endif /* MGLRenderer_Lifecycle_Private_h */
