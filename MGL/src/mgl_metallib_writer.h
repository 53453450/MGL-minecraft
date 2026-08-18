/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
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
 * mgl_metallib_writer.h
 * MGL - .metallib container serializer for the AIR backend.
 *
 * Layout is reverse-engineered from `xcrun metallib` output (macOS 26
 * SDK) and empirically validated with newLibraryWithData + PSO creation:
 * each shader entry gets its own standalone LLVM bitcode module (blob);
 * the function list records carry NAME/TYPE/HASH/SIZE/OFFSET/VERSION tags
 * and each record is followed by the byte length of the next record
 * (the last record is followed by the fourcc "ENDT").
 */

#ifndef MGL_METALLIB_WRITER_H
#define MGL_METALLIB_WRITER_H

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

namespace mgl {

enum MTLBFunctionType {
    MTLB_FN_VERTEX = 0,
    MTLB_FN_FRAGMENT = 1,
    MTLB_FN_KERNEL = 2,
};

struct MTLBFunction {
    std::string name;
    uint8_t type;                    /* MTLBFunctionType */
    uint8_t tessellation = 0;        /* 4 * control points + patch kind */
    std::vector<uint8_t> bitcode;    /* standalone module blob */
};

/* Serialize a function list into the MTLB container.  Each function's
 * `bitcode` blob must be a self-contained LLVM bitcode module. */
void mglMTLBWrite(const std::vector<MTLBFunction> &fns, llvm::raw_ostream &os);

} /* namespace mgl */

#endif /* MGL_METALLIB_WRITER_H */
