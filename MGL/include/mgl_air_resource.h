/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_resource.h
 *
 * C1c domain strip from mgl_air_backend.cpp — uniform / opaque resource
 * collection helpers (UBO instance-array shape, plain-uniform pack,
 * nested sampler/image leaf flatten, constant sampler-access path).
 * Do not sink these back into mgl_air_backend.cpp; do not put emitExpr
 * or matrix builtins here.
 */

#ifndef MGL_AIR_RESOURCE_H
#define MGL_AIR_RESOURCE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mgl_air_codegen.h"
#include "mgl_glsl_ast.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"

namespace mgl {
namespace air {

const MGLIRType *uniformBlockType(const MGLIRType *type);
uint32_t uniformBlockElementCount(const MGLIRType *type);
bool uniformBlockIsInstanceArray(const MGLIRType *type);

/* One implicit buffer holds every plain uniform, packed with std140
 * alignment in declaration order.  Returns 0 on success. */
int collectUniforms(const MGLIRModule *mod, std::vector<Uniform> *out,
                    uint32_t *bufferSize, char *err, size_t errCap);

/* Flatten sampler/image leaves inside named struct uniforms into TEXTURE /
 * IMAGE VarSyms (`s.c`, `s[0].c`, `s.b.a`) so Metal args and texture()
 * lookups match GL reflection names. */
void appendOpaqueUniformLeaves(std::vector<VarSym> &syms,
                               const MGLIRType *t,
                               const std::string &prefix);

/* Resolve texture()/texelFetch sampler argument to a flattened uniform name
 * (`tex`, `s.c`, `s[1].c`).  Only constant indices are supported. */
bool resolveSamplerAccessName(const MGLExpr *e, std::string *out);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_RESOURCE_H */
