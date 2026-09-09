/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_varsym.h
 *
 * C1e domain strip from mgl_air_backend.cpp — module-assembly VarSym stage
 * classify + location assign (plus varyingLocationSpan / attrib preference /
 * record stride helpers).  Backend keeps a thin call-site facade.
 * Do not sink these back into mgl_air_backend.cpp; do not move emitExpr
 * here (matrix builtins → mgl_air_matrix / C1f).
 */

#ifndef MGL_AIR_VARSYM_H
#define MGL_AIR_VARSYM_H

#include <cstdint>
#include <vector>

#include "mgl_air_codegen.h"
#include "mgl_glsl_ast.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"

namespace mgl {
namespace air {

/* Match MGL_STAGE_* (mgl_shader_abi.h) without pulling Mach/GL headers. */
enum AirStageId : int {
    AIR_STAGE_VERTEX = 0,
    AIR_STAGE_FRAGMENT = 1,
    AIR_STAGE_COMPUTE = 2,
    AIR_STAGE_TESS_CONTROL = 3,
    AIR_STAGE_TESS_EVALUATION = 4,
    AIR_STAGE_GEOMETRY = 5,
};

/* Thin peer view for FS←GS / TES←TCS location remapping (avoids GLboolean). */
struct AirIfaceLocationPeer {
    const char *name;
    uint32_t location;
    bool isPerPatch;
};

/* GLSL matrix varyings consume one location per column (GL 4.6 §4.4.1). */
uint32_t varyingLocationSpan(const MType &t);

/* Desired vertex attribute location: explicit glBindAttribLocation bindings
 * first, then Mojang stable names.  UINT32_MAX = no preference. */
uint32_t airAttribLocation(const char *name, const char *const *attrib_names,
                           int maxAttribs);

/* Classify IR symbols into stage VarSyms (kind/type/qualifiers). Clears *out
 * then fills; flattens opaque leaves inside struct uniforms. */
void collectStageVarSyms(const MGLIRModule *mod, const MGLTranslationUnit *tu,
                         int stage, std::vector<VarSym> *out);

/* Assign input/output (and patch) locations; optional attrib binds + peers. */
void assignStageVarSymLocations(std::vector<VarSym> &syms, int stage,
                                bool isKernel, bool has_gs,
                                const char *const *attrib_names,
                                int maxAttribs,
                                const AirIfaceLocationPeer *peers,
                                uint32_t peerCount);

/* Max end-of-record byte offset for symbols of `kind` (patch or plain). */
uint32_t stageRecordStride(const std::vector<VarSym> &syms, VarSym::Kind kind,
                           bool patch, uint32_t baseStride);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_VARSYM_H */
