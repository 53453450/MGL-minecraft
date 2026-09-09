/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_varsym.cpp
 * C1e — module-assembly VarSym classify / location assign extracted from
 * mgl_air_backend.cpp.  Backend keeps thin using-declarations + call site.
 */

#include "mgl_air_varsym.h"
#include "mgl_air_resource.h"
#include "mgl_air_type.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace mgl {
namespace air {

/* Keep aligned with MGL_AIR_GS_MAX_STREAMS in mgl_air_gs_abi.h (avoid that
 * header — it pulls mgl_shader_abi → Mach/GL). */
static constexpr uint32_t kGsMaxStreams = 4u;

uint32_t varyingLocationSpan(const MType &t)
{
    /* GL 4.6 §4.4.1: a matrix consumes one location per column; an array
     * of matrices consumes cols*N.  Non-matrix arrays consume one per
     * element (each element is one location / record slot). */
    uint32_t elem = (t.isMatrix() && t.cols > 0) ? t.cols : 1u;
    if (t.isArray() && t.arr > 0) return elem * (uint32_t)t.arr;
    return elem;
}

uint32_t airAttribLocation(const char *name, const char *const *attrib_names,
                           int maxAttribs)
{
    if (name && attrib_names && maxAttribs > 0) {
        for (int i = 0; i < maxAttribs; i++) {
            if (attrib_names[i] && strcmp(attrib_names[i], name) == 0) {
                return (uint32_t)i;
            }
        }
    }
    if (name) {
        static const struct { const char *n; uint32_t l; } def[] = {
            {"Position", 0}, {"Color", 1}, {"UV0", 2},
            {"UV1", 3}, {"UV2", 4}, {"Normal", 5},
        };
        for (const auto &d : def) {
            if (strcmp(d.n, name) == 0) {
                return d.l;
            }
        }
    }
    return UINT32_MAX;
}

void collectStageVarSyms(const MGLIRModule *mod, const MGLTranslationUnit *tu,
                         int stage, std::vector<VarSym> *out)
{
    if (!out) return;
    out->clear();
    if (!mod) return;

    const bool isVS = (stage == AIR_STAGE_VERTEX);
    const bool isTCS = (stage == AIR_STAGE_TESS_CONTROL);
    const bool isTES = (stage == AIR_STAGE_TESS_EVALUATION);
    const bool isGS = (stage == AIR_STAGE_GEOMETRY);

    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function) {
            continue;
        }
        /* Builtin interface-block shells/members are lowered through the
         * dedicated gl_Position/gl_PointSize/gl_CullDistance paths below,
         * never as user varyings.  EXCEPTIONS (must become real kernel
         * parameters, mirroring mgl_air_reflect.c's refined skip):
         * uniform-qualified gl_ symbols (legacy fixed-function matrix
         * uniforms injected verbatim) and explicitly-located gl_ symbols
         * (legacy gl_Vertex injected with layout(location = 0)). */
        if (s->name && strncmp(s->name, "gl_", 3) == 0 &&
            !(s->qualifiers & MGL_AST_Q_UNIFORM) &&
            s->location == UINT32_MAX) {
            continue;
        }
        /* GS interface-block instances flatten into per-member VARYING
         * symbols (block_name set); the struct-typed instance symbol
         * itself carries no interface storage. */
        {
            const MGLIRType *it = s->type;
            bool structShaped =
                it->kind == MGLIR_TYPE_STRUCT ||
                (it->kind == MGLIR_TYPE_ARRAY && it->elem_type &&
                 it->elem_type->kind == MGLIR_TYPE_STRUCT);
            if (!s->block_name && structShaped &&
                (s->qualifiers & (MGL_AST_Q_IN | MGL_AST_Q_OUT)) &&
                !(s->qualifiers & (MGL_AST_Q_UNIFORM | MGL_AST_Q_BUFFER))) {
                continue;
            }
        }
        /* Flattened anonymous UBO/SSBO members (block_name set) are not
         * Metal buffer arguments — only the owning block instance is.
         * Emitting a slot per member shifts bindings and leaves the
         * member buffers unbound (CTS unnamed `buffer Data {…};`). */
        if (s->block_name &&
            (s->qualifiers & (MGL_AST_Q_UNIFORM | MGL_AST_Q_BUFFER))) {
            continue;
        }
        VarSym v;
        v.name = s->name;
        v.type = typeFromIR(s->type);
        v.location = s->location;
        v.locationExplicit = (s->location != UINT32_MAX);
        v.stream = s->stream;
        v.blockName = s->block_name ? s->block_name : "";
        uint32_t q = s->qualifiers;
        v.isPatch = (q & MGL_AST_Q_PATCH) != 0;
        v.isSample = (q & MGL_AST_Q_SAMPLE) != 0;
        if (q & MGL_AST_Q_UNIFORM) {
            const MGLIRType *ut = s->type;
            if (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type)
                ut = ut->elem_type; /* UBO instance array */
            if (ut->kind == MGLIR_TYPE_SAMPLER) {
                v.kind = VarSym::TEXTURE;
            } else if (ut->kind == MGLIR_TYPE_IMAGE) {
                v.kind = VarSym::IMAGE;
            } else if (ut->kind == MGLIR_TYPE_ATOMIC_COUNTER ||
                       (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type &&
                        ut->elem_type->kind == MGLIR_TYPE_ATOMIC_COUNTER)) {
                v.kind = VarSym::ATOMIC_COUNTER;
            } else if (ut->kind == MGLIR_TYPE_STRUCT &&
                       ut->member_count > 0 &&
                       s->is_interface_block) {
                v.kind = VarSym::UBO;
            } else {
                v.kind = VarSym::BUFFER;
            }
        } else if (q & MGL_AST_Q_BUFFER) {
            v.kind = VarSym::SSBO;
        } else if (isTCS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isTCS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isGS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
            /* Plain gl_in-style input arrays index by input vertex; keep
             * the element type.  Interface-block members (block_name set)
             * keep their array shape: indexing selects the element slot
             * at base location + index. */
            if (!s->block_name &&
                s->type->kind == MGLIR_TYPE_ARRAY && s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isGS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
            if (v.stream < 0) {
                v.stream = (tu && tu->layout_stream >= 0)
                    ? tu->layout_stream : 0;
            }
            if (v.stream < 0 || v.stream >= (int32_t)kGsMaxStreams) {
                v.stream = 0;
            }
        } else if (isTES && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::CONTROL_POINT_INPUT;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::ATTR;
        } else if ((isVS || isTES) && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
        }
        out->push_back(v);
        if (v.kind == VarSym::BUFFER && (q & MGL_AST_Q_UNIFORM)) {
            const MGLIRType *bt = s->type;
            const MGLIRType *base = bt;
            while (base && base->kind == MGLIR_TYPE_ARRAY)
                base = base->elem_type;
            if (base && base->kind == MGLIR_TYPE_STRUCT)
                appendOpaqueUniformLeaves(*out, bt, s->name);
        }
    }
}

void assignStageVarSymLocations(std::vector<VarSym> &syms, int stage,
                                bool isKernel, bool has_gs,
                                const char *const *attrib_names,
                                int maxAttribs,
                                const AirIfaceLocationPeer *peers,
                                uint32_t peerCount)
{
    const bool isVS = (stage == AIR_STAGE_VERTEX);
    const bool isTCS = (stage == AIR_STAGE_TESS_CONTROL);
    const bool isTES = (stage == AIR_STAGE_TESS_EVALUATION);
    const bool isGS = (stage == AIR_STAGE_GEOMETRY);

    uint32_t nextInputLocation = 0;
    uint32_t nextOutputLocation = 0;
    uint32_t nextPatchInputLocation = 0;
    uint32_t nextPatchOutputLocation = 0;
    for (VarSym &v : syms) {
        bool input = ((isTCS || isGS) && v.kind == VarSym::VARYING) ||
                     (isTES && v.kind == VarSym::CONTROL_POINT_INPUT) ||
                     /* Fragment inputs are VARYING on the FS; assign
                      * locations so has_gs location tags (mgl_loc_N)
                      * can pair with the GS passthrough VS. */
                     (!isVS && !isTES && !isTCS && !isGS && !isKernel &&
                      v.kind == VarSym::VARYING);
        bool output = ((isVS || isTES) && v.kind == VarSym::VARYING) ||
                      ((isTCS || isGS) && v.kind == VarSym::OUTPUT) ||
                      (!isVS && !isTES && !isKernel &&
                       v.kind == VarSym::OUTPUT);
        if (input) {
            uint32_t &next = v.isPatch
                ? nextPatchInputLocation : nextInputLocation;
            /* VS ATTR: prefer glBindAttribLocation (attrib_names) over
             * declaration-order auto-assign, matching mgl_air_reflect.c.
             * Sparse binds (CTS enable_disable even/odd locations) must
             * put [[attribute(N)]] at the bound N; the vertex descriptor
             * is driven by reflection of those same binds. */
            if (isVS && v.kind == VarSym::ATTR && !v.locationExplicit &&
                attrib_names) {
                uint32_t want =
                    airAttribLocation(v.name.c_str(), attrib_names,
                                      maxAttribs);
                if (want != UINT32_MAX)
                    v.location = want;
            }
            if (v.location == UINT32_MAX) v.location = next;
            next = std::max(next, v.location + varyingLocationSpan(v.type));
        }
        if (output) {
            uint32_t &next = v.isPatch
                ? nextPatchOutputLocation : nextOutputLocation;
            if (v.location == UINT32_MAX) v.location = next;
            next = std::max(next, v.location + varyingLocationSpan(v.type));
        }
    }
    /* GS-expansion passthrough VS tags outputs as mgl_loc_N using the
     * GS reflection locations.  FS with has_gs must use the same N for
     * each varying; declaration-order auto-assign can disagree when GS
     * and FS list the same names in different order (CTS utf8_characters
     * gs_fs_tex_coord before/after gs_fs_result).  Remap by name.
     * TES←TCS uses the same peer list: a TES that omits some TCS outs
     * (e.g. only `test_vector2`) must still read the producer location. */
    if (peers && peerCount > 0 &&
        ((has_gs && stage == AIR_STAGE_FRAGMENT) ||
         stage == AIR_STAGE_TESS_EVALUATION)) {
        for (VarSym &v : syms) {
            const bool fsVarying =
                stage == AIR_STAGE_FRAGMENT && v.kind == VarSym::VARYING;
            const bool tesCpIn =
                stage == AIR_STAGE_TESS_EVALUATION &&
                v.kind == VarSym::CONTROL_POINT_INPUT;
            if ((!fsVarying && !tesCpIn) || v.locationExplicit)
                continue;
            for (uint32_t i = 0; i < peerCount; i++) {
                const AirIfaceLocationPeer &peer = peers[i];
                if (!peer.name)
                    continue;
                if (peer.isPerPatch != v.isPatch)
                    continue;
                if (strcmp(peer.name, v.name.c_str()) != 0)
                    continue;
                if (peer.location != UINT32_MAX)
                    v.location = peer.location;
                break;
            }
        }
    }
}

uint32_t stageRecordStride(const std::vector<VarSym> &syms, VarSym::Kind kind,
                           bool patch, uint32_t baseStride)
{
    uint32_t stride = baseStride;
    for (const VarSym &v : syms) {
        if (v.kind != kind || v.isPatch != patch ||
            v.location == UINT32_MAX) continue;
        uint64_t end = patch
            ? ((uint64_t)v.location + varyingLocationSpan(v.type)) * 16u
            : (uint64_t)baseStride +
              ((uint64_t)v.location + varyingLocationSpan(v.type)) * 16u;
        if (end > UINT32_MAX) return 0u;
        stride = std::max(stride, (uint32_t)end);
    }
    return stride;
}

} /* namespace air */
} /* namespace mgl */
