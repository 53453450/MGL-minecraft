/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_resource.cpp
 * C1c — AIR resource collection extracted from mgl_air_backend.cpp
 * (banner ~152–306).  Backend keeps thin using-declarations into mgl::air.
 */

#include "mgl_air_resource.h"
#include "mgl_air_type.h"

#include <cstdio>
#include <cstring>

namespace mgl {
namespace air {

const MGLIRType *uniformBlockType(const MGLIRType *type) {
    if (type && type->kind == MGLIR_TYPE_ARRAY)
        type = type->elem_type;
    return type && type->kind == MGLIR_TYPE_STRUCT && type->member_count > 0
        ? type : nullptr;
}

uint32_t uniformBlockElementCount(const MGLIRType *type) {
    /* `uniform Block { } name[1]` is still an instance array — keep the
     * declared length so blockC[0].x can index element slots. */
    if (type && type->kind == MGLIR_TYPE_ARRAY && type->array_size > 0u)
        return type->array_size;
    return 1u;
}

bool uniformBlockIsInstanceArray(const MGLIRType *type) {
    return type && type->kind == MGLIR_TYPE_ARRAY && type->array_size > 0u;
}

int collectUniforms(const MGLIRModule *mod, std::vector<Uniform> *out,
                    uint32_t *bufferSize, char *err, size_t errCap) {
    uint32_t off = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function || !(s->qualifiers & MGL_AST_Q_UNIFORM))
            continue;
        const MGLIRType *ut = s->type;
        while (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type)
            ut = ut->elem_type;
        if (ut->kind == MGLIR_TYPE_SAMPLER ||
            ut->kind == MGLIR_TYPE_IMAGE ||
            ut->kind == MGLIR_TYPE_ATOMIC_COUNTER) {
            continue;   /* texture/sampler/atomic-counter params are separate
                         * AIR args; packing them into the plain uniform blob
                         * would desync reflection offsets. */
        }
        /* Uniform blocks (interface-block structs) and their anonymous-block
         * members are independent device buffers, not part of the plain
         * uniform pack.  Named struct uniforms (`uniform S s`) stay here. */
        if (s->block_name ||
            (s->is_interface_block && uniformBlockType(s->type)))
            continue;
        uint32_t size = 0;
        if (mglIRComputeLayout(s->type, MGLIR_LAYOUT_STD140, &size) != 0) {
            snprintf(err, errCap, "layout failed for uniform %s", s->name);
            return -1;
        }
        off = (off + s->type->layout.alignment - 1) &
              ~(s->type->layout.alignment - 1);
        Uniform u;
        u.name = s->name;
        u.type = typeFromIR(s->type);
        u.offset = off;
        u.size = size;
        out->push_back(u);
        off += size;
    }
    *bufferSize = off;
    return 0;
}

void appendOpaqueUniformLeaves(std::vector<VarSym> &syms,
                               const MGLIRType *t,
                               const std::string &prefix)
{
    if (!t || prefix.empty())
        return;
    if (t->kind == MGLIR_TYPE_STRUCT) {
        for (uint32_t i = 0; i < t->member_count; i++) {
            const char *mn = t->member_names ? t->member_names[i] : nullptr;
            appendOpaqueUniformLeaves(
                syms, t->members[i],
                prefix + "." + (mn ? mn : "?"));
        }
        return;
    }
    if (t->kind == MGLIR_TYPE_ARRAY && t->elem_type &&
        t->elem_type->kind == MGLIR_TYPE_STRUCT) {
        uint32_t n = t->array_size ? t->array_size : 1u;
        for (uint32_t el = 0; el < n; el++) {
            appendOpaqueUniformLeaves(syms, t->elem_type,
                                      prefix + "[" + std::to_string(el) + "]");
        }
        return;
    }
    const MGLIRType *base = t;
    while (base && base->kind == MGLIR_TYPE_ARRAY)
        base = base->elem_type;
    if (!base)
        return;
    if (base->kind != MGLIR_TYPE_SAMPLER && base->kind != MGLIR_TYPE_IMAGE)
        return;
    VarSym ov;
    ov.name = prefix;
    ov.type = typeFromIR(t);
    ov.kind = base->kind == MGLIR_TYPE_SAMPLER ? VarSym::TEXTURE
                                               : VarSym::IMAGE;
    ov.opaqueType = t;
    syms.push_back(ov);
}

bool resolveSamplerAccessName(const MGLExpr *e, std::string *out)
{
    if (!e || !out)
        return false;
    struct Piece {
        enum { Field, Index } kind;
        std::string field;
        int64_t index;
    };
    std::vector<Piece> pieces;
    const MGLExpr *cur = e;
    while (cur) {
        if (cur->kind == MGL_EXPR_MEMBER) {
            if (!cur->u.member.field)
                return false;
            pieces.push_back({Piece::Field, cur->u.member.field, 0});
            cur = cur->u.member.object;
            continue;
        }
        if (cur->kind == MGL_EXPR_INDEX) {
            const MGLExpr *ix = cur->u.index.index;
            if (!ix || ix->kind != MGL_EXPR_LITERAL)
                return false;
            pieces.push_back(
                {Piece::Index, "", (int64_t)ix->u.literal.value});
            cur = cur->u.index.object;
            continue;
        }
        if (cur->kind == MGL_EXPR_VAR_REF) {
            if (!cur->u.var_ref.name)
                return false;
            std::string path = cur->u.var_ref.name;
            for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
                if (it->kind == Piece::Field)
                    path += "." + it->field;
                else
                    path += "[" + std::to_string(it->index) + "]";
            }
            *out = std::move(path);
            return true;
        }
        return false;
    }
    return false;
}

} /* namespace air */
} /* namespace mgl */
