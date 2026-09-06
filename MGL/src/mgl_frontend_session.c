/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * FrontendSession: legacy rewrite with a growing buffer, then one parse+sema.
 */

#include "mgl_frontend_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_glsl_parser.h"
#include "mgl_legacy_compat.h"
#include "mgl_shader_abi.h"
#include "glcorearb.h"

static GLuint mglFrontendStageToGLShaderType(int air_stage)
{
    switch (air_stage) {
    case MGL_STAGE_VERTEX:
        return GL_VERTEX_SHADER;
    case MGL_STAGE_FRAGMENT:
        return GL_FRAGMENT_SHADER;
    case MGL_STAGE_TESS_CONTROL:
        return GL_TESS_CONTROL_SHADER;
    case MGL_STAGE_TESS_EVALUATION:
        return GL_TESS_EVALUATION_SHADER;
    case MGL_STAGE_GEOMETRY:
        return GL_GEOMETRY_SHADER;
    case MGL_STAGE_COMPUTE:
        return GL_COMPUTE_SHADER;
    default:
        return 0;
    }
}

static int mglFrontendGLSLVersionOf(const char *src)
{
    if (!src)
        return 110;
    const char *v = strstr(src, "#version");
    if (!v)
        return 110;
    int ver = 0;
    char prof[32] = {0};
    if (sscanf(v + 8, "%d %31s", &ver, prof) >= 1 && ver > 0)
        return ver;
    return 110;
}

/* 0 = no rewrite (out NULL), 1 = *out owned rewritten source, -1 = fail. */
int mglFrontendRewriteLegacy(const char *src, int air_stage,
                                      char **out, char *err, size_t err_cap)
{
    if (out)
        *out = NULL;
    if (!src || !out)
        return 0;
    if (strstr(src, "/* MGL legacy GLSL translation: renamed builtins declared as"))
        return 0;

    mgl_legacy_features_t features;
    memset(&features, 0, sizeof(features));
    mgl_legacy_detect(src, &features);
    if (!features.needs_translation)
        return 0;

    const GLuint shader_type = mglFrontendStageToGLShaderType(air_stage);
    const int version = mglFrontendGLSLVersionOf(src);
    const size_t len = strlen(src);
    size_t cap = len + 4096u;
    const size_t cap_max = (size_t)16u * 1024u * 1024u;
    char *stable = NULL;

    for (;;) {
        char *buf = (char *)malloc(cap);
        if (!buf) {
            free(stable);
            if (err && err_cap)
                snprintf(err, err_cap, "legacy GLSL translation: out of memory");
            return -1;
        }
        memcpy(buf, src, len + 1u);
        int ret = mgl_translate_legacy_glsl(buf, cap, shader_type, version,
                                            &features);
        if (ret < 0) {
            free(buf);
            free(stable);
            if (err && err_cap)
                snprintf(err, err_cap, "legacy GLSL translation failed");
            return -1;
        }
        if (ret == 0) {
            free(buf);
            free(stable);
            return 0;
        }
        if (stable && strcmp(stable, buf) == 0) {
            free(buf);
            *out = stable;
            return 1;
        }
        free(stable);
        stable = buf;
        if (cap >= cap_max) {
            free(stable);
            if (err && err_cap)
                snprintf(err, err_cap,
                         "legacy GLSL translation overflow (source %zu bytes)",
                         len);
            return -1;
        }
        cap *= 2u;
        if (cap > cap_max)
            cap = cap_max;
    }
}

void mglFrontendSessionInit(MGLFrontendSession *s)
{
    if (!s)
        return;
    memset(s, 0, sizeof(*s));
}

void mglFrontendSessionDestroy(MGLFrontendSession *s)
{
    if (!s)
        return;
    mglIRModuleDestroy(&s->mod);
    mglGLSLTranslationUnitDestroy(s->tu);
    s->tu = NULL;
    free(s->legacy_src);
    s->legacy_src = NULL;
    s->src = NULL;
    s->ready = 0;
    s->stage = 0;
}

MGLTranslationUnit *mglFrontendSessionStealTU(MGLFrontendSession *s)
{
    MGLTranslationUnit *tu;
    if (!s)
        return NULL;
    tu = s->tu;
    s->tu = NULL;
    return tu;
}

uint32_t mglFrontendIRBuiltinArrayCount(const MGLIRModule *mod,
                                        const char *name)
{
    if (!mod || !name)
        return 0u;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        const MGLIRSymbol *sym = mod->symbols[i];
        if (!sym || !sym->name || sym->is_function)
            continue;
        if (strcmp(sym->name, name) != 0)
            continue;
        if (sym->type && sym->type->kind == MGLIR_TYPE_ARRAY)
            return sym->type->array_size > 0u ? sym->type->array_size : 8u;
        return 8u;
    }
    return 0u;
}

static void mglFrontendNoteBuiltinUse(const MGLExpr *e, const char *name,
                                      int *used, uint32_t *max_need);

static void mglFrontendNoteBuiltinStmt(const MGLStmt *st, const char *name,
                                       int *used, uint32_t *max_need);

static void mglFrontendNoteBuiltinDecl(const MGLDecl *d, const char *name,
                                       int *used, uint32_t *max_need)
{
    for (; d; d = d->next_declarator) {
        if (d->name && strcmp(d->name, name) == 0) {
            *used = 1;
            if (d->array_count > 0u && d->array_dims && d->array_dims[0] > *max_need)
                *max_need = d->array_dims[0];
        }
        if (d->init)
            mglFrontendNoteBuiltinUse(d->init, name, used, max_need);
        if (d->body)
            mglFrontendNoteBuiltinStmt(d->body, name, used, max_need);
        for (uint32_t i = 0; i < d->struct_member_count; i++)
            mglFrontendNoteBuiltinDecl(d->struct_members[i], name, used,
                                       max_need);
        for (uint32_t i = 0; i < d->param_count; i++)
            mglFrontendNoteBuiltinDecl(d->params[i], name, used, max_need);
    }
}

static void mglFrontendNoteBuiltinUse(const MGLExpr *e, const char *name,
                                      int *used, uint32_t *max_need)
{
    if (!e)
        return;
    switch (e->kind) {
    case MGL_EXPR_VAR_REF:
        if (e->u.var_ref.name && strcmp(e->u.var_ref.name, name) == 0)
            *used = 1;
        break;
    case MGL_EXPR_MEMBER:
        mglFrontendNoteBuiltinUse(e->u.member.object, name, used, max_need);
        if (e->u.member.field && strcmp(e->u.member.field, name) == 0)
            *used = 1;
        break;
    case MGL_EXPR_INDEX:
        if (e->u.index.object &&
            e->u.index.object->kind == MGL_EXPR_VAR_REF &&
            e->u.index.object->u.var_ref.name &&
            strcmp(e->u.index.object->u.var_ref.name, name) == 0) {
            *used = 1;
            if (e->u.index.index &&
                e->u.index.index->kind == MGL_EXPR_LITERAL) {
                uint32_t need =
                    (uint32_t)e->u.index.index->u.literal.value + 1u;
                if (need > *max_need)
                    *max_need = need;
            }
        } else if (e->u.index.object &&
                   e->u.index.object->kind == MGL_EXPR_MEMBER &&
                   e->u.index.object->u.member.field &&
                   strcmp(e->u.index.object->u.member.field, name) == 0) {
            *used = 1;
            if (e->u.index.index &&
                e->u.index.index->kind == MGL_EXPR_LITERAL) {
                uint32_t need =
                    (uint32_t)e->u.index.index->u.literal.value + 1u;
                if (need > *max_need)
                    *max_need = need;
            }
        }
        mglFrontendNoteBuiltinUse(e->u.index.object, name, used, max_need);
        mglFrontendNoteBuiltinUse(e->u.index.index, name, used, max_need);
        break;
    case MGL_EXPR_CALL:
        for (uint32_t i = 0; i < e->u.call.arg_count; i++)
            mglFrontendNoteBuiltinUse(e->u.call.args[i], name, used, max_need);
        break;
    case MGL_EXPR_UNARY:
        mglFrontendNoteBuiltinUse(e->u.unary.operand, name, used, max_need);
        break;
    case MGL_EXPR_BINARY:
        mglFrontendNoteBuiltinUse(e->u.binary.lhs, name, used, max_need);
        mglFrontendNoteBuiltinUse(e->u.binary.rhs, name, used, max_need);
        break;
    case MGL_EXPR_ASSIGN:
        mglFrontendNoteBuiltinUse(e->u.assign.lhs, name, used, max_need);
        mglFrontendNoteBuiltinUse(e->u.assign.rhs, name, used, max_need);
        break;
    case MGL_EXPR_TERNARY:
        mglFrontendNoteBuiltinUse(e->u.ternary.cond, name, used, max_need);
        mglFrontendNoteBuiltinUse(e->u.ternary.then, name, used, max_need);
        mglFrontendNoteBuiltinUse(e->u.ternary.else_, name, used, max_need);
        break;
    case MGL_EXPR_INIT_LIST:
        for (uint32_t i = 0; i < e->u.init_list.arg_count; i++)
            mglFrontendNoteBuiltinUse(e->u.init_list.args[i], name, used,
                                      max_need);
        break;
    default:
        break;
    }
}

static void mglFrontendNoteBuiltinStmt(const MGLStmt *st, const char *name,
                                       int *used, uint32_t *max_need)
{
    if (!st)
        return;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            mglFrontendNoteBuiltinStmt(st->u.compound.stmts[i], name, used,
                                       max_need);
        break;
    case MGL_STMT_EXPR:
        mglFrontendNoteBuiltinUse(st->u.expr.expr, name, used, max_need);
        break;
    case MGL_STMT_DECL:
        mglFrontendNoteBuiltinDecl(st->u.decl.decl, name, used, max_need);
        break;
    case MGL_STMT_IF:
        mglFrontendNoteBuiltinUse(st->u.ifs.cond, name, used, max_need);
        mglFrontendNoteBuiltinStmt(st->u.ifs.then, name, used, max_need);
        mglFrontendNoteBuiltinStmt(st->u.ifs.else_, name, used, max_need);
        break;
    case MGL_STMT_FOR:
        mglFrontendNoteBuiltinStmt(st->u.loop.init, name, used, max_need);
        mglFrontendNoteBuiltinUse(st->u.loop.cond, name, used, max_need);
        mglFrontendNoteBuiltinUse(st->u.loop.incr, name, used, max_need);
        mglFrontendNoteBuiltinStmt(st->u.loop.body, name, used, max_need);
        break;
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        mglFrontendNoteBuiltinUse(st->u.whilex.cond, name, used, max_need);
        mglFrontendNoteBuiltinStmt(st->u.whilex.body, name, used, max_need);
        break;
    case MGL_STMT_SWITCH:
        mglFrontendNoteBuiltinUse(st->u.switchx.cond, name, used, max_need);
        mglFrontendNoteBuiltinStmt(st->u.switchx.body, name, used, max_need);
        break;
    case MGL_STMT_CASE:
        mglFrontendNoteBuiltinUse(st->u.casex.value, name, used, max_need);
        break;
    case MGL_STMT_RETURN:
        mglFrontendNoteBuiltinUse(st->u.ret.value, name, used, max_need);
        break;
    default:
        break;
    }
}

uint32_t mglFrontendBuiltinArrayCount(const MGLIRModule *mod,
                                      const MGLTranslationUnit *tu,
                                      const char *name)
{
    uint32_t ir = mglFrontendIRBuiltinArrayCount(mod, name);
    if (ir > 0u)
        return ir;
    if (!tu || !name)
        return 0u;
    int used = 0;
    uint32_t need = 0u;
    for (uint32_t i = 0; i < tu->decl_count; i++)
        mglFrontendNoteBuiltinDecl(tu->decls[i], name, &used, &need);
    if (!used)
        return 0u;
    return need > 0u ? need : 8u;
}

int mglFrontendSessionBuild(MGLFrontendSession *s, const char *src, int stage,
                            char *err, size_t err_cap)
{
    if (!s || !src) {
        if (err && err_cap)
            snprintf(err, err_cap, "FrontendSession: bad args");
        return -1;
    }
    mglFrontendSessionDestroy(s);
    mglFrontendSessionInit(s);
    s->stage = stage;

    char *translated = NULL;
    if (mglFrontendRewriteLegacy(src, stage, &translated, err, err_cap) < 0)
        return -1;
    s->legacy_src = translated;
    s->src = translated ? translated : src;

    s->tu = mglGLSLParse(s->src, strlen(s->src));
    if (!s->tu) {
        if (err && err_cap)
            snprintf(err, err_cap, "parse: out of memory");
        mglFrontendSessionDestroy(s);
        return -1;
    }
    if (s->tu->error) {
        if (err && err_cap)
            snprintf(err, err_cap, "parse line %u: %s", s->tu->error_line,
                     s->tu->error);
        mglFrontendSessionDestroy(s);
        return -1;
    }

    MGLSemaError *errors = NULL;
    uint32_t error_count = 0;
    int hard = mglGLSLSemanticCheck(s->tu, stage, &s->mod, &errors,
                                    &error_count);
    if (hard) {
        if (err && err_cap && errors && error_count)
            snprintf(err, err_cap, "line %u: %s", errors[0].line,
                     errors[0].message);
        mglGLSLSemanticCheckDestroy(errors, error_count);
        mglFrontendSessionDestroy(s);
        return -1;
    }
    mglGLSLSemanticCheckDestroy(errors, error_count);
    s->ready = 1;
    return 0;
}
