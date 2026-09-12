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
        /* Builtin *functions* (interpolateAtSample / interpolateAtOffset) are
         * called, not declared, so the callee name is the usage signal. */
        if (e->u.call.name && strcmp(e->u.call.name, name) == 0) {
            *used = 1;
            if (*max_need == 0u)
                *max_need = 1u;
        }
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

/* Recursive `sample`-qualifier search over the AST declarations. */
static int mglFrontendDeclSampleQualified(const MGLDecl *d)
{
    for (; d; d = d->next_declarator) {
        /* `sample` is only legal on interface (in/out) declarations, so the
         * declaration-level qualifier is the whole answer -- no statement
         * walk needed. */
        if (d->qualifiers & MGL_AST_Q_SAMPLE)
            return 1;
        for (uint32_t i = 0; i < d->struct_member_count; i++)
            if (mglFrontendDeclSampleQualified(d->struct_members[i]))
                return 1;
        for (uint32_t i = 0; i < d->param_count; i++)
            if (mglFrontendDeclSampleQualified(d->params[i]))
                return 1;
    }
    return 0;
}

int mglFrontendStageUsesSampleInterpolation(const MGLIRModule *mod,
                                            const MGLTranslationUnit *tu)
{
    if (mod) {
        for (uint32_t i = 0; i < mod->symbol_count; i++) {
            const MGLIRSymbol *sym = mod->symbols[i];
            if (sym && !sym->is_function &&
                (sym->qualifiers & MGL_AST_Q_SAMPLE))
                return 1;
        }
    }
    if (!tu)
        return 0;
    for (uint32_t i = 0; i < tu->decl_count; i++)
        if (mglFrontendDeclSampleQualified(tu->decls[i]))
            return 1;
    return 0;
}

/* ---- Reference query over a stage's declaration bodies ---- */
/* Answers "does this stage's code reference <name> / <instance>.<member>?"
 * from the parsed TU instead of scanning the GLSL text.  Declarations never
 * count (the queries ask whether the stage USES the resource) and every
 * function body is walked, so a helper called from the entry point counts.
 * The retired source-text scan had to cut the source at the first "void main"
 * (a helper defined above main was invisible) and had to special-case comments
 * and string literals, which could fake a reference; it also matched only the
 * LEAF of a member name ("d[0]"), so an unrelated object with a member of the
 * same name counted as a use.
 *
 * Both sides are compared component-wise:
 *
 *   component := name ( '[' index ']' )*      index := literal | '?'
 *
 * The AST flattens to components (VAR_REF/MEMBER add a name, INDEX appends an
 * index to the last component) and a reflected query name ("colors[0]",
 * "a[0].b[0].d[0]") parses to the same shape.  A query matches an access when
 * its components appear as an aligned contiguous run, so the access may select
 * deeper (query "colors[0]" vs access "colors[0].rgb"), the query may omit the
 * trailing index (query "colors" vs access "colors[0]"), a dynamic index ('?',
 * from a non-literal subscript) is a wildcard, while two literal indices must
 * agree - "colors[1]" never matches an access to colors[0]. */

#define MGL_FRONTEND_REF_MAX_COMPONENTS 16u
#define MGL_FRONTEND_REF_MAX_INDICES 8u
#define MGL_FRONTEND_REF_NAME_CAP 64u

typedef struct MGLFrontendRefComponent {
    char name[MGL_FRONTEND_REF_NAME_CAP];
    uint32_t index_count;
    long long indices[MGL_FRONTEND_REF_MAX_INDICES]; /* -1 == '?' (dynamic) */
} MGLFrontendRefComponent;

typedef struct MGLFrontendRefPath {
    uint32_t count;
    MGLFrontendRefComponent comps[MGL_FRONTEND_REF_MAX_COMPONENTS];
} MGLFrontendRefPath;

typedef struct MGLFrontendRefQuery {
    const MGLFrontendRefPath *path; /* components to look for, in order */
    int hit;
} MGLFrontendRefQuery;

/* Parse a dotted, subscripted name into path components.  With `append` the
 * parsed components extend what `out` already holds (building "<instance>.<member>");
 * on any malformed input `out` is left with the component count it had. */
static int mglFrontendRefParseInto(const char *s, MGLFrontendRefPath *out,
                                   int append)
{
    const uint32_t base = append ? out->count : 0u;
    uint32_t count = base;
    if (!s || !s[0])
        return 0;
    for (;;) {
        const char *start = s;
        while (*s && *s != '.' && *s != '[')
            s++;
        const size_t n = (size_t)(s - start);
        if (n == 0u || n + 1u > MGL_FRONTEND_REF_NAME_CAP ||
            count == MGL_FRONTEND_REF_MAX_COMPONENTS)
            goto fail;
        MGLFrontendRefComponent *c = &out->comps[count++];
        memcpy(c->name, start, n);
        c->name[n] = '\0';
        c->index_count = 0u;
        while (*s == '[') {
            s++;
            long long v = -1;
            if (*s == '?') {
                s++;
            } else {
                int neg = 0;
                if (*s == '-') {
                    neg = 1;
                    s++;
                }
                if (*s < '0' || *s > '9')
                    goto fail;
                long long acc = 0;
                while (*s >= '0' && *s <= '9')
                    acc = acc * 10 + (*s++ - '0');
                v = neg ? -acc : acc;
            }
            if (*s != ']' || c->index_count == MGL_FRONTEND_REF_MAX_INDICES)
                goto fail;
            s++;
            c->indices[c->index_count++] = v;
        }
        if (*s == '.') {
            s++;
            continue;
        }
        if (*s == '\0') {
            out->count = count;
            return 1;
        }
        goto fail;
    }
fail:
    out->count = base;
    return 0;
}

/* Flatten an expression into its access path; 0 when it is not a plain path
 * (a call result, a literal, ...). */
static int mglFrontendRefFlatten(const MGLExpr *e, MGLFrontendRefPath *out)
{
    if (!e)
        return 0;
    switch (e->kind) {
    case MGL_EXPR_VAR_REF:
        /* The frontend may keep a block-instance member access as one
         * qualified symbol name, so the name text is parsed into components
         * too; a path always starts at a variable reference. */
        return (out->count == 0u)
                   ? mglFrontendRefParseInto(e->u.var_ref.name, out, 1)
                   : 0;
    case MGL_EXPR_MEMBER:
        return mglFrontendRefFlatten(e->u.member.object, out) &&
               mglFrontendRefParseInto(e->u.member.field, out, 1);
    case MGL_EXPR_INDEX: {
        if (!mglFrontendRefFlatten(e->u.index.object, out) || out->count == 0u)
            return 0;
        MGLFrontendRefComponent *c = &out->comps[out->count - 1u];
        const MGLExpr *ix = e->u.index.index;
        long long v = -1;
        if (ix && ix->kind == MGL_EXPR_LITERAL &&
            (ix->u.literal.base == MGL_AST_TYPE_INT ||
             ix->u.literal.base == MGL_AST_TYPE_UINT))
            v = (long long)ix->u.literal.value;
        if (c->index_count == MGL_FRONTEND_REF_MAX_INDICES)
            return 0;
        c->indices[c->index_count++] = v;
        return 1;
    }
    default:
        return 0;
    }
}

static int mglFrontendRefComponentMatches(const MGLFrontendRefComponent *q,
                                          const MGLFrontendRefComponent *p)
{
    if (strcmp(q->name, p->name) != 0)
        return 0;
    for (uint32_t i = 0; i < q->index_count; i++) {
        if (i >= p->index_count)
            return 0; /* the access is indexed shallower than the query: the
                       * path lost an index (the walker does not match the
                       * unindexed prefix of an indexed access), so a query for
                       * one element must not answer for its siblings */
        const long long qi = q->indices[i];
        const long long pi = p->indices[i];
        if (qi >= 0 && pi >= 0 && qi != pi)
            return 0;
    }
    return 1;
}

/* True when the query's components appear in the access path as an aligned
 * contiguous run. */
static int mglFrontendRefPathMatches(const MGLFrontendRefPath *path,
                                     const MGLFrontendRefPath *query)
{
    if (path->count == 0u || query->count == 0u ||
        query->count > path->count)
        return 0;
    for (uint32_t s = 0; s + query->count <= path->count; s++) {
        uint32_t i = 0;
        for (; i < query->count; i++) {
            if (!mglFrontendRefComponentMatches(&query->comps[i],
                                                &path->comps[s + i]))
                break;
        }
        if (i == query->count)
            return 1;
    }
    return 0;
}

/* Walk every expression of a body.  `match_self` is 0 for a path that dropped
 * an index on the way up (an indexed object seen from its parent), where the
 * prefix path must not be matched against the query. */
static void mglFrontendRefWalkExpr(const MGLExpr *e, MGLFrontendRefQuery *q,
                                   int match_self)
{
    if (!e || q->hit)
        return;
    if (match_self) {
        MGLFrontendRefPath path;
        path.count = 0u;
        if (mglFrontendRefFlatten(e, &path) &&
            mglFrontendRefPathMatches(&path, q->path))
            q->hit = 1;
    }
    switch (e->kind) {
    case MGL_EXPR_MEMBER:
    case MGL_EXPR_VAR_REF:
        /* The object chain keeps the parent's suppression state. */
        mglFrontendRefWalkExpr(e->kind == MGL_EXPR_MEMBER ? e->u.member.object
                                                          : NULL,
                               q, match_self);
        break;
    case MGL_EXPR_INDEX:
        /* The indexed object's own path is a prefix that dropped this index
         * ("blk.colors" under "blk.colors[0]"): it must not match on its own,
         * or a query for one element would answer for its siblings.  It is
         * still walked for expressions of its own (its subscripts, calls). */
        mglFrontendRefWalkExpr(e->u.index.object, q, 0);
        mglFrontendRefWalkExpr(e->u.index.index, q, 1);
        break;
    case MGL_EXPR_CALL:
        for (uint32_t i = 0; i < e->u.call.arg_count; i++)
            mglFrontendRefWalkExpr(e->u.call.args[i], q, 1);
        break;
    case MGL_EXPR_UNARY:
        mglFrontendRefWalkExpr(e->u.unary.operand, q, 1);
        break;
    case MGL_EXPR_BINARY:
        mglFrontendRefWalkExpr(e->u.binary.lhs, q, 1);
        mglFrontendRefWalkExpr(e->u.binary.rhs, q, 1);
        break;
    case MGL_EXPR_ASSIGN:
        mglFrontendRefWalkExpr(e->u.assign.lhs, q, 1);
        mglFrontendRefWalkExpr(e->u.assign.rhs, q, 1);
        break;
    case MGL_EXPR_TERNARY:
        mglFrontendRefWalkExpr(e->u.ternary.cond, q, 1);
        mglFrontendRefWalkExpr(e->u.ternary.then, q, 1);
        mglFrontendRefWalkExpr(e->u.ternary.else_, q, 1);
        break;
    case MGL_EXPR_INIT_LIST:
        for (uint32_t i = 0; i < e->u.init_list.arg_count; i++)
            mglFrontendRefWalkExpr(e->u.init_list.args[i], q, 1);
        break;
    default:
        break;
    }
}

static void mglFrontendRefWalkStmt(const MGLStmt *st, MGLFrontendRefQuery *q)
{
    if (!st || q->hit)
        return;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            mglFrontendRefWalkStmt(st->u.compound.stmts[i], q);
        break;
    case MGL_STMT_EXPR:
        mglFrontendRefWalkExpr(st->u.expr.expr, q, 1);
        break;
    case MGL_STMT_DECL:
        if (st->u.decl.decl && st->u.decl.decl->init)
            mglFrontendRefWalkExpr(st->u.decl.decl->init, q, 1);
        break;
    case MGL_STMT_IF:
        mglFrontendRefWalkExpr(st->u.ifs.cond, q, 1);
        mglFrontendRefWalkStmt(st->u.ifs.then, q);
        mglFrontendRefWalkStmt(st->u.ifs.else_, q);
        break;
    case MGL_STMT_FOR:
        mglFrontendRefWalkStmt(st->u.loop.init, q);
        mglFrontendRefWalkExpr(st->u.loop.cond, q, 1);
        mglFrontendRefWalkExpr(st->u.loop.incr, q, 1);
        mglFrontendRefWalkStmt(st->u.loop.body, q);
        break;
    case MGL_STMT_WHILE:
        mglFrontendRefWalkExpr(st->u.whilex.cond, q, 1);
        mglFrontendRefWalkStmt(st->u.whilex.body, q);
        break;
    case MGL_STMT_DO_WHILE:
        mglFrontendRefWalkStmt(st->u.body.body, q);
        mglFrontendRefWalkExpr(st->u.whilex.cond, q, 1);
        break;
    case MGL_STMT_SWITCH:
        mglFrontendRefWalkExpr(st->u.switchx.cond, q, 1);
        mglFrontendRefWalkStmt(st->u.switchx.body, q);
        break;
    case MGL_STMT_CASE:
        mglFrontendRefWalkExpr(st->u.casex.value, q, 1);
        break;
    case MGL_STMT_RETURN:
        mglFrontendRefWalkExpr(st->u.ret.value, q, 1);
        break;
    default:
        break;
    }
}

static int mglFrontendStageHasPath(const MGLTranslationUnit *tu,
                                   const MGLFrontendRefPath *query)
{
    if (!tu || query->count == 0u)
        return 0;
    MGLFrontendRefQuery q;
    q.path = query;
    q.hit = 0;
    for (uint32_t i = 0; i < tu->decl_count && !q.hit; i++) {
        for (const MGLDecl *d = tu->decls[i]; d && !q.hit;
             d = d->next_declarator)
            mglFrontendRefWalkStmt(d->body, &q);
    }
    return q.hit ? 1 : 0;
}

int mglFrontendStageReferencesName(const MGLTranslationUnit *tu,
                                   const char *name)
{
    MGLFrontendRefPath query;
    query.count = 0u;
    if (!tu || !name || !name[0] ||
        !mglFrontendRefParseInto(name, &query, 0))
        return 0;
    return mglFrontendStageHasPath(tu, &query);
}

int mglFrontendStageReferencesMember(const MGLTranslationUnit *tu,
                                     const char *instance,
                                     const char *member)
{
    MGLFrontendRefPath query;
    query.count = 0u;
    if (!tu || !member || !member[0])
        return 0;
    /* The query path is "<instance>.<member>"; the instance is optional (an
     * unnamed block interface leaves only the member name to match on). */
    if (instance && instance[0] &&
        !mglFrontendRefParseInto(instance, &query, 0))
        return 0;
    if (!mglFrontendRefParseInto(member, &query, 1))
        return 0;
    return mglFrontendStageHasPath(tu, &query);
}

int mglFrontendBuiltinUsed(const MGLIRModule *mod,
                           const MGLTranslationUnit *tu, const char *name)
{
    return mglFrontendBuiltinArrayCount(mod, tu, name) > 0u ? 1 : 0;
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
